# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-contained ATK executor for aclnnMlaPrologV3WeightNz.

Inlines golden / non-contiguous cache helpers so this file can be used as the
only ATK plugin.  Tests precision (CPU + NPU small-op double benchmark) and
kvCache/krCache dim0 non-contiguous stride.  cacheMode covers
BSND/TND/PA_BSND/PA_NZ/PA_BLK_BSND/PA_BLK_NZ; tokenX covers COMBINE (T,He)
and unmerged (B,S,He) where the ABI allows it.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import re
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import torch_npu

    _HAS_TORCH_NPU = True
except ImportError:
    torch_npu = None
    _HAS_TORCH_NPU = False

from atk.common.connection import RemoteManager
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface import acl_wrapper as _acl_wrapper
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclDataType,
    AclFormat,
    AclTensor,
    AclTensorStruct,
    NNOPBASE_PATH,
    TORCH_TO_ACLTYPE,
    nnopbase,
)

_acl_wrapper.TORCH_TO_ACLTYPE["torch.float8_e8m0fnu"] = int(AclDataType.ACL_FLOAT8_E8M0)
if hasattr(_acl_wrapper, "ACLTYPE_TO_CTYPE"):
    _acl_wrapper.ACLTYPE_TO_CTYPE.setdefault(
        int(AclDataType.ACL_FLOAT8_E8M0), ctypes.c_ubyte
    )
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.double_benchmark_compare import DoubleBenchmarkAccuracyCompare


# ---------------------------------------------------------------------------
# Non-contiguous cache helpers (inlined)
# ---------------------------------------------------------------------------

PADDING_SENTINEL = 3.0

_NC_NAME_RE = re.compile(r"\bnc(\d+)\b")


def parse_stride_factor(case_name: str) -> int:
    """从用例名解析首轴 stride 放大倍数，``nc1``/无标记表示连续。

    环境变量 MLA_PROLOG_NC_FACTOR 可整体覆盖，便于同一批用例跑连续/非连续两轮。
    """
    override = os.environ.get("MLA_PROLOG_NC_FACTOR")
    if override:
        return max(1, int(override))
    match = _NC_NAME_RE.search(case_name or "")
    return max(1, int(match.group(1))) if match else 1


@dataclass
class NonContigCache:
    """一个 cache 的非连续视图及其 backing。"""

    view: torch.Tensor
    backing: torch.Tensor
    factor: int

    @property
    def padding_rows(self) -> torch.Tensor:
        """backing 中不属于逻辑视图的行（拉回 CPU，避免 NPU 上 fp8 不支持 index）。

        PyTorch CPU 同样不支持 float8_e4m3fn 的 index 操作，
        借 uint8（同为 1 字节存储）做中转后再 view 回原 dtype，保证比特一致。
        """
        backing = self.backing.detach().cpu()
        keep = [i for i in range(backing.shape[0]) if i % self.factor != 0]
        if backing.dtype == torch.float8_e4m3fn and keep:
            return backing.view(torch.uint8)[keep].view(torch.float8_e4m3fn)
        return backing[keep]

    def padding_is_intact(self) -> bool:
        rows = self.padding_rows
        if rows.numel() == 0:
            return True
        if rows.dtype == torch.float8_e4m3fn:
            sentinel = (
                torch.tensor(PADDING_SENTINEL, dtype=torch.float32)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .item()
            )
            return bool(torch.all(rows.view(torch.uint8) == sentinel))
        return bool(torch.all(rows.float() == PADDING_SENTINEL))


def make_dim0_noncontig(tensor: torch.Tensor, factor: int) -> NonContigCache:
    """把 tensor 放进一个首轴放大 factor 倍的 backing，返回只有 dim0 非连续的视图。

    视图 stride = (factor * 连续stride0, 其余连续 stride)，逻辑行落在 backing 的
    第 0, factor, 2*factor ... 行，行间空洞填 sentinel。

    构造一律在 CPU 上做：NPU 上对 float8 的 as_strided/copy_ 会走 index_high_dims
    并触发 161002。
    """
    if factor <= 1:
        raise ValueError(f"stride factor must be > 1, got {factor}")
    if tensor.numel() == 0:
        return NonContigCache(view=tensor, backing=tensor, factor=1)

    shape = tuple(tensor.shape)
    rest = shape[1:]
    inner = 1
    for dim in rest:
        inner *= dim

    device = tensor.device
    src = tensor.detach().cpu()
    backing = torch.full(
        (shape[0] * factor,) + rest, PADDING_SENTINEL, dtype=torch.float32
    ).to(dtype=src.dtype)
    strides = (factor * inner,) + tuple(backing.stride()[1:])
    view = backing.as_strided(shape, strides)
    view.copy_(src)
    if view.is_contiguous():
        raise RuntimeError("failed to build a dim0 non-contiguous view")

    if device.type != "cpu":
        backing = backing.to(device)
        view = backing.as_strided(shape, strides)
    return NonContigCache(view=view, backing=backing, factor=factor)


def describe(caches: Dict[str, Optional[NonContigCache]]) -> str:
    parts = []
    for name, cache in caches.items():
        if cache is None:
            parts.append(f"{name}=contiguous")
        else:
            parts.append(
                f"{name}=stride0:{cache.view.stride(0)}(x{cache.factor}) "
                f"backing_dim0:{cache.backing.shape[0]}"
            )
    return " ".join(parts)


def padding_report(caches: Dict[str, Optional[NonContigCache]]) -> Tuple[bool, str]:
    ok = True
    details = []
    for name, cache in caches.items():
        if cache is None:
            details.append(f"{name}:n/a")
            continue
        intact = cache.padding_is_intact()
        ok = ok and intact
        details.append(f"{name}:{'intact' if intact else 'POLLUTED'}")
    return ok, " ".join(details)


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------

MXFP8_GRP = 32
FP8_E4M3_MAX = 448.0
E4M3_EMAX = 8
FRACTAL_NZ_FORMAT = 29
# Cube MMA C0 / BLOCK_CUBE_SIZE. fused MatmulCkvKr keeps L0C in fp32 and reduces
# K in 16-wide tiles before a single F322BF16 FixPipe.
BLOCK_CUBE_K = 16


@dataclass
class CaseConfig:
    t: int = 8
    he: int = 7168
    hcq: int = 1536
    hckv: int = 512
    d: int = 128
    dr: int = 64
    n: int = 8
    nkv: int = 1
    block_num: int = 2
    block_size: int = 128
    eps_cq: float = 1e-5
    eps_ckv: float = 1e-5
    qc_qr_scale: float = 1.0
    kc_scale: float = 1.0
    do_rope: bool = True
    rope_style: str = "original"
    seed: int = 2025
    weight_quant_mode: int = 0
    kv_cache_quant_mode: int = 0
    query_quant_mode: int = 0
    ckvkr_repo_mode: int = 0
    quant_scale_repo_mode: int = 0
    tile_size: int = 128
    query_norm_flag: bool = False
    cache_mode: str = "PA_BSND"
    b: int = 1
    unmerged: bool = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def interleave_to_half(x: torch.Tensor) -> torch.Tensor:
    dr = x.shape[-1]
    return x.reshape(*x.shape[:-1], dr // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], dr)


def apply_rope_original(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    q = interleave_to_half(x)
    if q.dim() == 3:
        cos_e = cos.unsqueeze(1).expand_as(q)
        sin_e = sin.unsqueeze(1).expand_as(q)
        return (q * cos_e) + (rotate_half(q) * sin_e)
    return (q * cos) + (rotate_half(q) * sin)


def apply_rope_vf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Match arch35 vf_rope.h RopeVFImpl: same as the original layout except the low half adds
    sin*odd instead of subtracting it."""
    even, odd = x[..., ::2], x[..., 1::2]
    half = x.shape[-1] // 2
    cos_l, cos_u = cos[..., :half], cos[..., half:]
    sin_l, sin_u = sin[..., :half], sin[..., half:]
    if x.dim() == 3:
        cos_l, cos_u = cos_l.unsqueeze(1).expand_as(even), cos_u.unsqueeze(1).expand_as(even)
        sin_l, sin_u = sin_l.unsqueeze(1).expand_as(even), sin_u.unsqueeze(1).expand_as(even)
    return torch.cat([cos_l * even + sin_l * odd, sin_u * even + cos_u * odd], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, do_rope: bool,
               rope_style: str = "original") -> torch.Tensor:
    x = x.to(torch.float32)
    if not do_rope:
        return x
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    if rope_style == "vf":
        return apply_rope_vf(x, cos, sin)
    return apply_rope_original(x, cos, sin)


def matmul_l0c_splitk(a: torch.Tensor, b: torch.Tensor, k_tile: int = BLOCK_CUBE_K) -> torch.Tensor:
    """Emulate fused MatmulCkvKr L0C: sequential fp32 sum of K-tiles, then caller F322BF16.

    One-shot ND ``A@B`` and Cube ``torch.matmul(bf16, NZ)`` both reduce K in one shot.
    fused ``MatmulSplitK`` accumulates 16-wide MMA tiles in L0C fp32 and FixPipes once.
    On case 42 that changes Kr dim8 from ``-64.249977`` (RNE ``-64.0``) to
    ``-64.25003`` (RNE ``-64.5``), which is the 78 vs 78.5 on ``kr_cache``.
    """
    a32 = a.to(device="cpu", dtype=torch.float32).contiguous()
    b32 = b.to(device="cpu", dtype=torch.float32).contiguous()
    m, k = a32.shape[-2], a32.shape[-1]
    n = b32.shape[-1]
    if k_tile <= 0 or k <= k_tile:
        return torch.matmul(a32, b32)
    ntiles, rem = divmod(k, k_tile)
    acc = None
    if ntiles:
        a_b = a32[:, : ntiles * k_tile].reshape(m, ntiles, k_tile).permute(1, 0, 2).contiguous()
        b_b = b32[: ntiles * k_tile].reshape(ntiles, k_tile, n).contiguous()
        parts = torch.bmm(a_b, b_b)
        acc = parts[0]
        for i in range(1, ntiles):
            acc = acc + parts[i]
    if rem:
        tail = torch.matmul(a32[:, ntiles * k_tile :], b32[ntiles * k_tile :])
        acc = tail if acc is None else acc + tail
    return acc


def _maybe_dump_kr_rope(tag: str, payload: Dict[str, Any]) -> None:
    path = os.environ.get("MLA_PROLOG_DUMP_ROPE", "").strip()
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            blob = {}
    clean = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().cpu()
        else:
            clean[k] = v
    blob[tag] = clean
    torch.save(blob, path)
    logging.info("[mla_prolog] dumped kr-rope intermediates tag=%s -> %s", tag, path)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float, scale: float) -> torch.Tensor:
    y = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return y * gamma * scale


def s8_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -128, 127).to(torch.int8)


def s9_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -256, 255)


def quant_s8(x: torch.Tensor, qscale: torch.Tensor) -> torch.Tensor:
    scaled = (x.to(torch.float32) * qscale.to(torch.float32)).round()
    return s8_saturation(s9_saturation(scaled))


def _cast_f32_to_i8_kernel(v: torch.Tensor) -> torch.Tensor:
    """对齐 vf_dynamic_quant.h：f32 → f16(CAST_ODD) → i8(CAST_RINT)。

    PyTorch 无 CAST_ODD。先 round-to-nearest-even 到 f16 量级（近似 Odd/RNE 落点），
    再 CAST_RINT 到 int8。环境变量 MLA_PROLOG_DQ_CAST=rint|f16|rhz 可切换近似。
    """
    # 实测：纯 rint 对 query 绝对误差更小；f16 两级 cast 近似反而拉大与 Cube 的差。
    # 可用 MLA_PROLOG_DQ_CAST=f16|rint|rhz 切换。
    mode = os.environ.get("MLA_PROLOG_DQ_CAST", "rint").lower()
    if mode == "rhz":
        return (torch.ceil(v.abs() - 0.5) * torch.sign(v)).clamp(-128, 127)
    if mode == "f16":
        return v.to(torch.float16).to(torch.float32).round().clamp(-128, 127)
    return torch.round(v).clamp(-128, 127)


def dynamic_quant(inputs: torch.Tensor, smooth_scale: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quant，对齐 arch35 DynamicQuantPerTokenVf。

    scale = amax / 127；q = cast_i8(x / scale)；dequant 时再乘回 scale。
    """
    t, h = inputs.shape
    y = torch.zeros(t, h, dtype=torch.int32)
    scale = torch.zeros(t, 1, dtype=torch.float32)
    x = inputs.to(torch.float32)
    ss = None
    if smooth_scale is not None:
        ss = smooth_scale.to(torch.float32).reshape(1, h)
    for i in range(t):
        row = x[i] * ss[0] if ss is not None else x[i]
        amax = row.abs().max().clamp(min=1e-12)
        scale[i, 0] = amax / 127.0
        y[i] = _cast_f32_to_i8_kernel(row / scale[i, 0])
    return y, scale


def int8_matmul_i32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8×int8 → int32 累加，对齐 Cube L0C=int32（不做 dequant）。"""
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def quantize_weight_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    wf = w.to(torch.float32)
    scale = wf.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0
    wq = torch.round(wf / scale).clamp(-128, 127).to(torch.int8)
    return wq, scale.to(torch.float32)


def quantize_activation_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    xq = torch.round(xf / scale).clamp(-128, 127).to(torch.int8)
    return xq, scale.to(torch.float32)


def _e8m0_scale_from_amax(amax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = amax.to(torch.float32)
    tiny = 2.0 ** (-126)
    log2a = torch.floor(torch.log2(a + tiny * (a == 0).to(torch.float32)))
    share = (log2a - float(E4M3_EMAX) + 127.0).clamp(0.0, 254.0)
    share = torch.where(a == 0, torch.full_like(share, 127.0), share)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), share - 127.0)
    scale_e8m0 = torch.empty(share.shape, dtype=torch.float8_e8m0fnu)
    scale_e8m0.view(torch.uint8).copy_(share.to(torch.uint8))
    return scale, scale_e8m0


def quantize_mxfp8_along_last(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    *lead, h = xf.shape
    assert h % MXFP8_GRP == 0
    g = h // MXFP8_GRP
    blocks = xf.reshape(*lead, g, MXFP8_GRP)
    amax = blocks.abs().amax(dim=-1)
    scale_fp, scale_e8 = _e8m0_scale_from_amax(amax)
    q = (blocks / scale_fp.unsqueeze(-1).clamp(min=1e-30)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q8 = q.to(torch.float8_e4m3fn).reshape(*lead, h)
    return q8, scale_e8, scale_fp


def quantize_weight_mxfp8(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wf = w.to(torch.float32)
    k, n = wf.shape
    assert k % MXFP8_GRP == 0
    g = k // MXFP8_GRP
    blocks = wf.reshape(g, MXFP8_GRP, n)
    amax = blocks.abs().amax(dim=1)
    scale_fp, scale_e8 = _e8m0_scale_from_amax(amax)
    scale_fp_out = scale_fp.transpose(0, 1).contiguous()
    scale_e8_out = scale_e8.transpose(0, 1).contiguous()
    q = (blocks / scale_fp.unsqueeze(1).clamp(min=1e-30)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q8 = q.to(torch.float8_e4m3fn).reshape(k, n)
    return q8, scale_e8_out, scale_fp_out


def dequant_mxfp8_along_last(x_fp8: torch.Tensor, scale_fp: torch.Tensor) -> torch.Tensor:
    xf = x_fp8.to(torch.float32)
    *lead, h = xf.shape
    g = h // MXFP8_GRP
    blocks = xf.reshape(*lead, g, MXFP8_GRP)
    return (blocks * scale_fp.unsqueeze(-1)).reshape(*lead, h)


def dequant_weight_mxfp8(w_fp8: torch.Tensor, scale_fp: torch.Tensor) -> torch.Tensor:
    wf = w_fp8.to(torch.float32)
    k, n = wf.shape
    g = k // MXFP8_GRP
    blocks = wf.reshape(g, MXFP8_GRP, n)
    s = scale_fp.transpose(0, 1).unsqueeze(1)
    return (blocks * s).reshape(k, n)


def dynamic_mx_quant_dequant_cq(x: torch.Tensor) -> torch.Tensor:
    q8, _, scale_fp = quantize_mxfp8_along_last(x)
    return dequant_mxfp8_along_last(q8, scale_fp)


def quant_ckv_fp8_per_tensor(x: torch.Tensor, quant_scale: torch.Tensor) -> torch.Tensor:
    y = (x.to(torch.float32) * quant_scale.to(torch.float32).reshape(1, -1))
    y = torch.round(y * 1e8) / 1e8
    return y.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def dynamic_quant_q_nope_fp8(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qf = q.to(torch.float32)
    amax = qf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    y = (qf / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return y, scale.squeeze(-1).unsqueeze(-1)


def gen_rope_cos_sin(t: int, dr: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    half = dr // 2
    pos = torch.arange(t, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = pos * inv_freq.unsqueeze(0)
    cos_h = torch.cos(angles)
    sin_h = torch.sin(angles)
    cos = torch.cat([cos_h, cos_h], dim=-1).to(torch.bfloat16)
    sin = torch.cat([sin_h, sin_h], dim=-1).to(torch.bfloat16)
    return cos, sin


def gen_inputs(cfg: CaseConfig, device: str = "cpu") -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)

    cos, sin = gen_rope_cos_sin(cfg.t, cfg.dr, cfg.seed)
    token_x_bf16 = rnd(cfg.t, cfg.he)
    w_dq_bf16 = rnd(cfg.he, cfg.hcq)
    w_uq_qr_bf16 = rnd(cfg.hcq, cfg.n * (cfg.d + cfg.dr))
    w_uk = rnd(cfg.n, cfg.d, cfg.hckv)
    w_dkv_kr_bf16 = rnd(cfg.he, cfg.hckv + cfg.dr)

    out: Dict[str, torch.Tensor] = {
        "rmsnorm_gamma_cq": torch.ones(cfg.hcq, dtype=torch.bfloat16),
        "rmsnorm_gamma_ckv": torch.ones(cfg.hckv, dtype=torch.bfloat16),
        "rope_sin": sin,
        "rope_cos": cos,
        "cache_index": torch.arange(cfg.t, dtype=torch.int64),
        "weight_uk": w_uk,
    }

    wq, kvq = cfg.weight_quant_mode, cfg.kv_cache_quant_mode

    if wq == 0:
        out["token_x"] = token_x_bf16
        out["weight_dq"] = w_dq_bf16
        out["weight_uq_qr"] = w_uq_qr_bf16
        out["weight_dkv_kr"] = w_dkv_kr_bf16
    elif wq == 1:
        out["token_x"] = token_x_bf16
        out["weight_dq"] = w_dq_bf16
        w_uq_q, deq_uq = quantize_weight_per_channel(w_uq_qr_bf16)
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_kr_bf16
        out["dequant_scale_w_uq_qr"] = deq_uq
    elif wq == 2:
        token_q, deq_x = quantize_activation_per_token(token_x_bf16)
        w_dq_q, deq_dq = quantize_weight_per_channel(w_dq_bf16)
        w_uq_q, deq_uq = quantize_weight_per_channel(w_uq_qr_bf16)
        w_dkv_q, deq_dkv = quantize_weight_per_channel(w_dkv_kr_bf16)
        out["token_x"] = token_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
    elif wq == 3:
        tok_q, deq_x, deq_x_fp = quantize_mxfp8_along_last(token_x_bf16)
        w_dq_q, deq_dq, deq_dq_fp = quantize_weight_mxfp8(w_dq_bf16)
        w_uq_q, deq_uq, deq_uq_fp = quantize_weight_mxfp8(w_uq_qr_bf16)
        w_dkv_q, deq_dkv, deq_dkv_fp = quantize_weight_mxfp8(w_dkv_kr_bf16)
        out["token_x"] = tok_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
        out["_dequant_scale_x_fp"] = deq_x_fp
        out["_dequant_scale_w_dq_fp"] = deq_dq_fp
        out["_dequant_scale_w_uq_qr_fp"] = deq_uq_fp
        out["_dequant_scale_w_dkv_kr_fp"] = deq_dkv_fp
    else:
        raise ValueError(f"unsupported weight_quant_mode={wq}")

    kv_shape, kr_shape = _cache_shapes(cfg)
    if kvq == 0:
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.bfloat16)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.bfloat16)
    elif kvq == 1:
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.float8_e4m3fn)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.bfloat16)
        out["quant_scale_ckv"] = torch.tensor([1.0], dtype=torch.float32)
    elif kvq == 2:
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.int8)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.int8)
        out["quant_scale_ckv"] = torch.full((1, cfg.hckv), 10.0, dtype=torch.float32)
        out["quant_scale_ckr"] = torch.full((1, cfg.dr), 10.0, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported kv_cache_quant_mode={kvq} in ATK golden")

    _layout_token_and_index(out, cfg)

    if device != "cpu":
        out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
    return out



def _cache_shapes(cfg: CaseConfig) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    mode = cfg.cache_mode
    if mode == "BSND":
        prefix = (cfg.b, cfg.t // cfg.b, cfg.nkv)
    elif mode == "TND":
        prefix = (cfg.t, cfg.nkv)
    else:
        prefix = (cfg.block_num, cfg.block_size, cfg.nkv)
    return prefix + (cfg.hckv,), prefix + (cfg.dr,)


def _layout_token_and_index(out: Dict[str, torch.Tensor], cfg: CaseConfig) -> None:
    """Match public ABI token / rope / cache_index / actual_seq_len to cacheMode + COMBINE."""
    mode = cfg.cache_mode
    unmerged = bool(cfg.unmerged) or mode == "BSND"
    b = max(1, cfg.b)
    s = cfg.t // b
    if unmerged:
        out["token_x"] = out["token_x"].reshape(b, s, cfg.he)
        out["rope_sin"] = out["rope_sin"].reshape(b, s, cfg.dr)
        out["rope_cos"] = out["rope_cos"].reshape(b, s, cfg.dr)
    if mode in ("BSND", "TND"):
        out.pop("cache_index", None)
        out.pop("actual_seq_len", None)
        return
    if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        if unmerged:
            out["cache_index"] = torch.arange(b, dtype=torch.int64).reshape(b, 1)
            out.pop("actual_seq_len", None)
        else:
            out["cache_index"] = torch.arange(b, dtype=torch.int64)
            out["actual_seq_len"] = torch.tensor(
                [(i + 1) * s for i in range(b)], dtype=torch.int32
            )
        return
    # PA_BSND / PA_NZ
    if unmerged:
        out["cache_index"] = torch.arange(cfg.t, dtype=torch.int64).reshape(b, s)
    out.pop("actual_seq_len", None)


def _flatten_tokens(
    inp: Dict[str, torch.Tensor], cfg: CaseConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token = inp["token_x"]
    cos, sin = inp["rope_cos"], inp["rope_sin"]
    if token.dim() == 3:
        t = int(token.shape[0] * token.shape[1])
        token = token.reshape(t, token.shape[-1])
        if cos.dim() == 3:
            cos = cos.reshape(t, cos.shape[-1])
            sin = sin.reshape(t, sin.shape[-1])
    return token, cos, sin


def _prep_golden_inputs(
    inp: Dict[str, torch.Tensor], cfg: CaseConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    token_x, rope_cos, rope_sin = _flatten_tokens(inp, cfg)
    cache_index = inp.get("cache_index")
    if not isinstance(cache_index, torch.Tensor):
        cache_index = None
    return token_x, rope_cos, rope_sin, cache_index


def _unmerge_query(
    out_q: torch.Tensor, out_qrope: torch.Tensor, inp: Dict[str, torch.Tensor], cfg: CaseConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    token = inp["token_x"]
    if token.dim() == 3:
        b, s = int(token.shape[0]), int(token.shape[1])
        out_q = out_q.reshape(b, s, cfg.n, out_q.shape[-1])
        out_qrope = out_qrope.reshape(b, s, cfg.n, out_qrope.shape[-1])
    return out_q, out_qrope


def _nz_data_size(dtype: torch.dtype) -> int:
    return 16 if dtype == torch.bfloat16 else 32


def _scatter_nz(
    rows: torch.Tensor,
    block_rows: List[Tuple[int, int]],
    block_num: int,
    block_size: int,
) -> torch.Tensor:
    t, nkv, h = rows.shape
    data_size = _nz_data_size(rows.dtype)
    data_num = math.ceil(h / data_size)
    cache = torch.zeros(block_num, block_size, nkv, h, dtype=rows.dtype, device=rows.device)
    for token, (block, row_in_block) in enumerate(block_rows):
        for data_index in range(data_num):
            source = data_index * data_size
            width = min(data_size, h - source)
            packed_index = data_index * block_size + row_in_block
            cache_row = packed_index // data_num
            cache_col = (packed_index % data_num) * data_size
            cache[block, cache_row, :, cache_col : cache_col + width] = rows[
                token, :, source : source + width
            ]
    return cache


def _pa_blk_ranges(cfg: CaseConfig) -> List[Tuple[int, int]]:
    b = max(1, cfg.b)
    s = cfg.t // b
    return [(i * s, (i + 1) * s) for i in range(b)]


def _cache_store_dtypes(cfg: CaseConfig) -> Tuple[torch.dtype, torch.dtype]:
    kvq = cfg.kv_cache_quant_mode
    if kvq == 2:
        return torch.int8, torch.int8
    if kvq == 1:
        return torch.float8_e4m3fn, torch.bfloat16
    return torch.bfloat16, torch.bfloat16


def _scatter_kv(
    norm2: torch.Tensor,
    rotary_k: torch.Tensor,
    cache_index: Optional[torch.Tensor],
    cfg: CaseConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t, nkv, hckv, dr = cfg.t, cfg.nkv, cfg.hckv, cfg.dr
    mode = cfg.cache_mode
    device = norm2.device
    ckv_w = norm2.unsqueeze(1).expand(t, nkv, hckv).contiguous()
    kr_w = rotary_k.unsqueeze(1).expand(t, nkv, dr).contiguous()

    if mode == "TND":
        return ckv_w, kr_w
    if mode == "BSND":
        b, s = cfg.b, t // cfg.b
        return ckv_w.reshape(b, s, nkv, hckv), kr_w.reshape(b, s, nkv, dr)

    block_num, block_size = cfg.block_num, cfg.block_size
    if cache_index is None:
        raise ValueError(f"cache_index is required for cacheMode={mode}")
    idx = cache_index.to(device=device, dtype=torch.long).reshape(-1)

    if mode == "PA_BSND":
        kv_cache = torch.zeros(block_num, block_size, nkv, hckv, dtype=norm2.dtype, device=device)
        kr_cache = torch.zeros(block_num, block_size, nkv, dr, dtype=rotary_k.dtype, device=device)
        kv_cache.reshape(block_num * block_size, nkv, hckv).index_copy_(0, idx, ckv_w)
        kr_cache.reshape(block_num * block_size, nkv, dr).index_copy_(0, idx, kr_w)
        return kv_cache, kr_cache

    if mode == "PA_BLK_BSND":
        kv_cache = torch.zeros(block_num, block_size, nkv, hckv, dtype=norm2.dtype, device=device)
        kr_cache = torch.zeros(block_num, block_size, nkv, dr, dtype=rotary_k.dtype, device=device)
        for batch, (start, end) in enumerate(_pa_blk_ranges(cfg)):
            block = int(idx[batch].item())
            kv_cache[block, : end - start] = ckv_w[start:end]
            kr_cache[block, : end - start] = kr_w[start:end]
        return kv_cache, kr_cache

    if mode == "PA_NZ":
        kv_dt, kr_dt = _cache_store_dtypes(cfg)
        block_rows = [(int(i.item()) // block_size, int(i.item()) % block_size) for i in idx]
        return (
            _scatter_nz(ckv_w.to(kv_dt), block_rows, block_num, block_size),
            _scatter_nz(kr_w.to(kr_dt), block_rows, block_num, block_size),
        )

    if mode == "PA_BLK_NZ":
        kv_dt, kr_dt = _cache_store_dtypes(cfg)
        block_rows: List[Tuple[int, int]] = []
        for batch, (start, end) in enumerate(_pa_blk_ranges(cfg)):
            block = int(idx[batch].item())
            for row in range(end - start):
                block_rows.append((block, row))
        return (
            _scatter_nz(ckv_w.to(kv_dt), block_rows, block_num, block_size),
            _scatter_nz(kr_w.to(kr_dt), block_rows, block_num, block_size),
        )

    raise ValueError(f"unsupported cacheMode={mode}")


def cal_mlaprolog_cpu(inp: Dict[str, torch.Tensor], cfg: CaseConfig
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, hckv, d, dr, n = cfg.t, cfg.hckv, cfg.d, cfg.dr, cfg.n
    nkv, block_size = cfg.nkv, cfg.block_size
    wq, kvq = cfg.weight_quant_mode, cfg.kv_cache_quant_mode
    qq = cfg.query_quant_mode

    token_x, rope_cos, rope_sin = _flatten_tokens(inp, cfg)
    gamma_cq = inp["rmsnorm_gamma_cq"].to(torch.float32)
    gamma_ckv = inp["rmsnorm_gamma_ckv"].to(torch.float32)
    cos = rope_cos.to(torch.float32)
    sin = rope_sin.to(torch.float32)
    cache_index = inp["cache_index"].cpu() if "cache_index" in inp else None
    w_uk = inp["weight_uk"].to(torch.float32)

    # ---- mm1 (MatmulCq) ----
    # wq=1: bf16×bf16, fp32 L0C, 落盘 bf16（F322BF16）后再进 RMS
    # wq=2: int8×int8 → int32，dequant(scale_x×scale_w) 后进 RMS（与 RmsNormCq 内 dequant 等价）
    if wq == 2:
        token_x_i8 = token_x.to(torch.int8)
        w_dq_i8 = inp["weight_dq"].to(torch.int8)
        mm1_i32 = int8_matmul_i32(token_x_i8, w_dq_i8)
        mm1 = mm1_i32.to(torch.float32)
        mm1 = mm1 * inp["dequant_scale_x"].to(torch.float32)
        mm1 = mm1 * inp["dequant_scale_w_dq"].to(torch.float32)
        token_x_for_mm4 = token_x_i8
    elif wq == 3:
        token_x_dq = dequant_mxfp8_along_last(token_x, inp["_dequant_scale_x_fp"])
        w_dq_dq = dequant_weight_mxfp8(inp["weight_dq"], inp["_dequant_scale_w_dq_fp"])
        mm1 = torch.matmul(token_x_dq, w_dq_dq)
        token_x_for_mm4 = token_x_dq
    else:
        token_x_fp = token_x.to(torch.float32)
        w_dq = inp["weight_dq"].to(torch.float32)
        # wq=0/1：与 Cube F322BF16 落盘一致
        mm1 = torch.matmul(token_x_fp, w_dq).to(torch.bfloat16).to(torch.float32)
        token_x_for_mm4 = token_x_fp

    # ---- RmsNormCq（fp32）+ 可选 qcQrScale；wq=1/2 再 fused dynamic quant ----
    norm1 = rms_norm(mm1, gamma_cq, cfg.eps_cq, cfg.qc_qr_scale)

    if wq in (1, 2):
        # int8×int8 → int32，再 per-token×per-channel dequant（对齐 DequantPerTokenQc）
        norm1_q, deq_qcqr = dynamic_quant(norm1, inp.get("smooth_scales_cq"))
        w_uq_i8 = inp["weight_uq_qr"].to(torch.int8)
        mm2_i32 = int8_matmul_i32(norm1_q.to(torch.int8), w_uq_i8)
        scale_w = inp["dequant_scale_w_uq_qr"].to(torch.float32)
        mm2 = mm2_i32.to(torch.float32) * deq_qcqr * scale_w
    elif wq == 3:
        w_uq = dequant_weight_mxfp8(inp["weight_uq_qr"], inp["_dequant_scale_w_uq_qr_fp"])
        norm1_dq = dynamic_mx_quant_dequant_cq(norm1.to(torch.bfloat16).to(torch.float32))
        mm2 = torch.matmul(norm1_dq, w_uq)
    else:
        norm1 = norm1.to(torch.bfloat16).to(torch.float32)
        w_uq = inp["weight_uq_qr"].to(torch.float32)
        mm2 = torch.matmul(norm1, w_uq).to(torch.bfloat16).to(torch.float32)

    mm2 = mm2.reshape(t, n, d + dr)
    q_nope, q_rope_in = mm2[:, :, :d], mm2[:, :, d:]

    # ---- MatmulQn：bf16×bf16，fp32 累加，bf16 输出（F322BF16）----
    # wq=1/2 的 Q_nope 在 kernel 里先 CAST_RINT 到 bf16 再进 mm3
    q_nope_bf16 = q_nope.to(torch.bfloat16)
    w_uk_b = w_uk.to(torch.bfloat16)
    out_q = torch.zeros((n, t, hckv), dtype=torch.bfloat16)
    q_nope_t = q_nope_bf16.transpose(0, 1)
    for i in range(n):
        out_q[i] = torch.matmul(
            q_nope_t[i].to(torch.float32), w_uk_b[i].to(torch.float32)
        ).to(torch.bfloat16)
    out_q = out_q.transpose(0, 1).to(torch.float32)

    deq_scale_q_nope = None
    if qq == 1 and wq == 3 and kvq == 1:
        out_q_fp8, deq_scale_q_nope = dynamic_quant_q_nope_fp8(out_q)
        out_q = out_q_fp8.to(torch.float32)

    # ---- RopeQr：fp32 计算（original/interleave-half），bf16 输出 ----
    out_qrope = apply_rope(q_rope_in, cos, sin, cfg.do_rope,
                           cfg.rope_style).to(torch.bfloat16).to(torch.float32)
    if deq_scale_q_nope is not None:
        qckv = inp["quant_scale_ckv"].to(torch.float32).reshape(1, 1).item()
        out_qrope = out_qrope * (qckv / deq_scale_q_nope.squeeze(-1).unsqueeze(-1))

    # ---- mm4 (MatmulCkvKr) ----
    # fused Rope(Kr) 用的是 Cube L0C fp32，存 kr_cache 时才 CAST_RINT 到 bf16。
    # k_nope / RMS 仍走 F322BF16 后的 mm4，避免动 kv_cache。
    mm4_fp32 = None
    if wq == 2:
        w_dkv_i8 = inp["weight_dkv_kr"].to(torch.int8)
        mm4_i32 = int8_matmul_i32(token_x_for_mm4.to(torch.int8), w_dkv_i8)
        mm4 = mm4_i32.to(torch.float32)
        mm4 = mm4 * inp["dequant_scale_x"].to(torch.float32)
        mm4 = mm4 * inp["dequant_scale_w_dkv_kr"].to(torch.float32)
        mm4_fp32 = mm4
    elif wq == 3:
        w_dkv = dequant_weight_mxfp8(inp["weight_dkv_kr"], inp["_dequant_scale_w_dkv_kr_fp"])
        mm4 = torch.matmul(token_x_for_mm4, w_dkv)
        mm4_fp32 = mm4
    else:
        w_dkv = inp["weight_dkv_kr"].to(torch.float32)
        # wq=0/1: 对齐 fused MatmulSplitK 的 L0C 累加（K=16 tile）再 F322BF16，不是 one-shot ND。
        mm4_fp32 = matmul_l0c_splitk(token_x_for_mm4, w_dkv)
        mm4 = mm4_fp32.to(torch.bfloat16).to(torch.float32)

    k_nope = mm4[:, :hckv]
    k_rope_in = mm4[:, hckv:]
    rotary_k = apply_rope(k_rope_in, cos, sin, cfg.do_rope, cfg.rope_style).to(torch.float32)
    if wq == 1 and kvq == 2:
        rotary_k_store = quant_s8(rotary_k, inp["quant_scale_ckr"]).to(torch.float32)
    else:
        rotary_k_store = rotary_k.to(torch.bfloat16).to(torch.float32)
    _maybe_dump_kr_rope("cpu", {
        "k_rope_in": k_rope_in,
        "mm4_fp32_kr": mm4_fp32[:, hckv:],
        "mm4_bf16_kr": mm4[:, hckv:],
        "token_x_for_mm4": token_x_for_mm4.detach().cpu(),
        "weight_dkv_kr": inp["weight_dkv_kr"].detach().cpu(),
        "rotary_k_fp32": rotary_k,
        "rotary_k_bf16": rotary_k.to(torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "rope_style": cfg.rope_style,
        "t": cfg.t,
        "b": cfg.b,
        "hckv": hckv,
    })

    norm2 = rms_norm(k_nope, gamma_ckv, cfg.eps_ckv, cfg.kc_scale).to(torch.float32)
    if kvq == 2:
        norm2 = quant_s8(norm2, inp["quant_scale_ckv"]).to(torch.float32)
    elif kvq == 1 and wq == 3:
        norm2 = quant_ckv_fp8_per_tensor(norm2, inp["quant_scale_ckv"]).to(torch.float32)
    else:
        norm2 = norm2.to(torch.bfloat16).to(torch.float32)

    kv_cache, kr_cache = _scatter_kv(norm2, rotary_k_store, cache_index, cfg)
    out_q, out_qrope = _unmerge_query(out_q, out_qrope, inp, cfg)
    return out_q, out_qrope, kv_cache, kr_cache


def _npu_rms(x: torch.Tensor, gamma: torch.Tensor, eps: float, scale: float) -> torch.Tensor:
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")
    if x.dtype == torch.float32:
        y = rms_norm(x, gamma.to(dtype=x.dtype, device=x.device), eps, 1.0)
        return y * scale if scale != 1.0 else y
    y, _ = torch_npu.npu_rms_norm(x.to(torch.bfloat16), gamma.to(torch.bfloat16), eps)
    return y * scale if scale != 1.0 else y


def _npu_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, do_rope: bool,
              rope_style: str = "original") -> torch.Tensor:
    if not do_rope:
        return x
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")
    if rope_style == "vf" or x.dtype == torch.float32:
        return apply_rope(x, cos, sin, True, rope_style)
    x = x.to(torch.bfloat16)
    cos = cos.to(torch.bfloat16)
    sin = sin.to(torch.bfloat16)
    xx = interleave_to_half(x)
    if xx.dim() == 2:
        xx3 = xx.unsqueeze(1)
        c = cos.unsqueeze(1) if cos.dim() == 2 else cos
        s = sin.unsqueeze(1) if sin.dim() == 2 else sin
        y = torch_npu.npu_rotary_mul(xx3, c, s, rotary_mode="half")
        return y.squeeze(1)
    c = cos.unsqueeze(1).expand_as(xx) if cos.dim() == 2 else cos
    s = sin.unsqueeze(1).expand_as(xx) if sin.dim() == 2 else sin
    return torch_npu.npu_rotary_mul(xx, c, s, rotary_mode="half")


def _npu_matmul_qn(q_nope: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
    n, t, hckv = w_uk.shape[0], q_nope.shape[0], w_uk.shape[-1]
    q_nope_n = q_nope.transpose(0, 1).contiguous()
    out_n = torch.empty((n, t, hckv), dtype=q_nope.dtype, device=q_nope.device)
    for i in range(n):
        out_n[i] = torch.matmul(q_nope_n[i], w_uk[i])
    return out_n.transpose(0, 1).contiguous()


def _npu_to_nz(weight: torch.Tensor, device: str) -> torch.Tensor:
    """Cast ND weight to FRACTAL_NZ for Cube-aligned matmul (independent of fused kernel)."""
    w = weight.contiguous().to(device)
    return torch_npu.npu_format_cast(w, FRACTAL_NZ_FORMAT)


def _npu_i8_mm_i32(x_i8: torch.Tensor, w_nz: torch.Tensor, n_out: int) -> torch.Tensor:
    """int8×int8(NZ) → int32 via npu_quant_matmul (scale=1, dequant done by caller)."""
    scale1 = torch.ones(int(n_out), dtype=torch.float32, device=x_i8.device)
    return torch_npu.npu_quant_matmul(x_i8, w_nz, scale1, output_dtype=torch.int32)


def _e8m0_bytes_to_fp_scale(scale: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu / uint8 share → float32 scale (= 2^(share-127))."""
    if scale.dtype == torch.float8_e8m0fnu:
        u = scale.view(torch.uint8)
    else:
        u = scale.to(torch.uint8)
    return torch.pow(
        torch.tensor(2.0, device=scale.device, dtype=torch.float32),
        u.to(torch.float32) - 127.0,
    )


def _mx_fp_scale(inp: Dict[str, torch.Tensor], fp_key: str, e8_key: str,
                 device: str) -> torch.Tensor:
    if fp_key in inp and isinstance(inp[fp_key], torch.Tensor):
        return inp[fp_key].contiguous().to(device=device, dtype=torch.float32)
    return _e8m0_bytes_to_fp_scale(inp[e8_key].contiguous().to(device))


def _npu_dynamic_mx_dequant(x: torch.Tensor) -> torch.Tensor:
    """Hardware mxfp8 dynamic quant then dequant back to float32 (block=32)."""
    x_bf16 = x.to(torch.bfloat16)
    y_fp8, scale_u8 = torch_npu.npu_dynamic_mx_quant(
        x_bf16,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=MXFP8_GRP,
    )
    scale_fp = _e8m0_bytes_to_fp_scale(scale_u8).reshape(x_bf16.shape[0], -1)
    return dequant_mxfp8_along_last(y_fp8, scale_fp)


def _cal_mlaprolog_npu_mxfp8(
    inp: Dict[str, torch.Tensor], cfg: CaseConfig, device: str = "npu:0"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU small-op golden for wq=3 (mxfp8): dequant → FRACTAL_NZ bf16 matmul + npu_dynamic_mx_quant.

    Independent of fused MlaPrologV3. Restores double-benchmark (CPU fp32 dequant-mm vs this
    NPU path) so compare_cv no longer collapses to err_threshold.
    """
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU mxfp8 golden")

    t, hckv, d, dr, n = cfg.t, cfg.hckv, cfg.d, cfg.dr, cfg.n
    kvq, qq = cfg.kv_cache_quant_mode, cfg.query_quant_mode

    def dev(x: torch.Tensor, dtype=None) -> torch.Tensor:
        y = x.contiguous().to(device=device)
        return y.to(dtype) if dtype is not None else y

    token_x_nd, rope_cos, rope_sin, cache_index_t = _prep_golden_inputs(inp, cfg)
    gamma_cq = dev(inp["rmsnorm_gamma_cq"], torch.float32)
    gamma_ckv = dev(inp["rmsnorm_gamma_ckv"], torch.float32)
    cos = dev(rope_cos, torch.float32)
    sin = dev(rope_sin, torch.float32)
    cache_index = dev(cache_index_t) if cache_index_t is not None else None
    w_uk = dev(inp["weight_uk"], torch.bfloat16)

    scale_x_fp = _mx_fp_scale(inp, "_dequant_scale_x_fp", "dequant_scale_x", device)
    scale_w_dq_fp = _mx_fp_scale(inp, "_dequant_scale_w_dq_fp", "dequant_scale_w_dq", device)
    scale_w_uq_fp = _mx_fp_scale(inp, "_dequant_scale_w_uq_qr_fp", "dequant_scale_w_uq_qr", device)
    scale_w_dkv_fp = _mx_fp_scale(
        inp, "_dequant_scale_w_dkv_kr_fp", "dequant_scale_w_dkv_kr", device
    )

    # Dequant on host (large fp8→fp32 expand), then bf16 NPU matmul.
    # Optional NZ via MLA_PROLOG_MX_NZ=1 (default off: large FRACTAL_NZ cast can hang/timeout).
    use_weight_nz = os.environ.get("MLA_PROLOG_MX_NZ", "0").strip() in ("1", "true", "True")

    def maybe_nz(w_bf16: torch.Tensor) -> torch.Tensor:
        return _npu_to_nz(w_bf16, device) if use_weight_nz else w_bf16.contiguous().to(device)

    token_dq = dequant_mxfp8_along_last(
        token_x_nd.detach().cpu(), scale_x_fp.detach().cpu()
    ).to(device=device, dtype=torch.bfloat16)
    w_dq_bf16 = dequant_weight_mxfp8(
        inp["weight_dq"].detach().cpu(), scale_w_dq_fp.detach().cpu()
    ).to(torch.bfloat16)
    mm1 = torch.matmul(token_dq, maybe_nz(w_dq_bf16)).to(torch.float32)

    # ---- RmsNormCq + hardware dynamic mx quant/dequant → bf16 mm2 ----
    norm1 = _npu_rms(mm1, gamma_cq, cfg.eps_cq, cfg.qc_qr_scale).to(torch.float32)
    norm1_dq = _npu_dynamic_mx_dequant(norm1).to(torch.bfloat16)
    w_uq_bf16 = dequant_weight_mxfp8(
        inp["weight_uq_qr"].detach().cpu(), scale_w_uq_fp.detach().cpu()
    ).to(torch.bfloat16)
    mm2 = torch.matmul(norm1_dq, maybe_nz(w_uq_bf16)).to(torch.float32)
    mm2 = mm2.reshape(t, n, d + dr)
    q_nope, q_rope_in = mm2[:, :, :d], mm2[:, :, d:]

    out_q = _npu_matmul_qn(q_nope.to(torch.bfloat16), w_uk).to(torch.float32)
    deq_scale_q_nope = None
    if qq == 1 and kvq == 1:
        out_q_fp8, deq_scale_q_nope = dynamic_quant_q_nope_fp8(out_q)
        out_q = out_q_fp8.to(torch.float32)

    out_qrope = _npu_rope(
        q_rope_in.contiguous().to(torch.bfloat16),
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
        cfg.do_rope,
        cfg.rope_style,
    ).to(torch.float32)
    if deq_scale_q_nope is not None:
        qckv = dev(inp["quant_scale_ckv"], torch.float32).reshape(1, 1).item()
        out_qrope = out_qrope * (qckv / deq_scale_q_nope.to(device).squeeze(-1).unsqueeze(-1))

    # ---- mm4 ----
    w_dkv_bf16 = dequant_weight_mxfp8(
        inp["weight_dkv_kr"].detach().cpu(), scale_w_dkv_fp.detach().cpu()
    ).to(torch.bfloat16)
    mm4 = torch.matmul(token_dq, maybe_nz(w_dkv_bf16)).to(torch.float32)
    k_nope, k_rope_in = mm4[:, :hckv], mm4[:, hckv:]
    rotary_k = _npu_rope(
        k_rope_in.contiguous().to(torch.float32),
        cos.to(torch.float32),
        sin.to(torch.float32),
        cfg.do_rope,
        cfg.rope_style,
    ).to(torch.float32)
    rotary_k_store = rotary_k.to(torch.bfloat16).to(torch.float32)

    norm2 = _npu_rms(k_nope, gamma_ckv, cfg.eps_ckv, cfg.kc_scale).to(torch.float32)
    if kvq == 1:
        norm2 = quant_ckv_fp8_per_tensor(norm2, dev(inp["quant_scale_ckv"], torch.float32)).to(
            torch.float32
        )
    else:
        norm2 = norm2.to(torch.bfloat16).to(torch.float32)

    kv_cache, kr_cache = _scatter_kv(norm2, rotary_k_store, cache_index, cfg)
    out_q, out_qrope = _unmerge_query(out_q, out_qrope, inp, cfg)
    torch.npu.synchronize()
    return out_q, out_qrope, kv_cache, kr_cache


def _cal_mlaprolog_npu_quant(
    inp: Dict[str, torch.Tensor], cfg: CaseConfig, device: str = "npu:0"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU small-op golden for wq=1/2: FRACTAL_NZ weights + npu_quant_matmul / npu_dynamic_quant.

    Still an independent op chain (not the fused MlaPrologV3 output). Matmul numerics track
    Cube more closely than CPU ND int32 matmul.
    """
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU NZ golden")

    t, hckv, d, dr, n = cfg.t, cfg.hckv, cfg.d, cfg.dr, cfg.n
    wq, kvq = cfg.weight_quant_mode, cfg.kv_cache_quant_mode

    def dev(x: torch.Tensor, dtype=None) -> torch.Tensor:
        y = x.contiguous().to(device=device)
        return y.to(dtype) if dtype is not None else y

    token_x_nd, rope_cos, rope_sin, cache_index_t = _prep_golden_inputs(inp, cfg)
    gamma_cq = dev(inp["rmsnorm_gamma_cq"], torch.float32)
    gamma_ckv = dev(inp["rmsnorm_gamma_ckv"], torch.float32)
    cos = dev(rope_cos, torch.float32)
    sin = dev(rope_sin, torch.float32)
    cache_index = dev(cache_index_t) if cache_index_t is not None else None
    w_uk = dev(inp["weight_uk"], torch.bfloat16)

    # ---- mm1 (MatmulCq) with NZ weight ----
    if wq == 2:
        token_x_i8 = dev(token_x_nd, torch.int8)
        w_dq_nd = inp["weight_dq"].to(torch.int8)
        w_dq_nz = _npu_to_nz(w_dq_nd, device)
        mm1_i32 = _npu_i8_mm_i32(token_x_i8, w_dq_nz, int(w_dq_nd.shape[-1]))
        scale_x = dev(inp["dequant_scale_x"], torch.float32)
        scale_w_dq = dev(inp["dequant_scale_w_dq"], torch.float32)
        mm1 = mm1_i32.to(torch.float32) * scale_x * scale_w_dq
        token_x_for_mm4 = token_x_i8
    else:
        # wq=1: bf16×bf16 NZ → bf16 (Cube F322BF16)
        token_x_bf16 = dev(token_x_nd, torch.bfloat16)
        w_dq_nz = _npu_to_nz(inp["weight_dq"].to(torch.bfloat16), device)
        mm1 = torch.matmul(token_x_bf16, w_dq_nz).to(torch.float32)
        token_x_for_mm4 = token_x_bf16

    # ---- RmsNormCq + dynamic quant (hardware) ----
    norm1 = _npu_rms(mm1, gamma_cq, cfg.eps_cq, cfg.qc_qr_scale).to(torch.float32)
    norm1_bf16 = norm1.to(torch.bfloat16)
    smooth = inp.get("smooth_scales_cq")
    smooth_arg = None
    if smooth is not None:
        smooth_arg = dev(smooth, torch.bfloat16).reshape(1, -1)
    yq, deq_pt = torch_npu.npu_dynamic_quant(norm1_bf16, smooth_scales=smooth_arg)
    # deq_pt: [T] ; weight scale: [1,N] or [N]
    scale_w_uq = dev(inp["dequant_scale_w_uq_qr"], torch.float32).reshape(1, -1)
    w_uq_nd = inp["weight_uq_qr"].to(torch.int8)
    w_uq_nz = _npu_to_nz(w_uq_nd, device)
    mm2_i32 = _npu_i8_mm_i32(yq, w_uq_nz, int(w_uq_nd.shape[-1]))
    mm2 = mm2_i32.to(torch.float32) * deq_pt.reshape(-1, 1) * scale_w_uq
    mm2 = mm2.reshape(t, n, d + dr)
    q_nope, q_rope_in = mm2[:, :, :d], mm2[:, :, d:]

    # ---- MatmulQn (ND w_uk; not NZ in op layout) ----
    out_q = _npu_matmul_qn(q_nope.to(torch.bfloat16), w_uk).to(torch.float32)
    out_qrope = _npu_rope(
        q_rope_in.contiguous().to(torch.bfloat16),
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
        cfg.do_rope,
        cfg.rope_style,
    ).to(torch.float32)

    # ---- mm4 (MatmulCkvKr) ----
    if wq == 2:
        w_dkv_nd = inp["weight_dkv_kr"].to(torch.int8)
        w_dkv_nz = _npu_to_nz(w_dkv_nd, device)
        mm4_i32 = _npu_i8_mm_i32(token_x_for_mm4, w_dkv_nz, int(w_dkv_nd.shape[-1]))
        scale_x = dev(inp["dequant_scale_x"], torch.float32)
        scale_w_dkv = dev(inp["dequant_scale_w_dkv_kr"], torch.float32)
        mm4 = mm4_i32.to(torch.float32) * scale_x * scale_w_dkv
        mm4_fp32 = mm4
    else:
        mm4_fp32 = matmul_l0c_splitk(
            token_x_for_mm4.detach().cpu(),
            inp["weight_dkv_kr"].detach().cpu(),
        )
        mm4 = mm4_fp32.to(torch.bfloat16).to(torch.float32).to(device)

    k_nope = mm4[:, :hckv]
    if wq == 2:
        k_rope_in = mm4_fp32[:, hckv:].contiguous()
        rotary_k = _npu_rope(
            k_rope_in.to(torch.float32),
            cos.to(torch.float32),
            sin.to(torch.float32),
            cfg.do_rope,
            cfg.rope_style,
        ).to(torch.float32)
    else:
        k_rope_in = mm4[:, hckv:]
        rotary_k = apply_rope(
            k_rope_in.detach().cpu().to(torch.float32),
            cos.detach().cpu().to(torch.float32),
            sin.detach().cpu().to(torch.float32),
            cfg.do_rope,
            cfg.rope_style,
        ).to(device=device, dtype=torch.float32)
    if wq == 1 and kvq == 2:
        rotary_k_store = quant_s8(rotary_k, dev(inp["quant_scale_ckr"], torch.float32)).to(
            torch.float32
        )
    else:
        rotary_k_store = rotary_k.to(torch.bfloat16).to(torch.float32)
    _maybe_dump_kr_rope("npu", {
        "k_rope_in": k_rope_in.detach().float(),
        "mm4_fp32_kr": mm4_fp32[:, hckv:].detach().float(),
        "mm4_cube_kr": mm4[:, hckv:].detach().float(),
        "rotary_k_fp32": rotary_k.detach(),
        "rotary_k_bf16": rotary_k_store.detach().to(torch.bfloat16),
        "cos": cos.detach(),
        "sin": sin.detach(),
    })

    norm2 = _npu_rms(k_nope, gamma_ckv, cfg.eps_ckv, cfg.kc_scale).to(torch.float32)
    if kvq == 2:
        norm2 = quant_s8(norm2, dev(inp["quant_scale_ckv"], torch.float32)).to(torch.float32)
    else:
        norm2 = norm2.to(torch.bfloat16).to(torch.float32)

    kv_cache, kr_cache = _scatter_kv(norm2, rotary_k_store, cache_index, cfg)
    out_q, out_qrope = _unmerge_query(out_q, out_qrope, inp, cfg)
    torch.npu.synchronize()
    return out_q, out_qrope, kv_cache, kr_cache


def cal_mlaprolog_npu(inp: Dict[str, torch.Tensor], cfg: CaseConfig, device: str = "npu:0"
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU small-op golden using torch_npu (no ATB).

    wq=1/2: FRACTAL_NZ + npu_quant_matmul / npu_dynamic_quant
    wq=3:   dequant + FRACTAL_NZ bf16 matmul + npu_dynamic_mx_quant
    Set MLA_PROLOG_NPU_NZ=0 to fall back to CPU algebra for all quant modes.
    """
    use_nz = os.environ.get("MLA_PROLOG_NPU_NZ", "1").strip() not in ("0", "false", "False")
    if cfg.weight_quant_mode in (1, 2) and use_nz:
        return _cal_mlaprolog_npu_quant(inp, cfg, device=device)
    if cfg.weight_quant_mode == 3 and use_nz:
        return _cal_mlaprolog_npu_mxfp8(inp, cfg, device=device)

    if cfg.weight_quant_mode != 0:
        # Callers may already have moved inputs to NPU; CPU algebra needs host tensors.
        cpu_inp = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
        }
        cpu_out = cal_mlaprolog_cpu(cpu_inp, cfg)
        return tuple(x.to(device) for x in cpu_out)

    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")

    t, hckv, d, dr, n = cfg.t, cfg.hckv, cfg.d, cfg.dr, cfg.n
    compute_dtype = torch.float32

    def to_dev(x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().to(device=device, dtype=compute_dtype)

    token_x_nd, rope_cos, rope_sin, cache_index_t = _prep_golden_inputs(inp, cfg)
    token_x = to_dev(token_x_nd)
    w_dq = to_dev(inp["weight_dq"])
    w_uq = to_dev(inp["weight_uq_qr"])
    w_dkv = to_dev(inp["weight_dkv_kr"])
    w_uk = to_dev(inp["weight_uk"])
    gamma_cq = to_dev(inp["rmsnorm_gamma_cq"])
    gamma_ckv = to_dev(inp["rmsnorm_gamma_ckv"])
    cos = to_dev(rope_cos)
    sin = to_dev(rope_sin)
    cache_index = (
        cache_index_t.contiguous().to(device=device) if cache_index_t is not None else None
    )

    mm1 = torch.matmul(token_x, w_dq)
    norm1 = _npu_rms(mm1, gamma_cq, cfg.eps_cq, cfg.qc_qr_scale)
    mm2 = torch.matmul(norm1, w_uq).reshape(t, n, d + dr)
    q_nope, q_rope_in = mm2[:, :, :d], mm2[:, :, d:]
    out_q = _npu_matmul_qn(q_nope, w_uk)
    out_qrope = _npu_rope(q_rope_in.contiguous(), cos, sin, cfg.do_rope, cfg.rope_style)

    mm4_fp32 = matmul_l0c_splitk(token_x.detach().cpu(), w_dkv.detach().cpu())
    mm4 = mm4_fp32.to(torch.bfloat16).to(torch.float32).to(device)
    k_nope = mm4[:, :hckv]
    k_rope_in = mm4[:, hckv:]
    norm2 = _npu_rms(k_nope, gamma_ckv, cfg.eps_ckv, cfg.kc_scale)
    rotary_k = _npu_rope(
        k_rope_in.contiguous().to(torch.bfloat16),
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
        cfg.do_rope,
        cfg.rope_style,
    )
    kv_cache, kr_cache = _scatter_kv(norm2, rotary_k, cache_index, cfg)
    out_q, out_qrope = _unmerge_query(out_q, out_qrope, inp, cfg)

    torch.npu.synchronize()
    return out_q, out_qrope, kv_cache, kr_cache


def cast_weight_nz(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if not _HAS_TORCH_NPU:
        return tensor.to(device)
    return torch_npu.npu_format_cast(tensor.contiguous().to(device), FRACTAL_NZ_FORMAT)


# ---------------------------------------------------------------------------
# ATK executor
# ---------------------------------------------------------------------------

logging = Logger().get_logger()
MIN_ERR = 1e-7
CACHE_INDEX_SEED = 2025
_SEED_NAME_RE = re.compile(r"\.seed(\d+)\.")


def parse_case_seed(case_name: str, default: int = CACHE_INDEX_SEED) -> int:
    """从用例名解析 seed<N>；泛化矩阵里同 shape 的 nc* 共用同一 seed。"""
    match = _SEED_NAME_RE.search(case_name or "")
    return int(match.group(1)) if match else default


def _task_case_name(task_result: TaskResult) -> str:
    case_config = getattr(task_result, "case_config", None)
    return str(getattr(case_config, "name", "") or "")

# Positional layout of the aclnn prototypes. Every tensor slot has to be passed, so the
# executor fills the slots a case does not declare with a null aclTensor pointer.
V1_TENSOR_SLOTS = (
    "token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos",
    "cache_index", "kv_cache", "kr_cache",
    "dequant_scale_x", "dequant_scale_w_dq", "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr", "quant_scale_ckv", "quant_scale_ckr", "smooth_scales_cq",
)
V3_TENSOR_SLOTS = (
    "token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos",
    "kv_cache", "kr_cache", "cache_index",
    "dequant_scale_x", "dequant_scale_w_dq", "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr", "quant_scale_ckv", "quant_scale_ckr", "smooth_scales_cq",
    "actual_seq_len", "k_nope_clip_alpha",
)

V1_ATTR_SLOTS = ("rmsnormEpsilonCq", "rmsnormEpsilonCkv", "cacheMode")
V3_ATTR_SLOTS = V1_ATTR_SLOTS + (
    "weightQuantMode", "kvCacheQuantMode", "queryQuantMode",
    "ckvkrRepoMode", "quantScaleRepoMode", "tileSize", "qcQrScale", "kcScale",
)

# queryOut / queryRopeOut are produced by the reference node; the remaining output slots
# (dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut) stay null in these cases.
VERSION_LAYOUT = {
    "v1": (V1_TENSOR_SLOTS, V1_ATTR_SLOTS, 0, ()),
    "v2": (V1_TENSOR_SLOTS, V1_ATTR_SLOTS, 1, ("weight_dq", "weight_uq_qr", "weight_dkv_kr")),
    "v3": (V3_TENSOR_SLOTS, V3_ATTR_SLOTS, 3, ("weight_dq", "weight_uq_qr", "weight_dkv_kr")),
}


def _null_tensor_ptr():
    return ctypes.POINTER(AclTensor)()


def _resolve_enable_rope(cos, sin):
    """RoPE is off when both rope tensors have an empty token axis (shape (0, Dr))."""
    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        raise ValueError("rope_cos and rope_sin are required inputs and must be tensors")
    has_cos = cos.numel() > 0
    has_sin = sin.numel() > 0
    if has_cos != has_sin:
        raise ValueError("rope_cos and rope_sin must both be empty or both be non-empty")
    return has_cos



def _apply_deterministic_cache_index(input_data: InputDataset, seed: int = CACHE_INDEX_SEED):
    """Every node must scatter to the same slots. COMBINE vs unmerged only changes index rank."""
    kw = input_data.kwargs
    mode = str(kw.get("cacheMode", "PA_BSND"))
    if mode in ("BSND", "TND"):
        kw.pop("cache_index", None)
        kw.pop("actual_seq_len", None)
        return

    token = kw["token_x"]
    kv = kw["kv_cache"]
    rng = np.random.RandomState(seed)
    unmerged = token.dim() == 3
    if token.dim() == 3:
        b, s = int(token.shape[0]), int(token.shape[1])
        t = b * s
    else:
        t = int(token.shape[0])
        if "actual_seq_len" in kw and isinstance(kw["actual_seq_len"], torch.Tensor):
            b = int(kw["actual_seq_len"].numel())
        elif unmerged:
            b = int(token.shape[0])
        else:
            b = 2 if t % 2 == 0 else 1
        s = t // max(b, 1)

    if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        block_num = int(kv.shape[0])
        if block_num < b:
            raise ValueError(f"PA_BLK needs blockNum>={b}, got {block_num}")
        index = torch.from_numpy(rng.choice(block_num, b, replace=False).astype(np.int64))
        if unmerged:
            index = index.reshape(b, 1)
            kw.pop("actual_seq_len", None)
        else:
            kw["actual_seq_len"] = torch.tensor(
                [(i + 1) * s for i in range(b)], dtype=torch.int32
            )
        if "cache_index" in kw and isinstance(kw["cache_index"], torch.Tensor):
            index = index.to(kw["cache_index"].dtype)
        kw["cache_index"] = index
        return

    slots = int(kv.shape[0]) * int(kv.shape[1])
    index = torch.from_numpy(rng.choice(slots, t, replace=False).astype(np.int64))
    if unmerged:
        index = index.reshape(b, s)
    if "cache_index" in kw and isinstance(kw["cache_index"], torch.Tensor):
        index = index.to(kw["cache_index"].dtype)
    kw["cache_index"] = index




class OpTypes(Enum):
    NA = 0
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9


def get_eb_threshold(dtype: torch.dtype):
    if dtype in [torch.bfloat16]:
        return 2 ** (-7)
    if dtype in [torch.float16]:
        return 2 ** (-10)
    if dtype in [torch.float32]:
        return 2 ** (-14)
    return 0


def get_err_threshold(op_type: OpTypes, dtype: torch.dtype):
    err_threshold = 0
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2 ** (-7)
        if dtype in [torch.float16]:
            err_threshold = 2 ** (-8)
        if dtype in [torch.float32]:
            err_threshold = 2 ** (-11)
    if op_type in [OpTypes.CV_FUSION]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2 ** (-8)
        if dtype in [torch.float16]:
            err_threshold = 2 ** (-11)
        if dtype in [torch.float32]:
            err_threshold = 2 ** (-14)
    return err_threshold


def get_mare(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.max(abs_error.flatten())


def get_mere(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.mean(abs_error)


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    return torch.sqrt(torch.mean(sqr_err))


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    return torch.mean(actual_error / golden_nmax)


def compare_cv(golden: torch.Tensor, npu_ref: torch.Tensor, actual: torch.Tensor):
    op_type = OpTypes.CV_FUSION
    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)
    mare_npu = get_mare(golden, actual)
    mare_gpu = get_mare(golden, npu_ref)
    mere_npu = get_mere(golden, actual)
    mere_gpu = get_mere(golden, npu_ref)
    rmse_npu = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, npu_ref)
    mare_rate = mare_npu / max(mare_gpu, err_threshold)
    mere_rate = mere_npu / max(mere_gpu, err_threshold)
    rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)
    eb = get_eb(npu_ref, actual)
    return bool((mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2) and (eb < eb_threshold))


def compare_int8(actual: torch.Tensor, npu_ref: torch.Tensor) -> bool:
    diff = actual.to(torch.float32).flatten() - npu_ref.to(torch.float32).flatten()
    return bool(torch.max(torch.abs(diff)) <= 1)


def compare_fp8_e4m3(
    golden: torch.Tensor, npu_ref: torch.Tensor, actual: torch.Tensor
) -> Tuple[bool, str]:
    """fp8e4m3 double-benchmark on uint8 code points.

    Baseline ±2 vs golden is too brittle for mxfp8 query outliers; when the NPU small-op
    ref already diverges from CPU golden, allow the same slack on actual vs golden.
    Also accept a tight match to either reference.
    """
    def _codes(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).to(torch.int16)

    g8, r8, a8 = _codes(golden), _codes(npu_ref), _codes(actual)
    diff_ag = int(torch.max(torch.abs(a8 - g8)))
    diff_ar = int(torch.max(torch.abs(a8 - r8)))
    diff_gr = int(torch.max(torch.abs(g8 - r8)))
    tol = max(2, diff_gr)
    ok = bool(diff_ag <= tol or diff_ar <= 2)
    detail = f"max_code_diff_vs_golden={diff_ag}/{tol} vs_npu={diff_ar} golden_vs_npu={diff_gr}"
    return ok, detail


def _infer_quant_modes(kw: Dict[str, Any]) -> Tuple[int, int]:
    """V1/V2 have no quant-mode attributes, so derive the mode from the input dtypes.
    The same derivation matches the attributes V3 carries."""
    token_x, w_uq = kw["token_x"], kw["weight_uq_qr"]
    if token_x.dtype == torch.int8:
        wq = 2
    elif w_uq.dtype == torch.int8:
        wq = 1
    elif w_uq.dtype == torch.float8_e4m3fn:
        wq = 3
    else:
        wq = 0

    kv_dtype = kw["kv_cache"].dtype
    if kv_dtype == torch.int8:
        kvq = 2
    elif kv_dtype == torch.float8_e4m3fn:
        kvq = 1
    else:
        kvq = 0
    return wq, kvq


def _infer_token_layout(kw: Dict[str, Any], mode: str) -> Tuple[int, int, int, int, bool]:
    token = kw["token_x"]
    if token.dim() == 3:
        b, s, he = int(token.shape[0]), int(token.shape[1]), int(token.shape[2])
        return b * s, he, b, s, True
    t, he = int(token.shape[0]), int(token.shape[1])
    seq = kw.get("actual_seq_len")
    if isinstance(seq, torch.Tensor) and seq.numel() > 0:
        b = int(seq.numel())
        return t, he, b, t // max(b, 1), False
    if mode == "BSND":
        return t, he, t, 1, True
    b = 2 if t % 2 == 0 else 1
    return t, he, b, t // b, False


def _build_case_config(input_data: InputDataset) -> CaseConfig:
    kw = input_data.kwargs
    hckv = int(kw["weight_uk"].shape[2])
    wq, kvq = _infer_quant_modes(kw)
    mode = str(kw.get("cacheMode", "PA_BSND"))
    t, he, b, _s, unmerged = _infer_token_layout(kw, mode)
    kv = kw["kv_cache"]
    if kv.dim() == 3:
        nkv = int(kv.shape[1])
        block_num, block_size = int(kv.shape[0]), 1
    else:
        nkv = int(kv.shape[2])
        block_num, block_size = int(kv.shape[0]), int(kv.shape[1])
    return CaseConfig(
        t=t,
        he=he,
        hcq=int(kw["weight_dq"].shape[1]),
        hckv=hckv,
        d=int(kw["weight_uk"].shape[1]),
        dr=int(kw["weight_dkv_kr"].shape[1]) - hckv,
        n=int(kw["weight_uk"].shape[0]),
        nkv=nkv,
        block_num=block_num,
        block_size=block_size,
        eps_cq=float(kw.get("rmsnormEpsilonCq", 1e-5)),
        eps_ckv=float(kw.get("rmsnormEpsilonCkv", 1e-5)),
        qc_qr_scale=float(kw.get("qcQrScale", 1.0)),
        kc_scale=float(kw.get("kcScale", 1.0)),
        do_rope=_resolve_enable_rope(kw.get("rope_cos"), kw.get("rope_sin")),
        seed=CACHE_INDEX_SEED,
        weight_quant_mode=wq,
        kv_cache_quant_mode=kvq,
        query_quant_mode=int(kw.get("queryQuantMode", 0)),
        ckvkr_repo_mode=int(kw.get("ckvkrRepoMode", 0)),
        quant_scale_repo_mode=int(kw.get("quantScaleRepoMode", 0)),
        tile_size=int(kw.get("tileSize", 128)),
        query_norm_flag=False,
        rope_style=os.environ.get("MLA_PROLOG_ROPE_STYLE", "original"),
        cache_mode=mode,
        b=b,
        unmerged=unmerged,
    )


def _cast_outputs(outs: Sequence[torch.Tensor], kw: Dict[str, Any]) -> List[torch.Tensor]:
    """The golden computes in fp32; ATK sizes the aclnn output buffers from these dtypes,
    so they have to be the dtypes the operator actually writes."""
    qq = int(kw.get("queryQuantMode", 0))
    if qq == 1:
        # mxfp8 + kv pertensor：queryOut 是 per-token-head 的 fp8e4m3
        query_dtype = torch.float8_e4m3fn
    elif kw["token_x"].dtype == torch.int8:
        query_dtype = torch.bfloat16
    elif kw["token_x"].dtype in (torch.bfloat16, torch.float16):
        query_dtype = kw["token_x"].dtype
    else:
        query_dtype = torch.bfloat16
    rope_dtype = torch.bfloat16
    dtypes = [query_dtype, rope_dtype, kw["kv_cache"].dtype, kw["kr_cache"].dtype]
    return [t.to(dtype) for t, dtype in zip(outs, dtypes)]


def _move_to_npu(input_data: InputDataset, nz_names: Sequence[str] = ()):
    """Only the operator under test consumes FRACTAL_NZ weights; the torch_npu reference
    chain needs plain ND tensors it can matmul and copy back."""
    kw = input_data.kwargs
    for name, tensor in kw.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "npu":
            tensor = tensor.npu()
        if name in nz_names:
            tensor = cast_weight_nz(tensor, tensor.device)
        kw[name] = tensor


def _is_mxfp8_case(kw: Dict[str, Any]) -> bool:
    if int(kw.get("weightQuantMode", -1)) == 3:
        return True
    # 占位阶段 scale 仍是 uint8，靠权重 dtype 识别
    w_uq = kw.get("weight_uq_qr")
    return isinstance(w_uq, torch.Tensor) and w_uq.dtype == torch.float8_e4m3fn


def _rewrite_mxfp8_inputs(input_data: InputDataset, seed: int = CACHE_INDEX_SEED) -> bool:
    """ATK 造不出 e8m0 scale：用 golden.gen_inputs 整包替换 mxfp8 相关张量。

    参考节点和 pyaclnn 节点都走这里，且共用同一 seed，两边输入一致。
    """
    kw = input_data.kwargs
    if not _is_mxfp8_case(kw):
        return False

    # 属性先写进 kwargs，再 build cfg（queryQuantMode 等）
    cfg = _build_case_config(input_data)
    cfg.weight_quant_mode = 3
    cfg.kv_cache_quant_mode = int(kw.get("kvCacheQuantMode", cfg.kv_cache_quant_mode))
    cfg.query_quant_mode = int(kw.get("queryQuantMode", cfg.query_quant_mode))
    cfg.seed = seed

    generated = gen_inputs(cfg, device="cpu")
    for name, value in generated.items():
        kw[name] = value
    logging.info(
        "[mla_prolog][mxfp8] rewrote inputs via gen_inputs "
        "wq=%s kvq=%s qq=%s seed=%s scale_dtype=%s",
        cfg.weight_quant_mode, cfg.kv_cache_quant_mode, cfg.query_quant_mode,
        seed, kw["dequant_scale_x"].dtype,
    )
    return True


def _load_nnopbase() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(NNOPBASE_PATH), mode=ctypes.RTLD_GLOBAL)
    except Exception:  # noqa: BLE001
        home = os.environ.get("ASCEND_HOME_PATH", "")
        for rel in ("lib64", "x86_64-linux/lib64", "aarch64-linux/lib64"):
            path = os.path.join(home, rel, "libnnopbase.so")
            if os.path.isfile(path):
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)


def _acl_create_tensor(tensor: torch.Tensor, acl_dtype: int, acl_format: int = 2):
    """自建 aclTensor；ATK 的 TORCH_TO_ACLTYPE 没有 float8_e8m0fnu。"""
    lib = _load_nnopbase()
    lib.aclCreateTensor.restype = ctypes.POINTER(AclTensor)
    lib.aclCreateTensor.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,
    ]
    shape = tuple(int(x) for x in tensor.shape)
    stride = tuple(int(x) for x in tensor.stride())
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    stride_arr = (ctypes.c_int64 * len(stride))(*stride)
    ptr = lib.aclCreateTensor(
        shape_arr, len(shape), int(acl_dtype), stride_arr,
        int(tensor.storage_offset()), int(acl_format),
        shape_arr, len(shape),
        ctypes.c_void_p(tensor.untyped_storage().data_ptr()),
    )
    if not ptr:
        raise RuntimeError(
            f"aclCreateTensor failed: shape={shape} stride={stride} acl_dtype={acl_dtype}"
        )
    return ptr


class FunctionMlaPrologBaseApi(BaseApi):
    """CPU / torch_npu reference node."""

    op_version: str = "v1"

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.set_seeds(parse_case_seed(_task_case_name(task_result)))

    def set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)

    def init_by_input_data(self, input_data: InputDataset):
        seed = parse_case_seed(_task_case_name(self.task_result))
        self.set_seeds(seed)
        _rewrite_mxfp8_inputs(input_data, seed=seed)
        _apply_deterministic_cache_index(input_data, seed=seed)
        self.enable_rope = _resolve_enable_rope(
            input_data.kwargs.get("rope_cos"), input_data.kwargs.get("rope_sin")
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = [n for n in self.task_result.nodes.nodes if n != post_node]
        remote_manager = RemoteManager(self.task_result.case_config, post_node, remote_nodes[0])
        _, remote_dir = remote_manager.get_local_and_remote_dir()

        kw = input_data.kwargs
        cfg = _build_case_config(input_data)
        cpu_inp = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}

        cpu_outs = _cast_outputs(cal_mlaprolog_cpu(cpu_inp, cfg), cpu_inp)
        torch.save(cpu_outs, f"{remote_dir}/result_cpu.bin")

        if self.device != "npu":
            return cpu_outs[0], cpu_outs[1]

        # The double benchmark needs a second reference from the same inputs. Running it here
        # rather than on its own node keeps the run at two nodes: celery turns a group of two
        # reference nodes into a chord, and the chord_unlock task lands on the default queue
        # that no ATK worker consumes, so the aclnn node would never start.
        _move_to_npu(input_data)
        npu_outs = cal_mlaprolog_npu(dict(kw), cfg, device=str(kw["token_x"].device))
        npu_outs = _cast_outputs([x.cpu() for x in npu_outs], cpu_inp)
        torch.save(npu_outs, f"{remote_dir}/result_npu.bin")
        return npu_outs[0].npu(), npu_outs[1].npu()


class AclnnMlaPrologBaseApi(AclnnBaseApi):
    """pyaclnn node: the operator under test."""

    op_version: str = "v1"

    def _layout(self):
        return VERSION_LAYOUT[self.op_version]

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name in self._layout()[3]:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def _case_name(self) -> str:
        case_config = getattr(self.task_result, "case_config", None)
        return str(getattr(case_config, "name", "") or "")

    def torch_tensor_to_acl(self, tensor, fmt=AclFormat.ACL_FORMAT_ND):
        """ATK 不认识 float8_e8m0fnu，scale 张量走自建 aclCreateTensor。"""
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float8_e8m0fnu:
            held = tensor.contiguous()
            self._acl_holders.append(held)
            ptr = _acl_create_tensor(
                held, int(AclDataType.ACL_FLOAT8_E8M0), int(fmt)
            )
            logging.info(
                "[mla_prolog][mxfp8] acl e8m0 tensor shape=%s", tuple(held.shape)
            )
            return AclTensorStruct(
                tensor=ptr,
                addr=int(held.untyped_storage().data_ptr()),
                pytensor=held,
                data_size=int(held.numel()),
            )
        return super().torch_tensor_to_acl(tensor, fmt)

    def _apply_noncontig_cache(self, input_data: InputDataset):
        """把 kv/kr cache 换成只有 dim0 非连续的视图。

        参考节点仍然拿到逻辑值，所以 golden 不受布局影响；这里改的只是被测算子
        看到的物理布局。ATK 建 aclTensor 时会带上 torch tensor 的真实 stride，
        算子 Tiling 侧再用 GetInputStride 取到 stride(0)。

        MLA_PROLOG_NC_SELFTEST=1 时故意把算子的入参换成同一 backing 上的连续视图，
        制造"stride 没生效"的场景，用来验证下面的等价性和 padding 检查确实会报错。
        """
        kw = input_data.kwargs
        self.nc_caches: Dict[str, Optional[NonContigCache]] = {"kv_cache": None, "kr_cache": None}
        if self.nc_factor <= 1:
            return
        for name in ("kv_cache", "kr_cache"):
            tensor = kw.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            cache = make_dim0_noncontig(tensor, self.nc_factor)
            self.nc_caches[name] = cache
            # 自检时喂连续切片（等价于 stride 丢失），正常时喂真正的非连续视图
            kw[name] = cache.backing[: cache.view.shape[0]] if self.nc_selftest else cache.view
        logging.info(
            "[mla_prolog][nc] %s selftest=%s", describe(self.nc_caches), self.nc_selftest
        )

    def init_by_input_data(self, input_data: InputDataset):
        self._acl_holders: List[torch.Tensor] = []
        seed = parse_case_seed(self._case_name())
        _rewrite_mxfp8_inputs(input_data, seed=seed)
        # gen_inputs 塞进 kwargs 的 _dequant_scale_*_fp 只给 golden 用，不能进 acl 原型
        kw = input_data.kwargs
        for key in [k for k in list(kw) if k.startswith("_")]:
            kw.pop(key)

        _apply_deterministic_cache_index(input_data, seed=seed)
        self.enable_rope = _resolve_enable_rope(
            input_data.kwargs.get("rope_cos"), input_data.kwargs.get("rope_sin")
        )
        self.nc_factor = parse_stride_factor(self._case_name())
        self.nc_selftest = os.environ.get("MLA_PROLOG_NC_SELFTEST", "0") == "1"
        self.query_quant_mode = int(kw.get("queryQuantMode", 0))

        tensor_slots, attr_slots, _, nz_names = self._layout()
        declared = [n for n, v in kw.items() if isinstance(v, torch.Tensor)]
        _move_to_npu(input_data, nz_names)
        self._apply_noncontig_cache(input_data)
        # kvCacheRef / krCacheRef are updated in place, so keep the device tensors around to
        # read the operator result back after the call. 非连续用例读的始终是逻辑视图。
        self.cache_refs = tuple(
            self.nc_caches[name].view if self.nc_caches.get(name) else input_data.kwargs[name]
            for name in ("kv_cache", "kr_cache")
        )

        if self.query_quant_mode == 1:
            token = kw["token_x"]
            t = int(token.shape[0] * token.shape[1]) if token.dim() >= 3 else int(token.shape[0])
            n = int(kw["weight_uk"].shape[0])
            self.dequant_q_nope = torch.zeros(
                (t, n, 1), dtype=torch.float32, device=kw["token_x"].device
            )
        else:
            self.dequant_q_nope = None

        # 对齐 mlapo：输出缓冲用 zeros 而不是 empty，避免未写满时读到脏数据。
        input_args, output_packages = self._init_with_zero_outputs(input_data)
        # optional 输出槽位自己拼：qq=1 时 dequantScaleQNopeOut 必须非空
        input_args = self._expand_to_prototype(
            list(input_args), declared, tensor_slots, attr_slots, len(output_packages), 0
        )
        if self.dequant_q_nope is not None:
            dq_struct = self.torch_tensor_to_acl(
                self.dequant_q_nope, AclFormat.ACL_FORMAT_ND
            )
            input_args.append(dq_struct.tensor)
            self._acl_holders.append(self.dequant_q_nope)
        else:
            input_args.append(_null_tensor_ptr())
        input_args.append(_null_tensor_ptr())  # queryNormOutOptional
        input_args.append(_null_tensor_ptr())  # dequantScaleQNormOutOptional
        # Sync expanded args back to backend so ATK's before_call type check sees the
        # full prototype-aligned list (21 tensor slots + 11 attrs + 5 outputs), not the
        # raw yaml-declared count.
        self.backend.input_args = input_args
        return input_args, output_packages

    @staticmethod
    def _expand_to_prototype(input_args, declared, tensor_slots, attr_slots,
                             n_outputs, null_outputs):
        """ATK converts the declared inputs positionally; splice in a null aclTensor for every
        optional slot the case leaves out so the argument list matches the aclnn prototype."""
        unknown = [n for n in declared if n not in tensor_slots]
        if unknown:
            raise ValueError(f"case declares tensors that are not in the prototype: {unknown}")
        if list(declared) != [n for n in tensor_slots if n in declared]:
            raise ValueError("case tensors must be declared in aclnn prototype order")

        n_declared = len(declared)
        tail = input_args[n_declared:]
        if len(tail) != len(attr_slots) + n_outputs:
            raise ValueError(
                f"expected {len(attr_slots)} attrs + {n_outputs} outputs, got {len(tail)}"
            )

        it = iter(input_args[:n_declared])
        remaining = list(declared)
        expanded = []
        for slot in tensor_slots:
            if remaining and remaining[0] == slot:
                expanded.append(next(it))
                remaining.pop(0)
            else:
                expanded.append(_null_tensor_ptr())
        expanded.extend(tail)
        expanded.extend(_null_tensor_ptr() for _ in range(null_outputs))
        return expanded

    def _init_with_zero_outputs(self, input_data: InputDataset):
        """参考 mlapo_atk/exec.py：把 ATK 的 output 分配改成 torch.zeros。"""
        backend = self.backend
        if not hasattr(backend, "convert_output_data"):
            return super().init_by_input_data(input_data)

        def my_convert_output_data(backend_instance, output_data, index):
            if isinstance(output_data, (list, tuple)):
                data_list = []
                for tmp in output_data:
                    data_list.extend(my_convert_output_data(backend_instance, tmp, index))
                return [nnopbase.create_x_list(data_list)]
            dtype = output_data.dtype
            if dtype not in TORCH_TO_ACLTYPE:
                raise ValueError(f"TORCH_TO_ACLTYPE不支持的dtype：{dtype}")
            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            empty_tensor = torch.zeros(tuple(output_data.shape), dtype=torch_dtype, device="npu")
            backend_instance.output_cache.append(empty_tensor)
            cur_index = index + len(backend_instance.input_args)
            fmt = backend_instance.get_format(index=cur_index)
            storage_shape = backend_instance.get_storage_shape(index=cur_index)
            out_tensor = nnopbase.create_acl_tensor(empty_tensor, fmt, storage_shape)
            return [out_tensor]

        orig = backend.convert_output_data
        try:
            backend.convert_output_data = types.MethodType(my_convert_output_data, backend)
            return super().init_by_input_data(input_data)
        finally:
            backend.convert_output_data = orig

    def __call__(self):
        torch.npu.synchronize()
        self.backend.aclnn_x_get_workspace_size()
        torch.npu.synchronize()
        self.backend.aclnn_x()
        torch.npu.synchronize()

    def after_call(self, output_packages):
        torch.npu.synchronize()
        outs = list(super().after_call(output_packages))
        torch.npu.synchronize()
        kv_cache, kr_cache = self.cache_refs
        result = [outs[0].cpu(), outs[1].cpu(), kv_cache.cpu(), kr_cache.cpu()]

        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = [n for n in self.task_result.nodes.nodes if n != post_node]
        remote_manager = RemoteManager(self.task_result.case_config, post_node, remote_nodes[0])
        local_dir, _ = remote_manager.get_local_and_remote_dir()
        tmp_path = os.path.join(local_dir, "aclnn_tmp")
        os.makedirs(tmp_path, exist_ok=True)
        torch.save(result, f"{tmp_path}/result.bin")

        padding_ok, padding_detail = padding_report(self.nc_caches)
        meta = {
            "stride_factor": self.nc_factor,
            "selftest": self.nc_selftest,
            "kv_stride0": int(kv_cache.stride(0)) if kv_cache.dim() else 0,
            "kr_stride0": int(kr_cache.stride(0)) if kr_cache.dim() else 0,
            "padding_ok": padding_ok,
            "padding_detail": padding_detail,
        }
        torch.save(meta, f"{tmp_path}/nc_meta.bin")
        logging.info("[mla_prolog][nc] %s", meta)
        return tuple(outs)


class CvFusedMlaPrologAccuracyCompare(DoubleBenchmarkAccuracyCompare):
    """query / query_rope / kv_cache / kr_cache under the CV-fusion double benchmark."""

    output_names = ("query", "query_rope", "kv_cache", "kr_cache")

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        local_dir, remote_dir = self.remote_manager.get_local_and_remote_dir()
        cpu_result = torch.load(f"{remote_dir}/result_cpu.bin")
        npu_result = torch.load(f"{remote_dir}/result_npu.bin")
        aclnn_result = torch.load(f"{local_dir}/aclnn_tmp/result.bin")

        dump_path = os.environ.get("MLA_PROLOG_DUMP_COMPARE", "").strip()
        if dump_path:
            try:
                case_name = ""
                tr = getattr(self, "task_result", None)
                if tr is not None:
                    case_name = _task_case_name(tr)
                payload = {
                    "case": case_name or remote_dir,
                    "cpu": [t.detach().cpu() for t in cpu_result],
                    "npu": [t.detach().cpu() for t in npu_result],
                    "aclnn": [t.detach().cpu() for t in aclnn_result],
                }
                os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
                torch.save(payload, dump_path)
                logging.info("[mla_prolog] dumped compare tensors to %s", dump_path)
            except Exception as exc:  # noqa: BLE001
                logging.exception("[mla_prolog] dump compare failed: %s", exc)

        failures = []
        for i, name in enumerate(self.output_names):
            golden, ref, actual = cpu_result[i].cpu(), npu_result[i].cpu(), aclnn_result[i].cpu()
            if actual.dtype == torch.int8:
                ok = compare_int8(actual, ref)
                detail = f"max_abs_diff_vs_npu={torch.max(torch.abs(actual.float() - ref.float())):.4g}"
            elif actual.dtype == torch.float8_e4m3fn:
                ok, detail = compare_fp8_e4m3(golden, ref, actual)
            else:
                ok = compare_cv(golden, ref, actual)
                detail = (f"mare={get_mare(golden, actual):.4g}/{get_mare(golden, ref):.4g} "
                          f"mere={get_mere(golden, actual):.4g}/{get_mere(golden, ref):.4g} "
                          f"rmse={get_rmse(golden, actual):.4g}/{get_rmse(golden, ref):.4g} "
                          f"eb={get_eb(ref, actual):.4g}")
            logging.info("[mla_prolog] %s %s %s", name, "PASS" if ok else "FAIL", detail)
            if not ok:
                failures.append(name)
                # 绝对/相对误差分布，便于区分「脏读/未同步」与「真精度偏差」
                a = actual.to(torch.float32).reshape(-1)
                g = golden.to(torch.float32).reshape(-1)
                r = ref.to(torch.float32).reshape(-1)
                abs_ag = torch.abs(a - g)
                abs_ar = torch.abs(a - r)
                rel_ag = abs_ag / (torch.abs(g) + MIN_ERR)
                logging.info(
                    "[mla_prolog][diff] %s vs_golden: max_abs=%.6g mean_abs=%.6g "
                    "p99_abs=%.6g max_rel=%.6g | vs_npu_ref: max_abs=%.6g mean_abs=%.6g | "
                    "golden_|max|=%.6g actual_|max|=%.6g nonzero_actual=%d/%d",
                    name,
                    float(abs_ag.max()), float(abs_ag.mean()),
                    float(torch.quantile(abs_ag, 0.99)), float(rel_ag.max()),
                    float(abs_ar.max()), float(abs_ar.mean()),
                    float(g.abs().max()), float(a.abs().max()),
                    int((a != 0).sum()), int(a.numel()),
                )
                # Element-level localization: count, coords, bf16 bits, ULP distance.
                mis = torch.nonzero(abs_ag > 0, as_tuple=False).flatten()
                n_mis = int(mis.numel())
                cpu_eq_npu = bool(torch.equal(golden.to(actual.dtype), ref.to(actual.dtype)))
                logging.info(
                    "[mla_prolog][elem] %s n_mismatch=%d/%d cpu_eq_npu_ref=%s shape=%s",
                    name, n_mis, int(a.numel()), cpu_eq_npu, tuple(actual.shape),
                )
                act_bf = actual.detach().cpu().to(torch.bfloat16).reshape(-1)
                gold_bf = golden.detach().cpu().to(torch.bfloat16).reshape(-1)
                ref_bf = ref.detach().cpu().to(torch.bfloat16).reshape(-1)
                act_u = act_bf.view(torch.int16)
                gold_u = gold_bf.view(torch.int16)
                ref_u = ref_bf.view(torch.int16)
                for k in mis[:8].tolist():
                    coord = tuple(int(x) for x in np.unravel_index(int(k), tuple(actual.shape)))
                    av, gv, rv = float(act_bf[k]), float(gold_bf[k]), float(ref_bf[k])
                    ab, gb, rb = int(act_u[k].item()) & 0xFFFF, int(gold_u[k].item()) & 0xFFFF, int(ref_u[k].item()) & 0xFFFF
                    ulp_ag = abs(ab - gb) if (ab >> 15) == (gb >> 15) else -1
                    ulp_ar = abs(ab - rb) if (ab >> 15) == (rb >> 15) else -1
                    mag = max(abs(av), abs(gv), 1e-30)
                    exp = int(math.floor(math.log2(mag))) if mag > 0 else -126
                    bf16_ulp = 2.0 ** (max(exp, -126) - 7)
                    logging.info(
                        "[mla_prolog][elem] %s idx=%d coord=%s fused=%g(0x%04x) "
                        "cpu=%g(0x%04x) npu_ref=%g(0x%04x) |diff|=%g ulp_vs_cpu=%s "
                        "ulp_vs_npu=%s bf16_ulp_at_mag=%g",
                        name, int(k), coord, av, ab, gv, gb, rv, rb,
                        abs(av - gv), ulp_ag, ulp_ar, bf16_ulp,
                    )

        meta_path = f"{local_dir}/aclnn_tmp/nc_meta.bin"
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path)
            if int(meta.get("stride_factor", 1)) > 1:
                # 非连续用例除了数值等价，还必须证明算子没有越界写到首轴空洞里。
                ok = bool(meta.get("padding_ok"))
                logging.info(
                    "[mla_prolog] nc_padding %s factor=%s kv_stride0=%s kr_stride0=%s %s",
                    "PASS" if ok else "FAIL", meta.get("stride_factor"),
                    meta.get("kv_stride0"), meta.get("kr_stride0"),
                    meta.get("padding_detail"),
                )
                if not ok:
                    failures.append("nc_padding")

        if not failures:
            return AccuracyConfig(result=True, error_info="Accuracy pass")
        return AccuracyConfig(result=False, error_info="Accuracy failed: " + ",".join(failures))


def register_mla_prolog_accuracy(registry_name: str):
    @ACCURACY_REGISTRY.register(registry_name)
    class _Compare(CvFusedMlaPrologAccuracyCompare):
        pass


@register("function_aclnn_mla_prolog_v3")
class FunctionMlaPrologV3Api(FunctionMlaPrologBaseApi):
    op_version = "v3"


@register("function_pyaclnn_mla_prolog_v3")
class AclnnMlaPrologV3Api(AclnnMlaPrologBaseApi):
    op_version = "v3"

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize("
            "const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, "
            "const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, "
            "const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos, "
            "aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, "
            "const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional, "
            "const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional, "
            "const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, "
            "const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional, "
            "const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, "
            "char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode, "
            "int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, "
            "double qcQrScale, double kcScale, const aclTensor *queryOut, const aclTensor *queryRopeOut, "
            "const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional, "
            "const aclTensor *dequantScaleQNormOutOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )


register_mla_prolog_accuracy("cv_fused_double_benchmark_aclnn_mla_prolog_v3")
