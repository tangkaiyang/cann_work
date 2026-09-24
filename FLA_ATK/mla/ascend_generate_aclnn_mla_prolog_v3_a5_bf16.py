# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ATK CaseGenerator for aclnnMlaPrologV3WeightNz on A5 (ascend950).

Active profile: BF16 non-quant only (0,0,0); quantization branches below are inactive.
The full A5 implementation supports the 16 legal (wq,kvq,qq)
combos in aclnnMlaPrologV4WeightNz.md (A5):
  wq=0 kvq=0
  wq=1 kvq=0/2/3
  wq=2/4/5 kvq=0 | kvq=1 qq=1 | kvq=3
  wq=3 kvq=0 | kvq=1 qq=1 | kvq=3
mxfp8 scales use uint8 placeholders; the plugin rewrites them to e8m0.
fp8-full uses fp8e4m3 + fp32 scales (same family as int8/hif8, not mxfp8).
hif8 tensors use ATK dtype uint8 (HiFloat8 1-byte storage; ATK has no hifp8
DataType). Plugin rewrites values and builds ACL_HIFLOAT8.

cacheMode (doc): PA_BSND / PA_NZ / PA_BLK_BSND / PA_BLK_NZ / BSND / TND.
Shapes, cacheIndex and actualSeqLen are bound to the picked mode.
pertoken-pergroup (kvq=3) only allows PA_BSND/BSND/TND, and forces
ckvkrRepoMode=quantScaleRepoMode=1 with Dtile=656 empty krCache.
kNopeClipAlphaOptional: only wq=1 (partial quant) or wq=2 (int8 full) with kvq=3;
wq=3/4/5 + kvq=3 must leave inputs[20] null (doc EZ0026).
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

HCKV = 512
DR = 64
NKV = 1
MXFP8_GRP = 32
HE_CHOICES = [1024, 2048, 4096, 7168]
# mxfp8 cases in validated JSON fix He=7168 (model path)
HE_MXFP8_CHOICES = [7168]
HCQ_CHOICES = [1536, 2048]
D_CHOICES = [128, 192]
T_CHOICES = [1, 2, 4, 8, 16]
B_CHOICES = [1, 2]
S_CHOICES = [1, 2, 4, 8]
BLOCK_SIZE_CHOICES = [16, 32, 64, 128, 256]
BLOCK_NUM = 2

CACHE_MODES = [
    "PA_BSND",
    "PA_NZ",
    "PA_BLK_BSND",
    "PA_BLK_NZ",
    "BSND",
    "TND",
]
PERTILE_CACHE_MODES = ["PA_BSND", "BSND", "TND"]
DTILE_PERTILE = 656  # Hckv + Dr*2 + (Hckv/tileSize)*4, tileSize=128

# Active profile: BF16 non-quant only (weightQuantMode, kvCacheQuantMode, queryQuantMode)
A5_SCENES: List[Tuple[int, int, int]] = [(0, 0, 0)]


def _legal_modes(kvq: int) -> List[str]:
    return PERTILE_CACHE_MODES if kvq == 3 else CACHE_MODES


# BF16 non-quant scene across all six cache modes
SCENE_MODE_PAIRS: List[Tuple[int, int, int, str]] = [
    (wq, kvq, qq, mode)
    for wq, kvq, qq in A5_SCENES
    for mode in _legal_modes(kvq)
]


def _null_optional(inp) -> None:
    inp.dtype = "bool"
    inp.shape = []
    inp.required = False


def _enable_optional(inp, dtype: str, shape: list, range_values=None) -> None:
    inp.dtype = dtype
    inp.shape = list(shape)
    inp.required = False
    if range_values is not None:
        inp.range_values = range_values


def _bind_k_nope_clip_alpha(case_config: CaseConfig, wq: int, kvq: int) -> None:
    """kNopeClipAlphaOptional：仅 wq=1(部分量化) 或 wq=2(int8 全量化) 且 kvq=3(pertile)。

    mxfp8/fp8/hif8 全量化 + pertile(wq=3/4/5, kvq=3) 与其它场景必须为 ATK null，否则 EZ0026。
    """
    inp = case_config.inputs[20]
    if wq in (1, 2) and kvq == 3:
        _enable_optional(inp, "fp32", [1], [0.5, 2.0])
    else:
        _null_optional(inp)


def _pick_bs() -> Tuple[int, int, int]:
    b = random.choice(B_CHOICES)
    s = random.choice(S_CHOICES)
    return b, s, b * s


@GENERATOR_REGISTRY.register("ascend_generate_aclnn_mla_prolog_v3_a5_bf16")
class MlaPrologV3A5Bf16Generator(CaseGenerator):
    """Shape / dtype binding for A5 precision cases (incl. mxfp8)."""

    _seq = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        seq = MlaPrologV3A5Bf16Generator._seq
        MlaPrologV3A5Bf16Generator._seq += 1
        wq, kvq, qq, mode = SCENE_MODE_PAIRS[seq % len(SCENE_MODE_PAIRS)]
        if mode in ("BSND", "PA_BLK_BSND", "PA_BLK_NZ"):
            b, s, t = _pick_bs()
        else:
            t = random.choice(T_CHOICES)
            b, s = 1, t

        he = random.choice(HE_MXFP8_CHOICES if wq == 3 else HE_CHOICES)
        hcq = random.choice(HCQ_CHOICES)
        d = random.choice(D_CHOICES)
        n = random.randint(1, 128)  # Doc: N 取值范围 1-128
        block_size = random.choice(BLOCK_SIZE_CHOICES)
        if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
            pages = b * math.ceil(s / block_size)
            block_num = max(BLOCK_NUM, pages)
            pa_slots = block_num
        else:
            block_num = max(BLOCK_NUM, math.ceil(t / block_size))
            pages = t
            pa_slots = block_num * block_size
        qk_dim = n * (d + DR)

        case_config.inputs[24].range_values = wq
        case_config.inputs[25].range_values = kvq
        case_config.inputs[26].range_values = qq
        case_config.inputs[23].range_values = mode
        case_config.inputs[27].range_values = 1 if kvq == 3 else 0
        case_config.inputs[28].range_values = 1 if kvq == 3 else 0
        case_config.inputs[29].range_values = 128
        case_config.inputs[21].range_values = 1e-5
        case_config.inputs[22].range_values = 1e-5
        case_config.inputs[30].range_values = 1.0
        case_config.inputs[31].range_values = 1.0

        if wq == 0:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "bf16"
        elif wq == 1:
            token_dtype = w_dq_dtype = w_dkv_dtype = "bf16"
            w_uq_dtype = "int8"
        elif wq == 2:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "int8"
        elif wq == 3:  # mxfp8 (E8M0 block scales)
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "fp8e4m3"
        elif wq == 4:  # fp8 full quant (FLOAT per-token/per-channel scales)
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "fp8e4m3"
        elif wq == 5:  # hif8 full quant
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "uint8"  # HiFloat8 storage
        else:
            raise ValueError(f"unsupported weightQuantMode={wq}")

        if kvq == 2:
            kv_dtype = kr_dtype = "int8"
        elif kvq == 1:
            kr_dtype = "bf16"
            if wq == 5:
                kv_dtype = "uint8"  # HiFloat8 storage
            elif wq == 2:
                kv_dtype = "int8"
            else:
                kv_dtype = "fp8e4m3"
        elif kvq == 3:
            kr_dtype = "bf16"
            if wq == 5:
                kv_dtype = "uint8"  # HiFloat8 storage
            elif wq in (3, 4):
                kv_dtype = "fp8e4m3"
            else:
                kv_dtype = "int8"
        else:
            kv_dtype = kr_dtype = "bf16"

        case_config.inputs[0].dtype = token_dtype
        case_config.inputs[0].shape = [b, s, he] if mode == "BSND" else [t, he]
        case_config.inputs[1].dtype = w_dq_dtype
        case_config.inputs[1].shape = [he, hcq]
        case_config.inputs[2].dtype = w_uq_dtype
        case_config.inputs[2].shape = [hcq, qk_dim]
        case_config.inputs[3].dtype = "bf16"
        case_config.inputs[3].shape = [n, d, HCKV]
        case_config.inputs[4].dtype = w_dkv_dtype
        case_config.inputs[4].shape = [he, HCKV + DR]
        case_config.inputs[5].dtype = "bf16"
        case_config.inputs[5].shape = [hcq]
        case_config.inputs[6].dtype = "bf16"
        case_config.inputs[6].shape = [HCKV]

        # V3：ropeSin/ropeCos 始终为有效输入（无 doRope 开关）
        case_config.inputs[7].dtype = "bf16"
        case_config.inputs[8].dtype = "bf16"
        if mode == "BSND":
            case_config.inputs[7].shape = [b, s, DR]
            case_config.inputs[8].shape = [b, s, DR]
        else:
            case_config.inputs[7].shape = [t, DR]
            case_config.inputs[8].shape = [t, DR]

        kv_last = DTILE_PERTILE if kvq == 3 else HCKV
        case_config.inputs[9].dtype = kv_dtype
        case_config.inputs[9].range_values = 0
        case_config.inputs[10].dtype = kr_dtype
        case_config.inputs[10].range_values = 0
        if kvq == 3:
            case_config.inputs[10].shape = [0]
        elif mode == "TND":
            case_config.inputs[10].shape = [t, NKV, DR]
        elif mode == "BSND":
            case_config.inputs[10].shape = [b, s, NKV, DR]
        else:
            case_config.inputs[10].shape = [block_num, block_size, NKV, DR]
        if mode == "TND":
            case_config.inputs[9].shape = [t, NKV, kv_last]
        elif mode == "BSND":
            case_config.inputs[9].shape = [b, s, NKV, kv_last]
        else:
            case_config.inputs[9].shape = [block_num, block_size, NKV, kv_last]

        if mode in ("BSND", "TND"):
            _null_optional(case_config.inputs[11])
        else:
            case_config.inputs[11].dtype = "int64"
            case_config.inputs[11].required = False
            if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
                case_config.inputs[11].shape = [pages]
                case_config.inputs[11].range_values = [0, max(pa_slots - 1, 0)]
            else:
                case_config.inputs[11].shape = [t]
                case_config.inputs[11].range_values = [0, max(pa_slots - 1, 0)]

        for idx in (12, 13, 14, 15, 16, 17, 18, 19, 20):
            _null_optional(case_config.inputs[idx])

        if wq in (2, 4, 5):
            _enable_optional(case_config.inputs[12], "fp32", [t, 1], [0.001, 0.01])
            _enable_optional(case_config.inputs[13], "fp32", [1, hcq], [0.001, 0.01])
            _enable_optional(case_config.inputs[14], "fp32", [1, qk_dim], [0.001, 0.01])
            _enable_optional(case_config.inputs[15], "fp32", [1, HCKV + DR], [0.001, 0.01])
        elif wq == 1:
            _enable_optional(case_config.inputs[14], "fp32", [1, qk_dim], [0.001, 0.01])
        elif wq == 3:
            # Placeholder shapes for mxfp8 e8m0 scales; plugin gen_inputs rewrites values.
            _enable_optional(case_config.inputs[12], "uint8", [t, he // MXFP8_GRP], [0, 254])
            _enable_optional(case_config.inputs[13], "uint8", [hcq, he // MXFP8_GRP], [0, 254])
            _enable_optional(case_config.inputs[14], "uint8", [qk_dim, hcq // MXFP8_GRP], [0, 254])
            _enable_optional(case_config.inputs[15], "uint8", [HCKV + DR, he // MXFP8_GRP], [0, 254])

        if kvq == 2:
            _enable_optional(case_config.inputs[16], "fp32", [1, HCKV], [0.001, 0.01])
            _enable_optional(case_config.inputs[17], "fp32", [1, DR], [0.001, 0.01])
        elif kvq == 1:
            _enable_optional(case_config.inputs[16], "fp32", [1], [0.001, 0.01])

        if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
            _enable_optional(case_config.inputs[19], "int32", [b], [1, t])

        _bind_k_nope_clip_alpha(case_config, wq, kvq)
        return case_config
