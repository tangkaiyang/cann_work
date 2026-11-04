# ATK executor for flash_attn_combine.
#
# CPU  : high-precision (float64) golden, then cast to out_dtype.
# NPU  : the shipped torch extension (flash_attn_npu_3 / flash_attn_npu_4).
#
# Both paths return (out, lse); ATK compares them output-by-output.

import importlib
import os
import sys

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


OUT_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_dtype(value):
    if value is None:
        return torch.float32
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return OUT_DTYPES.get(value.lower(), torch.float32)
    # ATK may hand back an enum-ish object; fall back to its string form.
    return OUT_DTYPES.get(str(value).lower(), torch.float32)


def _pick(kwargs, args, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if index is not None and len(args) > index:
        return args[index]
    return default


def _as_bool(value, default=False):
    """ATK hands bool attrs back as bool / int / str depending on the path."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _alloc_out(out_partial, out_dtype):
    """Build the preallocated `out` buffer with the shape the kernel expects.

    Dense (5D) out_partial -> (batch, seqlen, nheads, headdim)
    Varlen (4D) out_partial -> (total_q, nheads, headdim)
    """
    if out_partial.dim() == 5:
        shape = (out_partial.shape[1], out_partial.shape[2], out_partial.shape[3], out_partial.shape[4])
    else:
        shape = (out_partial.shape[1], out_partial.shape[2], out_partial.shape[3])
    # A few leading zeros are legal (kernel returns early), keep the shape usable.
    if any(dim == 0 for dim in shape):
        shape = tuple(dim if dim else 1 for dim in shape)
    return torch.empty(shape, dtype=out_dtype, device=out_partial.device)


def _build_cu_seqlens(seqlen, device):
    """1-D int32 with >= 2 elements, monotonically increasing."""
    return torch.tensor([0, seqlen], dtype=torch.int32, device=device)


def _build_seqused(row_count, device):
    """1-D int32 with >= 1 element."""
    return torch.tensor([row_count], dtype=torch.int32, device=device)


def _build_virtual_batch_idx(batch, device):
    """Identity mapping for the dense layout: numel must equal batch_size."""
    return torch.arange(batch, dtype=torch.int32, device=device)


def _to_seqlen_contiguous(tensor):
    """Re-lay lse_partial so its seqlen dim is contiguous.

    A plain transpose+contiguous is a no-op when the seqlen (or head) dim is 1,
    because size-1 strides are ignored by torch's contiguity check. Build an
    explicit (..., H, S) buffer instead so stride(-2) is always 1.
    """
    if tensor.dim() < 2:
        # 0/1 维张量没有 seqlen 维，stride(-2) 会抛
        # IndexError: Dimension specified as -1 but tensor has no dimensions
        return tensor
    if tensor.stride(-2) == 1:
        return tensor
    shape = list(tensor.shape)
    buf = torch.empty(shape[:-2] + [shape[-1], shape[-2]], dtype=tensor.dtype, device=tensor.device)
    buf.copy_(tensor.transpose(-1, -2))
    return buf.transpose(-1, -2)


def _reference(out_partial, lse_partial, out_dtype):
    op = out_partial.to(torch.float64)
    lp = lse_partial.to(torch.float64)
    lse = torch.logsumexp(lp, dim=0)
    scale = torch.exp(lp - lse)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    out = (scale.unsqueeze(-1) * op).sum(0)
    return out.to(out_dtype), lse.to(torch.float32)


def _small_op_combine(out_partial, lse_partial, out_dtype):
    """Second implementation used as the benchmark (small-op composition).

    Mirrors what a GPU/reference implementation would do: the same math
    expressed with logsumexp/exp/mul/sum instead of the fused kernel. Its error
    against the CPU golden is the `Benchmark_*` term of the L2 ratio.
    """
    op = out_partial.to(torch.float32)
    lp = lse_partial.to(torch.float32)
    lse = torch.logsumexp(lp, dim=0)
    weight = torch.exp(lp - lse)
    weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
    out = (weight.unsqueeze(-1) * op).sum(0)
    return out.to(out_dtype), lse.to(torch.float32)


def _nv_combine_fn():
    """Locate NVIDIA flash-attention's combine function, if installed.

    FA3 (hopper) exposes it from `flash_attn_interface`; FA4 (cute) from
    `flash_attn.cute.interface`; some builds re-export it from `flash_attn`.
    """
    for module_name in ("flash_attn_interface", "flash_attn.cute.interface", "flash_attn"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, "flash_attn_combine", None)
        if fn is not None:
            return fn, module_name
    return None, None


class _FlashAttnCombineBase(BaseApi):
    module_name = None  # set by subclasses
    # FA3 的公开接口只透传 4 个参数；FA4 透传全部 8 个。
    # 参见仓库 README 与 flash_attn_npu_interface{,_950}.py。
    supports_extended_args = False

    def __init__(self, task_result=None):
        """Bind to the ATK task so `self.device` reflects the DUT node.

        Without this the base class keeps its default `device == "cpu"` and every
        node — including `--backend npu` — silently runs the CPU golden path, so
        the report ends up comparing CPU against CPU with no NPU column filled.
        """
        super(_FlashAttnCombineBase, self).__init__(task_result)

    def _extended_kwargs(self, kwargs, args, out_partial, out_dtype, device):
        """收集 FA4 独有的可选参数；FA3 返回空 dict。

        这些参数在 FA4 的 Python 签名里是可选关键字，yaml 里配成
        required: false，未配置时 ATK 不会放进 kwargs，此时保持 None。
        """
        if not self.supports_extended_args:
            return {}

        seqlen = out_partial.shape[2] if out_partial.dim() == 5 else out_partial.shape[1]
        batch = out_partial.shape[1] if out_partial.dim() == 5 else None
        row_count = out_partial.shape[1] if out_partial.dim() == 4 else None

        cu_raw = _pick(kwargs, args, "cu_seqlens", 4, None)
        seqused_raw = _pick(kwargs, args, "seqused", 5, None)
        vbi_raw = _pick(kwargs, args, "virtual_batch_idx", 6, None)
        return_lse_raw = _pick(kwargs, args, "return_lse", 7, None)

        # yaml 里配成 0 占位时表示"要造一个合法的元数据张量"，而不是真的传 0。
        # 注意：ATK 把占位 0 传进来时可能是 0 维张量，直接透传会让底层
        # fwd_combine 在 stride(-1) 上报
        # "IndexError: Dimension specified as -1 but tensor has no dimensions"。
        # 因此只有 dim>=1 的张量才视为真实输入，0 维按占位重建。
        # ATK 把 `type: attrs` 的整数元组以 Python list 形式传进来
        # （实测 kw[cu_seqlens]=[0]，不是张量）。这里统一成 int32 张量，
        # 0 维标量或空 list 视为"未指定"，改由 executor 造合法元数据。
        def _to_i32(value, fallback, min_numel=1):
            """把 ATK 传来的元数据统一成合法 int32 张量。

            yaml 里这些参数是 required:false + 极小的占位范围（如 [0]），
            占位值本身并不满足底层约束（例如 cu_seqlens 至少 2 个元素），
            因此长度不足 min_numel 时按占位处理、交给 fallback 重建。
            """
            if value is None:
                return None
            if torch.is_tensor(value):
                if value.dim() < 1 or value.numel() < min_numel:
                    return fallback()
                return value.to(dtype=torch.int32, device=device)
            if isinstance(value, (list, tuple)):
                if len(value) < min_numel:
                    return fallback()
                return torch.tensor(list(value), dtype=torch.int32, device=device)
            return fallback()

        # cu_seqlens 要求 1-D int32 且至少 2 个元素（起止偏移）
        cu_seqlens = _to_i32(cu_raw, lambda: _build_cu_seqlens(seqlen, device), min_numel=2)
        seqused = _to_i32(
            seqused_raw,
            lambda: _build_seqused(row_count if row_count is not None else seqlen, device),
        )
        # dense 布局要求 numel 严格等于 batch_size（C++ TORCH_CHECK）。
        # 长度依赖 out_partial 的 batch 维，yaml 无法静态声明，
        # 因此传进来的值长度不对时必须重建，不能原样透传。
        _vbi_n = batch if batch is not None else seqlen
        if batch is not None:
            # dense：必须恰好 batch_size 个元素，否则 C++ 直接报错
            virtual_batch_idx = _to_i32(
                vbi_raw, lambda: _build_virtual_batch_idx(_vbi_n, device), min_numel=_vbi_n
            )
            if virtual_batch_idx is not None and virtual_batch_idx.numel() != _vbi_n:
                virtual_batch_idx = _build_virtual_batch_idx(_vbi_n, device)
        else:
            virtual_batch_idx = _to_i32(
                vbi_raw, lambda: _build_virtual_batch_idx(_vbi_n, device)
            )

        return_lse = True if return_lse_raw is None else _as_bool(return_lse_raw, True)

        return {
            "cu_seqlens": cu_seqlens,
            "seqused": seqused,
            "virtual_batch_idx": virtual_batch_idx,
            "return_lse": return_lse,
        }

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = dict(getattr(input_data, "kwargs", {}) or {})
        args = list(getattr(input_data, "args", []) or [])

        out_partial = _pick(kwargs, args, "out_partial", 0)
        lse_partial = _pick(kwargs, args, "lse_partial", 1)

        if out_partial is None or lse_partial is None:
            raise RuntimeError("flash_attn_combine executor: missing tensor inputs")

        # 位置索引以 yaml 的 inputs 声明顺序为准（不是 Python 函数签名顺序）：
        #   FA3/FA4: [0]=out_partial [1]=lse_partial [2]=out_dtype
        #   FA4 追加: [3]=out [4]=cu_seqlens [5]=seqused [6]=virtual_batch_idx [7]=return_lse
        # 实测 ATK 恒以 kwargs 传入（args 为空），位置索引仅作兼容兜底。
        out_dtype_raw = _pick(kwargs, args, "out_dtype", 2, None)
        # 未指定 out_dtype 时，内核默认取 out_partial.dtype（fp32）作为输出类型
        out_dtype = _resolve_dtype(out_dtype_raw) if out_dtype_raw is not None else None
        eff_dtype = out_dtype if out_dtype is not None else out_partial.dtype

        # ---- out 预分配的统一判定（所有设备共用）----
        # C++ 语义：out 一旦传入，out.dtype 即权威，覆盖 out_dtype。
        # 判定必须在设备分支之前完成，否则 CPU golden 仍按 out_dtype 计算，
        # 会与 NPU 侧按 out.dtype 的输出因 dtype 不同而无法比对。
        # 有效性条件：形状严格等于 out_partial.shape[1:]（去掉 num_splits）。
        #   ATK 对 required:false 的 out 可能给出 0 维占位或形状不符的张量，
        #   这类必须丢弃，按"未指定 out"处理。
        prealloc_raw = _pick(kwargs, args, "out", 3 if self.supports_extended_args else 2, None)
        want_out_shape = tuple(out_partial.shape[1:]) if out_partial.dim() >= 2 else None
        prealloc = None
        if (
            torch.is_tensor(prealloc_raw)
            and prealloc_raw.dim() >= 1
            and prealloc_raw.numel() > 0
            and want_out_shape is not None
            and tuple(prealloc_raw.shape) == want_out_shape
        ):
            prealloc = prealloc_raw
            eff_dtype = prealloc.dtype  # out.dtype 权威
            out_dtype = prealloc.dtype if out_dtype is None else out_dtype

        if self.device in {"npu", "pyaclnn"}:
            # Import lazily so CPU-only nodes do not need torch_npu.
            try:
                module = __import__(self.module_name, fromlist=["flash_attn_combine"])
            except ModuleNotFoundError:
                # The repo ships the extension as an in-place build; allow the
                # ATK worker (launched from another directory) to find it.
                repo_root = os.environ.get("FLASH_ATTN_NPU_ROOT")
                if not repo_root:
                    raise
                sys.path.insert(0, repo_root)
                module = __import__(self.module_name, fromlist=["flash_attn_combine"])
            flash_attn_combine = module.flash_attn_combine

            dev = out_partial.npu().device
            op = out_partial.npu()
            # ATK materialises lse_partial as a contiguous (NS, B, S, H)
            # tensor. Old builds required the seqlen dim to be contiguous;
            # 0.5.0 addresses it by stride, so this is now a cheap no-op on
            # tensors that are already seqlen-contiguous.
            lse_kernel = _to_seqlen_contiguous(lse_partial.npu())

            ext = self._extended_kwargs(kwargs, args, out_partial, eff_dtype, dev)

            # prealloc / eff_dtype 已在函数开头统一解析（所有设备共用），
            # 这里只需把 out 搬到 NPU。再次强调：不要在这里改 out 的 dtype，
            # 否则会破坏 "out.dtype 权威" 语义。
            out = prealloc.npu() if prealloc is not None else None

            call_kwargs = {}
            if out is not None:
                call_kwargs["out"] = out
            elif out_dtype is not None:
                call_kwargs["out_dtype"] = out_dtype
            if self.supports_extended_args:
                call_kwargs.update(ext)

            result = flash_attn_combine(op, lse_kernel, **call_kwargs)

            if isinstance(result, tuple):
                out_res, lse_res = result[0], result[1]
            else:
                out_res, lse_res = result, None

            # return_lse=False 时底层返回 lse=None，这本身就是被测语义的一部分：
            # 断言"不返回 lse"。ATK 需要固定数量的张量输出做比对，因此用
            # None 声明输出"不存在"，与 golden 侧保持同一种表示，
            # 不能补全零占位（那会让 DUT 的 lse 恒为 0、与真实 lse 比较必然失败）。
            return out_res, lse_res

        if self.device == "gpu":
            # GPU node = the "second benchmark" of the L2 double-benchmark
            # standard. Prefer NVIDIA's own combine implementation when the
            # package is available; otherwise fall back to the small-op
            # composition (same math expressed with logsumexp/exp/mul/sum).
            lse_kernel = _to_seqlen_contiguous(lse_partial)
            nv_combine, module_name = _nv_combine_fn()
            if nv_combine is not None:
                out, lse = nv_combine(out_partial, lse_kernel, None, eff_dtype)
                return out, lse.to(torch.float32)
            return _small_op_combine(out_partial, lse_partial, eff_dtype)

        if self.device == "cpu":
            return _reference(out_partial.cpu(), lse_partial.cpu(), eff_dtype)

        raise RuntimeError(f"Unsupported device: {self.device}")


@register("function_flash_attn_combine")
class FlashAttnCombineApi(_FlashAttnCombineBase):
    """FA3 interface (flash_attn_npu_3).

    公开签名只有 4 个参数，因此这里不开启 extended args。
    底层 fwd_combine 虽然支持 cu_seqlens/seqused/return_lse，
    但仓库未在 Python 层暴露（见 common/ISSUE_提交草稿.md）。
    """

    module_name = "flash_attn_npu_3"
    supports_extended_args = False


@register("function_flash_attn_combine_v4")
class FlashAttnCombineV4Api(_FlashAttnCombineBase):
    """FA4 interface (flash_attn_npu_4).

    FA4 支持 out_partial 为 fp16/bf16/fp32，并额外接受
    cu_seqlens / seqused / virtual_batch_idx / return_lse。
    """

    module_name = "flash_attn_npu_4"
    supports_extended_args = True


class _FlashAttnCombineRefBase(BaseApi):
    """Small-op composition benchmark (second benchmark of the L2 standard)."""

    def __init__(self, task_result=None):
        super(_FlashAttnCombineRefBase, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = dict(getattr(input_data, "kwargs", {}) or {})
        args = list(getattr(input_data, "args", []) or [])
        out_partial = _pick(kwargs, args, "out_partial", 0)
        lse_partial = _pick(kwargs, args, "lse_partial", 1)
        out_dtype_raw = _pick(kwargs, args, "out_dtype", 2, None)

        # 与 DUT 侧保持一致的三条约定：
        #  1) out 存在时 out.dtype 权威，覆盖 out_dtype
        #  2) 未指定 out_dtype 时输出类型取 out_partial.dtype
        #  3) out 为 0 维占位张量时视为未指定
        prealloc = _pick(kwargs, args, "out", None, None)
        # 与 DUT 侧一致：out 形状是 out_partial.shape[1:]（去掉 num_splits）
        _want = tuple(out_partial.shape[1:]) if out_partial is not None else None
        if (
            torch.is_tensor(prealloc)
            and prealloc.dim() >= 1
            and prealloc.numel() > 0
            and _want is not None
            and tuple(prealloc.shape) == _want
        ):
            out_dtype = prealloc.dtype
        elif out_dtype_raw is not None:
            out_dtype = _resolve_dtype(out_dtype_raw)
        else:
            out_dtype = out_partial.dtype
        if out_partial is None or lse_partial is None:
            raise RuntimeError("flash_attn_combine reference: missing tensor inputs")

        out, lse = _small_op_combine(out_partial, lse_partial, out_dtype)

        # 与 DUT 侧对称：return_lse=False 时约定不返回 lse，
        # 两侧都用 None 表示"该输出不存在"，否则会比较真实 lse 与占位值。
        return_lse_raw = _pick(kwargs, args, "return_lse", 7, None)
        if return_lse_raw is not None and not _as_bool(return_lse_raw, True):
            return out, None
        return out, lse


@register("function_flash_attn_combine_ref")
class FlashAttnCombineRefApi(_FlashAttnCombineRefBase):
    """FA3 second benchmark (NPU/CPU small-op composition)."""


@register("function_flash_attn_combine_v4_ref")
class FlashAttnCombineV4RefApi(_FlashAttnCombineRefBase):
    """FA4 second benchmark (NPU/CPU small-op composition)."""