"""ATK generator for flash_attn_combine — FA3（自包含，不依赖任何共享模块）。

注册两个约束名：
    generate_flash_attn_combine_accuracy_fa3
    generate_flash_attn_combine_memory_fa3

FA3 的真实约束（flash-attention-npu 0.5.0, commit 82a9a8d）：
  csrc/ascend950/flash_attn_npu_3/flash_api.cpp:143    mha_combine
  flash_attn_npu_3/flash_attn_npu_interface.py:1333
  flash_attn_npu_3/flash_attn_npu_interface_950.py:1262

  * out_partial 只能是 fp32
      TORCH_CHECK(out_partial.scalar_type() == at::kFloat,
                  "Attention combine function only support fp32 data type")
  * lse_partial 必须 fp32，且形状 = out_partial.shape[:-1]
  * rank 只能是 4(TND) 或 5(BSND)
  * 1 <= num_splits <= 256
  * 公开签名只有 4 个参数：out_partial, lse_partial, out, out_dtype
    没有 cu_seqlens / seqused / return_lse，因此不生成这些输入。

为什么需要 generator：lse_partial 与 out_partial 形状强耦合
    lse_partial.shape == out_partial.shape[:-1]
ATK 的独立 shape 列表表达不了这个依赖。

UB 预算（决定 head_dim 上限）：
    per_row = ns_align*16 + (16 + out_bytes + 4) * d_align
    ns_align = ceil(num_splits/8)*8
    tile_rows * per_row <= 150 KiB，tile_rows 取 {32,24,16,8}（最小 8 档）
    超限报 "combine tile exceeds 150 KiB UB budget for head_dim=..."
注意 out_bytes 取决于「输出」dtype，不是 partial dtype。
"""

import zlib

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


MIB = 1024 * 1024

# FA3 的 out_partial 恒为 fp32
PARTIAL_DTYPE = "fp32"
PARTIAL_BYTES = 4

# 输出 dtype -> 字节宽度（用于 UB 计算）
OUT_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

NUM_SPLITS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 64, 128, 256)
BATCH = (1, 2, 3, 4, 6, 8, 12, 16, 32)
SEQLEN = (1, 2, 4, 7, 8, 17, 32, 33, 65, 128, 192, 256, 257, 512, 513, 1000)
HEADS = (1, 2, 3, 4, 5, 8, 12, 16, 32, 48, 64, 128)
HEAD_DIMS = (32, 48, 64, 80, 96, 113, 128, 160, 192, 200, 256)
LAYOUTS = (5, 4)

ACCURACY_MAX_ELEMS = 8 * 1024 * 1024

# memory 套件
MEMORY_NUM_SPLITS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MEMORY_HEADS = (8, 12, 16, 24, 32, 48, 64)
MEMORY_HEAD_DIMS = (64, 96, 128, 160, 192, 256)
MEMORY_BATCH = (1, 4)
MEMORY_MIN_PARTIAL_BYTES = 550 * MIB
MEMORY_TARGET_STEP_BYTES = 32 * MIB
# 单个张量字节上限（768 MiB）。FA3 的 partial 恒为 fp32，
# 折算成元素数即 192M，与原实现等价。
MEMORY_MAX_PARTIAL_BYTES = 768 * MIB

ND_RANGE = {
    "name": "nd",
    "mean": [-100.0, 100.0],
    "std": [1.0, 25.0],
    "range": None,
}

UB_BUDGET = 150 * 1024


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _pick(values, index):
    return values[index % len(values)]


def _iter_objects(case_config):
    """展开 case_config.inputs，兼容两种形态。

    * `type: tensor` / `type: attr` 的元素是带属性的对象
    * `type: attrs` 的元素被 ATK 包成 list（内含若干对象），需要拆开

    直接对 list 取 .type 会抛 AttributeError，所以统一走这个入口。
    """
    for item in case_config.inputs:
        if isinstance(item, (list, tuple)):
            for sub in item:
                if hasattr(sub, "type") and hasattr(sub, "name"):
                    yield sub
        elif hasattr(item, "type") and hasattr(item, "name"):
            yield item


def _tensor_map(case_config):
    return {
        item.name: item
        for item in _iter_objects(case_config)
        if item.type == "tensor" and item.name
    }


def _attr_map(case_config):
    return {
        item.name: item
        for item in _iter_objects(case_config)
        if item.type == "attr" and item.name
    }


def _seed(original_shape, out_dtype):
    marker = f"fa3|{len(original_shape)}|{tuple(original_shape)}|{out_dtype}"
    return zlib.crc32(marker.encode("utf-8"))


# ---------------------------------------------------------------------------
# UB 预算：算 head_dim 上限
# ---------------------------------------------------------------------------
def max_head_dim(num_splits, out_bytes):
    """给定 num_splits 与输出字节宽度，返回可用的最大 head_dim（已按 elem_align 对齐）。"""
    ns_align = (int(num_splits) + 7) // 8 * 8
    elem_align = 8 if out_bytes == 4 else 16
    coef = 16 + out_bytes + 4
    budget = UB_BUDGET // 8 - ns_align * 16
    if budget <= 0:
        return 0
    return (budget // coef) // elem_align * elem_align


def clamp_head_dim(num_splits, head_dim, out_bytes):
    """把 head_dim 收敛到 UB 允许范围内，保持对齐特性。"""
    limit = max_head_dim(num_splits, out_bytes)
    if limit <= 0:
        return 0
    if head_dim <= limit:
        return head_dim
    elem_align = 8 if out_bytes == 4 else 16
    return max(elem_align, (limit // elem_align) * elem_align)


# ---------------------------------------------------------------------------
# 共用设置（本文件内私有，不对外）
# ---------------------------------------------------------------------------
def _set_common_inputs(case_config, shape, out_dtype, partial_dtype):
    tensors = _tensor_map(case_config)
    attrs = _attr_map(case_config)

    out_partial = tensors.get("out_partial")
    lse_partial = tensors.get("lse_partial")
    if out_partial is None or lse_partial is None:
        return case_config

    out_partial.shape = list(shape)
    # FA3 硬约束：只接受 fp32
    out_partial.dtype = partial_dtype

    # 关键耦合：lse 比 out_partial 少一维，且必须是 fp32
    lse_partial.shape = list(shape[:-1])
    lse_partial.dtype = "fp32"

    out_dtype_attr = attrs.get("out_dtype")
    if out_dtype_attr is not None:
        out_dtype_attr.range_values = out_dtype

    return case_config


def _choose_out(case_config, shape, seed):
    """约 1/5 的 case 启用 out 预分配，覆盖 out.dtype 权威这条分支。"""
    tensors = _tensor_map(case_config)
    out_item = tensors.get("out")
    if out_item is None:
        return
    if seed % 5 != 0:
        out_item.shape = []
        return
    # C++ 语义：out 不含 num_splits 维，即 shape[1:]
    #   5D (NS,B,S,H,D) -> (B,S,H,D)
    #   4D (NS,T,H,D)   -> (T,H,D)
    out_item.shape = [int(v) for v in shape[1:]]


def _fit_accuracy_budget(shape):
    shape = list(shape)
    zero_positions = [i for i, value in enumerate(shape) if value == 0]
    if zero_positions:
        for i, value in enumerate(shape):
            if value == 0 and i != zero_positions[0]:
                shape[i] = 1
            elif value < 0:
                shape[i] = 1
        return shape

    while _prod(shape) > ACCURACY_MAX_ELEMS:
        token_axis = 2 if len(shape) == 5 else 1
        if shape[token_axis] > 1:
            shape[token_axis] = max(1, shape[token_axis] // 2)
        elif shape[0] > 1:
            shape[0] = max(1, shape[0] // 2)
        else:
            break
    return shape


def _accuracy_range(seed):
    # 注意：不生成 [0, 0] 全零范围。全零 partial 的输出恒为 0，
    # 相对误差分母趋零会被放大成假失败（历史上有 9 条此类误报）。
    bucket = seed % 100
    if bucket < 10:
        return [-0.001, 0.001]
    if bucket < 50:
        return [-5, 5]
    return dict(ND_RANGE)


def _lse_range(seed):
    # 约 5% 的全 -inf，对应整行被 mask；内核把 den clamp 到 1，输出 0 而非 NaN
    if seed % 19 == 0:
        return "-inf"
    if seed % 5 < 3:
        return [-5, 5]
    return dict(ND_RANGE)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
@GENERATOR_REGISTRY.register("generate_flash_attn_combine_accuracy_fa3")
class FlashAttnCombineFA3Accuracy(CaseGenerator):
    """FA3 accuracy 约束。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = _tensor_map(case_config)
        attrs = _attr_map(case_config)
        out_partial = tensors.get("out_partial")
        if out_partial is None:
            return case_config

        original_shape = [int(value) for value in (out_partial.shape or [])]
        attr = attrs.get("out_dtype")
        out_dtype = str(attr.range_values) if attr is not None else "fp16"
        seed = _seed(original_shape, out_dtype)
        is_boundary = bool(getattr(case_config, "is_boundary", False))

        selected_out_dtype = _pick(("fp16", "bf16", "fp32"), seed)
        out_bytes = OUT_BYTES.get(selected_out_dtype, 2)

        if is_boundary:
            rank = _pick(LAYOUTS, seed)
            num_splits = _pick(NUM_SPLITS, seed)
            head_dim = clamp_head_dim(num_splits, _pick(HEAD_DIMS, seed // 5), out_bytes)
            if head_dim <= 0:
                head_dim = _pick(HEAD_DIMS, seed // 5)
            if rank == 5:
                shape = [
                    num_splits,
                    1,
                    0 if seed % 3 == 0 else 1,
                    _pick(HEADS, seed // 3),
                    head_dim,
                ]
            else:
                shape = [
                    num_splits,
                    0 if seed % 3 == 0 else 1,
                    _pick(HEADS, seed // 3),
                    head_dim,
                ]
            out_partial.range_values = [-5, 5]
        else:
            rank = 5 if seed % 200 < 113 else 4
            num_splits = _pick(NUM_SPLITS, seed)
            heads = _pick(HEADS, seed // 3)
            head_dim = clamp_head_dim(num_splits, _pick(HEAD_DIMS, seed // 5), out_bytes)
            if head_dim <= 0:
                head_dim = _pick(HEAD_DIMS, seed // 5)
            if rank == 5:
                shape = [
                    num_splits,
                    _pick(BATCH, seed // 7),
                    _pick(SEQLEN, seed // 11),
                    heads,
                    head_dim,
                ]
            else:
                shape = [
                    num_splits,
                    _pick(SEQLEN, seed // 11),
                    heads,
                    head_dim,
                ]
            out_partial.range_values = _accuracy_range(seed)

        shape = _fit_accuracy_budget(shape)
        _set_common_inputs(case_config, shape, selected_out_dtype, PARTIAL_DTYPE)

        lse_partial = _tensor_map(case_config).get("lse_partial")
        if lse_partial is not None:
            lse_partial.range_values = _lse_range(seed)

        _choose_out(case_config, shape, seed)
        return case_config


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def _make_memory_shape(seed):
    rank = 5 if seed % 2 == 0 else 4
    num_splits = _pick(MEMORY_NUM_SPLITS, seed)
    heads = _pick(MEMORY_HEADS, seed // 3)
    head_dim = _pick(MEMORY_HEAD_DIMS, seed // 5)

    # FA3 的 partial 恒为 fp32
    out_dtype = _pick(("fp16", "bf16", "fp32"), seed)
    out_bytes = OUT_BYTES.get(out_dtype, 2)

    head_dim = clamp_head_dim(num_splits, head_dim, out_bytes)
    if head_dim <= 0:
        head_dim = max_head_dim(num_splits, out_bytes)
    head_dim = max(16, head_dim)

    target_partial = MEMORY_MIN_PARTIAL_BYTES + (seed % 6) * MEMORY_TARGET_STEP_BYTES
    bytes_per_row = num_splits * heads * head_dim * PARTIAL_BYTES
    rows = max(1, (target_partial + bytes_per_row - 1) // bytes_per_row)
    rows += seed % 7

    if rank == 5:
        batch = _pick(MEMORY_BATCH, seed // 7)
        seqlen = max(1, (rows + batch - 1) // batch)
        shape = [num_splits, batch, seqlen, heads, head_dim]
    else:
        shape = [num_splits, rows, heads, head_dim]

    _max_elems = MEMORY_MAX_PARTIAL_BYTES // PARTIAL_BYTES
    while _prod(shape) > _max_elems:
        token_axis = 2 if rank == 5 else 1
        if shape[token_axis] > 1:
            shape[token_axis] -= 1
        else:
            break

    # 若 UB 收敛后的 head_dim 偏小导致不足 550 MiB，按缺口直接补行数（闭式解）
    base_elems = _prod(shape) / max(1, shape[2] if rank == 5 else shape[1])
    need_elems = -(-MEMORY_MIN_PARTIAL_BYTES // PARTIAL_BYTES)
    if _prod(shape) < need_elems and base_elems > 0:
        token_axis = 2 if rank == 5 else 1
        shape[token_axis] = max(shape[token_axis], int(-(-need_elems // base_elems)))
        while _prod(shape) > max_elems and shape[token_axis] > 1:
            shape[token_axis] = max(1, shape[token_axis] - max(1, shape[token_axis] // 100))

    partial_bytes = _prod(shape) * PARTIAL_BYTES
    if partial_bytes < MEMORY_MIN_PARTIAL_BYTES:
        raise ValueError(
            "memory constraint produced out_partial below 550 MiB: "
            f"shape={shape}, bytes={partial_bytes}. "
            "该组合下 head_dim 受 UB 预算限制过小，无法在元素上限内达到 550 MiB。"
        )
    return shape, out_dtype


@GENERATOR_REGISTRY.register("generate_flash_attn_combine_memory_fa3")
class FlashAttnCombineFA3Memory(CaseGenerator):
    """FA3 memory 约束：out_partial 单独就超过 550 MiB。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = _tensor_map(case_config)
        out_partial = tensors.get("out_partial")
        if out_partial is None:
            return case_config

        original_shape = [int(value) for value in (out_partial.shape or [])]
        seed = _seed(original_shape, "memory")

        shape, out_dtype = _make_memory_shape(seed)
        _set_common_inputs(case_config, shape, out_dtype, PARTIAL_DTYPE)

        refreshed = _tensor_map(case_config)
        refreshed["out_partial"].range_values = (
            [-5, 5] if seed % 2 == 0 else dict(ND_RANGE)
        )
        refreshed["lse_partial"].range_values = [-5, 5]

        # memory 套件同样让部分 case 走 out 预分配
        _choose_out(case_config, shape, seed)
        return case_config
