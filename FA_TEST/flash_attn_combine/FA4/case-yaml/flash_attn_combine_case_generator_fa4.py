"""ATK generator for flash_attn_combine — FA4（自包含，不依赖任何共享模块）。

注册两个约束名：
    generate_flash_attn_combine_accuracy_fa4
    generate_flash_attn_combine_memory_fa4

FA4 的真实约束（flash-attention-npu 0.5.0, commit 82a9a8d）：
  csrc/ascend950/flash_attn_npu_4/flash_api.cpp:138    mha_combine
  flash_attn_npu_4/flash_attn_npu_interface.py:1439
  flash_attn_npu_4/flash_attn_npu_interface_950.py:1116

与 FA3 的差异（本文件的增量）：
  1. out_partial 放开到 fp16 / bf16 / fp32，低精度内部转 fp32 累加
       TORCH_CHECK(partial_dtype == at::kFloat || at::kHalf || at::kBFloat16,
                   "out_partial must be fp16, bf16, or fp32")
       if (partial_dtype != at::kFloat) out_partial = out_partial.to(at::kFloat);
  2. 多出 4 个可选参数：
       cu_seqlens        1-D int32, numel >= 2      （只校验，不限制行数）
       seqused           1-D int32, numel >= 1      （只校验，不限制行数）
       virtual_batch_idx 连续 1-D int32 NPU tensor；
                         dense(5D) 时 numel == batch_size
       return_lse        False 时返回 (out, None)
  共同点：lse_partial 仍必须 fp32；num_splits 1~256；UB 预算 150KiB 相同。

UB 预算里的 out_bytes 取决于「输出」dtype（out 或 out_dtype），
与 partial 的 dtype 无关。这一点在 memory 套件里尤其要注意：
partial 若是 bf16（2 字节），要让 out_partial 达到 550 MiB 需要更多元素。
"""

import zlib

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


MIB = 1024 * 1024

# FA4 允许的 partial dtype
PARTIAL_DTYPES = ("fp32", "fp16", "bf16")
PARTIAL_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

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
# 内存用例单个张量的字节上限。原来写死 192M 元素是按 fp32 折算（=768 MiB），
# 但 FA4 允许 bf16/fp16 partial：同样 550 MiB 需要 288M 元素，
# 与 192M 元素上限直接冲突。因此改成按字节数封顶，再按 dtype 折算成元素数。
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


def _attrs_map(case_config):
    """`type: attrs` 是 ATK 里表示「整数元组类可选参数」的写法。

    这类元素在 case_config.inputs 里可能是 list 而非对象，
    因此使用 _iter_objects 过滤后再按 type 匹配。
    """
    return {
        item.name: item
        for item in _iter_objects(case_config)
        if item.type == "attrs" and item.name
    }


def _seed(original_shape, out_dtype):
    marker = f"fa4|{len(original_shape)}|{tuple(original_shape)}|{out_dtype}"
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
    # FA4 允许 fp16/bf16/fp32
    out_partial.dtype = partial_dtype

    # lse 仍必须 fp32，且比 out_partial 少一维
    lse_partial.shape = list(shape[:-1])
    lse_partial.dtype = "fp32"

    out_dtype_attr = attrs.get("out_dtype")
    if out_dtype_attr is not None:
        out_dtype_attr.range_values = out_dtype

    return case_config


def _set_extended_inputs(case_config, shape, seed):
    """FA4 专属：cu_seqlens / seqused / virtual_batch_idx / return_lse。

    这些参数在 C++ 里只做校验、不参与行数限制，所以造的是
    「合法且不改变数值结果」的值，用于覆盖校验分支与传参路径。
    """
    meta = _attrs_map(case_config)
    is_dense = len(shape) == 5
    seqlen = shape[2] if is_dense else shape[1]
    batch = shape[1] if is_dense else None

    # cu_seqlens: 1-D int32, numel>=2，用 [0, seqlen] 前缀和
    if "cu_seqlens" in meta:
        meta["cu_seqlens"].range_values = [0, int(seqlen)]

    # seqused: 1-D int32, numel>=1
    if "seqused" in meta:
        row_count = int(batch) if is_dense else int(seqlen)
        meta["seqused"].range_values = [max(1, row_count)]

    # virtual_batch_idx: dense 时必须 numel == batch_size，用恒等映射
    if "virtual_batch_idx" in meta:
        n = int(batch) if is_dense else 1
        meta["virtual_batch_idx"].range_values = list(range(max(1, n)))

    # return_lse: 约 1/4 的 case 走 False，覆盖「返回 (out, None)」分支
    attrs = _attr_map(case_config)
    if "return_lse" in attrs and seed % 4 == 0:
        attrs["return_lse"].range_values = False

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
    # 不生成 [0, 0] 全零范围：全零 partial 输出恒为 0，
    # 相对误差分母趋零被放大成假失败（历史上有 5 条此类误报）。
    bucket = seed % 100
    if bucket < 10:
        return [-0.001, 0.001]
    if bucket < 50:
        return [-5, 5]
    return dict(ND_RANGE)


def _lse_range(seed):
    if seed % 19 == 0:
        return "-inf"
    if seed % 5 < 3:
        return [-5, 5]
    return dict(ND_RANGE)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
@GENERATOR_REGISTRY.register("generate_flash_attn_combine_accuracy_fa4")
class FlashAttnCombineFA4Accuracy(CaseGenerator):
    """FA4 accuracy 约束。"""

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

        # FA4 增量：partial 可以是低精度
        selected_partial_dtype = _pick(PARTIAL_DTYPES, seed)
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
        _set_common_inputs(case_config, shape, selected_out_dtype, selected_partial_dtype)

        lse_partial = _tensor_map(case_config).get("lse_partial")
        if lse_partial is not None:
            lse_partial.range_values = _lse_range(seed)

        _set_extended_inputs(case_config, shape, seed)
        _choose_out(case_config, shape, seed)
        return case_config


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def _make_memory_shape(seed, partial_width):
    rank = 5 if seed % 2 == 0 else 4
    num_splits = _pick(MEMORY_NUM_SPLITS, seed)
    heads = _pick(MEMORY_HEADS, seed // 3)
    head_dim = _pick(MEMORY_HEAD_DIMS, seed // 5)

    out_dtype = _pick(("fp16", "bf16", "fp32"), seed)
    out_bytes = OUT_BYTES.get(out_dtype, 2)

    head_dim = clamp_head_dim(num_splits, head_dim, out_bytes)
    if head_dim <= 0:
        head_dim = max_head_dim(num_splits, out_bytes)
    head_dim = max(16, head_dim)

    target_partial = MEMORY_MIN_PARTIAL_BYTES + (seed % 6) * MEMORY_TARGET_STEP_BYTES
    # 按 partial 的实际字节宽度换算行数（bf16 partial 需要更多元素才能到 550 MiB）
    bytes_per_row = num_splits * heads * head_dim * partial_width
    rows = max(1, (target_partial + bytes_per_row - 1) // bytes_per_row)
    rows += seed % 7

    if rank == 5:
        batch = _pick(MEMORY_BATCH, seed // 7)
        seqlen = max(1, (rows + batch - 1) // batch)
        shape = [num_splits, batch, seqlen, heads, head_dim]
    else:
        shape = [num_splits, rows, heads, head_dim]

    _max_elems = MEMORY_MAX_PARTIAL_BYTES // max(1, partial_width)
    while _prod(shape) > _max_elems:
        token_axis = 2 if rank == 5 else 1
        if shape[token_axis] > 1:
            shape[token_axis] -= 1
        else:
            break

    # 若 UB 收敛后的 head_dim 偏小导致不足 550 MiB，按缺口直接补行数。
    # 前两维（num_splits、batch/heads）之外只有 token_axis 可以拉长，
    # 所以用闭式解：需要的总元素数 / 每行元素数 = 需要的最小行数。
    base_elems = _prod(shape) / max(1, shape[2] if rank == 5 else shape[1])
    need_elems = -(-MEMORY_MIN_PARTIAL_BYTES // partial_width)  # 向上取整
    if _prod(shape) < need_elems and base_elems > 0:
        token_axis = 2 if rank == 5 else 1
        shape[token_axis] = max(shape[token_axis], int(-(-need_elems // base_elems)))
        # 夹总元素上限（注意用 _max_elems，与上面定义的名字一致，
        # 否则会 NameError: name 'max_elems' is not defined）
        while _prod(shape) > _max_elems and shape[token_axis] > 1:
            shape[token_axis] = max(1, shape[token_axis] - max(1, shape[token_axis] // 100))

    actual_bytes = _prod(shape) * partial_width
    if actual_bytes < MEMORY_MIN_PARTIAL_BYTES:
        raise ValueError(
            "memory constraint produced out_partial below 550 MiB: "
            f"shape={shape}, dtype_bytes={partial_width}, bytes={actual_bytes}. "
            "该组合下 head_dim 受 UB 预算限制过小，无法在元素上限内达到 550 MiB。"
        )
    return shape, out_dtype


@GENERATOR_REGISTRY.register("generate_flash_attn_combine_memory_fa4")
class FlashAttnCombineFA4Memory(CaseGenerator):
    """FA4 memory 约束：out_partial 单独就超过 550 MiB。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = _tensor_map(case_config)
        out_partial = tensors.get("out_partial")
        if out_partial is None:
            return case_config

        original_shape = [int(value) for value in (out_partial.shape or [])]
        seed = _seed(original_shape, "memory")

        partial_dtype = _pick(PARTIAL_DTYPES, seed)
        partial_width = PARTIAL_BYTES.get(partial_dtype, 4)

        shape, out_dtype = _make_memory_shape(seed, partial_width)
        _set_common_inputs(case_config, shape, out_dtype, partial_dtype)

        refreshed = _tensor_map(case_config)
        refreshed["out_partial"].range_values = (
            [-5, 5] if seed % 2 == 0 else dict(ND_RANGE)
        )
        refreshed["lse_partial"].range_values = [-5, 5]

        _set_extended_inputs(case_config, shape, seed)
        _choose_out(case_config, shape, seed)
        return case_config
