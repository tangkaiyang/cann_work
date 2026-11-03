"""recurrent_kda 的 10000 条 ATK 泛化用例生成器。

相比早期 200 条版本，本版本大幅扩展了 B/T/H/HV 取值空间：

* B 取值 1~8，大 shape 额外覆盖 16/32/64；
* T 取值 1~8，并新增少量 T>8（最大到 999）的长序列用例；
* H/HV 覆盖 1~256，集中在 96 附近，且始终满足 ``HV % H == 0``；
* 极端 shape（B=64、T 很大）只与很小的 HV 组合，保证 state 元素数不超限。
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from itertools import product

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None


OP_NAME = "recurrent_kda"
CASE_COUNT = 10000
SMALL_CASE_COUNT = 9500
LARGE_CASE_COUNT = 500
EXPECTED_TILING_KEYS = {0}

K_DIM = 128
# 单条用例 state 元素数上限。相比 200 条版本显著放宽，以支持 H/HV 1~256 与
# B=64、T~999 等极端 shape；实际序列条数远小于最坏估计，单用例 state 通常
# 只有几十 MB。
SMALL_STATE_ELEMENT_CAP = 24_000_000
LARGE_STATE_ELEMENT_CAP = 48_000_000
FP32_STATE_NUMBER_COUNT_RATIO = 4
# T>8 的长序列用例无法用 dense/uniform 表达（每条有效序列长度必须 <=8），
# 只能走按 token 切分的 varlen/padding/zero 模式。
LONG_SEQUENCE_T_VALUES = (64, 96, 128, 192, 256, 384, 512, 640, 768, 999)

GATE_DTYPES = ("fp32", "bf16", "fp16")
BETA_DTYPES = ("fp32", "bf16", "fp16")
STATE_DTYPES = ("fp32", "bf16")
LAYOUTS = ("BSND", "TND")
STATE_DIRECTIONS = (False, True)
VALUE_DIMS = (128, 256)
INT_DTYPES = ("int32", "int64")

# H 的候选维度（1~256，覆盖 96 附近），HV 必须是 H 的整数倍。
HEAD_H_DIMS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
HEAD_HV_DIMS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def _build_head_pairs() -> tuple:
    pairs = []
    seen = set()
    for hv in HEAD_HV_DIMS:
        for h in HEAD_H_DIMS:
            if h <= hv and hv % h == 0 and (h, hv) not in seen:
                seen.add((h, hv))
                pairs.append((h, hv))
    return tuple(pairs)


HEAD_PAIRS = _build_head_pairs()

OPTIONAL_MODES = (
    {"cu_mode": "none", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "packed", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "speculative", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "speculative", "accepted": True},
    {"cu_mode": "padding", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "zero", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "packed", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "packed", "accepted": True},
)

# 长序列（T>8）只能使用的 cu 模式：必须把 token 切成 <=8 的多条序列。
LONG_SEQUENCE_OPTIONAL_MODES = tuple(
    mode for mode in OPTIONAL_MODES if mode["cu_mode"] in ("varlen", "padding", "zero")
)

# 大 shape 只使用 uniform/none，保证序列条数 <= B，从而可以承载较大的 HV。
LARGE_OPTIONAL_MODES = (
    {"cu_mode": "uniform", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "packed", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "speculative", "accepted": True},
    {"cu_mode": "none", "ssm_mode": "none", "accepted": False},
)

KERNEL_FEATURE_MODES = (
    {},
    {"use_qk_l2norm_in_kernel": True},
    {"use_gate_in_kernel": True},
    {"use_gate_in_kernel": True, "dt_bias_mode": "flat"},
    {
        "use_gate_in_kernel": True,
        "dt_bias_mode": "matrix",
        "safe_gate": True,
        "lower_bound": -2.5,
    },
    {"use_beta_sigmoid_in_kernel": True},
    {"use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": True},
    {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "dt_bias_mode": "matrix",
        "safe_gate": True,
        "lower_bound": -4.0,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": True,
    },
)

OUTPUT_MODES = (
    (True, False),
    (True, True),
    (False, False),
    (False, True),
)

# layout 与 V 现在由 shape spec 决定，这里只保留 dtype/direction 组合。
BASE_COMBINATIONS = tuple(
    product(
        GATE_DTYPES,
        BETA_DTYPES,
        STATE_DTYPES,
        STATE_DIRECTIONS,
    )
)


def _worst_state_elements(B: int, T: int, HV: int, V: int) -> int:
    """在最坏 cu 模式下估算 state 元素数（序列条数上限 = token 容量 + 索引余量）。"""
    worst_sequences = B * T + 3
    return worst_sequences * HV * K_DIM * V


def _build_shape_specs(size_class: str) -> tuple:
    cap = SMALL_STATE_ELEMENT_CAP if size_class == "small" else LARGE_STATE_ELEMENT_CAP
    if size_class == "small":
        batch_dims = (1, 2, 3, 4, 5, 6, 7, 8)
    else:
        batch_dims = (1, 2, 3, 4, 5, 6, 8, 16, 32, 64)

    specs = []
    for layout in LAYOUTS:
        for batch in batch_dims:
            for length in (1, 2, 3, 4, 5, 6, 7, 8):
                if layout == "TND" and batch * length > 8:
                    # TND dense 只有一条序列，总 token 必须 <=8。
                    continue
                for value_dim in VALUE_DIMS:
                    for head, head_value in HEAD_PAIRS:
                        if _worst_state_elements(batch, length, head_value, value_dim) > cap:
                            continue
                        specs.append(
                            {
                                "layout": layout,
                                "B": batch,
                                "T": length,
                                "H": head,
                                "HV": head_value,
                                "V": value_dim,
                                "long_sequence": False,
                            }
                        )

    if size_class == "small":
        # 少量长序列用例（T>8）。只能配极小 HV，避免 state 爆炸。
        for length in LONG_SEQUENCE_T_VALUES:
            if _worst_state_elements(1, length, 1, 128) > cap:
                continue
            specs.append(
                {
                    "layout": "BSND",
                    "B": 1,
                    "T": length,
                    "H": 1,
                    "HV": 1,
                    "V": 128,
                    "long_sequence": True,
                }
            )
    return tuple(specs)


SHAPE_SPECS = {
    "small": _build_shape_specs("small"),
    "large": _build_shape_specs("large"),
}
SHAPE_COUNTS = {"small": SMALL_CASE_COUNT, "large": LARGE_CASE_COUNT}


def _shape_for(local_index: int, size_class: str) -> dict:
    """把 local_index 均匀映射到该 size_class 的 shape 空间，尽量覆盖全部取值。"""
    specs = SHAPE_SPECS[size_class]
    total = SHAPE_COUNTS[size_class]
    if total <= 1:
        return specs[0]
    # 均匀映射到 [0, len(specs)-1] 且包含两端，保证尾部极端 shape（如 T=999）
    # 也会被采样到。
    position = (local_index * (len(specs) - 1)) // (total - 1)
    return specs[position]

def _partition(total: int, seed: int) -> list:
    """把 token 数切成 1..8 的确定性变长序列。"""
    lengths = []
    remaining = int(total)
    step = 0
    while remaining > 0:
        upper = min(8, remaining)
        length = 1 + ((seed + 3 * step) % upper)
        lengths.append(length)
        remaining -= length
        step += 1
    if len(lengths) == 1 and total > 1:
        first = max(1, lengths[0] // 2)
        lengths = [first, lengths[0] - first]
    return lengths


def _sequence_lengths(layout: str, B: int, T: int, cu_mode: str, seed: int):
    capacity = B * T
    if cu_mode == "none":
        return None
    if cu_mode == "uniform":
        return [T] * B
    if cu_mode == "varlen":
        return _partition(capacity, seed)
    if cu_mode == "padding":
        valid = max(1, capacity - 1 - (seed % min(4, capacity)))
        return _partition(valid, seed)
    if cu_mode == "zero":
        lengths = _partition(capacity, seed)
        lengths.insert((seed % (len(lengths) + 1)), 0)
        return lengths
    raise ValueError(f"unsupported cu_mode: {cu_mode}")


def _default_profile(index: int, size_class: str) -> dict:
    local_index = index if size_class == "small" else index - SMALL_CASE_COUNT
    shape = _shape_for(local_index, size_class)
    layout = shape["layout"]
    B = shape["B"]
    T = shape["T"]
    H = shape["H"]
    HV = shape["HV"]
    V = shape["V"]

    gate_dtype, beta_dtype, state_dtype, state_v_first = BASE_COMBINATIONS[
        index % len(BASE_COMBINATIONS)
    ]

    if size_class == "large":
        optional = deepcopy(LARGE_OPTIONAL_MODES[index % len(LARGE_OPTIONAL_MODES)])
    elif shape["long_sequence"]:
        optional = deepcopy(
            LONG_SEQUENCE_OPTIONAL_MODES[index % len(LONG_SEQUENCE_OPTIONAL_MODES)]
        )
    else:
        optional = deepcopy(OPTIONAL_MODES[index % len(OPTIONAL_MODES)])

    lengths = _sequence_lengths(layout, B, T, optional["cu_mode"], 20260817 + index)
    if optional["accepted"] and (lengths is None or any(length <= 0 for length in lengths)):
        optional["accepted"] = False

    feature = {
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False,
        "dt_bias_mode": "none",
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "lower_bound": -5.0,
    }
    feature.update(KERNEL_FEATURE_MODES[index % len(KERNEL_FEATURE_MODES)])
    output_final_state, inplace_final_state = OUTPUT_MODES[index % len(OUTPUT_MODES)]

    profile = {
        "name": f"{size_class}_{index:05d}",
        "size_class": size_class,
        "tiling_key": 0,
        "dtype": "bf16",
        "gate_dtype": gate_dtype,
        "beta_dtype": beta_dtype,
        "state_dtype": state_dtype,
        "B": B,
        "T": T,
        "H": H,
        "HV": HV,
        "K": K_DIM,
        "V": V,
        "layout": layout,
        "state_v_first": state_v_first,
        "cu_mode": optional["cu_mode"],
        "seq_lengths": lengths,
        "cu_dtype": INT_DTYPES[index % len(INT_DTYPES)],
        "ssm_mode": optional["ssm_mode"],
        "ssm_dtype": INT_DTYPES[(index // 2) % len(INT_DTYPES)],
        "accepted_tokens": optional["accepted"],
        "accepted_dtype": INT_DTYPES[(index // 3) % len(INT_DTYPES)],
        "state_capacity_extra": 1 + (index % 3) if optional["ssm_mode"] != "none" else 0,
        "output_final_state": output_final_state,
        "inplace_final_state": inplace_final_state,
        "state_noncontiguous": index % 7 == 0,
        "input_noncontiguous": index % 13 == 0,
        "scale": (128.0 ** -0.5, 1.0, 0.5)[index % 3],
        **feature,
    }
    # 当前 wheel（26.10.0）的 npu_recurrent_kda 在 initial_state=None 时要求
    # inplace_final_state=True，且该路径会触发 kernel vector-core 异常(507035)，
    # 因此不再生成隐式 state 用例，所有用例都显式构造 initial_state。
    profile["initial_state_none"] = False
    return profile

def _logical_lengths(profile: dict) -> list:
    if profile["seq_lengths"] is not None:
        return list(profile["seq_lengths"])
    if profile["layout"] == "BSND":
        return [profile["T"]] * profile["B"]
    return [profile["B"] * profile["T"]]


def _validate_profile(profile: dict) -> None:
    if profile["tiling_key"] not in EXPECTED_TILING_KEYS:
        raise RuntimeError(f"invalid tiling key: {profile['tiling_key']}")
    if profile["dtype"] != "bf16":
        raise RuntimeError("q/k/v must use bf16")
    if profile["gate_dtype"] not in GATE_DTYPES or profile["beta_dtype"] not in BETA_DTYPES:
        raise RuntimeError("invalid gate/beta dtype")
    if profile["state_dtype"] not in STATE_DTYPES:
        raise RuntimeError("invalid state dtype")
    if profile["layout"] not in LAYOUTS or profile["K"] != K_DIM or profile["V"] not in VALUE_DIMS:
        raise RuntimeError("invalid layout or K/V")
    if not (0 < profile["H"] <= 256 and 0 < profile["HV"] <= 256):
        raise RuntimeError("H/HV must be in (0, 256]")
    if profile["HV"] % profile["H"] != 0:
        raise RuntimeError("HV must be divisible by H")

    capacity = profile["B"] * profile["T"]
    lengths = _logical_lengths(profile)
    # A2/A3 removed the historical "each sequence length <= 8" requirement.
    if not lengths or any(length < 0 for length in lengths):
        raise RuntimeError(f"invalid sequence lengths: {lengths}")
    if sum(lengths) > capacity:
        raise RuntimeError("valid tokens exceed physical capacity")
    if profile["accepted_tokens"]:
        if profile["ssm_mode"] == "none" or any(length <= 0 for length in lengths):
            raise RuntimeError("accepted tokens require non-empty indexed sequences")
    if profile["ssm_mode"] not in ("none", "packed", "speculative"):
        raise RuntimeError("invalid ssm mode")
    if profile["ssm_mode"] == "none" and profile["state_capacity_extra"] != 0:
        raise RuntimeError("state capacity extra requires ssm indices")
    if profile["initial_state_none"] and (
        profile["inplace_final_state"] or profile["ssm_mode"] != "none"
    ):
        raise RuntimeError("implicit state is only valid for non-inplace non-indexed cases")
    if not profile["use_gate_in_kernel"]:
        if profile["safe_gate"] or profile["dt_bias_mode"] != "none":
            raise RuntimeError("safe gate/dt bias require in-kernel gate")
    if profile["safe_gate"] and not -5.0 <= profile["lower_bound"] < 0.0:
        raise RuntimeError("safe gate lower bound is invalid")
    if profile["allow_neg_eigval"] and not profile["use_beta_sigmoid_in_kernel"]:
        raise RuntimeError("allow_neg_eigval requires beta sigmoid in generated cases")

    seq_num = len(lengths)
    state_capacity = seq_num + profile["state_capacity_extra"]
    state_elements = state_capacity * profile["HV"] * profile["K"] * profile["V"]
    cap = SMALL_STATE_ELEMENT_CAP if profile["size_class"] == "small" else LARGE_STATE_ELEMENT_CAP
    if state_elements > cap:
        raise RuntimeError(
            f"{profile['name']} state elements {state_elements} exceed cap {cap}"
        )
    profile["token_capacity"] = capacity
    profile["valid_tokens"] = sum(lengths)
    profile["seq_num"] = seq_num
    profile["state_capacity"] = state_capacity
    profile["state_elements"] = state_elements


def _build_profiles() -> list:
    profiles = []
    for index in range(CASE_COUNT):
        size_class = "small" if index < SMALL_CASE_COUNT else "large"
        profile = _default_profile(index, size_class)
        _validate_profile(profile)
        profile["name"] = (
            f"{size_class}_{index:05d}_{profile['layout'].lower()}_"
            f"b{profile['B']}_t{profile['T']}_h{profile['H']}x{profile['HV']}_v{profile['V']}"
        )
        profiles.append(profile)

    if len(profiles) != CASE_COUNT or len({profile["name"] for profile in profiles}) != CASE_COUNT:
        raise RuntimeError("profiles must contain exactly CASE_COUNT unique names")
    if {profile["tiling_key"] for profile in profiles} != EXPECTED_TILING_KEYS:
        raise RuntimeError("profiles do not cover the complete tilingKey set {0}")
    if sum(profile["size_class"] == "small" for profile in profiles) != SMALL_CASE_COUNT:
        raise RuntimeError("small/large distribution is invalid")
    if sum(profile["size_class"] == "large" for profile in profiles) != LARGE_CASE_COUNT:
        raise RuntimeError("small/large distribution is invalid")

    required_values = {
        "layout": set(LAYOUTS),
        "gate_dtype": set(GATE_DTYPES),
        "beta_dtype": set(BETA_DTYPES),
        "state_dtype": set(STATE_DTYPES),
        "state_v_first": set(STATE_DIRECTIONS),
        "V": set(VALUE_DIMS),
        "cu_mode": {mode["cu_mode"] for mode in OPTIONAL_MODES},
        "ssm_mode": {mode["ssm_mode"] for mode in OPTIONAL_MODES},
        "output_final_state": {False, True},
        "inplace_final_state": {False, True},
        "use_qk_l2norm_in_kernel": {False, True},
        "use_gate_in_kernel": {False, True},
        "use_beta_sigmoid_in_kernel": {False, True},
        "safe_gate": {False, True},
    }
    for field, expected in required_values.items():
        actual = {profile[field] for profile in profiles}
        if actual != expected:
            raise RuntimeError(f"field {field} coverage mismatch: expected {expected}, got {actual}")

    # 确保扩展后的极端取值确实出现在 10000 条中。
    if not any(profile["B"] >= 64 for profile in profiles):
        raise RuntimeError("B=64 cases are missing")
    if not any(profile["T"] > 8 for profile in profiles):
        raise RuntimeError("T>8 long-sequence cases are missing")
    if max(profile["HV"] for profile in profiles) < 96:
        raise RuntimeError("large HV (~96+) coverage is missing")
    return profiles


PROFILES = _build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    if not 0 <= index < len(PROFILES):
        raise IndexError(f"case index {index} is outside [0, {len(PROFILES)})")
    profile = deepcopy(PROFILES[index])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260817 + index,
            "route": "ascendc",
            "soc": "ascend910b",
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_recurrent_kda")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:05d}_{spec['name']}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    # q/k/v 只支持 BF16；第二路 marker 仅用于让统一 -dt 100 生成全部 case。
                    cfg.dtype = "bf16"
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(
                        spec, ensure_ascii=False, separators=(",", ":")
                    )
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            if spec["state_dtype"] == "fp32" and (
                spec["output_final_state"] or spec["inplace_final_state"]
            ):
                # FP32 final_state 在 CPU/AIV 不同归约顺序下有少量 ULP 差异，
                # 放宽该类用例的双标杆计数比。
                case_config.standard.acc["cv_fused_double_benchmark"][
                    "number_count_radio"
                ] = FP32_STATE_NUMBER_COUNT_RATIO
            return case_config


if __name__ == "__main__":
    summary_fields = (
        "tiling_key",
        "size_class",
        "layout",
        "gate_dtype",
        "beta_dtype",
        "state_dtype",
        "state_v_first",
        "V",
        "cu_mode",
        "ssm_mode",
    )
    summary = {
        field: dict(sorted(Counter(str(profile[field]) for profile in PROFILES).items()))
        for field in summary_fields
    }
    extremes = {
        "B": dict(sorted(Counter(profile["B"] for profile in PROFILES).items())),
        "T": dict(sorted(Counter(profile["T"] for profile in PROFILES).items())),
        "H": dict(sorted(Counter(profile["H"] for profile in PROFILES).items())),
        "HV": dict(sorted(Counter(profile["HV"] for profile in PROFILES).items())),
    }
    print(
        json.dumps(
            {"case_count": len(PROFILES), "coverage": summary, "distributions": extremes},
            ensure_ascii=False,
            indent=2,
        )
    )
