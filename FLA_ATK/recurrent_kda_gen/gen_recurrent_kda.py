"""recurrent_kda 的 ATK 泛化用例生成器（补充泛化版）。

相对源仓 gen_recurrent_kda.py（仅 1 个 B=2,T=2 平凡用例）的扩展：
- 覆盖 ssm_state_indices 三种模式：
  * none        —— 不传索引，state_capacity == seq_num；
  * packed      —— packed [T]（每个 token 一个槽索引，含共享/重复槽）；
  * speculative —— [seq_num, max_step] + num_accepted_tokens；
- 覆盖 layout：BSND / TND；V：128 / 256；HV/H 比：1/2/4；
- 覆盖 dtBias 形状：flat [H_v*K] 与 matrix [H_v, K]；
- 多段 cu_seqlens：varlen 1~4 段（BSND 下段数 <= B），每段长度 1~8（满足算子段长 <= 8 约束）；
- q/k/v 仅 BF16（算子限制），g/beta 使用 FP32。

case_spec 携带全部确定性元数据（cu_seqlens、ssm_state_indices 数组、
num_accepted_tokens 数组），executor 只按 spec 消费，离线可完整校验。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy

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
SEED_BASE = 20260817
# 算子约束：K 固定 128，V ∈ {128, 256}，每段长度 <= 8
K_DIM = 128
V_VALUES = (128, 256)
MAX_SEGMENT = 8
LAYOUTS = ("BSND", "TND")
SSM_MODES = ("none", "packed", "speculative")
DT_BIAS_MODES = ("flat", "matrix")


def _build_multi_segments(num_segments: int, total_tokens: int | None, rng: random.Random):
    """构造 cu_seqlens。

    - total_tokens=None：num_segments 段，每段长度 1..MAX_SEGMENT（TND 场景，
      packed 总数由段长决定）；
    - total_tokens=int：段长和恰好等于 total_tokens（BSND 场景，packed 容量
      B*T 必须被 cu_seqlens 末项用满），每段长度仍 1..MAX_SEGMENT。
    """
    if total_tokens is None:
        lens = [rng.randint(1, MAX_SEGMENT) for _ in range(num_segments)]
        if sum(lens) == 0:
            lens[0] = 1
    else:
        assert total_tokens >= num_segments
        # 把 total_tokens 拆成 num_segments 段，每段 1..MAX_SEGMENT
        lens = []
        remaining = total_tokens
        for i in range(num_segments - 1):
            low = max(1, remaining - MAX_SEGMENT * (num_segments - 1 - i))
            high = min(MAX_SEGMENT, remaining - (num_segments - 1 - i))
            length = rng.randint(low, high)
            lens.append(length)
            remaining -= length
        lens.append(remaining)
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    return cu


def _build_state_indices(mode: str, total_tokens: int, num_segments: int, state_capacity: int,
                         rng: random.Random, max_step: int | None = None) -> list[int] | list[list[int]] | None:
    """构造 ssm_state_indices。"""
    if mode == "none":
        return None
    if mode == "packed":
        # packed [T]：每个 token 一个槽索引；允许重复槽（state 共享），覆盖 capacity > seq_num
        indices = [rng.randrange(state_capacity) for _ in range(total_tokens)]
        # 保证每个出现的槽都合法
        return [min(int(i), state_capacity - 1) for i in indices]
    # speculative [seq_num, max_step]：max_step 必须 >= 每段最大长度
    # （标杆按 token - start 索引列，num_accepted 最大取段长）
    max_step = max(1, max_step) if max_step else min(MAX_SEGMENT, 4)
    matrix = [[rng.randrange(state_capacity) for _ in range(max_step)] for _ in range(num_segments)]
    return matrix


def _build_num_accepted(mode: str, num_segments: int, seg_lens: list[int],
                        rng: random.Random) -> list[int] | None:
    """构造 num_accepted_tokens：每段 1..seg_len，speculative 模式专用。"""
    if mode != "speculative":
        return None
    return [rng.randint(1, max(1, length)) for length in seg_lens]


def _build_profiles():
    """确定性生成 profile 矩阵。"""
    profiles = []
    rng = random.Random(SEED_BASE)
    case_i = 0
    for layout in LAYOUTS:
        for ssm_mode in SSM_MODES:
            for v_dim in V_VALUES:
                for dt_mode in DT_BIAS_MODES:
                    # HV/H 组合：HV ∈ {2,4,8}，HV % H == 0
                    hv_h_pairs = ((2, 1), (4, 2), (8, 4), (4, 4))
                    for hv, h in hv_h_pairs:
                        seg_count = 1 + (case_i % 4) if layout == "TND" else (1, 2, 3, 4)[case_i % 4]
                        spec_rng = random.Random(SEED_BASE + 31 * case_i)
                        if layout == "TND":
                            cu = _build_multi_segments(seg_count, None, spec_rng)
                            total_tokens = cu[-1]
                            B, T = 1, total_tokens
                        else:
                            # BSND：dense 形状 [B, T]，packed 容量 B*T 用满
                            B = seg_count
                            T = 8 - (case_i % 8)  # 每段 1..8（降序轮转，case 0 = T8）
                            total_tokens = B * T
                            cu = _build_multi_segments(seg_count, total_tokens, spec_rng)
                        seg_lens = [cu[i + 1] - cu[i] for i in range(seg_count)]
                        # speculative: state_capacity >= seq_num；packed: 允许 capacity > seq_num
                        state_capacity = seg_count if ssm_mode == "none" else max(seg_count, 4 + (case_i % 5))
                        profile = {
                            "name": f"bf16_{layout.lower()}_{ssm_mode}_V{v_dim}_dt{dt_mode}_H{h}_HV{hv}_S{seg_count}",
                            "dtype": "bf16",
                            "B": B,
                            "T": T,
                            "H": h,
                            "HV": hv,
                            "K": K_DIM,
                            "V": v_dim,
                            "layout": layout,
                            "ssm_mode": ssm_mode,
                            "dt_bias_mode": dt_mode,
                            "state_capacity": state_capacity,
                            "total_tokens": total_tokens,
                            "cu_seqlens": cu,
                            "state_v_first": (case_i % 2 == 0),
                        }
                        indices = _build_state_indices(ssm_mode, total_tokens, seg_count, state_capacity, spec_rng,
                                                       max_step=max(seg_lens) if ssm_mode == "speculative" else None)
                        if indices is not None:
                            profile["ssm_state_indices"] = indices
                        accepted = _build_num_accepted(ssm_mode, seg_count, seg_lens, spec_rng)
                        if accepted is not None:
                            profile["num_accepted_tokens"] = accepted
                        profiles.append(profile)
                        case_i += 1
    return profiles


PROFILES = _build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"op": OP_NAME, "case_id": index, "seed": SEED_BASE + index, "route": "ascendc", "soc": "ascend910b"})
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
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config
