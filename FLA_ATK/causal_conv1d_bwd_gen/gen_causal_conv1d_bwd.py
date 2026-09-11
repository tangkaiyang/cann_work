"""causal_conv1d_bwd 的 ATK 泛化用例生成器（补充泛化版）。

相对源仓 gen_causal_conv1d_bwd.py 的扩展：
- 新增 layout 维度：BSND / BSH / BNSD（定长）与 TND / NTD（varlen，query_start_loc）；
- 新增 activation 维度：0（无激活）/ 1（SiLU）/ 2（Swish），act=1/2 时 executor 会构造 y；
- 新增 with_state 维度：是否携带 initial_state / dht（[B, W, D]）；
- varlen（TND/NTD）场景下由生成器确定性构造 query_start_loc（每段长度 >= W）并写入 case_spec；
- B/T/D/W 网格与源仓保持一致（T >= W，BSND/BSH 下 D % 16 == 0）。

case_spec 是唯一的真值来源：所有 shape / layout / activation / 状态开关 / varlen
分段都在生成器中确定性计算，executor 只按 spec 消费，保证离线可复现。
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

OP_NAME = "causal_conv1d_bwd"
SEED_BASE = 20260817

DTYPES = ("bf16", "fp16")
B_VALUES = (1, 2, 4, 8)
T_VALUES = (16, 32, 64, 128, 256)
D_VALUES = (16, 32, 64, 128, 256)
W_VALUES = (2, 3, 4)
LAYOUTS = ("BSND", "BSH", "BNSD", "TND", "NTD")
ACTIVATIONS = (0, 1, 2)
WITH_STATE = (False, True)
# varlen 布局允许的最大段数（query_start_loc 长度 = 段数 + 1）
VARLEN_MAX_SEGMENTS = 4
# 定长与 varlen 的配比：每 5 个 layout 用例中 3 个定长、2 个 varlen
FIXED_LAYOUTS = ("BSND", "BSH", "BNSD")
VARLEN_LAYOUTS = ("TND", "NTD")


def _build_varlen_segments(B: int, T: int, W: int, rng: random.Random) -> list[int]:
    """把总 token 数 T 切成 B 个长度 >= W 的段，返回 query_start_loc（长度 B+1）。"""
    if B * W > T:
        raise ValueError(f"varlen needs B*W <= T, got B={B} W={W} T={T}")
    base = T // B
    lens = [base] * B
    remainder = T - base * B
    idx = 0
    while remainder > 0:
        lens[idx % B] += 1
        idx += 1
    # 少量扰动：在保证每段 >= W 的前提下把某段挪 1 个 token 到邻段
    if B >= 2:
        for i in range(B - 1):
            if rng.random() < 0.5 and lens[i] - 1 >= W:
                lens[i] -= 1
                lens[i + 1] += 1
    loc = [0]
    for length in lens:
        loc.append(loc[-1] + length)
    return loc


def _build_profiles():
    """确定性生成 profile 矩阵，保证任意用例数量下各维度均衡覆盖。"""
    shape_combos = [
        (B, T, D, W)
        for B in B_VALUES
        for T in T_VALUES
        for D in D_VALUES
        for W in W_VALUES
        if T >= W
    ]
    random.Random(SEED_BASE).shuffle(shape_combos)
    layouts_with_mode = [
        (layout, False) for layout in FIXED_LAYOUTS
    ] + [
        (layout, True) for layout in VARLEN_LAYOUTS
    ]
    profiles = []
    for B, T, D, W in shape_combos:
        for dtype in DTYPES:
            for act_i, act in enumerate(ACTIVATIONS):
                for mode_i, (layout, is_varlen) in enumerate(layouts_with_mode):
                    if is_varlen and B * W > T:
                        # varlen 要求每段 >= W，构造不出则跳过该组合
                        continue
                    with_state = ((act_i + mode_i) % 2 == 1)
                    profile = {
                        "name": f"{dtype}_{layout.lower()}_B{B}_T{T}_D{D}_W{W}_act{act}" + ("_st" if with_state else ""),
                        "dtype": dtype,
                        "B": B,
                        "T": T,
                        "D": D,
                        "W": W,
                        "layout": layout,
                        "activation": act,
                        "with_state": with_state,
                    }
                    if is_varlen:
                        rng = random.Random(SEED_BASE + 7 * len(profiles))
                        profile["query_start_loc"] = _build_varlen_segments(B, T, W, rng)
                    profiles.append(profile)
    return profiles


PROFILES = _build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": SEED_BASE + index,
            "route": "ascendc",
            "soc": "ascend910b",
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
    class CausalConv1dBwdGenerator(CaseGenerator):
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
