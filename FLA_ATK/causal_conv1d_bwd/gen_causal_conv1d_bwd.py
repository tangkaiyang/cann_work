"""causal_conv1d_bwd 的 ATK 泛化用例生成器。

生成确定性的泛化用例矩阵，覆盖 executor 当前支持的执行域：
- layout 固定为 BSND（逻辑 [B, T, D]），activation=0，不携带 initial_state / dht；
- dtype: bf16 / fp16 / fp32；
- B: 1 / 2 / 3 / 4 / 8，包含非二次幂 batch；
- T: 短序列、T < W / T == W，以及 16/32/64/128/256/1024 邻域；
- D: 16 的倍数，包含 48/80/112/144/240/272 等分块尾部；
- W: 1 / 2 / 3 / 4，包含单点卷积。

先生成 33 组定向 shape（99 条用例），使默认 -dt 100 覆盖关键边界；
其余 shape 经固定种子打乱、去重后追加。每组按三个 dtype 展开，任意
前缀的 dtype 数量相差不超过 1。限制单个 x 的元素数，控制 CPU 标杆开销。
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

DTYPES = ("bf16", "fp16", "fp32")
B_VALUES = (1, 2, 3, 4, 8)
T_VALUES = (1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
            127, 128, 129, 255, 256, 257, 1023, 1024, 1025)
D_VALUES = (16, 32, 48, 64, 80, 112, 128, 144, 240, 256, 272, 512)
W_VALUES = (1, 2, 3, 4)
MAX_INPUT_ELEMENTS = 1 << 20
TENSOR_NAMES = ("x", "weight", "dy")
TENSOR_RANGE_VALUES = (
    {"name": "nd", "mean": [-100, 100], "std": [1, 25]},
    [-0.001, 0.001],
    [-5, 5],
)


def _boundary_shapes():
    # README 默认 shape；逐一覆盖窗口宽度及窗口前后长度。
    shapes = [(1, 8, 16, 4)]
    shapes.extend((1, T, 16, W) for W in W_VALUES
                  for T in sorted({1, max(1, W - 1), W, W + 1}))
    # 序列分块边界，同时轮换 batch、特征尾部和窗口宽度。
    shapes.extend(
        (B_VALUES[i % len(B_VALUES)], T,
         D_VALUES[i % len(D_VALUES)], W_VALUES[i % len(W_VALUES)])
        for i, T in enumerate((15, 16, 17, 31, 32, 33, 63, 64, 65,
                               127, 128, 129, 255, 256, 257, 1023, 1024, 1025))
    )
    shapes.append((8, 256, 512, 4))
    return shapes


def _build_profiles():
    shape_combos = [
        (B, T, D, W)
        for B in B_VALUES
        for T in T_VALUES
        for D in D_VALUES
        for W in W_VALUES
        if B * T * D <= MAX_INPUT_ELEMENTS
    ]
    random.Random(SEED_BASE).shuffle(shape_combos)
    # 定向用例置前；保留旧矩阵全部 shape，并避免和随机矩阵重复。
    shape_combos = list(dict.fromkeys(_boundary_shapes() + shape_combos))
    profiles = []
    for B, T, D, W in shape_combos:
        for dtype in DTYPES:
            profiles.append(
                {
                    "name": f"{dtype}_bsnd_B{B}_T{T}_D{D}_W{W}",
                    "dtype": dtype,
                    "B": B,
                    "T": T,
                    "D": D,
                    "W": W,
                }
            )
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
                elif cfg.name in TENSOR_NAMES:
                    # YAML 声明占位 tensor；这里落定实际 shape、dtype 和分布。
                    # 按 shape 组轮换，使三种 dtype 都能覆盖全部分布。
                    cfg.dtype = _dtype(spec["dtype"])
                    cfg.shape = ([spec["W"], spec["D"]] if cfg.name == "weight"
                                 else [spec["B"], spec["T"], spec["D"]])
                    distribution_index = (
                        index // len(DTYPES) + TENSOR_NAMES.index(cfg.name)
                    ) % len(TENSOR_RANGE_VALUES)
                    cfg.range_values = deepcopy(TENSOR_RANGE_VALUES[distribution_index])
                    cfg.outlier_values = [0.001, 1000]
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config
