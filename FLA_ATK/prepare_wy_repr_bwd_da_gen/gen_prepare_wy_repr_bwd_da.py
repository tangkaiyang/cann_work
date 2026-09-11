"""prepare_wy_repr_bwd_da 的 ATK 泛化用例生成器。

原有 200 条最终配置内置于本文件，不依赖外部 JSON 或其他算子生成器。
保留原有用例的顺序、形状、历史溯源字段和种子修正，之后追加定长泛化矩阵。
新增用例固定 K=128，覆盖 V=128/256、chunk_size=64/128、chunk 前后
长度、非二次幂 batch、单头及分组头；beta 由 executor 固定为 FP32，
gtype 使用原有四种 dtype/gtype 配对。继续沿用已有精度组合过滤规则。
executor 尚未构造变长元数据，新增用例均为 varlen=False。
默认 BF16/FP16 各 5000 条，共 10000 条；双 marker 默认使用 -dt 5000。
支持更大的 -dt：超出默认池后按 BF16/FP16 交替复用合法 shape，使用新的
case_id 和 seed；生成数量不受默认池大小限制，shape 并非无限去重。
"""

from __future__ import annotations

import json
import random
from collections import Counter
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

OP_NAME = "prepare_wy_repr_bwd_da"
CASE_COUNT = 200
CASES_PER_DTYPE = 5000
TOTAL_CASE_COUNT = 2 * CASES_PER_DTYPE
SMALL_ELEMENT_LIMIT = 20_000_000


DTYPE_GTYPE_PAIRS = (
    ("fp16", "fp16"),
    ("fp16", "fp32"),
    ("bf16", "bf16"),
    ("bf16", "fp32"),
)
CHUNK_SIZES = (64, 128)
SMALL_V_VALUES = (128, 256)


SCALE_METADATA_KEYS = (
    "scale_reference",
    "scale_reference_case_id",
    "scale_reference_elements",
    "source_T",
    "scaled_da_elements",
    "candidate_min_t",
)
SEED_ADJUSTMENTS = {125: 2, 167: 1000, 168: 1}


def _profile_key(profile):
    return tuple(
        profile.get(key)
        for key in (
            "dtype",
            "gtype",
            "B",
            "HK",
            "HV",
            "T",
            "K",
            "V",
            "chunk_size",
            "varlen",
            "mean_len",
        )
    )


def _shape_elements(profile):
    return profile["B"] * profile["T"] * (
        profile["HK"] * profile["K"]
        + profile["HV"]
        * (2 * profile["V"] + profile["K"] + profile["chunk_size"] + 2)
    )


def _is_filtered_profile(profile):
    return (
        profile["dtype"] == "fp16"
        and profile["V"] == 256
        and profile["chunk_size"] == 128
    )


# 原有 200 条最终配置已固化；历史 scale_reference 字段仅用于溯源，不再加载外部文件。
_FROZEN_BASE_PROFILES = (
    {'name': 'small_0_000_b4_h16x16_t24_scaled_t2_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 0, 'scale_reference_elements': 98560, 'source_T': 24, 'scaled_da_elements': 82176, 'candidate_min_t': 48},
    {'name': 'small_0_001_b2_h2x2_t128_scaled_t13_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 1, 'scale_reference_elements': 43104, 'source_T': 128, 'scaled_da_elements': 43368, 'candidate_min_t': 48},
    {'name': 'small_0_002_b1_h16x16_t256_scaled_t26', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 2, 'scale_reference_elements': 236256, 'source_T': 256, 'scaled_da_elements': 240448},
    {'name': 'small_0_003_b1_h8x8_t24_scaled_t2_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 3, 'scale_reference_elements': 14368, 'source_T': 24, 'scaled_da_elements': 13344, 'candidate_min_t': 48},
    {'name': 'small_0_004_b4_h4x4_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 4, 'scale_reference_elements': 20544, 'source_T': 24, 'scaled_da_elements': 18496, 'candidate_min_t': 48},
    {'name': 'small_0_005_b1_h4x4_t128_scaled_t13_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 5, 'scale_reference_elements': 30816, 'source_T': 128, 'scaled_da_elements': 30056, 'candidate_min_t': 48},
    {'name': 'small_0_006_b2_h16x16_t256_scaled_t25', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 16, 'HV': 16, 'T': 25, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 6, 'scale_reference_elements': 722304, 'source_T': 256, 'scaled_da_elements': 718400},
    {'name': 'small_0_007_b4_h16x16_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 7, 'scale_reference_elements': 98560, 'source_T': 24, 'scaled_da_elements': 82176, 'candidate_min_t': 48},
    {'name': 'small_0_008_b4_h2x2_t196_scaled_t19', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 4, 'HK': 2, 'HV': 2, 'T': 19, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 8, 'scale_reference_elements': 98560, 'source_T': 196, 'scaled_da_elements': 97584},
    {'name': 'small_0_009_b4_h8x8_t512_scaled_t51', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 8, 'HV': 8, 'T': 51, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 9, 'scale_reference_elements': 945024, 'source_T': 512, 'scaled_da_elements': 943296},
    {'name': 'small_0_010_b2_h4x4_t256_scaled_t26', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 4, 'HV': 4, 'T': 26, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 10, 'scale_reference_elements': 172416, 'source_T': 256, 'scaled_da_elements': 173472},
    {'name': 'small_0_011_b4_h8x8_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 11, 'scale_reference_elements': 57472, 'source_T': 24, 'scaled_da_elements': 53376, 'candidate_min_t': 48},
    {'name': 'small_0_012_b2_h4x4_t128_scaled_t13', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 4, 'T': 13, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 12, 'scale_reference_elements': 90288, 'source_T': 128, 'scaled_da_elements': 93392},
    {'name': 'small_0_013_b1_h16x16_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 13, 'scale_reference_elements': 20544, 'source_T': 24, 'scaled_da_elements': 18496, 'candidate_min_t': 48},
    {'name': 'small_0_014_b4_h8x8_t196_scaled_t20', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 20, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 14, 'scale_reference_elements': 369792, 'source_T': 196, 'scaled_da_elements': 369920},
    {'name': 'small_0_015_b4_h16x16_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 15, 'scale_reference_elements': 114944, 'source_T': 24, 'scaled_da_elements': 106752, 'candidate_min_t': 48},
    {'name': 'small_0_016_b2_h16x16_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 16, 'scale_reference_elements': 41088, 'source_T': 24, 'scaled_da_elements': 36992, 'candidate_min_t': 48},
    {'name': 'small_0_017_b2_h2x2_t24_scaled_t2_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 17, 'scale_reference_elements': 6160, 'source_T': 24, 'scaled_da_elements': 5136, 'candidate_min_t': 48},
    {'name': 'small_0_018_b1_h2x2_t196_scaled_t19_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 18, 'scale_reference_elements': 34884, 'source_T': 196, 'scaled_da_elements': 34124, 'candidate_min_t': 48},
    {'name': 'small_0_019_b1_h8x8_t196_scaled_t19', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 19, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 19, 'scale_reference_elements': 129312, 'source_T': 196, 'scaled_da_elements': 126768},
    {'name': 'small_0_020_b1_h16x16_t512_scaled_t51', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 51, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 20, 'scale_reference_elements': 738720, 'source_T': 512, 'scaled_da_elements': 732768},
    {'name': 'small_0_021_b4_h4x4_t256_scaled_t26', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 4, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 21, 'scale_reference_elements': 236256, 'source_T': 256, 'scaled_da_elements': 240448},
    {'name': 'small_0_022_b1_h2x2_t512_scaled_t52', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 52, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 22, 'scale_reference_elements': 66220, 'source_T': 512, 'scaled_da_elements': 66768},
    {'name': 'small_0_023_b1_h4x4_t256_scaled_t25_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 23, 'scale_reference_elements': 90288, 'source_T': 256, 'scaled_da_elements': 89800, 'candidate_min_t': 48},
    {'name': 'small_0_024_b4_h4x4_t196_scaled_t19', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 4, 'T': 19, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 24, 'scale_reference_elements': 279072, 'source_T': 196, 'scaled_da_elements': 272992},
    {'name': 'small_0_025_b1_h8x8_t196_scaled_t19', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 19, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 25, 'scale_reference_elements': 98560, 'source_T': 196, 'scaled_da_elements': 97584},
    {'name': 'small_0_026_b2_h4x4_t128_scaled_t13_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 26, 'scale_reference_elements': 86208, 'source_T': 128, 'scaled_da_elements': 86736, 'candidate_min_t': 48},
    {'name': 'small_0_027_b2_h16x16_t24_scaled_t2_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 27, 'scale_reference_elements': 49280, 'source_T': 24, 'scaled_da_elements': 41088, 'candidate_min_t': 48},
    {'name': 'small_0_028_b1_h2x2_t512_scaled_t52', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 52, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 28, 'scale_reference_elements': 86208, 'source_T': 512, 'scaled_da_elements': 86736},
    {'name': 'small_0_029_b2_h2x2_t24_scaled_t2', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 2, 'HV': 2, 'T': 2, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 29, 'scale_reference_elements': 8208, 'source_T': 24, 'scaled_da_elements': 7184},
    {'name': 'small_0_030_b1_h2x2_t256_scaled_t26_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 30, 'scale_reference_elements': 29532, 'source_T': 256, 'scaled_da_elements': 30056, 'candidate_min_t': 48},
    {'name': 'small_0_031_b4_h4x4_t256_scaled_t26', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 4, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 31, 'scale_reference_elements': 236256, 'source_T': 256, 'scaled_da_elements': 240448},
    {'name': 'small_0_032_b2_h8x8_t128_scaled_t13', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 8, 'HV': 8, 'T': 13, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 32, 'scale_reference_elements': 172416, 'source_T': 128, 'scaled_da_elements': 173472},
    {'name': 'small_0_033_b2_h8x8_t256_scaled_t26', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 8, 'HV': 8, 'T': 26, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 33, 'scale_reference_elements': 344832, 'source_T': 256, 'scaled_da_elements': 346944},
    {'name': 'small_0_034_b1_h16x16_t196_scaled_t20', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 20, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 34, 'scale_reference_elements': 184896, 'source_T': 196, 'scaled_da_elements': 184960},
    {'name': 'small_0_035_b4_h4x4_t128_scaled_t13', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 13, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 35, 'scale_reference_elements': 123264, 'source_T': 128, 'scaled_da_elements': 120224},
    {'name': 'small_0_036_b2_h2x2_t24_scaled_t8', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 2, 'HV': 2, 'T': 8, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 36, 'scale_reference_elements': 6160, 'source_T': 24, 'scaled_da_elements': 20544},
    {'name': 'small_0_037_b4_h16x16_t196_scaled_t19', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 19, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 37, 'scale_reference_elements': 788480, 'source_T': 196, 'scaled_da_elements': 780672},
    {'name': 'small_0_038_b4_h8x8_t512_scaled_t52', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 52, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 38, 'scale_reference_elements': 1059520, 'source_T': 512, 'scaled_da_elements': 1068288},
    {'name': 'small_0_039_b2_h8x8_t256_scaled_t25', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 8, 'HV': 8, 'T': 25, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 39, 'scale_reference_elements': 258720, 'source_T': 256, 'scaled_da_elements': 256800},
    {'name': 'small_0_040_b1_h2x2_t256_scaled_t26_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 40, 'scale_reference_elements': 29532, 'source_T': 256, 'scaled_da_elements': 30056, 'candidate_min_t': 48},
    {'name': 'small_0_041_b2_h2x2_t512_scaled_t52', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 2, 'HV': 2, 'T': 52, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 41, 'scale_reference_elements': 172416, 'source_T': 512, 'scaled_da_elements': 173472},
    {'name': 'small_0_042_b2_h2x2_t196_scaled_t19_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 42, 'scale_reference_elements': 49280, 'source_T': 196, 'scaled_da_elements': 48792, 'candidate_min_t': 48},
    {'name': 'small_0_043_b1_h16x16_t128_scaled_t13', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 13, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 43, 'scale_reference_elements': 123264, 'source_T': 128, 'scaled_da_elements': 120224},
    {'name': 'small_0_044_b1_h2x2_t128_scaled_t13_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 44, 'scale_reference_elements': 16940, 'source_T': 128, 'scaled_da_elements': 16692, 'candidate_min_t': 48},
    {'name': 'small_1_000_b4_h8x32_t196_scaled_t1', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 8, 'HV': 32, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 45, 'scale_reference_elements': 14368, 'source_T': 196, 'scaled_da_elements': 61696},
    {'name': 'small_1_001_b2_h4x32_t128_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 46, 'scale_reference_elements': 86208, 'source_T': 128, 'scaled_da_elements': 92416, 'candidate_min_t': 48},
    {'name': 'small_1_002_b2_h4x16_t128_scaled_t22', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 4, 'HV': 16, 'T': 22, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 47, 'scale_reference_elements': 344832, 'source_T': 128, 'scaled_da_elements': 339328},
    {'name': 'small_1_003_b2_h2x4_t256_scaled_t256', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 2, 'HV': 4, 'T': 256, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 48, 'scale_reference_elements': 1188096, 'source_T': 256, 'scaled_da_elements': 1183744},
    {'name': 'small_1_004_b4_h4x8_t512_scaled_t24', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 8, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 49, 'scale_reference_elements': 603648, 'source_T': 512, 'scaled_da_elements': 591360},
    {'name': 'small_1_005_b4_h8x32_t24_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 50, 'scale_reference_elements': 192192, 'source_T': 24, 'scaled_da_elements': 205312, 'candidate_min_t': 48},
    {'name': 'small_1_006_b4_h4x16_t512_scaled_t4_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 51, 'scale_reference_elements': 118608, 'source_T': 512, 'scaled_da_elements': 123392, 'candidate_min_t': 48},
    {'name': 'small_1_007_b4_h4x32_t24_scaled_t14', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 32, 'T': 14, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 52, 'scale_reference_elements': 1254336, 'source_T': 24, 'scaled_da_elements': 1293824},
    {'name': 'small_1_008_b2_h4x8_t196_scaled_t26', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 8, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 53, 'scale_reference_elements': 238080, 'source_T': 196, 'scaled_da_elements': 240448},
    {'name': 'small_1_009_b2_h4x8_t512_scaled_t170', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 8, 'T': 170, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 54, 'scale_reference_elements': 1572480, 'source_T': 512, 'scaled_da_elements': 1572160},
    {'name': 'small_1_010_b2_h4x16_t512_scaled_t13', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 16, 'T': 13, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 55, 'scale_reference_elements': 201216, 'source_T': 512, 'scaled_da_elements': 200512},
    {'name': 'small_1_011_b1_h4x16_t196_scaled_t14', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 14, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 56, 'scale_reference_elements': 180736, 'source_T': 196, 'scaled_da_elements': 179648},
    {'name': 'small_1_012_b4_h8x16_t196_scaled_t9', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 16, 'T': 9, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 57, 'scale_reference_elements': 474432, 'source_T': 196, 'scaled_da_elements': 480384},
    {'name': 'small_1_013_b2_h4x16_t196_scaled_t33', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 16, 'T': 33, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 58, 'scale_reference_elements': 786240, 'source_T': 196, 'scaled_da_elements': 779328},
    {'name': 'small_1_014_b1_h4x32_t512_scaled_t17', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 32, 'T': 17, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 59, 'scale_reference_elements': 252960, 'source_T': 512, 'scaled_da_elements': 253504},
    {'name': 'small_1_015_b4_h2x4_t128_scaled_t85', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 2, 'HV': 4, 'T': 85, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 60, 'scale_reference_elements': 1046656, 'source_T': 128, 'scaled_da_elements': 1047200},
    {'name': 'small_1_016_b2_h2x4_t512_scaled_t100', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 2, 'HV': 4, 'T': 100, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 61, 'scale_reference_elements': 461952, 'source_T': 512, 'scaled_da_elements': 462400},
    {'name': 'small_1_017_b4_h8x32_t24_scaled_t8', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 32, 'T': 8, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 62, 'scale_reference_elements': 763200, 'source_T': 24, 'scaled_da_elements': 755712},
    {'name': 'small_1_018_b2_h4x16_t256_scaled_t9', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 16, 'T': 9, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 63, 'scale_reference_elements': 160128, 'source_T': 256, 'scaled_da_elements': 157248},
    {'name': 'small_1_019_b4_h4x32_t256_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 64, 'scale_reference_elements': 237216, 'source_T': 256, 'scaled_da_elements': 203520, 'candidate_min_t': 48},
    {'name': 'small_1_020_b4_h2x4_t128_scaled_t25', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 4, 'HK': 2, 'HV': 4, 'T': 25, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 65, 'scale_reference_elements': 205312, 'source_T': 128, 'scaled_da_elements': 205600},
    {'name': 'small_1_021_b2_h2x4_t256_scaled_t110', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 2, 'HV': 4, 'T': 110, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 66, 'scale_reference_elements': 452928, 'source_T': 256, 'scaled_da_elements': 452320},
    {'name': 'small_1_022_b1_h8x32_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 32, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 67, 'scale_reference_elements': 1768704, 'source_T': 24, 'scaled_da_elements': 615936},
    {'name': 'small_1_023_b1_h8x32_t24_scaled_t4_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 68, 'scale_reference_elements': 101728, 'source_T': 24, 'scaled_da_elements': 94464, 'candidate_min_t': 48},
    {'name': 'small_1_024_b2_h4x16_t24_scaled_t7_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 69, 'scale_reference_elements': 106352, 'source_T': 24, 'scaled_da_elements': 107968, 'candidate_min_t': 48},
    {'name': 'small_1_025_b1_h2x4_t196_scaled_t19', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 19, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 70, 'scale_reference_elements': 59520, 'source_T': 196, 'scaled_da_elements': 58520},
    {'name': 'small_1_026_b2_h8x32_t512_scaled_t2_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 71, 'scale_reference_elements': 51328, 'source_T': 512, 'scaled_da_elements': 61696, 'candidate_min_t': 48},
    {'name': 'small_1_027_b2_h4x32_t24_scaled_t1', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 32, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 72, 'scale_reference_elements': 34944, 'source_T': 24, 'scaled_da_elements': 33920},
    {'name': 'small_1_028_b4_h2x4_t256_scaled_t5_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 4, 'HK': 2, 'HV': 4, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 73, 'scale_reference_elements': 60048, 'source_T': 256, 'scaled_da_elements': 61600, 'candidate_min_t': 48},
    {'name': 'small_1_029_b2_h4x32_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 32, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 74, 'scale_reference_elements': 1572480, 'source_T': 24, 'scaled_da_elements': 814080},
    {'name': 'small_1_030_b2_h8x32_t512_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 75, 'scale_reference_elements': 84224, 'source_T': 512, 'scaled_da_elements': 69888, 'candidate_min_t': 48},
    {'name': 'small_1_031_b1_h2x4_t256_scaled_t139', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 139, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 76, 'scale_reference_elements': 320256, 'source_T': 256, 'scaled_da_elements': 321368},
    {'name': 'small_1_032_b1_h4x32_t512_scaled_t5_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 77, 'scale_reference_elements': 84224, 'source_T': 512, 'scaled_da_elements': 84800, 'candidate_min_t': 48},
    {'name': 'small_1_033_b1_h4x32_t128_scaled_t70', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 70, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 78, 'scale_reference_elements': 1768576, 'source_T': 128, 'scaled_da_elements': 1760640},
    {'name': 'small_1_034_b2_h8x16_t24_scaled_t4', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 8, 'HV': 16, 'T': 4, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 79, 'scale_reference_elements': 59304, 'source_T': 24, 'scaled_da_elements': 65792},
    {'name': 'small_1_035_b4_h2x4_t128_scaled_t105', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 2, 'HV': 4, 'T': 105, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 80, 'scale_reference_elements': 863296, 'source_T': 128, 'scaled_da_elements': 863520},
    {'name': 'small_1_036_b4_h4x16_t128_scaled_t10', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 16, 'T': 10, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 81, 'scale_reference_elements': 321728, 'source_T': 128, 'scaled_da_elements': 308480},
    {'name': 'small_1_037_b4_h2x4_t196_scaled_t4_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 4, 'HK': 2, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 82, 'scale_reference_elements': 36992, 'source_T': 196, 'scaled_da_elements': 36992, 'candidate_min_t': 48},
    {'name': 'small_1_038_b2_h8x16_t256_scaled_t4_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 2, 'HK': 8, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 83, 'scale_reference_elements': 101728, 'source_T': 256, 'scaled_da_elements': 98560, 'candidate_min_t': 48},
    {'name': 'small_1_039_b2_h4x8_t256_scaled_t42', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 8, 'T': 42, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 84, 'scale_reference_elements': 384384, 'source_T': 256, 'scaled_da_elements': 388416},
    {'name': 'small_1_040_b4_h4x16_t24_scaled_t6_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 4, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 85, 'scale_reference_elements': 180736, 'source_T': 24, 'scaled_da_elements': 185088, 'candidate_min_t': 48},
    {'name': 'small_1_041_b2_h4x16_t24_scaled_t24', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 2, 'HK': 4, 'HV': 16, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 86, 'scale_reference_elements': 640512, 'source_T': 24, 'scaled_da_elements': 370176},
    {'name': 'small_replacement_087_b1_h2x2_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 87, 'scale_reference_elements': 237216, 'source_T': 24, 'scaled_da_elements': 27744},
    {'name': 'small_1_043_b1_h2x4_t256_scaled_t21', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 21, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 88, 'scale_reference_elements': 69888, 'source_T': 256, 'scaled_da_elements': 70056},
    {'name': 'small_1_044_b2_h4x8_t512_scaled_t4_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 2, 'HK': 4, 'HV': 8, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'small', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 89, 'scale_reference_elements': 34944, 'source_T': 512, 'scaled_da_elements': 32896, 'candidate_min_t': 48},
    {'name': 'small_2_000_b1_h2x2_t256_scaled_t256', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 256, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 90, 'scale_reference_elements': 690176, 'source_T': 256, 'scaled_da_elements': 295936},
    {'name': 'small_2_001_b1_h4x4_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 91, 'scale_reference_elements': 84656, 'source_T': 24, 'scaled_da_elements': 80064},
    {'name': 'small_2_002_b1_h8x8_t512_scaled_t92', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 92, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 92, 'scale_reference_elements': 425408, 'source_T': 512, 'scaled_da_elements': 425408},
    {'name': 'small_2_003_b1_h2x2_t128_scaled_t66', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 66, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 93, 'scale_reference_elements': 118608, 'source_T': 128, 'scaled_da_elements': 118536},
    {'name': 'small_2_004_b1_h4x4_t256_scaled_t122', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 122, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 94, 'scale_reference_elements': 314432, 'source_T': 256, 'scaled_da_elements': 313296},
    {'name': 'small_2_005_b1_h16x16_t256_scaled_t21', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 21, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 95, 'scale_reference_elements': 192192, 'source_T': 256, 'scaled_da_elements': 194208},
    {'name': 'small_2_006_b1_h2x2_t24_scaled_t18_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 96, 'scale_reference_elements': 29532, 'source_T': 24, 'scaled_da_elements': 30024, 'candidate_min_t': 48},
    {'name': 'small_2_007_b1_h16x16_t512_scaled_t1', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 97, 'scale_reference_elements': 7184, 'source_T': 512, 'scaled_da_elements': 10272},
    {'name': 'small_2_008_b1_h16x16_t256_scaled_t26', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 98, 'scale_reference_elements': 236256, 'source_T': 256, 'scaled_da_elements': 240448},
    {'name': 'small_2_009_b1_h8x8_t256_scaled_t4_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 99, 'scale_reference_elements': 22572, 'source_T': 256, 'scaled_da_elements': 20544, 'candidate_min_t': 48},
    {'name': 'small_2_010_b1_h8x8_t512_scaled_t14', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 14, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 100, 'scale_reference_elements': 64680, 'source_T': 512, 'scaled_da_elements': 64736},
    {'name': 'small_2_011_b1_h8x8_t196_scaled_t46', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 46, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 101, 'scale_reference_elements': 236256, 'source_T': 196, 'scaled_da_elements': 236256},
    {'name': 'small_2_012_b1_h4x4_t24_scaled_t1', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 1, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 102, 'scale_reference_elements': 3592, 'source_T': 24, 'scaled_da_elements': 3592},
    {'name': 'small_2_013_b1_h4x4_t24_scaled_t24', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 103, 'scale_reference_elements': 529760, 'source_T': 24, 'scaled_da_elements': 55488},
    {'name': 'small_2_014_b1_h4x4_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 104, 'scale_reference_elements': 236256, 'source_T': 24, 'scaled_da_elements': 55488},
    {'name': 'small_2_015_b1_h2x2_t512_scaled_t72', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 72, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 105, 'scale_reference_elements': 129360, 'source_T': 512, 'scaled_da_elements': 129312},
    {'name': 'small_2_016_b1_h2x2_t128_scaled_t128', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 128, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 106, 'scale_reference_elements': 236256, 'source_T': 128, 'scaled_da_elements': 213504},
    {'name': 'small_2_017_b1_h16x16_t196_scaled_t7_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 107, 'scale_reference_elements': 98560, 'source_T': 196, 'scaled_da_elements': 100576, 'candidate_min_t': 48},
    {'name': 'small_2_018_b1_h2x2_t128_scaled_t5', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 5, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 108, 'scale_reference_elements': 8208, 'source_T': 128, 'scaled_da_elements': 8340},
    {'name': 'small_2_019_b1_h2x2_t256_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 109, 'scale_reference_elements': 5136, 'source_T': 256, 'scaled_da_elements': 5388, 'candidate_min_t': 48},
    {'name': 'small_2_020_b1_h16x16_t512_scaled_t1', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 110, 'scale_reference_elements': 5136, 'source_T': 512, 'scaled_da_elements': 9248},
    {'name': 'small_2_021_b1_h4x4_t196_scaled_t26', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 26, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 111, 'scale_reference_elements': 92340, 'source_T': 196, 'scaled_da_elements': 93392},
    {'name': 'small_2_022_b1_h8x8_t24_scaled_t3_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 112, 'scale_reference_elements': 21552, 'source_T': 24, 'scaled_da_elements': 20016, 'candidate_min_t': 48},
    {'name': 'small_2_023_b1_h16x16_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 113, 'scale_reference_elements': 279072, 'source_T': 24, 'scaled_da_elements': 246528},
    {'name': 'small_2_024_b1_h8x8_t128_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 114, 'scale_reference_elements': 21552, 'source_T': 128, 'scaled_da_elements': 20016, 'candidate_min_t': 48},
    {'name': 'small_2_025_b1_h2x2_t512_scaled_t27', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 27, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 115, 'scale_reference_elements': 45144, 'source_T': 512, 'scaled_da_elements': 45036},
    {'name': 'small_2_026_b1_h8x8_t24_scaled_t24', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 116, 'scale_reference_elements': 472512, 'source_T': 24, 'scaled_da_elements': 160128},
    {'name': 'small_2_027_b1_h2x2_t128_scaled_t54', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 54, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 117, 'scale_reference_elements': 69768, 'source_T': 128, 'scaled_da_elements': 69336},
    {'name': 'small_2_028_b1_h2x2_t512_scaled_t11', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 11, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 118, 'scale_reference_elements': 14368, 'source_T': 512, 'scaled_da_elements': 14124},
    {'name': 'small_2_029_b1_h16x16_t256_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 119, 'scale_reference_elements': 24640, 'source_T': 256, 'scaled_da_elements': 27744, 'candidate_min_t': 48},
    {'name': 'small_2_030_b1_h8x8_t256_scaled_t13', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 13, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 120, 'scale_reference_elements': 86208, 'source_T': 256, 'scaled_da_elements': 86736},
    {'name': 'small_2_031_b1_h2x2_t196_scaled_t48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 2, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 121, 'scale_reference_elements': 86208, 'source_T': 196, 'scaled_da_elements': 86208},
    {'name': 'small_2_032_b1_h16x16_t128_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 122, 'scale_reference_elements': 14368, 'source_T': 128, 'scaled_da_elements': 18496, 'candidate_min_t': 48},
    {'name': 'small_2_033_b1_h8x8_t512_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 123, 'scale_reference_elements': 16940, 'source_T': 512, 'scaled_da_elements': 20016, 'candidate_min_t': 48},
    {'name': 'small_2_034_b1_h16x16_t128_scaled_t5_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 124, 'scale_reference_elements': 66220, 'source_T': 128, 'scaled_da_elements': 66720, 'candidate_min_t': 48},
    {'name': 'small_2_035_b1_h2x2_t196_scaled_t184_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 184, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 125, 'scale_reference_elements': 236256, 'source_T': 196, 'scaled_da_elements': 236256, 'candidate_min_t': 48},
    {'name': 'small_2_036_b1_h16x16_t128_scaled_t12', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 16, 'T': 12, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 126, 'scale_reference_elements': 172416, 'source_T': 128, 'scaled_da_elements': 172416},
    {'name': 'small_2_037_b1_h4x4_t24_scaled_t14', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 14, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 127, 'scale_reference_elements': 34884, 'source_T': 24, 'scaled_da_elements': 35952},
    {'name': 'small_2_038_b1_h8x8_t256_scaled_t27', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 27, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 128, 'scale_reference_elements': 123264, 'source_T': 256, 'scaled_da_elements': 124848},
    {'name': 'small_2_039_b1_h2x2_t256_scaled_t256', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 2, 'T': 256, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 129, 'scale_reference_elements': 344832, 'source_T': 256, 'scaled_da_elements': 328704},
    {'name': 'small_2_040_b1_h4x4_t512_scaled_t52', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 52, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 130, 'scale_reference_elements': 172416, 'source_T': 512, 'scaled_da_elements': 173472},
    {'name': 'small_2_041_b1_h8x8_t196_scaled_t5_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 131, 'scale_reference_elements': 24640, 'source_T': 196, 'scaled_da_elements': 23120, 'candidate_min_t': 48},
    {'name': 'small_2_042_b1_h4x4_t196_scaled_t78', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 78, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 132, 'scale_reference_elements': 180576, 'source_T': 196, 'scaled_da_elements': 180336},
    {'name': 'small_2_043_b1_h4x4_t24_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 133, 'scale_reference_elements': 6160, 'source_T': 24, 'scaled_da_elements': 6936, 'candidate_min_t': 48},
    {'name': 'small_2_044_b1_h4x4_t512_scaled_t35', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 35, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 134, 'scale_reference_elements': 118128, 'source_T': 512, 'scaled_da_elements': 116760},
    {'name': 'small_3_000_b1_h2x4_t128_scaled_t14_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 135, 'scale_reference_elements': 32340, 'source_T': 128, 'scaled_da_elements': 32368, 'candidate_min_t': 48},
    {'name': 'small_3_001_b1_h8x32_t256_scaled_t10_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 136, 'scale_reference_elements': 172416, 'source_T': 256, 'scaled_da_elements': 174720, 'candidate_min_t': 48},
    {'name': 'small_3_002_b1_h4x8_t128_scaled_t20', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 8, 'T': 20, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 137, 'scale_reference_elements': 92448, 'source_T': 128, 'scaled_da_elements': 92480},
    {'name': 'small_3_003_b1_h4x16_t256_scaled_t4_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 138, 'scale_reference_elements': 46224, 'source_T': 256, 'scaled_da_elements': 51328, 'candidate_min_t': 48},
    {'name': 'small_3_004_b1_h4x16_t128_scaled_t1', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 16, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 139, 'scale_reference_elements': 5136, 'source_T': 128, 'scaled_da_elements': 7712},
    {'name': 'small_3_005_b1_h2x4_t128_scaled_t56', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 56, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 140, 'scale_reference_elements': 172416, 'source_T': 128, 'scaled_da_elements': 172480},
    {'name': 'small_3_006_b1_h4x16_t128_scaled_t2_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 141, 'scale_reference_elements': 24640, 'source_T': 128, 'scaled_da_elements': 23616, 'candidate_min_t': 48},
    {'name': 'small_3_007_b1_h4x8_t512_scaled_t26', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 8, 'T': 26, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 142, 'scale_reference_elements': 118128, 'source_T': 512, 'scaled_da_elements': 120224},
    {'name': 'small_3_008_b1_h8x16_t256_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 143, 'scale_reference_elements': 28240, 'source_T': 256, 'scaled_da_elements': 27744, 'candidate_min_t': 48},
    {'name': 'small_3_009_b1_h4x16_t256_scaled_t52', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 52, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 144, 'scale_reference_elements': 452928, 'source_T': 256, 'scaled_da_elements': 454272},
    {'name': 'small_3_010_b1_h4x16_t512_scaled_t6_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 145, 'scale_reference_elements': 56480, 'source_T': 512, 'scaled_da_elements': 52416, 'candidate_min_t': 48},
    {'name': 'small_3_011_b1_h4x32_t24_scaled_t22', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 22, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 146, 'scale_reference_elements': 327360, 'source_T': 24, 'scaled_da_elements': 328064},
    {'name': 'small_3_012_b1_h8x16_t256_scaled_t12', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 12, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 147, 'scale_reference_elements': 96096, 'source_T': 256, 'scaled_da_elements': 98688},
    {'name': 'small_3_013_b1_h8x16_t256_scaled_t4_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 148, 'scale_reference_elements': 40032, 'source_T': 256, 'scaled_da_elements': 36992, 'candidate_min_t': 48},
    {'name': 'small_3_014_b1_h4x16_t24_scaled_t12', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 12, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 149, 'scale_reference_elements': 153984, 'source_T': 24, 'scaled_da_elements': 153984},
    {'name': 'small_3_015_b1_h4x32_t196_scaled_t16', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 32, 'T': 16, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 150, 'scale_reference_elements': 237216, 'source_T': 196, 'scaled_da_elements': 238592},
    {'name': 'small_3_016_b1_h4x8_t128_scaled_t58', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 8, 'T': 58, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 151, 'scale_reference_elements': 237216, 'source_T': 128, 'scaled_da_elements': 238496},
    {'name': 'small_3_017_b1_h4x32_t196_scaled_t9_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 152, 'scale_reference_elements': 226464, 'source_T': 196, 'scaled_da_elements': 226368, 'candidate_min_t': 48},
    {'name': 'small_3_018_b1_h8x32_t512_scaled_t29', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 32, 'T': 29, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 153, 'scale_reference_elements': 442144, 'source_T': 512, 'scaled_da_elements': 447296},
    {'name': 'small_3_019_b1_h4x16_t24_scaled_t3_min48', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 154, 'scale_reference_elements': 33920, 'source_T': 24, 'scaled_da_elements': 35424, 'candidate_min_t': 48},
    {'name': 'small_3_020_b1_h8x16_t128_scaled_t23', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 23, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 155, 'scale_reference_elements': 212704, 'source_T': 128, 'scaled_da_elements': 212704},
    {'name': 'small_3_021_b1_h4x16_t196_scaled_t27', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 16, 'T': 27, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 156, 'scale_reference_elements': 237216, 'source_T': 196, 'scaled_da_elements': 235872},
    {'name': 'small_3_022_b1_h4x32_t256_scaled_t2', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 2, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 157, 'scale_reference_elements': 29760, 'source_T': 256, 'scaled_da_elements': 29824},
    {'name': 'small_3_023_b1_h8x16_t196_scaled_t35', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 35, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 158, 'scale_reference_elements': 288320, 'source_T': 196, 'scaled_da_elements': 287840},
    {'name': 'small_3_024_b1_h8x32_t512_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 159, 'scale_reference_elements': 50864, 'source_T': 512, 'scaled_da_elements': 52416, 'candidate_min_t': 48},
    {'name': 'small_3_025_b1_h8x16_t128_scaled_t54', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 16, 'T': 54, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 160, 'scale_reference_elements': 497216, 'source_T': 128, 'scaled_da_elements': 499392},
    {'name': 'small_3_026_b1_h2x4_t512_scaled_t340', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 4, 'T': 340, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 161, 'scale_reference_elements': 786240, 'source_T': 512, 'scaled_da_elements': 786080},
    {'name': 'small_3_027_b1_h8x16_t256_scaled_t3_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 162, 'scale_reference_elements': 25664, 'source_T': 256, 'scaled_da_elements': 27744, 'candidate_min_t': 48},
    {'name': 'small_3_028_b1_h2x4_t128_scaled_t55', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 55, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 163, 'scale_reference_elements': 112960, 'source_T': 128, 'scaled_da_elements': 113080},
    {'name': 'small_3_029_b1_h2x4_t256_scaled_t75', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 75, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 164, 'scale_reference_elements': 172544, 'source_T': 256, 'scaled_da_elements': 173400},
    {'name': 'small_3_030_b1_h8x32_t24_scaled_t17', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 17, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 165, 'scale_reference_elements': 390080, 'source_T': 24, 'scaled_da_elements': 401472},
    {'name': 'small_3_031_b1_h8x32_t128_scaled_t9_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 166, 'scale_reference_elements': 157216, 'source_T': 128, 'scaled_da_elements': 157248, 'candidate_min_t': 48},
    {'name': 'small_3_032_b1_h2x4_t128_scaled_t128_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 128, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 167, 'scale_reference_elements': 884288, 'source_T': 128, 'scaled_da_elements': 427008, 'candidate_min_t': 48},
    {'name': 'small_3_033_b1_h2x4_t256_scaled_t55_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 4, 'T': 55, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 168, 'scale_reference_elements': 112960, 'source_T': 256, 'scaled_da_elements': 113080, 'candidate_min_t': 48},
    {'name': 'small_3_034_b1_h8x16_t512_scaled_t14', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 14, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 169, 'scale_reference_elements': 118608, 'source_T': 512, 'scaled_da_elements': 115136},
    {'name': 'small_3_035_b1_h4x8_t24_scaled_t24', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 8, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 5, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 170, 'scale_reference_elements': 237216, 'source_T': 24, 'scaled_da_elements': 110976},
    {'name': 'small_3_036_b1_h4x8_t128_scaled_t4_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 8, 'T': 48, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 171, 'scale_reference_elements': 25432, 'source_T': 128, 'scaled_da_elements': 26688, 'candidate_min_t': 48},
    {'name': 'small_3_037_b1_h4x8_t512_scaled_t14_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 8, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 3, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 172, 'scale_reference_elements': 59304, 'source_T': 512, 'scaled_da_elements': 57568, 'candidate_min_t': 48},
    {'name': 'small_3_038_b1_h4x32_t512_scaled_t3_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 173, 'scale_reference_elements': 51328, 'source_T': 512, 'scaled_da_elements': 50880, 'candidate_min_t': 48},
    {'name': 'small_3_039_b1_h8x16_t128_scaled_t18', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 16, 'T': 18, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 174, 'scale_reference_elements': 215680, 'source_T': 128, 'scaled_da_elements': 221760},
    {'name': 'small_3_040_b1_h4x16_t24_scaled_t5_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 16, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 2, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 175, 'scale_reference_elements': 42328, 'source_T': 24, 'scaled_da_elements': 38560, 'candidate_min_t': 48},
    {'name': 'small_3_041_b1_h4x32_t256_scaled_t3_min48', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 9, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 176, 'scale_reference_elements': 53176, 'source_T': 256, 'scaled_da_elements': 50880, 'candidate_min_t': 48},
    {'name': 'small_3_042_b1_h2x4_t24_scaled_t24', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 2, 'HV': 4, 'T': 24, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'size_class': 'small', 'mean_len': 4, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 177, 'scale_reference_elements': 425408, 'source_T': 24, 'scaled_da_elements': 55488},
    {'name': 'small_3_043_b1_h8x16_t512_scaled_t1', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 8, 'HV': 16, 'T': 1, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 178, 'scale_reference_elements': 11296, 'source_T': 512, 'scaled_da_elements': 12320},
    {'name': 'small_3_044_b1_h4x8_t256_scaled_t14', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 8, 'T': 14, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'size_class': 'small', 'mean_len': 16, 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 179, 'scale_reference_elements': 84656, 'source_T': 256, 'scaled_da_elements': 86240},
    {'name': 'large_var_hk_eq_hv_4_scaled_t7', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 32, 'HV': 32, 'T': 7, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': True, 'mean_len': 5, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 180, 'scale_reference_elements': 212704, 'source_T': 1024, 'scaled_da_elements': 201152},
    {'name': 'large_fix_hk_eq_hv_5_scaled_t47', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 32, 'HV': 32, 'T': 47, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 181, 'scale_reference_elements': 863296, 'source_T': 2048, 'scaled_da_elements': 869312},
    {'name': 'large_var_hk_eq_hv_5_scaled_t9_min48', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 32, 'HV': 32, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'mean_len': 16, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 182, 'scale_reference_elements': 160128, 'source_T': 2048, 'scaled_da_elements': 166464, 'candidate_min_t': 48},
    {'name': 'large_fix_hk_eq_hv_4_scaled_t1', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 5, 'HK': 16, 'HV': 16, 'T': 1, 'K': 128, 'V': 256, 'chunk_size': 128, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 183, 'scale_reference_elements': 17472, 'source_T': 1024, 'scaled_da_elements': 71840},
    {'name': 'large_phase_1_fix_1_scaled_t6_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 16, 'HK': 8, 'HV': 8, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 184, 'scale_reference_elements': 442176, 'source_T': 1024, 'scaled_da_elements': 443904, 'candidate_min_t': 48},
    {'name': 'large_phase_1_fix_2_scaled_t1', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 1, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 185, 'scale_reference_elements': 5648, 'source_T': 2048, 'scaled_da_elements': 36992},
    {'name': 'large_phase_1_fix_6_scaled_t17', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 8, 'HK': 8, 'HV': 8, 'T': 17, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 186, 'scale_reference_elements': 627168, 'source_T': 2048, 'scaled_da_elements': 628864},
    {'name': 'large_phase_1_var_3_scaled_t9', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 32, 'HV': 32, 'T': 9, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'mean_len': 17, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 187, 'scale_reference_elements': 160128, 'source_T': 4096, 'scaled_da_elements': 166464},
    {'name': 'large_phase_1_fix_5_scaled_t8_min48', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 33, 'HK': 4, 'HV': 4, 'T': 48, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 188, 'scale_reference_elements': 615936, 'source_T': 1024, 'scaled_da_elements': 610368, 'candidate_min_t': 48},
    {'name': 'large_gva_fix_4_scaled_t1', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 73, 'HK': 2, 'HV': 64, 'T': 1, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 189, 'scale_reference_elements': 237216, 'source_T': 24, 'scaled_da_elements': 3317120},
    {'name': 'large_gva_fix_1_scaled_t120', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 32, 'T': 120, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 190, 'scale_reference_elements': 2954880, 'source_T': 3246, 'scaled_da_elements': 2956800},
    {'name': 'large_gva_var_1_scaled_t153', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 16, 'HV': 32, 'T': 153, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'mean_len': 128, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 191, 'scale_reference_elements': 3780096, 'source_T': 3246, 'scaled_da_elements': 3769920},
    {'name': 'large_gva_var_2_scaled_t157', 'dtype': 'bf16', 'gtype': 'fp32', 'B': 1, 'HK': 21, 'HV': 63, 'T': 157, 'K': 128, 'V': 256, 'chunk_size': 64, 'varlen': True, 'mean_len': 2, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 192, 'scale_reference_elements': 7387200, 'source_T': 1696, 'scaled_da_elements': 7405062},
    {'name': 'large_phase_1_fix_13_scaled_t409', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 409, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 193, 'scale_reference_elements': 7560192, 'source_T': 4325, 'scaled_da_elements': 7564864},
    {'name': 'large_phase_1_fix_14_scaled_t820', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 820, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 194, 'scale_reference_elements': 7580736, 'source_T': 8650, 'scaled_da_elements': 7583360},
    {'name': 'large_phase_1_var_4_scaled_t348', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 348, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': True, 'mean_len': 9, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 195, 'scale_reference_elements': 6438912, 'source_T': 4325, 'scaled_da_elements': 6436608},
    {'name': 'large_phase_1_var_5_scaled_t779', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 779, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'mean_len': 2, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 196, 'scale_reference_elements': 8006400, 'source_T': 7788, 'scaled_da_elements': 8001888},
    {'name': 'large_phase_1_var_6_scaled_t1559', 'dtype': 'fp16', 'gtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 1559, 'K': 128, 'V': 128, 'chunk_size': 128, 'varlen': True, 'mean_len': 2, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 197, 'scale_reference_elements': 8006400, 'source_T': 15576, 'scaled_da_elements': 8007024},
    {'name': 'large_phase_1_fix_15_scaled_t1727', 'dtype': 'bf16', 'gtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 1727, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 198, 'scale_reference_elements': 7986888, 'source_T': 17301, 'scaled_da_elements': 7985648},
    {'name': 'large_phase_1_fix_16_scaled_t3458', 'dtype': 'fp16', 'gtype': 'fp32', 'B': 1, 'HK': 4, 'HV': 4, 'T': 3458, 'K': 128, 'V': 128, 'chunk_size': 64, 'varlen': False, 'size_class': 'large', 'scale_reference': 'prepare_wy_repr_bwd_full_case_by_index', 'scale_reference_case_id': 199, 'scale_reference_elements': 7995680, 'source_T': 34602, 'scaled_da_elements': 7994896},
)


def _build_profiles():
    return deepcopy(list(_FROZEN_BASE_PROFILES))


GENERAL_BATCHES = (1, 2, 3, 4)
GENERAL_HEAD_PAIRS = ((1, 1), (1, 2), (1, 4), (2, 2), (2, 6), (3, 3), (3, 6), (4, 16))
GENERAL_ELEMENT_CAP = 2_000_000


def _build_generalized_profiles(existing_profiles, extra_lengths=()):
    """定向边界优先，其余候选固定种子打乱；不改变原有 case ID。"""
    shapes = []
    for chunk_size in CHUNK_SIZES:
        for T in (1, 2, chunk_size - 1, chunk_size, chunk_size + 1,
                  2 * chunk_size - 1, 2 * chunk_size, 2 * chunk_size + 1):
            for V in SMALL_V_VALUES:
                shapes.append((1, 1, 1, T, V, chunk_size))
    candidates = [
        (B, HK, HV, T, V, chunk_size)
        for B in GENERAL_BATCHES
        for HK, HV in GENERAL_HEAD_PAIRS
        for chunk_size in CHUNK_SIZES
        for T in (1, 2, 8, 24, chunk_size - 1, chunk_size, chunk_size + 1,
                  2 * chunk_size - 1, 2 * chunk_size, 2 * chunk_size + 1, 512) + tuple(extra_lengths)
        for V in SMALL_V_VALUES
    ]
    random.Random(20260911).shuffle(candidates)
    # 旧 varlen 标签在当前 executor 中不生效，按实际定长输入去重。
    seen = {
        _profile_key({**profile, "varlen": False, "mean_len": None})
        for profile in existing_profiles
    }
    profiles = []
    for B, HK, HV, T, V, chunk_size in dict.fromkeys(shapes + candidates):
        for dtype, gtype in DTYPE_GTYPE_PAIRS:
            profile = {
                "name": (f"general_{dtype}_g{gtype}_b{B}_h{HK}x{HV}"
                         f"_t{T}_k128_v{V}_c{chunk_size}"),
                "dtype": dtype,
                "gtype": gtype,
                "B": B,
                "HK": HK,
                "HV": HV,
                "T": T,
                "K": 128,
                "V": V,
                "chunk_size": chunk_size,
                "varlen": False,
                "size_class": "small",
            }
            key = _profile_key(profile)
            if (_is_filtered_profile(profile)
                    or _shape_elements(profile) > GENERAL_ELEMENT_CAP
                    or key in seen):
                continue
            seen.add(key)
            profiles.append(profile)
    return profiles


BASE_PROFILES = _build_profiles()


def _build_balanced_profiles():
    # 原有 4487 条保持顺序及编号，扩充短长度空间后按 dtype 配额追加。
    profiles = BASE_PROFILES + _build_generalized_profiles(BASE_PROFILES)
    counts = Counter(profile["dtype"] for profile in profiles)
    if any(count > CASES_PER_DTYPE for count in counts.values()):
        raise RuntimeError(f"existing profiles exceed dtype quota: {counts}")
    candidates = _build_generalized_profiles(profiles, extra_lengths=range(3, 64))
    for profile in candidates:
        dtype = profile["dtype"]
        if counts[dtype] < CASES_PER_DTYPE:
            profiles.append(profile)
            counts[dtype] += 1
        if len(profiles) == TOTAL_CASE_COUNT:
            break
    expected = {"bf16": CASES_PER_DTYPE, "fp16": CASES_PER_DTYPE}
    if dict(counts) != expected:
        raise RuntimeError(f"insufficient unique profiles: expected {expected}, got {counts}")
    return profiles


PROFILES = _build_balanced_profiles()
_EXTRA_DTYPES = ("bf16", "fp16")
_PROFILES_BY_DTYPE = {
    dtype: tuple(profile for profile in PROFILES if profile["dtype"] == dtype)
    for dtype in _EXTRA_DTYPES
}

TENSOR_NAMES = ("k", "v", "beta", "A", "g", "dw", "du")
TENSOR_RANGE_VALUES = (
    {"name": "nd", "mean": [-100, 100], "std": [1, 25]},
    [-0.001, 0.001],
    [-5, 5],
)


def _tensor_shape(spec, name):
    B, HK, HV, T, K, V = (spec[key] for key in ("B", "HK", "HV", "T", "K", "V"))
    return {
        "k": [B, HK, T, K],
        "v": [B, HV, T, V],
        "du": [B, HV, T, V],
        "dw": [B, HV, T, K],
        "A": [B, HV, T, spec["chunk_size"]],
        "beta": [B, HV, T],
        "g": [B, HV, T],
    }[name]


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    if index < 0:
        raise IndexError(f"case index must be non-negative, got {index}")
    if index < len(PROFILES):
        profile = deepcopy(PROFILES[index])
    else:
        extra_index = index - len(PROFILES)
        dtype = _EXTRA_DTYPES[extra_index % len(_EXTRA_DTYPES)]
        candidates = _PROFILES_BY_DTYPE[dtype]
        profile_index = (extra_index // len(_EXTRA_DTYPES)) % len(candidates)
        profile = deepcopy(candidates[profile_index])
        profile["name"] = f"{profile['name']}_repeat_{index}"
    scale_metadata = {
        key: profile.pop(key) for key in SCALE_METADATA_KEYS if key in profile
    }
    seed_adjustment = SEED_ADJUSTMENTS.get(index, 0)
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260817 + index + seed_adjustment,
            "route": "ascendc",
            "soc": "ascend910b",
        }
    )
    profile.update(scale_metadata)
    if seed_adjustment:
        profile["accuracy_seed_adjustment"] = seed_adjustment
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_prepare_wy_repr_bwd_da")
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
                elif cfg.name in TENSOR_NAMES:
                    cfg.dtype = ("fp32" if cfg.name == "beta" else
                                 _dtype(spec["gtype"] if cfg.name == "g" else spec["dtype"]))
                    cfg.shape = _tensor_shape(spec, cfg.name)
                    distribution_index = (index + TENSOR_NAMES.index(cfg.name)) % len(TENSOR_RANGE_VALUES)
                    cfg.range_values = deepcopy(TENSOR_RANGE_VALUES[distribution_index])
                    cfg.outlier_values = [0.001, 1000]
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config
