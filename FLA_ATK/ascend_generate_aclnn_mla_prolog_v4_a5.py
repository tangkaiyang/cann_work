# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ATK CaseGenerator for aclnnMlaPrologV3WeightNz on A5 (ascend950).

A5 supports all quant scenes; this generator emits modes covered by
function_mla_prolog_v3.py golden:
  wq=0 kvq=0 | wq=1 kvq=0 | wq=1 kvq=2 | wq=2 kvq=0 | wq=3 kvq=1 qq=1
mxfp8 scales use uint8 placeholders; the plugin rewrites them to e8m0.
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
# HE_CHOICES = [1024, 2048, 4096, 7168]
HE_CHOICES = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192]
# mxfp8 cases in validated JSON fix He=7168 (model path)
HE_MXFP8_CHOICES = [7168]
HCQ_CHOICES = [1536, 2048]
D_CHOICES = [128, 192]
T_CHOICES = [1, 2, 4, 8, 16]
BLOCK_SIZE_CHOICES = [16, 32, 64, 128, 256]
BLOCK_NUM = 2

# (weightQuantMode, kvCacheQuantMode, queryQuantMode)
A5_SCENES: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 2, 0),
    (2, 0, 0),
    (3, 1, 1),
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


@GENERATOR_REGISTRY.register("ascend_generate_aclnn_mla_prolog_v4_a5")
class MlaPrologV3A5Generator(CaseGenerator):
    """Shape / dtype binding for A5 precision cases (incl. mxfp8)."""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        wq, kvq, qq = random.choice(A5_SCENES)
        t = random.choice(T_CHOICES)
        he = random.choice(HE_MXFP8_CHOICES if wq == 3 else HE_CHOICES)
        hcq = random.choice(HCQ_CHOICES)
        d = random.choice(D_CHOICES)
        n = random.randint(1, 128)  # Doc: N 取值范围 1-128
        block_size = random.choice(BLOCK_SIZE_CHOICES)
        block_num = max(BLOCK_NUM, math.ceil(t / block_size))
        slots = block_num * block_size
        qk_dim = n * (d + DR)
        do_rope = random.random() < 0.6

        case_config.inputs[24].range_values = wq
        case_config.inputs[25].range_values = kvq
        case_config.inputs[26].range_values = qq
        case_config.inputs[23].range_values = "PA_BSND"
        case_config.inputs[27].range_values = 0
        case_config.inputs[28].range_values = 0
        case_config.inputs[29].range_values = 128
        case_config.inputs[21].range_values = 1e-5
        case_config.inputs[22].range_values = 1e-5
        case_config.inputs[30].range_values = 1.0
        case_config.inputs[31].range_values = 1.0
        case_config.inputs[32].range_values = bool(do_rope)

        if wq == 0:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "bf16"
        elif wq == 1:
            token_dtype = w_dq_dtype = w_dkv_dtype = "bf16"
            w_uq_dtype = "int8"
        elif wq == 2:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "int8"
        else:  # wq == 3 mxfp8
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "fp8e4m3"

        if kvq == 2:
            kv_dtype = kr_dtype = "int8"
        elif kvq == 1:
            kv_dtype = "fp8e4m3"
            kr_dtype = "bf16"
        else:
            kv_dtype = kr_dtype = "bf16"

        case_config.inputs[0].dtype = token_dtype
        case_config.inputs[0].shape = [t, he]
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

        # ropeSin/ropeCos 与 doRope 联动：doRope=true 时必选有效输入；doRope=false 时必须为空
        if do_rope:
            case_config.inputs[7].dtype = "bf16"
            case_config.inputs[7].shape = [t, DR]
            case_config.inputs[8].dtype = "bf16"
            case_config.inputs[8].shape = [t, DR]
        else:
            _null_optional(case_config.inputs[7])
            _null_optional(case_config.inputs[8])

        case_config.inputs[9].dtype = kv_dtype
        case_config.inputs[9].shape = [block_num, block_size, NKV, HCKV]
        case_config.inputs[9].range_values = 0
        case_config.inputs[10].dtype = kr_dtype
        case_config.inputs[10].shape = [block_num, block_size, NKV, DR]
        case_config.inputs[10].range_values = 0
        case_config.inputs[11].dtype = "int64"
        case_config.inputs[11].shape = [t]
        case_config.inputs[11].range_values = [0, slots - 1]

        for idx in (12, 13, 14, 15, 16, 17, 18, 19, 20):
            _null_optional(case_config.inputs[idx])

        if wq == 2:
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

        return case_config
