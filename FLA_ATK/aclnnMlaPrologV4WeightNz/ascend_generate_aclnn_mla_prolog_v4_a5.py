# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ATK CaseGenerator for aclnnMlaPrologV4WeightNz on A5 (ascend950).

Document-driven A5 cases: all 16 legal quantization scenes and six cache modes.
Default MLA_V4_PROFILE=golden restricts weight modes to 0..3; full enables 4/5
and requires golden/plugin support for fp8/hif8. MX scales retain the existing
uint8 placeholder convention (plugin must rewrite values and dtype to e8m0).
Weight shapes are logical ND shapes; the ACLNN adapter must convert to NZ.
"""

from __future__ import annotations

import os
import random
from math import prod
from typing import List, Tuple

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

HCKV = 512
DR = 64
NKV = 1
MXFP8_GRP = 32
HE_CHOICES = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192)
HCQ_CHOICES = [1536, 2048]
D_CHOICES = [128, 192]
BLOCK_SIZE_CHOICES = [16, 32, 48, 64, 128, 256, 512, 1008, 1024]
CACHE_MODES = ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"]
EPS_CHOICES = [1e-6, 1e-5, 1e-4]
SCALE_CHOICES = [0.5, 1.0, 2.0]
# Only combinations explicitly listed in the interface document.
A5_SCENES: List[Tuple[int, int, int]] = [
    (0, 0, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0),
] + [(wq, kvq, int(kvq == 1)) for wq in (2, 3, 4, 5) for kvq in (0, 1, 3)]


def _scenes():
    # Existing golden advertises wq=0..3 only. Enable 4/5 after adapting golden.
    profile = os.environ.get("MLA_V4_PROFILE", "golden")
    if profile not in ("golden", "full"):
        raise ValueError("MLA_V4_PROFILE must be golden or full")
    return [scene for scene in A5_SCENES if profile == "full" or scene[0] <= 3]


def _null_optional(inp) -> None:
    inp.dtype = "bool"
    inp.shape = []
    inp.required = False
    inp.range_values = 0


def _enable_optional(inp, dtype: str, shape: list, range_values=None) -> None:
    inp.dtype = dtype
    inp.shape = list(shape)
    inp.required = False
    if range_values is not None:
        inp.range_values = range_values


@GENERATOR_REGISTRY.register("ascend_generate_aclnn_mla_prolog_v4_a5")
class MlaPrologV4A5Generator(CaseGenerator):
    """Shape / dtype binding for A5 precision cases (incl. mxfp8)."""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        wq, kvq, qq = random.choice(_scenes())
        shape = [max(0, int(dim)) for dim in case_config.inputs[0].shape]
        # Preserve valid shapes; repair sampled dimensions before binding inputs.
        if not shape:
            shape = [1, HE_CHOICES[0]]
        elif len(shape) == 1:
            shape = [1, shape[0]]
        elif len(shape) > 3:
            shape = [prod(shape[:-1]), shape[-1]]
        shape[-1] = min(HE_CHOICES, key=lambda he: abs(he - shape[-1]))
        if len(shape) == 3:
            shape[0] = min(shape[0], 65536)
        case_config.inputs[0].shape = shape
        merged = len(shape) == 2
        if merged:
            t, he = shape
            batch, seq = 1, t
        else:
            batch, seq, he = shape
            t = batch * seq
        leading = shape[:-1]
        cache_modes = CACHE_MODES if kvq != 3 else ["PA_BSND", "BSND", "TND"]
        # Packed block layouts require actualSeqLen=[T] in INT32.
        if merged and t > 2**31 - 1:
            cache_modes = [mode for mode in cache_modes if not mode.startswith("PA_BLK")]
        cache_mode = random.choice([mode for mode in cache_modes
                                    if mode != ("BSND" if merged else "TND")])
        hcq = random.choice(HCQ_CHOICES)
        d = random.choice(D_CHOICES)
        n = random.choice([1, 2, 3, 7, 8, 16, 32, 64, 127, 128, random.randint(1, 128)])
        block_size = random.choice(BLOCK_SIZE_CHOICES)
        blocks_per_seq = (seq + block_size - 1) // block_size
        blocks_used = batch * blocks_per_seq
        block_num = max(1, blocks_used) + random.choice([0, 1, 3])
        slots = block_num * block_size
        qk_dim = n * (d + DR)
        do_rope = random.choice([True, False])

        attrs = {21: random.choice(EPS_CHOICES), 22: random.choice(EPS_CHOICES),
                 23: cache_mode, 24: wq, 25: kvq, 26: qq,
                 27: int(kvq == 3), 28: int(kvq == 3), 29: 128,
                 30: random.choice(SCALE_CHOICES), 31: random.choice(SCALE_CHOICES),
                 32: do_rope}
        for idx, value in attrs.items():
            case_config.inputs[idx].range_values = value
        # Restore mandatory flags when a CaseConfig instance is reused.
        for idx in range(12):
            case_config.inputs[idx].required = True

        if wq == 0:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "bf16"
        elif wq == 1:
            token_dtype = w_dq_dtype = w_dkv_dtype = "bf16"
            w_uq_dtype = "int8"
        elif wq == 2:
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = "int8"
        else:  # mxfp8 / fp8 / hif8
            token_dtype = w_dq_dtype = w_uq_dtype = w_dkv_dtype = ("hif8" if wq == 5 else "fp8e4m3")

        if kvq == 2:
            kv_dtype = kr_dtype = "int8"
        elif kvq in (1, 3):
            kv_dtype = {1: "int8", 2: "int8", 3: "fp8e4m3", 4: "fp8e4m3", 5: "hif8"}[wq]
            kr_dtype = "bf16"
        else:
            kv_dtype = kr_dtype = "bf16"

        case_config.inputs[0].dtype = token_dtype
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
            case_config.inputs[7].range_values = [-1, 1]
            case_config.inputs[8].range_values = [-1, 1]
            case_config.inputs[7].dtype = "bf16"
            case_config.inputs[7].shape = leading + [DR]
            case_config.inputs[8].dtype = "bf16"
            case_config.inputs[8].shape = leading + [DR]
        else:
            _null_optional(case_config.inputs[7])
            _null_optional(case_config.inputs[8])

        cache_leading = ([t] if cache_mode == "TND" else
                         [batch, seq] if cache_mode == "BSND" else [block_num, block_size])
        case_config.inputs[9].dtype = kv_dtype
        case_config.inputs[9].shape = cache_leading + [NKV, 656 if kvq == 3 else HCKV]
        case_config.inputs[9].range_values = 0
        case_config.inputs[10].dtype = kr_dtype
        # Empty typed Tensor, not the bool/[] sentinel used for nullptr.
        case_config.inputs[10].shape = [0] if kvq == 3 else cache_leading + [NKV, DR]
        case_config.inputs[10].range_values = 0

        for idx in range(11, 21):
            _null_optional(case_config.inputs[idx])
        if cache_mode.startswith("PA_BLK"):
            index_shape = [blocks_used] if merged else [batch, blocks_per_seq]
            _enable_optional(case_config.inputs[11], "int64", index_shape, [0, block_num - 1])
            if merged:
                _enable_optional(case_config.inputs[19], "int32", [1], t)
        elif cache_mode.startswith("PA_"):
            _enable_optional(case_config.inputs[11], "int64", leading, [0, slots - 1])

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

        if wq in (1, 2, 4, 5) and random.choice([True, False]):
            _enable_optional(case_config.inputs[18], "fp32", [1, hcq], [0.5, 1.5])
        if kvq == 3 and wq in (1, 2):
            _enable_optional(case_config.inputs[20], "fp32", [1], [0.5, 2.0])
        return case_config
