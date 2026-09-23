# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ATK plugin entry for aclnnMlaPrologV3WeightNz.

Golden / rewrite logic lives in function_mla_prolog_v4.py; V3 aclnn prototype ends at
kcScale (no doRope attr). op_version=v3 selects V3_ATTR_SLOTS in VERSION_LAYOUT.
"""

from __future__ import annotations

from atk.tasks.api_execute import register

from function_mla_prolog_v4 import AclnnMlaPrologBaseApi, FunctionMlaPrologBaseApi


@register("function_mla_prolog_v3")
class FunctionMlaPrologV3(FunctionMlaPrologBaseApi):
    """Reference node for ATK default cv_fused_double_benchmark (CPU / NPU small-op)."""

    op_version = "v3"


@register("pyaclnn_function_mla_prolog_v3")
class PyAclnnMlaPrologV3(AclnnMlaPrologBaseApi):
    """DUT: aclnnMlaPrologV3WeightNz via ATK opp path."""

    op_version = "v3"

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize("
            "const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, "
            "const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, "
            "const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos, "
            "aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, "
            "const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional, "
            "const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional, "
            "const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, "
            "const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional, "
            "const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, "
            "char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode, "
            "int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, "
            "double qcQrScale, double kcScale, const aclTensor *queryOut, "
            "const aclTensor *queryRopeOut, "
            "const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional, "
            "const aclTensor *dequantScaleQNormOutOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
