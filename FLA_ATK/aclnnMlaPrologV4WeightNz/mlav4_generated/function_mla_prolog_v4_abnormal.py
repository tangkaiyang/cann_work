"""Independent ATK reference for negative MLA V4 tests.

Load this plugin alongside the original function_mla_prolog_v4.py DUT plugin.
The normal mlav4.json case has T=21, N=1, Hckv=512, Dr=64 and
weightQuantMode=1 / queryQuantMode=0. Only its two output contracts
are reproduced here; zero values are placeholders, not a numerical golden.
"""

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# 异常场景说明（对应 mlav4_invalid.json 的 68 条用例）
# 用 save_name 定位用例，以下名称均省略统一前缀 "mlav4_"。
# 基线：原始 mlav4.json，token_x=[21,1024]，wq=1/kvq=2/qq=0，
# PA_BSND，doRope=false。每条用例从正常用例复制后施加对应变异。
# 源码根目录：D:\t30072652\cann\ops-transformer；框架：D:\t30072652\cann\opbase。
# 依据：mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md 参数表、shape/量化约束，以及
# mla_prolog_tiling_check.cpp 和 aclnn_mla_prolog_v4_weight_nz.cpp。
#
# 一、维度取值异常（18 条；关联 shape 随目标维度调整）
#   he_invalid_1008 / 8208：He 仅允许
#       {1024,2048,3072,4096,5120,6144,7168,7680,8192}。
#   hcq_invalid_1520 / 2064：Hcq 仅允许 {1536,2048}。
#   n_invalid_0 / 129：N 必须在 [1,128]。
#   d_invalid_112 / 208：D 仅允许 {128,192}。
#   h_invalid_496 / 528：Hckv 固定为 512。
#   dr_invalid_48 / 80：Dr 固定为 64。
#   nkv_invalid_2：Nkv 固定为 1。
#   bs_invalid_0 / 15 / 17 / 1040：BlockSize 在 [16,1024] 且为 16 倍数。
#   b_65537：B 不得超过 65536；大 shape 优先通过元数据/tiling 验证。
#
# 二、属性枚举及联动异常（18 条）
#   weightQuantMode_-1 / 6：超出 A5 的 0..5 枚举。
#   kvCacheQuantMode_-1 / 4：超出 0..3 枚举。
#   queryQuantMode_-1 / 2：超出 {0,1}；queryQuantMode_1：kvq=2 时应为 0。
#   ckvkrRepoMode_-1 / 2：超出 {0,1}；ckvkrRepoMode_1：kvq=2 时应为 0。
#   quantScaleRepoMode_-1 / 2：超出 {0,1}；quantScaleRepoMode_1：kvq=2 应为 0。
#   tileSize_0 / 64 / 256：tileSize 仅允许 128。
#   cacheMode_INVALID：不支持的缓存模式。
#   cacheMode_BSND：BSND 要求三维 token_x，当前正常基线为二维。
#
# 三、dtype 异常（13 条，名称为 <输入名>_dtype_fp16）
#   分别将 token_x、weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr、
#   rmsnorm_gamma_cq、rmsnorm_gamma_ckv、kv_cache、kr_cache、cache_index、
#   dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr 改为 fp16。
#   当前场景：token/非量化权重/gamma 为 bf16，weight_uq_qr/cache 为 int8，
#   反量化和缓存量化 scale 为 fp32，索引为整数类型；上述 fp16 均不匹配。
#
# 四、rank/shape 关联异常（11 条，名称为 <输入名>_shape_mismatch）
#   token_x=[21,1,1,1024]：rank=4，接口仅允许 rank=2/3。
#   weight_dq=[1024,1535]：Hcq 不合法且与 gamma/UQ 权重不匹配。
#   weight_uq_qr=[1535,256]：第一轴不匹配 Hcq=1536。
#   weight_uk=[1,191,512]：D 不合法且与 weight_uq_qr 的 N*(D+Dr) 不匹配。
#   weight_dkv_kr=[1024,575]：第二轴应为 Hckv+Dr=576。
#   rmsnorm_gamma_cq=[1535]：应为 [1536]。
#   rmsnorm_gamma_ckv=[511]：应为 [512]。
#   cache_index=[20]：应匹配 T=21。
#   dequant_scale_w_uq_qr=[1,255]：应为 [1,256]。
#   quant_scale_ckv=[1,511]：应为 [1,512]。
#   quant_scale_ckr=[1,63]：应为 [1,64]。
#
# 五、必要量化输入缺失（3 条，名称为 <输入名>_missing）
#   dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr 分别变为
#   bool/shape=[]/required=false 空指针哨兵；当前 wq=1/kvq=2 必须提供。
#
# 六、缓存索引数据越界（2 条）
#   cache_index_oob_-1 / 32：T=1、缓存容量=1*32，有效索引为 [0,32)。
#   不保证 host 参数检查读取索引内容；设备执行可能越界，须隔离验证。
#
# 七、RoPE 开关与输入空值异常（3 条）
#   rope_true_missing_both：doRope=true，却缺失 sin 和 cos。
#   rope_False_sin_only：doRope=false，却提供非空 sin。
#   rope_True_sin_only：doRope=true，仅提供 sin，缺失 cos。
#
# 执行约定：本标杆只提供正常用例的输出 shape/dtype 和零值占位，不判断异常。
# 不从非法输入推导输出，也不计算普通 golden；异常是否被正确拒绝由 DUT 验证。
# 具体状态码/报错阶段尚未设备实测，不能把造数或适配器报错算作 DUT 正确拒绝。
# 原 DUT 适配器可能修正负量化模式、RoPE 或 cacheIndex，应检查实际传入参数。
# 特殊值文件中的 scalar/超大轴长不在此 68 条中，也未切换到本注册名。
#
# expected_error_msg 填写依据：
#   weightQuantMode_-1/6、kvCacheQuantMode_-1/4、queryQuantMode_-1/2，
#   以及三条 RoPE 异常，公开 API 直接返回 ge::GRAPH_FAILED=0xFFFFFFFF，
#   JSON expected_error_msg 写为字符串 "4294967295"，共 9 条。
#   这是公开函数返回值，不是 OP_LOGE 中的 ACLNN_ERR_PARAM_INVALID=161002。
#   若 Python 包装按 int32 显示，会呈现 -1；本机无 ATK，未确认其消息格式。
#   13 条 fp16 异常：预期 161002，算子定义与 opbase dtype/format 支持表校验。
#   44 条属性/维度/缺失输入异常：预期 561002，tiling 返回失败，由
#   opbase/indv_executor.cpp:885 转为 ACLNN_ERR_INNER_TILING_ERROR。
#   后两组是源码路径推导的预期码，未经过 ATK/设备验证；适配器或生成接口
#   先行失败时实际码可能不同。仅两条索引值越界保持 null，未找到固定 host 错误。
#   每条的代码位置、候选码和不确定原因记录在 manifest.error_evidence。


@register('function_mla_prolog_v4_abnormal')
class FunctionMlaPrologV4Abnormal(BaseApi):
    """Never infer output sizes, rewrite inputs, or calculate from invalid data."""

    op_version = 'v4'
    OUTPUT_SHAPES = ((21, 1, 512), (21, 1, 64))

    def init_by_input_data(self, input_data):
        # Do not call the normal executor's sanitizers or shape inference.
        pass

    def __call__(self, input_data, with_output=False):
        # Fresh tensors on each invocation prevent mutation leaking across cases.
        device = getattr(self, 'device', 'cpu')
        return tuple(torch.zeros(shape, dtype=torch.bfloat16, device=device)
                     for shape in self.OUTPUT_SHAPES)
