# aclnn 文档与样例复扫报告

仅指出问题；未修改四个仓库的文档、实现或样例。所有结论针对下表本地版本。文档及源码链接指向 GitCode 对应提交，保留行号，避免后续更新导致定位漂移。

| 仓库 | 提交 | aclnn Markdown 文件数 |
|---|---|---:|
| ops-math | 052d8f3d | 417 |
| ops-nn | 6bf411a1 | 537 |
| ops-transformer | f2e0720e | 288 |
| ops-cv | ed2be3c | 53 |

## 范围与结论

- 合计 **92 个定位项，涉及 87 份文档**。按“文档 + 问题类别”归并，同一项可能列出多个出错位置；本轮新增 39 项。
- 文件级扫描覆盖 1,295 份 aclnn*.md；样例解析覆盖 1,278 份来源文档中的 1,468 个代码单元（含文档链接到的 C++ 样例，不等同于可独立编译程序数量）。
- 建立 1,056 份文档、1,215 处 API 到实现的映射，用于筛选正文类型、空 Tensor 等约束差异。映射成功不代表已逐行人工核验全部功能。
- 对下列确认项复核了相关正文、辅助函数、头文件或实现分支；候选扫描告警不计入问题总数。
- 本次是静态审查，未编译全部样例、未在昇腾设备上运行；数学公式、产品支持矩阵、动态 shape 和各硬件分支没有全部逐项验证，未列出问题不代表该文档已通过完整审查。

## 问题分类

| 类别 | 定位项数 |
|---|---:|
| 样例无法按当前头文件编译 | 2 |
| 样例主机缓冲区越界读取 | 14 |
| 样例调用错误算子 | 1 |
| 接口版本名错误 | 1 |
| 函数原型与头文件不一致 | 27 |
| 影响展示或复制的格式问题 | 9 |
| 正文与实现不一致 | 6 |
| 样例声明或宏定义错误 | 4 |
| 样例设备缓冲区大小与 Tensor 类型不符 | 11 |
| 样例输入按错误类型解释 | 1 |
| 样例正常路径遗漏资源释放 | 16 |

## 误报排除与前次漏检说明

- aclnnMul 前次漏检是检查规则没有覆盖变量声明与使用关系。仅检查参数数量、shape/初始化和文本模式，不能证明样例可编译。本轮补充作用域分析，并人工核对完整 main。
- 独立 examples 文件正确不代表 Markdown 内嵌样例正确；两者分别检查。
- 排除合法的文件末尾隐式代码围栏结束、故意省略上下文的代码片段、宏/命名空间符号误报。
- 排除辅助函数作用域混淆产生的类型误报；输出初始化使用更大的临时缓冲区，不直接判定为缓冲区不足。
- TensorList/ScalarList 销毁时会销毁成员，未单独 destroy 成员不直接计为泄漏；参见 [源码依据](https://gitcode.com/cann/opbase/blob/2cd7c3f6a0aa723a705a89d4736766132f85170e/src/nnopbase/common/api/acl_op_api.cpp#L157)。
- 量化接口存在先调整数据类型再校验的分支，不能直接用类型白名单全文差集判错；未确认的量化类型告警未收入清单。

## 正文与实现不一致（6 项）

### 1. aclnnIsNegInf.md · 本轮新增

位置：[ops-math/math/is_neg_inf/docs/aclnnIsNegInf.md:81](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/is_neg_inf/docs/aclnnIsNegInf.md#L81)

self 的类型表列出 DOUBLE，但实现的受支持类型列表均不包含 DOUBLE，且在计算前执行类型检查。按文档传 DOUBLE 会被拒绝。

依据：[源码依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/is_neg_inf/op_api/aclnn_isneginf.cpp#L53)

### 2. aclnnIsPosInf.md · 本轮新增

位置：[ops-math/math/is_pos_inf/docs/aclnnIsPosInf.md:81](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/is_pos_inf/docs/aclnnIsPosInf.md#L81)

self 的类型表列出 DOUBLE，但实现的受支持类型列表均不包含 DOUBLE，且在计算前执行类型检查。按文档传 DOUBLE 会被拒绝。

依据：[源码依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/is_pos_inf/op_api/aclnn_isposinf.cpp#L51)

### 3. aclnnIndexFillTensor&aclnnInplaceIndexFillTensor.md · 本轮新增

位置：[ops-nn/experimental/index/index_fill_d/docs/aclnnIndexFillTensor&aclnnInplaceIndexFillTensor.md:114](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/index/index_fill_d/docs/aclnnIndexFillTensor%26aclnnInplaceIndexFillTensor.md#L114)

文档列出的支持产品是 Atlas A2，self/selfRef 类型表（114、298 行）却包含 INT64；DAV_2201 路径的类型检查只允许 INT32、FLOAT16、FLOAT、BOOL、BF16，INT64 会返回参数错误。

依据：[源码依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/index/index_fill_d/op_host/op_api/aclnn_index_fill_tensor.cpp#L91)

### 4. aclnnPdist.md · 本轮新增

位置：[ops-math/experimental/math/pdist/docs/aclnnPdist.md:74](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/pdist/docs/aclnnPdist.md#L74)

错误码表称 N<2 会报错，参数表第 58 行限制 N≥2、M≥1。实现实际允许 N≤1 并返回空结果成功；M=0 时填充零。例如 self=[1,3]、out=[0] 可走成功分支，self=[2,0]、out=[1] 走填零分支。这里的方括号表示 shape。

依据：[源码依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/pdist/op_api/aclnn_pdist.cpp#L101)

### 5. aclnnSoftsign.md · 本轮新增

位置：[ops-nn/experimental/activation/softsign/docs/aclnnSoftsign.md:90](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign/docs/aclnnSoftsign.md#L90)

输入说明支持空 Tensor，输出却写“不支持空Tensor”并要求与输入 shape 相同，文档内部互相矛盾。实现对空输入有明确成功返回分支，没有按该输出限制拒绝；输入和输出均取 shape=[0] 时可进入此分支。

依据：[源码依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign/op_api/aclnn_softsign.cpp#L122)

### 6. aclnnSoftsignBackward.md · 本轮新增

位置：[ops-nn/experimental/activation/softsign_grad/docs/aclnnSoftsignBackward.md:101](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign_grad/docs/aclnnSoftsignBackward.md#L101)

输入说明支持空 Tensor，输出却写“不支持空Tensor”并要求与输入 shape 相同，文档内部互相矛盾。实现对空输入有明确成功返回分支，没有按该输出限制拒绝；输入和输出均取 shape=[0] 时可进入此分支。

依据：[源码依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign_grad/op_api/aclnn_softsign_backward.cpp#L138)

## 样例声明或宏定义错误（4 项）

### 7. aclnnMul&aclnnInplaceMul.md · 本轮新增

位置：[ops-math/math/mul/docs/aclnnMul&aclnnInplaceMul.md:503](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mul/docs/aclnnMul%26aclnnInplaceMul.md#L503)

aclnnMul 样例使用未声明的 out（503 行）、outShape（520 行）、outDeviceAddr（522 行），并且遗漏输出 Tensor/设备内存的创建，按文档复制无法编译。独立 examples/test_aclnn_mul.cpp 有相应声明和创建，不能据此认定文档样例正确。

依据：[源码依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mul/examples/test_aclnn_mul.cpp#L86)

### 8. aclnnAddcmul&aclnnInplaceAddcmul.md · 本轮新增

位置：[ops-math/math/addcmul/docs/aclnnAddcmul&aclnnInplaceAddcmul.md:760](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/addcmul/docs/aclnnAddcmul%26aclnnInplaceAddcmul.md#L760)

Inplace 样例声明的是 selfRef（749 行），创建 Tensor 时却传 &self（760 行），第一阶段接口又传 self（776 行）；该作用域没有声明 self，无法编译。

### 9. aclnnRecurrentGatedDeltaRule.md · 本轮新增

位置：[ops-transformer/attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md:516](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md#L516)

gkDeviceAddr 没有声明，却用于创建 gk Tensor（516 行）及释放（576 行），完整 main 样例无法编译。

### 10. aclnnReluGradV3.md · 本轮新增

位置：[ops-nn/experimental/activation/relu_grad_v3/docs/aclnnReluGradV3.md:205](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/relu_grad_v3/docs/aclnnReluGradV3.md#L205)

CHECK_RET 和 LOG_PRINT 的多行宏缺少行末续行符。预处理器只会定义空宏，后续 do/while、cond、return_expr 以及 ##__VA_ARGS__ 落到宏外，样例无法编译。

## 样例无法按当前头文件编译（2 项）

### 11. aclnnBlitzSparseAttention.md · 前轮保留

位置：[ops-transformer/experimental/attention/blitz_sparse_attention/docs/aclnnBlitzSparseAttention.md:889](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/experimental/attention/blitz_sparse_attention/docs/aclnnBlitzSparseAttention.md#L889)

aclnnBlitzSparseAttentionGetWorkspaceSize 样例传 23 个实参，声明要求 27 个。query/key/value 后的可选输入槽位少一个，且没有传 softmaxLseFlag、blockShape、softmaxLse。

依据：[头文件声明](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/experimental/attention/blitz_sparse_attention/op_host/op_api/aclnn_blitz_sparse_attention.h#L24)

### 12. aclnnMhcPreSinkhornPremix.md · 前轮保留

位置：[ops-transformer/experimental/mhc/mhc_pre_sinkhorn_premix/docs/aclnnMhcPreSinkhornPremix.md:768](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/experimental/mhc/mhc_pre_sinkhorn_premix/docs/aclnnMhcPreSinkhornPremix.md#L768)

aclnnMhcPreSinkhornPremixGetWorkspaceSize 样例传 19 个实参，声明要求 20 个。遗漏第 5 个参数 premix；即使不使用可选输入，也必须显式传 nullptr。后续参数整体错位。

依据：[头文件声明](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/experimental/mhc/mhc_pre_sinkhorn_premix/op_host/op_api/aclnn_mhc_pre_sinkhorn_premix.h#L36)

## 样例主机缓冲区越界读取（14 项）

### 13. aclnnIm2col.md · 前轮保留

位置：[ops-math/conversion/im2col/docs/aclnnIm2col.md:362](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/conversion/im2col/docs/aclnnIm2col.md#L362)

[362](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/conversion/im2col/docs/aclnnIm2col.md#L362)：`outHostData` 只有 1 个元素，但 `outShape` 要求 32 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 14. aclnnIsInTensorScalar.md · 前轮保留

位置：[ops-math/experimental/math/equal/docs/aclnnIsInTensorScalar.md:310](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/equal/docs/aclnnIsInTensorScalar.md#L310)

[310](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/equal/docs/aclnnIsInTensorScalar.md#L310)：`outHostData` 只有 2 个元素，但 `outShape` 要求 5 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 15. aclnnIsInTensorScalar.md · 前轮保留

位置：[ops-math/math/equal/docs/aclnnIsInTensorScalar.md:332](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/equal/docs/aclnnIsInTensorScalar.md#L332)

[332](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/equal/docs/aclnnIsInTensorScalar.md#L332)：`outHostData` 只有 2 个元素，但 `outShape` 要求 5 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 16. aclnnFmodScalar&aclnnInplaceFmodScalar.md · 前轮保留

位置：[ops-math/math/mod/docs/aclnnFmodScalar&aclnnInplaceFmodScalar.md:495](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mod/docs/aclnnFmodScalar%26aclnnInplaceFmodScalar.md#L495)

[495](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mod/docs/aclnnFmodScalar%26aclnnInplaceFmodScalar.md#L495)：`selfHostData` 只有 4 个元素，但 `selfShape` 要求 8 个；[501](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mod/docs/aclnnFmodScalar%26aclnnInplaceFmodScalar.md#L501)：`outHostData` 只有 4 个元素，但 `outShape` 要求 8 个；[634](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/mod/docs/aclnnFmodScalar%26aclnnInplaceFmodScalar.md#L634)：`selfHostData` 只有 4 个元素，但 `selfShape` 要求 8 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 17. aclnnSwiGluGrad.md · 前轮保留

位置：[ops-nn/activation/swi_glu_grad/docs/aclnnSwiGluGrad.md:334](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/activation/swi_glu_grad/docs/aclnnSwiGluGrad.md#L334)

[334](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/activation/swi_glu_grad/docs/aclnnSwiGluGrad.md#L334)：`outHostData` 只有 32 个元素，但 `outShape` 要求 64 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 18. aclnnBatchNormGatherStatsWithCounts.md · 前轮保留

位置：[ops-nn/norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md:457](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md#L457)

[457](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md#L457)：`meanHostData` 只有 8 个元素，但 `meanShape` 要求 16 个；[472](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md#L472)：`meanAllHostData` 只有 2 个元素，但 `meanAllShape` 要求 4 个；[475](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md#L475)：`invstdAllHostData` 只有 2 个元素，但 `invstdAllShape` 要求 4 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 19. aclnnFusedAdam.md · 前轮保留

位置：[ops-nn/optim/fused_adam/docs/aclnnFusedAdam.md:545](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adam/docs/aclnnFusedAdam.md#L545)

[545](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adam/docs/aclnnFusedAdam.md#L545)：`stepsHostData1` 只有 1 个元素，但 `inputShape1` 要求 8 个；[559](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adam/docs/aclnnFusedAdam.md#L559)：`stepsHostData2` 只有 1 个元素，但 `inputShape2` 要求 4 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 20. aclnnFusedAdamw.md · 前轮保留

位置：[ops-nn/optim/fused_adamw/docs/aclnnFusedAdamw.md:521](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adamw/docs/aclnnFusedAdamw.md#L521)

[521](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adamw/docs/aclnnFusedAdamw.md#L521)：`stepsHostData1` 只有 1 个元素，但 `inputShape1` 要求 8 个；[534](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/optim/fused_adamw/docs/aclnnFusedAdamw.md#L534)：`stepsHostData2` 只有 1 个元素，但 `inputShape2` 要求 4 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 21. aclnnSwigluMxQuant.md · 前轮保留

位置：[ops-nn/quant/swiglu_mx_quant/docs/aclnnSwigluMxQuant.md:626](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/quant/swiglu_mx_quant/docs/aclnnSwigluMxQuant.md#L626)

[626](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/quant/swiglu_mx_quant/docs/aclnnSwigluMxQuant.md#L626)：`outHostData` 只有 32 个元素，但 `outShape` 要求 64 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 22. aclnnMultiScaleDeformableAttnFunction.md · 前轮保留

位置：[ops-nn/vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md:455](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md#L455)

[455](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md#L455)：`valueHostData` 只有 2 个元素，但 `valueShape` 要求 64 个；[467](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md#L467)：`attnWeightHostData` 只有 2 个元素，但 `attnWeightShape` 要求 256 个；[470](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md#L470)：`outputHostData` 只有 2 个元素，但 `outputShape` 要求 2048 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 23. aclnnScatterPaCache.md · 前轮保留

位置：[ops-transformer/attention/scatter_pa_cache/docs/aclnnScatterPaCache.md:427](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/scatter_pa_cache/docs/aclnnScatterPaCache.md#L427)

[427](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/scatter_pa_cache/docs/aclnnScatterPaCache.md#L427)：`hostKey` 只有 1 个元素，但 `keyShape` 要求 84 个；[433](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/scatter_pa_cache/docs/aclnnScatterPaCache.md#L433)：`hostKeyCacheRef` 只有 1 个元素，但 `keyCacheShape` 要求 28 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 24. aclnnFFN.md · 前轮保留

位置：[ops-transformer/ffn/ffn/docs/aclnnFFN.md:609](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFN.md#L609)

[609](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFN.md#L609)：`outHostData` 只有 4 个元素，但 `outShape` 要求 8 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 25. aclnnFFNV2.md · 前轮保留

位置：[ops-transformer/ffn/ffn/docs/aclnnFFNV2.md:575](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFNV2.md#L575)

[575](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFNV2.md#L575)：`outHostData` 只有 4 个元素，但 `outShape` 要求 8 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

### 26. aclnnFFNV3.md · 前轮保留

位置：[ops-transformer/ffn/ffn/docs/aclnnFFNV3.md:581](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFNV3.md#L581)

[581](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/ffn/ffn/docs/aclnnFFNV3.md#L581)：`outHostData` 只有 4 个元素，但 `outShape` 要求 8 个。样例辅助函数按 shape 元素数拷贝 hostData.data()，会读过 vector 有效范围；即使是输出初始化也有此问题。

## 样例设备缓冲区大小与 Tensor 类型不符（11 项）

### 27. aclnnIndexFill&aclnnInplaceIndexFill.md · 本轮新增

位置：[ops-nn/experimental/index/index_fill/docs/aclnnIndexFill&aclnnInplaceIndexFill.md:523](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/index/index_fill/docs/aclnnIndexFill%26aclnnInplaceIndexFill.md#L523)

523、665 行：indexHostData 为 vector<int>，Tensor 却声明 ACL_INT64，单元素申请 4 字节、需要 8 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 28. aclnnIndexFill&aclnnInplaceIndexFill.md · 本轮新增

位置：[ops-nn/experimental/index/index_fill_d/docs/aclnnIndexFill&aclnnInplaceIndexFill.md:522](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/index/index_fill_d/docs/aclnnIndexFill%26aclnnInplaceIndexFill.md#L522)

522、664 行：indexHostData 为 vector<int>，Tensor 却声明 ACL_INT64，单元素申请 4 字节、需要 8 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 29. aclnnQuantMatmulV3.md · 本轮新增

位置：[ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV3.md:811](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV3.md#L811)

811、1072、1360 行：float scaleHostData 被声明为 ACL_UINT64，每元素申请 4 字节、需要 8 字节；1055 行 int8_t x2HostData 被声明为 ACL_INT32，每元素申请 1 字节、需要 4 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 30. aclnnQuantMatmulV4.md · 本轮新增

位置：[ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md:891](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md#L891)

891、1193、1484 行：float scaleHostData 被声明为 ACL_UINT64（4/8 字节）；1467 行 int8_t x2HostData 被声明为 ACL_INT32（1/4 字节）。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 31. aclnnQuantMatmulWeightNz.md · 本轮新增

位置：[ops-nn/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md:1111](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md#L1111)

1111、1405 行：float scaleHostData 被声明为 ACL_UINT64，每元素申请 4 字节、需要 8 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 32. aclnnQuantMatmulDequant.md · 本轮新增

位置：[ops-nn/matmul/quant_matmul_dequant/docs/aclnnQuantMatmulDequant.md:469](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/matmul/quant_matmul_dequant/docs/aclnnQuantMatmulDequant.md#L469)

469、471 行：uint16_t 的 weightScaleHostData、xScaleHostData 被声明为 ACL_FLOAT，每元素申请 2 字节、需要 4 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 33. aclnnAddRmsNormDynamicQuant.md · 本轮新增

位置：[ops-nn/norm/add_rms_norm_dynamic_quant/docs/aclnnAddRmsNormDynamicQuant.md:551](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/norm/add_rms_norm_dynamic_quant/docs/aclnnAddRmsNormDynamicQuant.md#L551)

551、554 行：short 的 scale1HostData、scale2HostData 被声明为 ACL_FLOAT，每元素申请 2 字节、需要 4 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 34. aclnnRingAttentionUpdate.md · 本轮新增

位置：[ops-transformer/attention/ring_attention_update/docs/aclnnRingAttentionUpdate.md:509](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/ring_attention_update/docs/aclnnRingAttentionUpdate.md#L509)

float actualSeqQlenOptionalHostData 被声明为 ACL_INT64，每元素申请 4 字节、需要 8 字节，浮点数的原始位模式也不是整数序列长度。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 35. aclnnRingAttentionUpdateV2.md · 本轮新增

位置：[ops-transformer/attention/ring_attention_update/docs/aclnnRingAttentionUpdateV2.md:523](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/ring_attention_update/docs/aclnnRingAttentionUpdateV2.md#L523)

float actualSeqQlenOptionalHostData 被声明为 ACL_INT64，每元素申请 4 字节、需要 8 字节，浮点数的原始位模式也不是整数序列长度。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 36. aclnnGroupedMatmulWeightNz.md · 本轮新增

位置：[ops-transformer/gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md:1756](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md#L1756)

1756 行 int8_t scaleHostData 被声明为 ACL_BF16（1/2 字节）；1761 行 int8_t pertokenHostData 被声明为 ACL_FLOAT（1/4 字节）。CreateAclTensorList 内部仍调用按 sizeof(T) 分配的 CreateAclTensor。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

### 37. aclnnAttentionToFFN.md · 本轮新增

位置：[ops-transformer/mc2/attention_to_ffn/docs/aclnnAttentionToFFN.md:581](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/mc2/attention_to_ffn/docs/aclnnAttentionToFFN.md#L581)

581、583、585 行：int16_t 的 sessionIdHostData、microBatchIdHostData、layerIdHostData 被声明为 ACL_INT32，每元素申请 2 字节、需要 4 字节。 本页辅助函数按 shape 元素数 × sizeof(T) 分配并原样复制，没有执行数值转换；Tensor 所需字节数大于申请量，存在越过所申请缓冲区访问的风险。

## 样例输入按错误类型解释（1 项）

### 38. aclnnMaskedScale.md · 本轮新增

位置：[ops-math/math/masked_scale/docs/aclnnMaskedScale.md:320](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/masked_scale/docs/aclnnMaskedScale.md#L320)

maskHostData 是 vector<float>{7,6,5,4,3,2,1,0}，辅助函数直接复制浮点原始字节，再把 Tensor 标为 ACL_UINT8；这不会把各个 float 转成整数 7、6、…、0。算子读到的是浮点编码中的字节，实际计算输入与样例构造意图不一致。辅助函数见 273–290 行。

## 样例调用错误算子（1 项）

### 39. aclnnUpsampleBilinear2dAA.md · 前轮保留

位置：[ops-cv/image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md:433](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md#L433)

AA 文档样例包含 aclnn_upsample_bilinear_2d.h，并调用非 AA 的 aclnnUpsampleBilinear2dGetWorkspaceSize / aclnnUpsampleBilinear2d（第 444 行）。运行的不是文档描述的抗混叠算子。

依据：[本页 AA 原型](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md#L80)

## 接口版本名错误（1 项）

### 40. aclnnSparseLightningIndexerGradKLLossV2.md · 前轮保留

位置：[ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/docs/aclnnSparseLightningIndexerGradKLLossV2.md:111](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/sparse_lightning_indexer_grad_kl_loss/docs/aclnnSparseLightningIndexerGradKLLossV2.md#L111)

V2 文档的第二段函数原型写成 aclnnSparseLightningIndexerGradKLLoss，遗漏 V2；第一段和调用样例使用 V2。

依据：[V2 头文件声明](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/sparse_lightning_indexer_grad_kl_loss/op_host/op_api/aclnn_sparse_lightning_indexer_grad_kl_loss_v2.h#L38)

## 函数原型与头文件不一致（27 项）

已排除按值参数顶层 const 的无害差异。指针所指对象 const 限定不一致属于接口描述问题，不代表所有样例都会编译失败。

### 41. aclnnIsInScalarTensor.md · 前轮保留

位置：[ops-math/experimental/math/equal/docs/aclnnIsInScalarTensor.md:18](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/equal/docs/aclnnIsInScalarTensor.md#L18)

第 1 项：文档 `const aclTensor* self`；头文件 `const aclScalar* element`；第 2 项：文档 `const aclScalar* element`；头文件 `const aclTensor* testElements`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/equal/op_api/aclnn_isin.h#L34)

### 42. aclnnIsInTensorScalar.md · 前轮保留

位置：[ops-math/experimental/math/equal/docs/aclnnIsInTensorScalar.md:18](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/equal/docs/aclnnIsInTensorScalar.md#L18)

第 5 项：文档 `aclScalar* out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/equal/op_api/aclnn_isin_tensor_scalar.h#L32)

### 43. aclnnKlDivV2.md · 前轮保留

位置：[ops-math/experimental/math/kl_div_v2/docs/aclnnKlDivV2.md:31](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/kl_div_v2/docs/aclnnKlDivV2.md#L31)

第 3 项：文档 `const char* reduction`；头文件 `int64_t reduction`；第 5 项：文档 `const aclTensor* out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/kl_div_v2/op_api/aclnn_kl_div.h#L24)

### 44. aclnnLogicalNot.md · 前轮保留

位置：[ops-math/experimental/math/logical_not/docs/aclnnLogicalNot.md:26](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/logical_not/docs/aclnnLogicalNot.md#L26)

第 2 项：文档 `const aclTensor* out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/logical_not/op_api/aclnn_logical_not.h#L49)

### 45. aclnnLogicalOr.md · 前轮保留

位置：[ops-math/experimental/math/logical_or/docs/aclnnLogicalOr.md:30](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/logical_or/docs/aclnnLogicalOr.md#L30)

第 3 项：文档 `const aclTensor* out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/logical_or/op_api/aclnn_logical_or.h#L48)

### 46. aclnn_log_add_exp.md · 前轮保留

位置：[ops-math/experimental/math/log_add_exp/docs/aclnn_log_add_exp.md:23](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/log_add_exp/docs/aclnn_log_add_exp.md#L23)

第 3 项：文档 `const aclTensor *out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/log_add_exp/op_api/aclnn_logaddexp.h#L39)

### 47. aclnnPdist.md · 前轮保留

位置：[ops-math/experimental/math/pdist/docs/aclnnPdist.md:36](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/pdist/docs/aclnnPdist.md#L36)

第 2 项：文档 `double p`；头文件 `float p`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/pdist/op_api/aclnn_pdist.h#L35)

### 48. aclnnSign.md · 前轮保留

位置：[ops-math/experimental/math/sign/docs/aclnnSign.md:29](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/sign/docs/aclnnSign.md#L29)

第 2 项：文档 `const aclTensor *result`；头文件 `aclTensor* result`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/sign/op_api/aclnn_sign.h#L25)

### 49. aclnnTan.md · 前轮保留

位置：[ops-math/experimental/math/tan/docs/aclnnTan.md:19](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/tan/docs/aclnnTan.md#L19)

第 2 项：文档 `const aclTensor *out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/tan/op_api/aclnn_tan.h#L45)

### 50. aclnnAngleV2.md · 前轮保留

位置：[ops-math/math/angle_v2/docs/aclnnAngleV2.md:39](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/angle_v2/docs/aclnnAngleV2.md#L39)

第 2 项：文档 `const aclTensor* out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/angle_v2/op_api/aclnn_angle_v2.h#L36)

### 51. aclnnSoftsign.md · 前轮保留

位置：[ops-nn/experimental/activation/softsign/docs/aclnnSoftsign.md:35](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign/docs/aclnnSoftsign.md#L35)

第 2 项：文档 `aclTensor *out`；头文件 `const aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign/op_api/aclnn_softsign.h#L46)

### 52. aclnnSoftsignBackward.md · 前轮保留

位置：[ops-nn/experimental/activation/softsign_grad/docs/aclnnSoftsignBackward.md:35](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign_grad/docs/aclnnSoftsignBackward.md#L35)

第 3 项：文档 `aclTensor *output`；头文件 `const aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softsign_grad/op_api/aclnn_softsign_backward.h#L49)

### 53. aclnnSoftMarginLoss.md · 前轮保留

位置：[ops-nn/experimental/loss/soft_margin_loss/docs/aclnnSoftMarginLoss.md:27](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/loss/soft_margin_loss/docs/aclnnSoftMarginLoss.md#L27)

第 4 项：文档 `const aclTensor *out`；头文件 `aclTensor* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/loss/soft_margin_loss/op_api/aclnn_soft_margin_loss.h#L37)

### 54. aclnnForeachAddcdivScalar.md · 前轮保留

位置：[ops-nn/foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalar.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalar.md#L46)

第 5 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_addcdiv_scalar/op_host/op_api/aclnn_foreach_addcdiv_scalar.h#L34)

### 55. aclnnForeachAddcdivScalarList.md · 前轮保留

位置：[ops-nn/foreach/foreach_addcdiv_scalar_list/docs/aclnnForeachAddcdivScalarList.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_addcdiv_scalar_list/docs/aclnnForeachAddcdivScalarList.md#L46)

第 5 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_addcdiv_scalar_list/op_host/op_api/aclnn_foreach_addcdiv_scalar_list.h#L34)

### 56. aclnnForeachAddList.md · 前轮保留

位置：[ops-nn/foreach/foreach_add_list/docs/aclnnForeachAddList.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_add_list/docs/aclnnForeachAddList.md#L46)

第 4 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_add_list/op_host/op_api/aclnn_foreach_add_list.h#L33)

### 57. aclnnForeachCopy.md · 前轮保留

位置：[ops-nn/foreach/foreach_copy/docs/aclnnForeachCopy.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_copy/docs/aclnnForeachCopy.md#L46)

第 2 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_copy/op_api/aclnn_foreach_copy.h#L34)

### 58. aclnnForeachDivList.md · 前轮保留

位置：[ops-nn/foreach/foreach_div_list/docs/aclnnForeachDivList.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_div_list/docs/aclnnForeachDivList.md#L46)

第 3 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_div_list/op_host/op_api/aclnn_foreach_div_list.h#L32)

### 59. aclnnForeachDivScalar.md · 前轮保留

位置：[ops-nn/foreach/foreach_div_scalar/docs/aclnnForeachDivScalar.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_div_scalar/docs/aclnnForeachDivScalar.md#L46)

第 3 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_div_scalar/op_host/op_api/aclnn_foreach_div_scalar.h#L32)

### 60. aclnnForeachLerpList.md · 前轮保留

位置：[ops-nn/foreach/foreach_lerp_list/docs/aclnnForeachLerpList.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_lerp_list/docs/aclnnForeachLerpList.md#L46)

第 4 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_lerp_list/op_api/aclnn_foreach_lerp_list.h#L33)

### 61. aclnnForeachLerpScalar.md · 前轮保留

位置：[ops-nn/foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md:45](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md#L45)

第 4 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_lerp_scalar/op_api/aclnn_foreach_lerp_scalar.h#L33)

### 62. aclnnForeachMulList.md · 前轮保留

位置：[ops-nn/foreach/foreach_mul_list/docs/aclnnForeachMulList.md:46](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_mul_list/docs/aclnnForeachMulList.md#L46)

第 3 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_mul_list/op_host/op_api/aclnn_foreach_mul_list.h#L32)

### 63. aclnnForeachSqrt.md · 前轮保留

位置：[ops-nn/foreach/foreach_sqrt/docs/aclnnForeachSqrt.md:45](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_sqrt/docs/aclnnForeachSqrt.md#L45)

第 2 项：文档 `const aclTensorList *out`；头文件 `aclTensorList* out`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/foreach/foreach_sqrt/op_host/op_api/aclnn_foreach_sqrt.h#L32)

### 64. aclnnGroupedQuantMax.md · 前轮保留

位置：[ops-nn/quant/grouped_quant_max/docs/aclnnGroupedQuantMax.md:53](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/quant/grouped_quant_max/docs/aclnnGroupedQuantMax.md#L53)

第 6 项：文档 `aclTensor *y`；头文件 `const aclTensor* y`；第 7 项：文档 `aclTensor *amax`；头文件 `const aclTensor* amax`

依据：[声明依据](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/quant/grouped_quant_max/op_api/aclnn_grouped_quant_max.h#L24)

### 65. aclnnMixedQuantSparseFlashMla.md · 前轮保留

位置：[ops-transformer/attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md:104](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md#L104)

第 27 项：文档 `char *layoutQOptional`；头文件 `const char *layoutQOptional`；第 28 项：文档 `char *layoutKvOptional`；头文件 `const char *layoutKvOptional`

依据：[声明依据](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/mixed_quant_sparse_flash_mla/op_api/aclnn_mixed_quant_sparse_flash_mla.h#L66)

### 66. aclnnRgb2yuv422.md · 前轮保留

位置：[ops-cv/experimental/image/rgb2yuv422/docs/aclnnRgb2yuv422.md:62](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/experimental/image/rgb2yuv422/docs/aclnnRgb2yuv422.md#L62)

第 2 项：文档 `const char* dataFormat`；头文件 `char *dataFormatOptional`

依据：[声明依据](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/experimental/image/rgb2yuv422/op_api/aclnn_rgb2yuv422.h#L27)

### 67. aclnnNonMaxSuppression.md · 前轮保留

位置：[ops-cv/objdetect/non_max_suppression_v6/docs/aclnnNonMaxSuppression.md:51](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/objdetect/non_max_suppression_v6/docs/aclnnNonMaxSuppression.md#L51)

第 4 项：文档 `aclFloatArray* iouThreshold`；头文件 `const aclFloatArray* iouThreshold`

依据：[声明依据](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/objdetect/non_max_suppression_v6/op_host/op_api/aclnn_non_max_suppression.h#L21)

## 样例正常路径遗漏资源释放（16 项）

### 68. aclnnUpsampleBicubic2dAAGrad.md · 本轮新增

位置：[ops-cv/image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md:451](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md#L451)

通过 aclCreateIntArray 创建 outputSizeArray、inputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 69. aclnnUpsampleBilinear2dAABackward.md · 本轮新增

位置：[ops-cv/image/upsample_bilinear2d_aa_backward/docs/aclnnUpsampleBilinear2dAABackward.md:454](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bilinear2d_aa_backward/docs/aclnnUpsampleBilinear2dAABackward.md#L454)

通过 aclCreateIntArray 创建 outputSizeArray、inputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 70. aclnnUpsampleBilinear2dAA.md · 本轮新增

位置：[ops-cv/image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md:424](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md#L424)

通过 aclCreateIntArray 创建 outputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 71. aclnnUpsampleBilinear2d.md · 本轮新增

位置：[ops-cv/image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md:421](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_bilinear2d/docs/aclnnUpsampleBilinear2d.md#L421)

通过 aclCreateIntArray 创建 outputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 72. aclnnUpsampleNearestExact1dBackward.md · 本轮新增

位置：[ops-cv/image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact1dBackward.md:372](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact1dBackward.md#L372)

通过 aclCreateIntArray 创建 outputSizeArray、inputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 73. aclnnUpsampleNearestExact2dBackward.md · 本轮新增

位置：[ops-cv/image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact2dBackward.md:403](https://gitcode.com/cann/ops-cv/blob/ed2be3ce7c60a683f54ce999965dc4571d2d62a8/image/upsample_nearest_exact2d_grad/docs/aclnnUpsampleNearestExact2dBackward.md#L403)

通过 aclCreateIntArray 创建 outputSizeArray、inputSizeArray，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 74. aclnnConfusionTranspose.md · 本轮新增

位置：[ops-math/conversion/confusion_transpose_d/docs/aclnnConfusionTranspose.md:354](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/conversion/confusion_transpose_d/docs/aclnnConfusionTranspose.md#L354)

通过 aclCreateIntArray 创建 perm、shape，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 75. aclnnAddrV2&aclnnInplaceAddrV2.md · 本轮新增

位置：[ops-math/experimental/math/addr_v2/docs/aclnnAddrV2&aclnnInplaceAddrV2.md:579](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/math/addr_v2/docs/aclnnAddrV2%26aclnnInplaceAddrV2.md#L579)

通过 aclCreateScalar 创建 beta、alpha，正常结束时没有对应 aclDestroyScalar。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 76. aclnnAddr&aclnnInplaceAddr.md · 本轮新增

位置：[ops-math/math/addr/docs/aclnnAddr&aclnnInplaceAddr.md:579](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/addr/docs/aclnnAddr%26aclnnInplaceAddr.md#L579)

通过 aclCreateScalar 创建 beta、alpha，正常结束时没有对应 aclDestroyScalar。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 77. aclnnSoftplusBackward.md · 本轮新增

位置：[ops-nn/activation/softplus_v2_grad/docs/aclnnSoftplusBackward.md:351](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/activation/softplus_v2_grad/docs/aclnnSoftplusBackward.md#L351)

通过 aclCreateScalar 创建 beta、threshold，正常结束时没有对应 aclDestroyScalar。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 78. aclnnSoftplusBackward.md · 本轮新增

位置：[ops-nn/experimental/activation/softplus_v2_grad/docs/aclnnSoftplusBackward.md:321](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/softplus_v2_grad/docs/aclnnSoftplusBackward.md#L321)

通过 aclCreateScalar 创建 beta、threshold，正常结束时没有对应 aclDestroyScalar。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 79. aclnnFlip.md · 本轮新增

位置：[ops-nn/index/reverse_v2/docs/aclnnFlip.md:313](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/index/reverse_v2/docs/aclnnFlip.md#L313)

通过 aclCreateIntArray 创建 dims，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 80. aclnnNsaCompressAttention.md · 本轮新增

位置：[ops-transformer/attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md:624](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md#L624)

通过 aclCreateIntArray 创建 actualSeqQLen、actualCmpKvSeqLen、actualSelKvSeqLen，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 81. aclnnNsaSelectedAttentionInfer.md · 本轮新增

位置：[ops-transformer/attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md:556](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md#L556)

通过 aclCreateIntArray 创建 actualCmpKvSeqLen、actualCmpQSeqLen，正常结束时没有对应 aclDestroyIntArray。它们也未交给列表或 RAII 对象管理。复用这段调用流程会积累未释放对象。

### 82. aclnnDequantBias.md · 本轮新增

位置：[ops-nn/quant/dequant_bias/docs/aclnnDequantBias.md:371](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/quant/dequant_bias/docs/aclnnDequantBias.md#L371)

样例创建 weight、activation、bias 三个 Tensor 及设备缓冲区，结尾只销毁/释放 input、y，遗漏上述三个 Tensor 和对应设备内存。

### 83. aclnnSearchSorted.md · 本轮新增

位置：[ops-math/math/search_sorted/docs/aclnnSearchSorted.md:411](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/search_sorted/docs/aclnnSearchSorted.md#L411)

selfDeviceAddr 已申请并使用，但正常释放段仅释放 sortedSequenceDeviceAddr、sorterDeviceAddr、outDeviceAddr，遗漏 selfDeviceAddr。

## 影响展示或复制的格式问题（9 项）

### 84. aclnnClampMaxTensor&aclnnInplaceClampMaxTensor.md · 前轮保留

位置：[ops-math/experimental/conversion/clip_by_value_v2/docs/aclnnClampMaxTensor&aclnnInplaceClampMaxTensor.md:35](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/experimental/conversion/clip_by_value_v2/docs/aclnnClampMaxTensor%26aclnnInplaceClampMaxTensor.md#L35)

第一段原型未结束便再次写入带语言的开始围栏，第二个 ```cpp 被当作代码内容。

### 85. aclnnMseLossGrad.md · 前轮保留

位置：[ops-nn/experimental/loss/mse_loss_grad/docs/aclnnMseLossGrad.md:43](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/loss/mse_loss_grad/docs/aclnnMseLossGrad.md#L43)

第一段原型未结束便再次写入带语言的开始围栏。

### 86. aclnnMaxPoolingGrad.md · 前轮保留

位置：[ops-nn/experimental/pooling/max_pooling_grad/docs/aclnnMaxPoolingGrad.md:62](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/pooling/max_pooling_grad/docs/aclnnMaxPoolingGrad.md#L62)

第一段原型未结束便再次写入带语言的开始围栏。

### 87. aclnnSwishBackward.md · 前轮保留

位置：[ops-nn/experimental/activation/swish_grad/docs/aclnnSwishBackward.md:58](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/experimental/activation/swish_grad/docs/aclnnSwishBackward.md#L58)

结束围栏误写为 ```l；这不是有效结束围栏，后续参数表、约束和样例标题被收入代码块。

### 88. aclnnConvertWeightToINT4Pack.md · 前轮保留

位置：[ops-nn/matmul/convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md:698](https://gitcode.com/cann/ops-nn/blob/6bf411a1c5c8bfde94573ad49f76628550c04c7f/matmul/convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md#L698)

上一段示例结束后缺少结束围栏，产品注释及下一段样例说明会进入代码块；第 706 行开始围栏不能正确开启新代码块。

### 89. aclnnNanToNum&aclnnInplaceNanToNum.md · 前轮保留

位置：[ops-math/math/nan_to_num/docs/aclnnNanToNum&aclnnInplaceNanToNum.md:702](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/nan_to_num/docs/aclnnNanToNum%26aclnnInplaceNanToNum.md#L702)

结尾写成 }```，围栏未独占一行，反引号进入 C++ 样例内容，复制后不能编译。

### 90. aclnnNeScalar&aclnnInplaceNeScalar.md · 前轮保留

位置：[ops-math/math/not_equal/docs/aclnnNeScalar&aclnnInplaceNeScalar.md:598](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/not_equal/docs/aclnnNeScalar%26aclnnInplaceNeScalar.md#L598)

结尾写成 }```，围栏未独占一行，反引号进入 C++ 样例内容，复制后不能编译。

### 91. aclnnNeTensor&aclnnInplaceNeTensor.md · 前轮保留

位置：[ops-math/math/not_equal/docs/aclnnNeTensor&aclnnInplaceNeTensor.md:595](https://gitcode.com/cann/ops-math/blob/052d8f3d08373538453a30cccdfd90fd750f3c23/math/not_equal/docs/aclnnNeTensor%26aclnnInplaceNeTensor.md#L595)

结尾写成 }```，围栏未独占一行，反引号进入 C++ 样例内容，复制后不能编译。

### 92. aclnnMoeUpdateExpert.md · 本轮新增

位置：[ops-transformer/mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md:356](https://gitcode.com/cann/ops-transformer/blob/f2e0720ee72b9e17fa26f7255e35cad58702261d/mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md#L356)

顶层样例的开始围栏缩进 4 个空格，CommonMark 将其解析为缩进代码块，字面量 ```Cpp 会成为代码内容。按渲染后的代码块复制会带入无效 C++ 字符。
