# aclnnMlaPrologV3/V4WeightNz 扫描记录

基于本地 ops-transformer 提交 `2322a9b6b`，扫描日期 2026-09-20。结论来自当前检出的文档和源码，未编译或运行 NPU。此次扫描不修改 YAML、generator 或仓库实现。

## 主要结论

两套公开接口都位于 `attention/mla_prolog_v3`。V4 没有独立的 op_host/op_kernel 目录；V3/V4 的公开 API 都调用 `aclnnInnerMlaPrologV3GetWorkspaceSize` 和 `aclnnInnerMlaPrologV3`。V3 将 doRope 固定为 true，V4 从公开接口传入该开关。

| 项目 | V3 | V4 |
| --- | --- | --- |
| 公开参数 | kcScale 后直接是输出 tensor | kcScale 后增加 bool doRope |
| RoPE | 固定开启 | 开启或关闭；关闭时 Q/K 位置编码分支直通 |
| sin/cos | 按开启 RoPE 的规则提供 | true 时必须是非 nullptr；false 时必须同时 nullptr |
| N 范围（当前版本） | 1–128 | 1–128 |
| 量化与缓存 | 共用底层场景规则 | 共用底层场景规则 |
| queryNormFlag | 由 queryNormOutOptional 是否为 nullptr 推导 | 同左 |

V4 文档仍描述“N 范围调整”为功能更新，但当前 V3 文档及共用 tiling 校验也已经允许 1–128，因此不能以该描述推断当前两个入口有不同的 N 范围。

## 源码阅读入口

下列路径均相对 `ops-transformer/`，行号以本次检出版本为准。

| 文件 | 阅读位置/用途 |
| --- | --- |
| attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md | V3 接口、维度、dtype、场景表；约 903 行为 N 范围 |
| attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md | V4 功能更新与参数规则 |
| attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp | 190 行公开入口；末尾调用 inner 时固定传 true |
| attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.cpp | 106 行起平台/量化组合校验；194 行入口；RoPE nullptr 校验及可选输出处理 |
| attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp | 算子定义；180 行 do_rope 默认 true |
| attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.cpp | 69 行起兼容缺失 do_rope 的旧图；关闭 RoPE 时由权重推导 Dr |
| attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling.h | 引用 mla_prolog 的共用 tiling 头文件 |
| attention/mla_prolog/op_host/mla_prolog_tiling.cpp | 场景识别、维度提取、tiling；380 行起处理 RoPE 开关 |
| attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp | 核心校验：223 行维度；306/322 行 Hcq/D；338 行 Dtile；374 行 NZ；981 行量化联动 |
| attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp | 带 EnableRope 模板参数的公共 kernel 入口，按架构引用 arch35/arch22 |
| attention/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h | 1294 行等位置包含关闭 RoPE 的分支；旋转函数带 enableRope 模板参数 |
| attention/mla_prolog_v3/torch_extension/csrc/mla_prolog.cpp | 53 行推导 RoPE 开关；173 行输入校验；453 行调用 V4 ACLNN |
| attention/mla_prolog_v3/op_graph/fallback_mla_prolog_v3.cpp | 图模式 fallback 调用 V3 公开接口 |

## 已核实的共用约束

- tokenX 为 `(T,He)` 或 `(B,S,He)`；B≤65536，He∈{1024,2048,3072,4096,5120,6144,7168,7680,8192}。
- Hcq∈{1536,2048}，N∈[1,128]，D∈{128,192}，Hckv=512，Dr=64，Nkv=1。
- BlockSize∈[16,1024] 且为 16 的倍数。tileSize 在共用 V3 属性校验中仅允许 128，不应随意泛化。
- DAV_3510 的 wq 支持 0–5，其余架构的公开 API 校验只允许 0–2。
- wq=0 仅 kvq=0；wq=1 支持 kvq=0/2/3；wq=2/3/4/5 支持 kvq=0/1/3。A5 共 16 种场景。
- queryQuantMode 必须在 kvq=1 时为 1，其他场景为 0。
- kvq=3 时两个 RepoMode 必须为 1，其他场景必须为 0。Dtile=512+64×2+(512/128)×4=656；合并存储时 krCache/krCacheOut 必须为零元素 Tensor。
- 文档限定 kvq=3 使用 PA_BSND/BSND/TND。BSND 对应三维 tokenX，TND 对应二维 tokenX；PA 模式按布局决定索引 shape。
- PA_BLK 合轴时需 actualSeqLen 前缀和，末项等于 T；cacheIndex 长度对应各序列所需块数之和。不能用独立随机值代替前缀和。
- 三个权重必须为 FRACTAL_NZ。逻辑二维 shape 不是实际格式转换；共用校验也处理四维 NZ storage shape，轴块大小与 dtype 字节数有关。
- MXFP8 的四组反量化 scale 实际 dtype 为 E8M0；uint8 只是当前 generator 的适配约定，不能直接当算子输入 dtype。
- `MAX_S1_SIZE`、`MAX_T_SIZE` 虽出现在共用头文件中，但本次检索该 op_host 目录仅发现定义，未发现它们被用于校验；不能仅依据常量断言 S/T 有该上限。维度内部使用 uint32 等类型也意味着“文档不限制”不等于任意大的 Python 整数均已验证可执行。

## 可选输出与空值区别

公开 API 使用 TensorHolder 将部分 nullptr 输出转换为内部占位空 Tensor，并根据原始指针是否存在判定合法性。因此看到后续的 nullptr 检查，不能误判所有 Optional 输出都必须提供。

- dequantScaleQNopeOutOptional：仅全量化且 kvq=1 时提供，其他场景应为 nullptr。
- queryNormOutOptional：是否提供决定内部 queryNormFlag；不是额外的公开 ACLNN 参数。
- dequantScaleQNormOutOptional：queryNormOut 存在且 wq≠0 时提供。
- V4 doRope=false：公开 ACLNN 严格要求 nullptr，而不是“指向零元素 tensor 的非空指针”。inner 接收到的空 Tensor 是 API 内部转换结果。
- 分组缓存 krCache：要求保留 dtype 的零元素 Tensor，不能直接混用 RoPE 的 nullptr 语义。

## PyTorch 扩展与现有 ATK 的关系

仓库新提供的 torch_extension 调用 V4，并根据 sin/cos 是否成对提供推导 do_rope，而非直接暴露同名布尔参数。其包装会将关闭 RoPE 的输入转为 nullopt 后传 ACLNN。

HIF8 路径在该扩展中支持 uint8 载体配合显式 ACL_HIFLOAT8 dtype 信息；这不能证明当前 ATK 已注册 `hif8` 别名。MX scale 路径显式检查 torch.float8_e8m0fnu。

本地 YAML 使用的 `function_mla_prolog_v4` / `pyaclnn_function_mla_prolog_v4` 不等同于这个扩展。仅扫描本仓库不能确认现有 ATK golden 的全场景支持能力。

## 测试和后续用例重点

仓库包含两版本的普通示例、部分量化 perchannel 示例以及 arch35 全量化 KV 示例。共用 `tests/ut/op_host` 中有 tiling、infershape 测试；`test_mla_prolog_v3_tiling.cpp` 3805 行起包括关闭 RoPE，3862 行起包括关闭 RoPE 且合并 krCache；infershape 测试还覆盖缺失 do_rope 属性的旧图兼容。测试存在不代表本机已执行或通过。

对当前 generator 的主要核对点：

1. shape 修正后的所有关联权重、rope、scale、cache 维度必须同步；现有 He/B 修正方向与代码检查一致。
2. 适配器必须将 bool/[] 的可选输入哨兵转换成真正 nullptr，不能生成 bool 标量或非空空张量指针传给关闭 RoPE 的接口。
3. 为 PA 索引生成无重复写入位置，避免精度比对受到并发覆盖影响；范围合法不等于数据无冲突。
4. 维持 NZ 转换、E8M0 转换、HIF8 dtype 处理与 profile 能力一致。
5. 补充 queryNorm 输出开关、独立空缓存、变长多序列 PA_BLK、首轴非连续缓存等适配器级验证。

此次只扫描和记录，未将未验证的适配器行为当作已经支持，也未运行 NPU 精度测试。
