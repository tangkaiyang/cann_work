# MlaPrologV4WeightNz A5 泛化用例

依据同目录 `aclnnMlaPrologV4WeightNz.md`，保留原有 33 个入参顺序和 generator 注册名。T、B、S、He 直接读取 token_x.shape，合法值不再由 generator 随机生成或覆盖，超限值自动修正。二维读取 (T, He)，三维读取 (B, S, He) 并计算 T=B*S；缓存模式随输入维数筛选。

- 默认 `MLA_V4_PROFILE=golden`：wq=0–3，共 10 种合法量化组合。
- `MLA_V4_PROFILE=full`：wq=0–5，共 16 种组合；使用前需确认 ATK 支持 `hif8` dtype 别名，并补齐 golden/适配器的 fp8、hif8 处理。
- 缓存布局：PA_BSND、PA_NZ、PA_BLK_BSND、PA_BLK_NZ、BSND、TND；分组量化仅生成文档允许的 PA_BSND、BSND、TND。
- 支持 BS 合轴/非合轴、零 token/零 batch/零序列长度；这些维度由输入 shape 决定。YAML 默认生成 (B,S,7168)，二维配置方式见 token_x 注释；He 应使用文档列出的合法值。其余维度覆盖 N=1/128 及中间值、Hcq/D 候选，以及 BlockSize=16/1024 和非二次幂的 16 倍数。
- 分组量化绑定两个 RepoMode=1、Dtile=656、tileSize=128 和 bf16 空 krCache；其他场景 RepoMode=0。INT8 pertensor 缓存为 int8，fp8/mxfp8 缓存为 fp8e4m3。
- 联动 RoPE 空指针、量化 scale、可选 smooth scale、clip alpha；扩充 epsilon 和 Query/Key 尺度参数。

## 运行和验证

generator 自动修正 token_x.shape：维度转整数、负值归零，He 映射到最近的合法枚举（等距取较小值），B 截断到 65536；合法 shape 保持不变。空 shape 补为 [1,1024]，一维补为 [1,He]，超过三维则合并前面的维度为 T。修正结果写回 token_x，其他输入全部按修正后的维度生成。S/T 按文档不额外设置上限；合轴 T 超出 INT32 时排除需要 actualSeqLen 的 PA_BLK 布局。

沿用原 ATK 调用方式；YAML 的随机配置数由 100 调整为 500。完整模式在启动 ATK 前设置环境变量 `MLA_V4_PROFILE=full`。随机采样不保证单批 500 条遍历全部组合，可使用 ATK 的随机种子机制复现。

本地配置检查：安装 PyYAML 后运行 `python -B test_generator.py`。测试使用轻量 ATK 替身，检查两个 profile 各 5000 次生成及配置复用，验证 YAML 类型/维数/属性枚举与接口联动约束，不分配张量。

## 适配边界

本目录没有真实 ATK、golden、ACLNN 适配器或 NPU 环境。默认 profile 仅限制权重量化模式，并不表示新增布局、空输入和分组量化已通过 golden 精度验证。

权重仍使用逻辑二维 shape，由现有适配器转换为 FRACTAL_NZ；MX scale 沿用 uint8 占位约定，由适配器转换 dtype 和数值为 e8m0。PA_NZ/PA_BLK_NZ 同样需要适配器正确处理缓存存储格式。

PA_BLK 合轴暂以单条序列生成 actualSeqLen=[T]，避免将随机范围误当作合法前缀和。多序列变长前缀和需要数据生成钩子。cacheIndex 沿用原有范围采样方式，只保证索引范围合法；精度运行如需无冲突写入，应由适配器生成不重复索引。未加入非连续 stride、最大 B=65536 压力用例和独立 Skv=0 场景。

文档 BSND/TND shape 约束小节将 kvCache 末维写为 Dr，与参数表及 Dtile 定义不一致；这里采用参数表的 Hckv=512（分组量化 656）。文档参数表出现 queryNormFlag，但函数原型和现有 33 入参均无此项，因此未擅自增加接口参数。
