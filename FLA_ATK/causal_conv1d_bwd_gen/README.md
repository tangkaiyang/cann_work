# causal_conv1d_bwd 泛化用例（补充泛化归档）

本目录是源仓 `flash-linear-attention-npu/tests/atk/causal_conv1d_bwd/` 的**补充泛化**归档，
保持源仓 spec-driven 风格（YAML 只含 marker 张量 + 镜像属性，真实输入由 executor 的
`build_inputs()` 按 `case_spec` JSON 构造）。部署时把三个文件拷入源仓 `tests/atk/causal_conv1d_bwd/`
覆盖同名文件即可（executor 依赖 `../common/_ascendc_common_executor.py`，目录层级不变）。

## 相对源仓的扩展

| 维度 | 源仓覆盖 | 本目录覆盖 |
| --- | --- | --- |
| layout | 仅 BSND | BSND / BSH / BNSD（定长）+ TND / NTD（varlen，query_start_loc） |
| activation | 仅 0 | 0（无）/ 1（SiLU）/ 2（Swish），act=1/2 时自动构造 yOptional |
| initial_state / dht | 不携带 | `with_state` 维度交替覆盖（[B, W, D]） |
| dtype | bf16 / fp16 | 同源仓 |
| B / T / D / W | 网格 B∈{1,2,4,8} T∈{16..256} D∈{16..256} W∈{2,3,4} | 同源仓（T>=W） |
| varlen 分段 | 无 | TND/NTD 下由生成器确定性构造 query_start_loc（每段 >= W） |

## 生成器设计（gen_causal_conv1d_bwd.py）

- `PROFILES` 由 `(B,T,D,W)` 网格（固定种子打乱）× 2 dtype × 3 activation × 5 layout 展开，
  varlen 组合要求 `B*W <= T`（每段至少 W 个 token），构造不出则跳过；
- `with_state` 按 (activation 序号 + 定长/varlen 序号) 奇偶交替，保证各维度正交覆盖；
- varlen 的 `query_start_loc` 用确定性 RNG 均分 T 并做少量扰动，写入 case_spec；
- 共 8880 个 profile，`dtype_numbers` 控制实际用例数量，按 index 取模循环。

## Executor 设计（executor_causal_conv1d_bwd.py）

- `run_npu`：按 layout 对逻辑 [B, T, D] 张量做 reshape/permute（BSH/BNSD/TND/NTD），
  传 `query_start_loc`（TND/NTD）、`activation`、`initial_state`/`dht`（with_state 时）；
  调用 `ascendc.causal_conv1d_bwd`，无 dht 时只取前 3 个输出（dx/dw/db），
  有 dht 时返回 4 个输出（含 dh0），CPU 标杆同步返回 4 元组；
- `run_cpu`：显式左填充卷积反传参考（源仓实现），varlen 场景按 query_start_loc
  逐段调用并叠加 dw/db（reduction 跨段累加）；
- activation=1/2 时用 FP32 前向预激活参考构造 yOptional 并量化回原始 dtype。

## 校验

见仓库根 `_sanity_check_gen.py`（离线桩校验：全部 8880 个 profile 的
维度/约束/镜像属性一致性，2026-09-11 全量通过）。
