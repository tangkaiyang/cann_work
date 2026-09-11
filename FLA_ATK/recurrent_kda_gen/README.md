# recurrent_kda 泛化用例（补充泛化归档）

本目录是源仓 `flash-linear-attention-npu/tests/atk/recurrent_kda/` 的**补充泛化**归档。
源仓仅有 1 个 BSND B=2,T=2,H=2,HV=4,K=128,V=128 的平凡用例（cu_seqlens 按 batch 平凡划分、
不传 ssm_state_indices）。本目录在保持 spec-driven 风格不变的前提下扩展能力域。
部署时把三个文件拷入源仓 `tests/atk/recurrent_kda/` 覆盖同名文件即可。

## 相对源仓的扩展

| 维度 | 源仓覆盖 | 本目录覆盖 |
| --- | --- | --- |
| ssm_state_indices | 不传（None） | none / packed [T]（含重复槽）/ speculative [seq_num, max_step] + numAcceptedTokens |
| cu_seqlens | 平凡按 batch 划分 | 多段（1~4 段），每段长度 1~8（满足算子"段长 <= 8"约束） |
| layout | 仅 BSND | BSND / TND |
| V | 仅 128 | 128 / 256（算子支持的两档） |
| dt_bias | 不传 | flat [H_v*K] 与 matrix [H_v, K] 两种形状（use_gate_in_kernel=True 路径） |
| state_v_first | 恒 true | true / false 交替（state pool [cap,HV,V,K] / [cap,HV,K,V]） |
| HV/H | 4/2 | (2,1) (4,2) (8,4) (4,4)，HV % H == 0 |

q/k/v 仅 BF16（算子限制，README "当前限制"），g/beta 用 FP32。

## 生成器设计（gen_recurrent_kda.py）

- 96 个 profile = 2 layout × 3 ssm_mode × 2 V × 2 dt_bias × 4 HV/H 组合；
- `cu_seqlens` 由确定性 RNG 构造（每段 1..8，严格递增），写入 case_spec；
- packed 索引允许重复槽（覆盖 state 共享路径），speculative 矩阵形状
  [seq_num, max_step] 且 `num_accepted_tokens ∈ [1, 段长]`；
- state_capacity：none 模式等于 seq_num（wrapper 强约束），索引模式 > seq_num
  （覆盖"capacity 大于段数"场景）。

## Executor 设计（executor_recurrent_kda.py）

- `build_inputs` 按 spec 构造全部张量与元数据：cu_seqlens INT64、
  ssm_state_indices（1D/2D INT64）、num_accepted_tokens、dt_bias（flat/matrix）、
  A_log（FP32 [HV]，use_gate_in_kernel=True 必选）、非零小量 initial_state；
- `run_npu` 把 ssm_state_indices / numAcceptedTokens / A_log / dtBias 真实传给
  `ascendc.recurrent_kda`（源仓固定传 None，本目录打开完整通路）；
- CPU 标杆与源仓逐行一致（索引寻址、dt_bias 门控、state_v_first 切换），
  run_cpu/run_npu 均以 `use_gate_in_kernel=True` 闭环 raw gate 路径。

## 校验

见仓库根 `_sanity_check_gen.py`（离线桩校验：96 个 profile 全量通过，
含 cu_seqlens 单调性、段长 <= 8、索引范围、accepted 与段长一致性、
HV % H == 0、K/V 合法性、TND/BSND 的 T/B 语义一致性）。
