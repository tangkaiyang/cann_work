# prepare_wy_repr_bwd_da 泛化用例（补充泛化归档）

本目录是源仓 `flash-linear-attention-npu/tests/atk/prepare_wy_repr_bwd_da/` 的**补充泛化**归档。
源仓仅有 2 个窄用例（B=1,HK=1,HV=1,T=128 的 bf16/fp16，且 executor 固定
`cu_seqlens=None, chunk_indices=None` 只跑定长）。本目录打开 varlen 与 GVA 通路。
部署时把三个文件拷入源仓 `tests/atk/prepare_wy_repr_bwd_da/` 覆盖同名文件即可。

## 相对源仓的扩展

| 维度 | 源仓覆盖 | 本目录覆盖 |
| --- | --- | --- |
| 模式 | 仅定长 | 定长 B∈{1,2,4} 与 varlen（B=1）交替 |
| cu_seqlens / chunk_indices | None | varlen 时传 python list（wrapper 要求），由生成器确定性构造 |
| varlen 分段 | 无 | 2~4 段，每段为 chunk_size 的整数倍（README 5.2 约束） |
| GVA | HK=HV=1 | (HK,HV) ∈ {(1,1) (2,2) (2,4) (4,8) (1,4)}，HV = group_size × HK |
| V | 仅 128 | 128 / 256 |
| T | 128（yaml 声明 16） | 64 / 128 / 256 |
| dtype | bf16 / fp16 | 交替覆盖 |

K 固定 128，chunk_size 64（算子常用值）。

## 生成器设计（gen_prepare_wy_repr_bwd_da.py）

- 30 个 profile = 3 T × 5 (HK,HV) × 2 V 展开，varlen/fixed 与 dtype 按序交替；
- varlen 的 `cu_seqlens` 用确定性 RNG 按"每段 1~3 个 chunk、缩放到总 chunk 数"构造，
  保证每段为 chunk_size 倍数且总和 = T；
- `chunk_indices` 为扁平化 [seq_id0, chunk_id0, seq_id1, chunk_id1, ...]
  （语义与 `test_da.py::prepare_chunk_indices` 一致），写入 case_spec。

## Executor 设计（executor_prepare_wy_repr_bwd_da.py）

- `build_inputs`：varlen 时 B 强制 1、T 取 cu_seqlens 末项（packed 长度）；
  g 用 `-arange` 构造（负数且沿 T 单调递减，满足 README g 约束）；
- `run_npu`：按 spec 把 `cu_seqlens` / `chunk_indices`（python list）真实传给
  `ascendc.prepare_wy_repr_bwd_da`（源仓固定 None）；
- CPU 标杆为 `test_da.py::compute_dA_cpu` 的完整移植（含 varlen 的
  `get_bos_eos` 切分语义：bos = cu[seq] + chunk*BT，eos 截断到段尾；
  causal mask 以段内 T 判定）。

## 校验

见仓库根 `_sanity_check_gen.py`（离线桩校验：30 个 profile 全量通过，
含 varlen B=1、段 chunk 对齐、chunk_indices 与 cu_seqlens 重放一致性、
HV % HK == 0、T % chunk_size == 0、V 合法性）。
