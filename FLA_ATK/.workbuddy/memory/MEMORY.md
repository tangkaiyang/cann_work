# FLA_ATK 项目长期笔记

## 环境要点
- git 仓库根在父目录 `D:\t30072652\Project\.git`（工作区是子目录 FLA_ATK），远程 origin = github.com/tangkaiyang/cann_work。
- 网络：环境变量默认代理 127.0.0.1:52470 已失效（502）；**可用代理是 127.0.0.1:7897**。git push/fetch 需要显式加 `https_proxy=http://127.0.0.1:7897`，否则连不上 GitHub。
- PortableGit 环境 bug：push/fetch 成功后 `refs/remotes/origin/main`（在 packed-refs 里）可能不更新，导致 status 误报 ahead。修复：直接改 `D:\t30072652\Project\.git\packed-refs`，或 `git update-ref`（update-ref 有时不生效）。
- SSH：`ssh 246` = root@192.168.13.246（密钥 ~/.ssh/servers/246）。服务器 sshd 间歇性 reset（疑似 MaxStartups 打满，80 用户/负载 35+）；KEX 需避开 sntrup761x25519，config 里已固定 curve25519-sha256。失败时多重试几次。

## PyCharm
- 本机 PyCharm CE 2024.3.1：`C:\Users\admin\AppData\Local\Programs\PyCharm Community Edition\bin\pycharm64.exe`，配置目录 `%APPDATA%\JetBrains\PyCharmCE2024.3`。
- 项目根 `D:\t30072652\Project` 已配好：解释器 Python 3.13.14（workbuddy 管理版，jdk.table.xml 名 "Python 3.13 (workbuddy)"）、UTF-8、Git 映射、三个 gen_*.py 运行配置。
- 项目代码 import `atk.*`（华为 ATK 用例生成框架），仅存在于服务器环境，本机导入报 unresolved reference 属预期，不要尝试 pip 安装。

## 泛化归档目录约定（2026-09-11 起）
- 源仓 spec-driven 用例（D:\生态仓测试\flash-linear-attention-npu\tests\atk\<op>\）的泛化版本归档在 FLA_ATK\<op>_gen\（如 causal_conv1d_bwd_gen / recurrent_kda_gen / prepare_wy_repr_bwd_da_gen），每个含 yaml+gen+executor+README；FLA_ATK 下不带 _gen 的算子目录是用户自己的 aclnn 风格工作，**不要修改**。
- 部署方式：把 <op>_gen 下三个同名文件（yaml/gen/executor）拷入源仓 tests/atk/<op>/ 覆盖即可，executor 依赖 ../common/_ascendc_common_executor.py，目录层级不变。
- 离线校验：_sanity_check_gen.py（atk 桩，全量 profile 约束校验）与 _sanity_check_exec.py（torch 真跑 run_cpu），依赖 _stub_common\ 下的 torch 版公共桩；managed python 已装 numpy + torch-cpu 2.14（pip 走代理 7897，--index-url https://download.pytorch.org/whl/cpu）。
- 关键语义陷阱：prepare_wy_repr_bwd_da 的 CPU golden（test_da.py::compute_dA_cpu）把 b_dA.T 存入 dA，即存储块严格上三角非零、下三角含对角恒零；recurrent_kda BSND 多段时 packed 容量 B*T 必须被 cu_seqlens 末项用满；speculative 索引的 max_step 必须 >= 最大段长。
