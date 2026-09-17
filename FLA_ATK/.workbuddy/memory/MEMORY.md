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

## 246 服务器 conda 环境（2026-09-15）
- conda 在 /root/miniconda3（非交互 shell 无 conda 命令，直接用绝对路径 /root/miniconda3/bin/conda；envs 在 /root/miniconda3/envs/，含 atk_fla、atk_fla_630 等）。
- **atk_fla_630**（2026-09-15 建）：`conda create -n atk_fla_630 --clone atk_fla`（Python 3.11.16, 2.8G, torch 2.12.0+cpu + triton_ascend 3.2.1），装了 flash-linear-attention-npu **26.6.0** 950.x86_64（源环境 atk_fla 是 26.7.0.dev0）。wheel 从 GitHub release v26.6.0 下载，本机走 7897 代理下载后 scp 上去（246 直连 GitHub 不通）。
- 两个环境裸 `import torch` 都报 torch_npu 后端加载错，是没 source CANN 的正常现象：先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 即可 torch+npu 正常。
- flash-linear-attention-npu 的顶层包是 `fla` 和 `fla_npu`（不是 flash_linear_attention_npu），fla_npu 导入前必须 source CANN set_env.sh。
- 246 下载 GitHub 受限；本机 PortableGit 偶发 ls/grep/sed 等基础命令丢失（PATH 异常），改用 PowerShell 或 python 单行命令兜底。
