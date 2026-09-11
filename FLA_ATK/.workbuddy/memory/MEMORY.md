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
