# aclnnMlaPrologV4WeightNz (A5) 项目约定

## 权威版本 = git HEAD（93493de「异常用例」）
用户会主动 `git checkout . && git clean -fdx` 还原工作区。**上次还原是用户有意为之**
（2026-09-17 确认），上轮把 generator 整体重写的改动已被丢弃 —— 不要从会话记录里恢复。
改动前先 `git status`/`git log -1`，确认自己在改哪个基线。

HEAD 内容（5 个文件）：
- `aclnnMlaPrologV4WeightNz.md`：官方文档本地副本（91KB，与 raw gitcode 下载的一致）。
- `aclnnMlaPrologV4WeightNz_a5.yaml`：ATK 元数据，33 项 inputs（21 tensor + 12 attr）。
- `ascend_generate_aclnn_mla_prolog_v4_a5.py`：注册名 `ascend_generate_aclnn_mla_prolog_v4_a5`，
  唯一入口 `after_case_config`；类名 `MlaPrologV4A5Generator`。
- `test_generator.py`：离线契约测试（需要 PyYAML）。
- `README.md`：设计说明 + 适配边界。

## 验证
隔离 venv：`C:/Users/admin/.workbuddy/binaries/python/envs/default/Scripts/python.exe`
（managed python 本体无 PyYAML，已在该 venv 装好 pyyaml 6.0.3）。
跑法：`cd 本目录 && <venv>/Scripts/python.exe -B test_generator.py` → 2 tests OK（两 profile 各 5000 次生成）。

## 约定（当前基线写法）
- generator **无条件覆盖** yaml 的 `ranges/dtypes/shapes`；yaml 只声明取值域，两处要同步。
- 几何由 generator 自己抽：先定 `merged`（合轴），再 `leading = [t] if merged else [batch, seq]`，
  `token_x.shape = leading + [he]`。**token_x 的 yaml shape 目前不是真值来源**（用户提过要改成
  由 yaml rank 决定 T/He vs B/S/He，尚未落地）。
- `merged` 规则：TND 强制合轴；BSND 强制非合轴；其余 50/50 随机。
- 场景 (wq,kvq,qq) 走 `MLA_V4_PROFILE` 环境变量：`golden`(默认) wq≤3 共 10 种，`full` wq≤5 共 16 种。
  非法值抛 ValueError。
- kvq=3 只允许 PA_BSND/BSND/TND；此时 `ckvkrRepoMode=quantScaleRepoMode=1`，krCache 为 `[0]` bf16，
  kvCache 末维 Dtile=656（其余 512）。
- mxfp8(wq=3) 的 e8m0 scale 用 `uint8` 占位，plugin 在 gen_inputs 阶段改数值与 dtype。
- PA_BLK 合轴只生成单序列 `actualSeqLen=[T]`（多序列前缀和需数据钩子）。
- 输出 tensor 不在 inputs 里声明，由 executor/plugin 按 case_config 推导。
- 文档两处不一致（已记录在 README）：kvCache 末维 "shape约束" 写 Dr 而参数表写 Dtile（取 Dtile）；
  参数表有 `queryNormFlag` 但函数原型与 33 入参都没有（不擅自加）。
