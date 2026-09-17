#!/bin/bash
# Confirm torch_npu import behavior is identical in BOTH envs (env issue, not clone issue)
set +e
echo "=== source env atk_fla: torch import ==="
/root/miniconda3/envs/atk_fla/bin/python -c "import torch; print('torch OK', torch.__version__)" 2>&1 | tail -1
echo "=== clone env atk_fla_630: torch import ==="
/root/miniconda3/envs/atk_fla_630/bin/python -c "import torch; print('torch OK', torch.__version__)" 2>&1 | tail -1
echo "=== clone env with CANN env sourced ==="
ls /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && \
bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && /root/miniconda3/envs/atk_fla_630/bin/python -c 'import torch; import torch_npu; print(\"torch+npu OK\", torch.__version__)'" 2>&1 | tail -1
