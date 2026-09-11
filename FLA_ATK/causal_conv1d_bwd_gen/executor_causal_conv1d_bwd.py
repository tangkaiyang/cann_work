"""causal_conv1d_bwd 的 ATK executor（补充泛化版）。

相对源仓 executor_causal_conv1d_bwd.py 的扩展：
- run_npu / run_cpu 按 case_spec 的 layout 传 input_layout（BSND/BSH/BNSD/TND/NTD）；
- activation=1/2 时，构造 yOptional（前向预激活输出）并传入算子与 CPU 标杆；
- with_state=True 时，构造 initial_state / dht（[B, W, D]），CPU 标杆返回 dh0；
- varlen（TND/NTD）时按 case_spec.query_start_loc 切段，CPU 标杆逐段复用定长参考；
- 固定长度 BSH/BNSD 输入由逻辑 [B, T, D] 张量 reshape/permute 得到。

CPU 标杆保持与源仓一致的显式左填充卷积实现；varlen 通过逐段调用该实现并
叠加 reduction（dw/db 跨段累加）完成。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "causal_conv1d_bwd"
FIXED_LAYOUTS = ("BSND", "BSH", "BNSD")
VARLEN_LAYOUTS = ("TND", "NTD")


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """按 spec 构造逻辑 [B, T, D] 视角的确定性输入（CPU 视角，供标杆使用）。"""
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, T, D, W = (int(spec[x]) for x in ("B", "T", "D", "W"))
    layout = str(spec.get("layout", "BSND"))
    activation = int(spec.get("activation", 0))
    with_state = bool(spec.get("with_state", False))
    qsl = spec.get("query_start_loc")

    total_tokens = T if qsl is None else int(qsl[-1])
    x = _randn((B, T, D), dtype_name, calc_dtype, device, seed + 1)
    weight = _randn((W, D), dtype_name, calc_dtype, device, seed + 2)
    dy = _randn((B, T, D), dtype_name, calc_dtype, device, seed + 3)
    inputs: dict[str, Any] = {
        "x": x,
        "weight": weight,
        "dy": dy,
        "layout": layout,
        "activation": activation,
        "initial_state": None,
        "dht": None,
        "y": None,
        "query_start_loc": list(int(v) for v in qsl) if qsl is not None else None,
    }
    if with_state:
        inputs["initial_state"] = _randn((B, W, D), dtype_name, calc_dtype, device, seed + 4, scale=0.2)
        inputs["dht"] = _randn((B, W, D), dtype_name, calc_dtype, device, seed + 5, scale=0.2)
    if activation in (1, 2):
        # yOptional：用 FP32 前向预激活参考并量化回原始 dtype
        y = _forward_preactivation(x, weight, inputs["initial_state"], dtype_name, device)
        inputs["y"] = y
    del total_tokens
    return inputs


def _forward_preactivation(x: torch.Tensor, weight: torch.Tensor, initial_state: Optional[torch.Tensor],
                           dtype_name: str, device: torch.device) -> torch.Tensor:
    """FP32 前向预激活（显式左填充 + 历史状态），量化回原始精度。"""
    calc = torch.float32
    B, T, D = x.shape
    W = weight.shape[0]
    x_c = x.detach().to(calc)
    w_c = weight.detach().to(calc)
    if initial_state is None:
        h0 = torch.zeros((B, W, D), dtype=calc, device=device)
    else:
        h0 = initial_state.detach().to(calc)
    hist = h0[:, -(W - 1):, :] if W > 1 else torch.zeros((B, 0, D), dtype=calc, device=device)
    x_pad = torch.cat([hist, x_c], dim=1) if W > 1 else x_c
    y = F.conv1d(
        x_pad.permute(0, 2, 1).contiguous(),
        w_c.t().unsqueeze(1).contiguous(),
        bias=None,
        padding=0,
        groups=D,
    ).permute(0, 2, 1).contiguous()[:, :T, :]
    return y.to(_orig_dtype(dtype_name))


def _to_layout(x: torch.Tensor, layout: str):
    """把逻辑 [B, T, D] 张量转成目标 layout 视角的输入。"""
    if layout == "BSND":
        return x
    if layout == "BSH":
        return x.reshape(x.shape[0], x.shape[1], -1)
    if layout == "BNSD":
        n_heads = 2
        if x.shape[-1] % n_heads != 0:
            n_heads = 1
        B, T, D = x.shape
        return x.reshape(B, T, n_heads, D // n_heads).permute(0, 2, 1, 3).contiguous()
    if layout == "TND":
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    if layout == "NTD":
        n_heads = 2
        if x.shape[-1] % n_heads != 0:
            n_heads = 1
        B, T, D = x.shape
        return x.reshape(B, T, n_heads, D // n_heads).permute(1, 0, 2, 3).contiguous().reshape(T, n_heads, D // n_heads).contiguous()
    raise ValueError(f"unsupported layout: {layout!r}")


def _activation_bwd_grad(dy, y, activation: int):
    """把上游梯度 dy 转成预激活位置的有效梯度 g。"""
    if activation == 0:
        return dy
    if activation in (1, 2):
        if y is None:
            raise ValueError("activation 为 1/2 时必须提供 yOptional。")
        sig = torch.sigmoid(y)
        return dy * sig * (1.0 + y * (1.0 - sig))
    raise ValueError(f"activation 仅支持 0/1/2，当前为 {activation!r}")


def _dense_bwd_ref(x, weight, dy, y_optional, initial_state, dht, activation):
    """单条定长序列的显式左填充反传参考（FP32 计算）。"""
    calc = torch.float32
    B, T, D = x.shape
    W = weight.shape[0]

    x_c = x.detach().clone().to(calc).requires_grad_(True)
    w_c = weight.detach().clone().to(calc).requires_grad_(True)
    dy_c = dy.detach().to(calc)

    if initial_state is None:
        h0 = torch.zeros((B, W, D), dtype=calc, device=x.device)
    else:
        h0 = initial_state.detach().clone().to(calc)
    h0.requires_grad_(True)

    hist = h0[:, -(W - 1):, :] if W > 1 else torch.zeros((B, 0, D), dtype=calc, device=x.device)
    x_pad = torch.cat([hist, x_c], dim=1) if W > 1 else x_c

    y = F.conv1d(
        x_pad.permute(0, 2, 1).contiguous(),
        w_c.t().unsqueeze(1).contiguous(),
        bias=None,
        padding=0,
        groups=D,
    ).permute(0, 2, 1).contiguous()[:, :T, :]

    g = _activation_bwd_grad(dy_c, y_optional.detach().to(calc) if y_optional is not None else None, activation)
    y.backward(g, retain_graph=dht is not None)

    dx = x_c.grad
    dw = w_c.grad
    db = g.sum(dim=(0, 1))
    dh0 = h0.grad if (h0.grad is not None and initial_state is not None) else torch.zeros((B, W, D), dtype=calc, device=x.device)

    if dht is not None:
        dht_c = dht.detach().to(calc)
        tail = min(W, T)
        dx = dx.clone()
        dx[:, T - tail:, :] += dht_c[:, W - tail:, :]

    return dx, dw, db, dh0


def _causal_conv1d_bwd_ref(
    inputs: dict[str, Any],
    y_optional=None,
    initial_state=None,
    dht=None,
    activation: int = 0,
    input_layout: str = "BSND",
    query_start_loc=None,
):
    """CPU 标杆：定长布局直接反传；varlen 布局逐段反传并叠加 reduction。"""
    if input_layout not in {"BSND", "BSH", "BNSD", "TND", "NTD"}:
        raise NotImplementedError(f"CPU 标杆不支持 layout={input_layout!r}")

    x_in, w_in, dy_in = inputs["x"], inputs["weight"], inputs["dy"]
    calc = torch.float32
    B, T, D = x_in.shape
    W = w_in.shape[0]

    if query_start_loc is None:
        dx, dw, db, dh0 = _dense_bwd_ref(x_in, w_in, dy_in, y_optional, initial_state, dht, activation)
        return dx.to(x_in.dtype), dw.to(w_in.dtype), db.to(dy_in.dtype), dh0.to(x_in.dtype)

    # varlen：按 query_start_loc 逐段处理；逻辑输入按 [T, D] 展开（B 维即段数）
    x_tok = x_in.reshape(B * T, D)
    dy_tok = dy_in.reshape(B * T, D)
    y_tok = y_optional.reshape(B * T, D) if y_optional is not None else None
    dx_tok = torch.zeros((B * T, D), dtype=calc, device=x_in.device)
    dw_acc = torch.zeros((W, D), dtype=calc, device=x_in.device)
    db_acc = torch.zeros((D,), dtype=calc, device=x_in.device)
    dh0_out = torch.zeros((B, W, D), dtype=calc, device=x_in.device)
    for b in range(B):
        start, end = int(query_start_loc[b]), int(query_start_loc[b + 1])
        seq_len = end - start
        x_seq = x_tok[start:end].unsqueeze(0)
        dy_seq = dy_tok[start:end].unsqueeze(0)
        y_seq = y_tok[start:end].unsqueeze(0) if y_tok is not None else None
        st_seq = initial_state[b:b + 1] if initial_state is not None else None
        dht_seq = dht[b:b + 1] if dht is not None else None
        dx_seq, dw_seq, db_seq, dh0_seq = _dense_bwd_ref(x_seq, w_in, dy_seq, y_seq, st_seq, dht_seq, activation)
        dx_tok[start:end] = dx_seq
        dw_acc += dw_seq
        db_acc += db_seq
        dh0_out[b] = dh0_seq[0]
    return dx_tok.to(x_in.dtype), dw_acc.to(w_in.dtype), db_acc.to(dy_in.dtype), dh0_out.to(x_in.dtype)


def _pack_outputs(result, dtype_name: str):
    dx, dw, db, dh0 = result
    return dx, dw, db, dh0


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    del high_precision  # 标杆统一 FP32 计算，与源仓实现一致
    inputs = build_inputs(spec, torch.device("cpu"))
    return _causal_conv1d_bwd_ref(
        inputs,
        y_optional=inputs["y"],
        initial_state=inputs["initial_state"],
        dht=inputs["dht"],
        activation=int(inputs["activation"]),
        input_layout=str(inputs["layout"]),
        query_start_loc=inputs["query_start_loc"],
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    layout = str(inputs["layout"])
    activation = int(inputs["activation"])
    qsl = inputs["query_start_loc"]
    x_layout = _to_layout(inputs["x"], layout)
    dy_layout = _to_layout(inputs["dy"], layout)
    y_layout = _to_layout(inputs["y"], layout) if inputs["y"] is not None else None

    outputs = ascendc.causal_conv1d_bwd(
        x_layout,
        y_layout,
        inputs["weight"],
        dy_layout,
        initial_state=inputs["initial_state"],
        dht=inputs["dht"],
        query_start_loc=qsl,
        activation=activation,
        input_layout=layout,
    )
    return outputs[:3] if inputs["dht"] is None else tuple(outputs)


@register("executor_causal_conv1d_bwd")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
