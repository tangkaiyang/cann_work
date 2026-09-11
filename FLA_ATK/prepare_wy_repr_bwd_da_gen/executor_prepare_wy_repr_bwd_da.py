"""ATK executor for prepare_wy_repr_bwd_da（补充泛化版）.

相对源仓 executor_prepare_wy_repr_bwd_da.py 的扩展：
- build_inputs 按 case_spec 构造定长与 varlen 两种模式：
  * fixed  : B>1（或 B=1），cu_seqlens/chunk_indices 均为 None；
  * varlen : B=1，按 spec.cu_seqlens / spec.chunk_indices 传入（NPU wrapper
    要求 python list；CPU 标杆按同一切分逻辑计算）；
- GVA：k 为 [B, HK, T, K]，其余张量为 [B, HV, T, ...]，group_size = HV // HK；
- g 满足 README 约束：负数且沿 T 单调递减（-cumsum 构造）；
- V 支持 128/256，K 固定 128，chunk_size 64；
- CPU 标杆为 test_da.py::compute_dA_cpu 的移植，支持 varlen 切分
  （get_bos_eos 语义：bos = cu[seq]+chunk*BT，eos 截断到段尾）。

The CPU golden is embedded from fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/test_da.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
    _rand,
)


OP_NAME = "prepare_wy_repr_bwd_da"


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    data_dtype = _orig_dtype(dtype_name)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    is_varlen = bool(spec.get("varlen", False))
    cu_seqlens: Optional[list[int]] = None
    chunk_indices: Optional[list[int]] = None
    if is_varlen:
        cu_seqlens = [int(x) for x in spec["cu_seqlens"]]
        chunk_indices = [int(x) for x in spec["chunk_indices"]]
        # varlen 模式 B 必须为 1，packed 总 token 数取 cu_seqlens 末项
        B = 1
        T = cu_seqlens[-1]

    inputs = {
        "k": _rand((B, HK, T, K), dtype_name, data_dtype, device, seed + 1, 0.0, 1.0),
        "v": _rand((B, HV, T, V), dtype_name, data_dtype, device, seed + 2, 0.0, 1.0),
        "beta": _rand((B, HV, T), "fp32", torch.float32, device, seed + 3, 0.0, 1.0),
        "A": _rand((B, HV, T, chunk_size), dtype_name, data_dtype, device, seed + 4, 0.0, 1.0),
        # g 约束：负数且沿 T 单调递减
        "g": -torch.arange(1, B * HV * T + 1, dtype=torch.float32)
        .reshape(B, HV, T)
        .to(device),
        "dw": _rand((B, HV, T, K), dtype_name, data_dtype, device, seed + 6, 0.0, 1.0),
        "du": _rand((B, HV, T, V), dtype_name, data_dtype, device, seed + 7, 0.0, 1.0),
        "chunk_size": chunk_size,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }
    return inputs


def _get_bos_eos(idx: int, T: int, chunk_size: int,
                 cu_seqlens: Optional[Sequence[int]],
                 chunk_indices: Optional[Sequence[int]]) -> tuple[int, int]:
    """与 test_da.py::get_bos_eos 一致。"""
    if cu_seqlens is not None:
        seq_idx = int(chunk_indices[idx * 2])
        chunk_idx = int(chunk_indices[idx * 2 + 1])
        bos = int(cu_seqlens[seq_idx]) + chunk_idx * chunk_size
        eos = bos + chunk_size
        if eos > int(cu_seqlens[seq_idx + 1]):
            eos = int(cu_seqlens[seq_idx + 1])
    else:
        bos = idx * chunk_size
        eos = min(bos + chunk_size, T)
    return bos, eos


def _compute_da_golden(inputs: dict[str, Any], high_precision: bool) -> torch.Tensor:
    """Port of test_da.py::compute_dA_cpu (支持 GVA 与 varlen)."""
    k, v = inputs["k"], inputs["v"]
    beta, A, g = inputs["beta"], inputs["A"], inputs["g"]
    dw, du = inputs["dw"], inputs["du"]
    B, HK, T, _ = k.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    cu_seqlens = inputs.get("cu_seqlens")
    chunk_indices = inputs.get("chunk_indices")
    if cu_seqlens is not None:
        num_chunks = len(chunk_indices) // 2
    else:
        num_chunks = (T + chunk_size - 1) // chunk_size
    calc_dtype = torch.float64 if high_precision else torch.float32
    dA = torch.zeros(A.shape, dtype=calc_dtype, device=A.device)
    group_size = HV // HK
    is_varlen = cu_seqlens is not None

    for idx in range(num_chunks):
        bos, eos = _get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        chunk_len = eos - bos
        if is_varlen:
            seq_idx = int(chunk_indices[idx * 2])
            chunk_idx = int(chunk_indices[idx * 2 + 1])
            i_t = chunk_idx
            seq_len = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        else:
            i_t = idx
            seq_len = T

        o_t = i_t * chunk_size + torch.arange(0, chunk_size, dtype=torch.int32)
        m_t = o_t < seq_len
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

        for b in range(B):
            for hv in range(HV):
                hk = hv // group_size
                dw_chunk = dw[b, hv, bos:eos, :].to(calc_dtype)
                k_chunk = k[b, hk, bos:eos, :].to(calc_dtype)
                beta_chunk = beta[b, hv, bos:eos].to(calc_dtype)
                g_chunk = g[b, hv, bos:eos].to(calc_dtype)
                du_chunk = du[b, hv, bos:eos, :].to(calc_dtype)
                v_chunk = v[b, hv, bos:eos, :].to(calc_dtype)
                a_chunk = A[b, hv, bos:eos, :chunk_len].to(calc_dtype)

                g_exp_chunk = torch.exp(g_chunk)
                b_k_beta_g = k_chunk * (beta_chunk * g_exp_chunk).unsqueeze(-1)
                b_v_beta = v_chunk * beta_chunk.unsqueeze(-1)
                b_dA_1 = torch.matmul(dw_chunk, b_k_beta_g.T)
                b_dA_2 = torch.matmul(du_chunk, b_v_beta.T)
                b_dA_3 = b_dA_1 + b_dA_2
                m_a_c = m_A[:chunk_len, :chunk_len]
                b_dA_4 = torch.where(m_a_c, b_dA_3, torch.zeros_like(b_dA_3))
                b_dA_5 = torch.matmul(b_dA_4, a_chunk.T)
                b_dA_6 = torch.matmul(a_chunk.T, b_dA_5)
                b_g_sub_exp = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                b_dA_7 = -b_dA_6 * b_g_sub_exp
                b_dA = torch.where(m_a_c, b_dA_7, torch.zeros_like(b_dA_7))
                dA[b, hv, bos:eos, :chunk_len] = b_dA.T

    return dA if high_precision else dA.to(A.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    return _compute_da_golden(build_inputs(spec, torch.device("cpu")), high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))

    # NPU wrapper 要求 cu_seqlens / chunk_indices 传 python list（或 None）
    return ascendc.prepare_wy_repr_bwd_da(
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["A"],
        inputs["dw"],
        inputs["du"],
        inputs["g"],
        chunk_size=inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
    )


@register("executor_prepare_wy_repr_bwd_da")
class FunctionApi(BaseApi):
    """ATK execution entry."""

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
            raise RuntimeError(f"{OP_NAME} supports only NPU DUT and CPU benchmark nodes")
        return _finite_tuple(outputs, golden=self.device == "cpu")
