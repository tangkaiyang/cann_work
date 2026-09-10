"""ATK executor for prepare_wy_repr_bwd_da.

The CPU golden is embedded from fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/test_da.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _chunks, _finite_tuple


OP_NAME = "prepare_wy_repr_bwd_da"


def _optional(value):
    return None if value is None or (isinstance(value, str) and value == "null") else value


def _configure_cuda_control(device: torch.device) -> None:
    """Use IEEE FP32 matmul for the kernel's FP32 Cube accumulation model."""
    if device.type != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False


def _compute_da_golden(inputs: dict[str, Any], high_precision: bool) -> torch.Tensor:
    """Compute the FP64 truth or the kernel-faithful mixed-precision control.

    The AscendC kernel stores k*beta*exp(g), v*beta, and every Cube matmul
    result as kType before the next stage consumes them. Mirror those
    round-trips only in the control path; the FP64 benchmark remains an
    unquantized mathematical truth.
    """
    k, v = inputs["k"], inputs["v"]
    beta, A, g = inputs["beta"], inputs["a"], inputs["g"]
    dw, du = inputs["dw"], inputs["du"]
    B, HK, T, _ = k.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunkSize"])
    calc_dtype = torch.float64 if high_precision else torch.float32
    dA = torch.zeros(A.shape, dtype=calc_dtype, device=A.device)
    group_size = HV // HK

    supplied_offsets = inputs.get("cuSeqlensOptional")
    if supplied_offsets is None:
        ranges = [(b, 0, T) for b in range(B)]
    else:
        offsets = [int(value) for value in supplied_offsets]
        if B != 1 or offsets[0] != 0 or offsets[-1] > T:
            raise ValueError("cuSeqlensOptional is invalid")
        ranges = [(0, start, end) for start, end in zip(offsets, offsets[1:])]

    for b, sequence_start, sequence_end in ranges:
        for hv in range(HV):
            hk = hv // group_size
            for local_start, local_end in _chunks(
                sequence_end - sequence_start, chunk_size
            ):
                start = sequence_start + local_start
                end = sequence_start + local_end
                length = end - start
                a_chunk = A[b, hv, start:end, :length].to(calc_dtype)
                dw_chunk = dw[b, hv, start:end].to(calc_dtype)
                du_chunk = du[b, hv, start:end].to(calc_dtype)
                k_chunk = k[b, hk, start:end].to(calc_dtype)
                v_chunk = v[b, hv, start:end].to(calc_dtype)
                beta_chunk = beta[b, hv, start:end].to(calc_dtype)
                g_chunk = g[b, hv, start:end].to(calc_dtype)

                causal = torch.tril(
                    torch.ones((length, length), dtype=torch.bool, device=A.device),
                    diagonal=-1,
                )
                if high_precision:
                    k_beta_g = k_chunk * (
                        beta_chunk * torch.exp(g_chunk)
                    ).unsqueeze(-1)
                    v_beta = v_chunk * beta_chunk.unsqueeze(-1)
                    raw = torch.matmul(
                        dw_chunk, k_beta_g.T
                    ) + torch.matmul(du_chunk, v_beta.T)
                    masked = torch.where(causal, raw, torch.zeros_like(raw))
                    transformed = torch.matmul(
                        a_chunk.T, torch.matmul(masked, a_chunk.T)
                    )
                    gate_ratio = torch.exp(
                        g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0)
                    )
                    result = torch.where(
                        causal,
                        -transformed * gate_ratio,
                        torch.zeros_like(transformed),
                    )
                    dA[b, hv, start:end, :length] = result.T
                    continue

                def quantize(value: torch.Tensor) -> torch.Tensor:
                    return value.to(A.dtype).to(torch.float32)

                k_beta_g = quantize(
                    k_chunk * (beta_chunk * torch.exp(g_chunk)).unsqueeze(-1)
                )
                v_beta = quantize(v_chunk * beta_chunk.unsqueeze(-1))
                dA1 = quantize(torch.matmul(dw_chunk, k_beta_g.T))
                dA2 = quantize(torch.matmul(du_chunk, v_beta.T))
                dA4 = quantize(
                    torch.where(causal, dA1 + dA2, torch.zeros_like(dA1))
                )
                dA5 = quantize(torch.matmul(dA4, a_chunk.T))
                dA6 = quantize(torch.matmul(dA5.T, a_chunk))

                output_causal = causal.T
                gate_delta = g_chunk.unsqueeze(0) - g_chunk.unsqueeze(1)
                gate_ratio = torch.exp(
                    torch.minimum(gate_delta, torch.zeros_like(gate_delta))
                )
                result = quantize(
                    torch.where(
                        output_causal,
                        -dA6 * gate_ratio,
                        torch.zeros_like(dA6),
                    )
                )
                dA[b, hv, start:end, :length] = result

    return dA if high_precision else dA.to(A.dtype)


def run_torch_reference(inputs: dict[str, Any], high_precision: bool = False):
    """Run the FP64 truth or same-precision Torch control on CPU/CUDA."""
    _configure_cuda_control(inputs["k"].device)
    return _compute_da_golden(inputs, high_precision)


def run_cpu(inputs: dict[str, Any], high_precision: bool = False):
    return run_torch_reference(inputs, high_precision)


def run_npu(inputs: dict[str, Any]):
    from fla_npu.ops import ascendc

    return ascendc.prepare_wy_repr_bwd_da(
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["a"],
        inputs["dw"],
        inputs["du"],
        inputs["g"],
        chunk_size=inputs["chunkSize"],
        cu_seqlens=inputs["cuSeqlensOptional"],
        chunk_indices=inputs["chunkIndicesOptional"],
    )


@register("executor_prepare_wy_repr_bwd_da")
class FunctionApi(BaseApi):
    """ATK execution entry."""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = (
            self.device in {"cpu", "gpu"} and self.is_benchmark_task
        )
        self.gpu_control = self.device == "gpu" and not self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        inputs = {
            name: _optional(input_data.kwargs.get(name))
            for name in (
                "k", "v", "beta", "a", "dw", "du", "g",
                "cuSeqlensOptional", "chunkIndicesOptional", "chunkSize",
            )
        }
        if any(inputs[name] is None for name in ("k", "v", "beta", "a", "dw", "du", "g", "chunkSize")):
            raise ValueError("missing required PrepareWyReprBwdDa input")
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(inputs)
        elif self.device in {"cpu", "gpu"}:
            outputs = run_torch_reference(
                inputs,
                self.high_precision,
            )
        else:
            raise RuntimeError(
                f"{OP_NAME} requires an NPU DUT and a CPU or GPU reference node"
            )
        return _finite_tuple(outputs)
