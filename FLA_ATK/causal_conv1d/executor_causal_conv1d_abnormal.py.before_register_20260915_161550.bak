# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from __future__ import annotations

import ctypes
from typing import Iterable

import torch
import torch.nn.functional as F

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray, AclTensor

_ABSENT_INT_ARRAY_SENTINEL = -(1 << 60)

_gpu_triton_fns = None

torch.backends.mkldnn.enabled = False


def _get_gpu_triton_fns():
    global _gpu_triton_fns
    if _gpu_triton_fns is None:
        import importlib, os, sys  # noqa: E401
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        mod = importlib.import_module("gpu_triton_causal_conv1d")
        _gpu_triton_fns = (mod.causal_conv1d_fn, mod.causal_conv1d_update)
    return _gpu_triton_fns


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        if x.shape[-1] < width:
            x = F.pad(x, (width - x.shape[-1], 0))
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    _, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape[:2] == (x.shape[0], dim)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
        copy_idx = copy_idx + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_update_spec_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    activation: str | None = None,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if num_accepted_tokens is None:
        raise ValueError("num_accepted_tokens must be provided for spec decode golden")

    dtype_in = x.dtype
    x = x.to(weight.dtype)
    conv_state = conv_state.to(weight.dtype)
    if not isinstance(num_accepted_tokens, torch.Tensor):
        num_accepted_tokens = torch.tensor(num_accepted_tokens, dtype=torch.long, device=x.device)
    else:
        num_accepted_tokens = num_accepted_tokens.to(device=x.device, dtype=torch.long)

    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    keep = width - 2
    required_state_len = (width - 1) + (seqlen - 1)
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    assert state_len >= required_state_len

    out = torch.empty_like(x)
    for seq_idx in range(batch):
        offset = int(num_accepted_tokens[seq_idx].item()) - 1
        assert 0 <= offset <= seqlen - 1
        hist = conv_state[seq_idx : seq_idx + 1, :, offset : offset + width - 1]
        x_cat = torch.cat([hist, x[seq_idx : seq_idx + 1]], dim=-1)
        out_seq = F.conv1d(x_cat, weight.unsqueeze(1), bias, padding=0, groups=dim)[..., :seqlen]
        if activation is not None:
            out_seq = F.silu(out_seq)
        out[seq_idx : seq_idx + 1] = out_seq

        if keep > 0:
            conv_state[seq_idx, :, :keep] = conv_state[seq_idx, :, offset + 1 : offset + 1 + keep]
        conv_state[seq_idx, :, keep : keep + seqlen] = x[seq_idx]
    return out.to(dtype=dtype_in)


def activation_from_mode(mode: int) -> str | None:
    return None if mode == 0 else "silu"


def op_weight_to_ref(weight_op: torch.Tensor) -> torch.Tensor:
    return weight_op.detach().cpu().float().transpose(0, 1).contiguous()


def op_conv_states_to_ref(conv_states_op: torch.Tensor) -> torch.Tensor:
    return conv_states_op.detach().cpu().float().permute(0, 2, 1).contiguous()


def ref_conv_states_to_op(conv_states_ref: torch.Tensor) -> torch.Tensor:
    return conv_states_ref.permute(0, 2, 1).contiguous()


def op_batch_x_to_ref(x_op: torch.Tensor) -> torch.Tensor:
    return x_op.detach().cpu().float().permute(0, 2, 1).contiguous()


def _is_absent_tensor(value) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() == 0


def _optional_tensor(value):
    if value is None or _is_absent_tensor(value):
        return None
    return value


def _optional_int_list(value) -> list[int] | None:
    value = _optional_tensor(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        result = [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
        if result == [_ABSENT_INT_ARRAY_SENTINEL]:
            return None
        return result or None
    if isinstance(value, Iterable):
        result = [int(v) for v in value]
        if result == [_ABSENT_INT_ARRAY_SENTINEL]:
            return None
        return result or None
    return [int(value)]


def _normalize_cache_indices(
    cache_indices: list[int] | None,
    *,
    batch: int,
    num_cache_lines: int,
    pad_slot_id: int,
) -> list[int]:
    if cache_indices is None:
        if num_cache_lines < batch:
            raise ValueError(
                "cache_indices is absent, requires conv_states.shape[0] >= batch for identity mapping"
            )
        return list(range(batch))
    if len(cache_indices) != batch:
        raise ValueError("cache_indices size must equal batch")
    for index in cache_indices:
        if index == pad_slot_id:
            continue
        if index < 0 or index >= num_cache_lines:
            raise ValueError(
                f"cache_indices contains out-of-range value {index}, num_cache_lines={num_cache_lines}"
            )
    return cache_indices


def _normalize_initial_state_mode(initial_state_mode: list[int] | None, batch: int) -> list[int]:
    if initial_state_mode is None:
        return [0] * batch
    if len(initial_state_mode) != batch:
        raise ValueError("initial_state_mode size must equal batch")
    for value in initial_state_mode:
        if value not in (0, 1):
            raise ValueError(f"initial_state_mode only supports 0/1, got {value}")
    return initial_state_mode


def _normalize_num_accepted_tokens(num_accepted_tokens: list[int] | None, batch: int) -> list[int] | None:
    if num_accepted_tokens is None:
        return None
    if len(num_accepted_tokens) != batch:
        raise ValueError("num_accepted_tokens size must equal batch")
    return num_accepted_tokens


def _is_varlen_2d_input(
    x: torch.Tensor,
    query_start_loc: list[int] | None,
    run_mode: int,
) -> bool:
    return (
        x.dim() == 2
        and query_start_loc is not None
        and (run_mode == 0 or len(query_start_loc) - 1 != x.shape[0])
    )


def _to_int32_tensor(value: list[int] | None, device: str) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor(value, dtype=torch.int32, device=device)


def _to_bool_tensor(value: list[int] | None, device: str) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor(value, dtype=torch.bool, device=device)


def _validate_gpu_varlen_num_accepted_tokens(
    num_accepted_tokens: list[int] | None,
    *,
    segment_lengths: list[int],
) -> list[int] | None:
    if num_accepted_tokens is None:
        return None
    if all(token == 0 for token in num_accepted_tokens):
        return None
    if any(token <= 0 for token in num_accepted_tokens):
        raise ValueError(
            "GPU Triton varlen update path does not support mixed zero/non-zero num_accepted_tokens; "
            "use None for plain update or strictly positive values for every sequence"
        )
    for seq_idx, (token, segment_len) in enumerate(zip(num_accepted_tokens, segment_lengths)):
        if token > segment_len:
            raise ValueError(
                f"num_accepted_tokens[{seq_idx}]={token} exceeds varlen segment length {segment_len}"
            )
    return num_accepted_tokens


def _mask_pad_slot_y(
    y: torch.Tensor,
    *,
    x: torch.Tensor,
    query_start_loc: list[int] | None,
    cache_indices: list[int] | None,
    pad_slot_id: int,
    run_mode: int,
    head_num: int,
) -> torch.Tensor:
    if cache_indices is None:
        return y

    pad_seq_indices = [seq_idx for seq_idx, cache_idx in enumerate(cache_indices) if cache_idx == pad_slot_id]
    if not pad_seq_indices:
        return y

    y_masked = y.clone()
    if x.dim() == 3:
        for seq_idx in pad_seq_indices:
            y_masked[seq_idx].zero_()
        return y_masked

    is_varlen_2d = _is_varlen_2d_input(x, query_start_loc, run_mode)
    if is_varlen_2d:
        for seq_idx in pad_seq_indices:
            start = int(query_start_loc[seq_idx])
            end = int(query_start_loc[seq_idx + 1])
            if head_num > 0:
                y_masked[:, start:end].zero_()
            else:
                y_masked[start:end].zero_()
        return y_masked

    for seq_idx in pad_seq_indices:
        y_masked[seq_idx].zero_()
    return y_masked


def _finalize_backend_outputs(
    y: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    x: torch.Tensor,
    query_start_loc: list[int] | None,
    cache_indices: list[int] | None,
    pad_slot_id: int,
    run_mode: int,
    head_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _mask_pad_slot_y(
            y,
            x=x,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            pad_slot_id=pad_slot_id,
            run_mode=run_mode,
            head_num=head_num,
        ),
        conv_states,
    )


def _to_head_major_output(y: torch.Tensor, *, head_num: int, run_mode: int) -> torch.Tensor:
    if head_num == 0:
        return y
    if head_num < 0:
        raise ValueError(f"head_num must be >= 0, got {head_num}")
    if run_mode != 0:
        raise ValueError("head_num > 0 is only supported when run_mode=0")
    dim = y.shape[-1]
    if dim % head_num != 0:
        raise ValueError(f"head_num must divide dim exactly, got head_num={head_num}, dim={dim}")
    head_dim = dim // head_num
    if head_dim % 16 != 0:
        raise ValueError(f"head_dim must be a multiple of 16, got head_dim={head_dim}")
    if y.dim() == 2:
        # TH -> NTD
        return y.reshape(y.shape[0], head_num, head_dim).permute(1, 0, 2).contiguous()
    if y.dim() == 3:
        # BSH -> BNSD
        return y.reshape(y.shape[0], y.shape[1], head_num, head_dim).permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"head-major output expects rank-2/rank-3 y, got rank={y.dim()}")


def _finalize_outputs(y: torch.Tensor, conv_states_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return y, ref_conv_states_to_op(conv_states_ref).to(dtype=y.dtype)


def _run_prefill_batch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cache_indices: list[int] | None,
    initial_state_mode: list[int] | None,
    activation_mode: int,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    conv_states_ref = op_conv_states_to_ref(conv_states.clone())
    weight_ref = op_weight_to_ref(weight)
    bias_ref = None if bias is None else bias.detach().cpu().float()
    x_ref = op_batch_x_to_ref(x)
    cache_indices = _normalize_cache_indices(
        cache_indices,
        batch=batch,
        num_cache_lines=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
    )
    initial_state_mode = _normalize_initial_state_mode(initial_state_mode, batch)

    y = torch.zeros_like(x)
    for seq_idx, cache_idx in enumerate(cache_indices):
        if cache_idx == pad_slot_id:
            continue
        initial_state = None
        if initial_state_mode[seq_idx] == 1:
            initial_state = conv_states_ref[cache_idx : cache_idx + 1]
        y_seq, _ = causal_conv1d_ref(
            x_ref[seq_idx : seq_idx + 1],
            weight_ref,
            bias=bias_ref,
            initial_states=initial_state,
            return_final_states=True,
            final_states_out=conv_states_ref[cache_idx : cache_idx + 1],
            activation=activation_from_mode(activation_mode),
        )
        y[seq_idx : seq_idx + 1] = y_seq.permute(0, 2, 1).contiguous().to(dtype=x.dtype)
    return _finalize_outputs(y, conv_states_ref)


def _run_prefill_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: list[int],
    cache_indices: list[int] | None,
    initial_state_mode: list[int] | None,
    activation_mode: int,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(query_start_loc) - 1
    conv_states_ref = op_conv_states_to_ref(conv_states.clone())
    weight_ref = op_weight_to_ref(weight)
    bias_ref = None if bias is None else bias.detach().cpu().float()
    x_ref = x.detach().cpu().float()
    cache_indices = _normalize_cache_indices(
        cache_indices,
        batch=batch,
        num_cache_lines=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
    )
    initial_state_mode = _normalize_initial_state_mode(initial_state_mode, batch)

    y = torch.zeros_like(x)
    for seq_idx, cache_idx in enumerate(cache_indices):
        start = query_start_loc[seq_idx]
        end = query_start_loc[seq_idx + 1]
        if cache_idx == pad_slot_id:
            continue
        x_seq = x_ref[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        initial_state = None
        if initial_state_mode[seq_idx] == 1:
            initial_state = conv_states_ref[cache_idx : cache_idx + 1]
        y_seq, _ = causal_conv1d_ref(
            x_seq,
            weight_ref,
            bias=bias_ref,
            initial_states=initial_state,
            return_final_states=True,
            final_states_out=conv_states_ref[cache_idx : cache_idx + 1],
            activation=activation_from_mode(activation_mode),
        )
        y[start:end] = y_seq.squeeze(0).transpose(0, 1).contiguous().to(dtype=x.dtype)
    return _finalize_outputs(y, conv_states_ref)


def _run_update_decode_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cache_indices: list[int] | None,
    num_accepted_tokens: list[int] | None,
    activation_mode: int,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    conv_states_ref = op_conv_states_to_ref(conv_states.clone())
    weight_ref = op_weight_to_ref(weight)
    bias_ref = None if bias is None else bias.detach().cpu().float()
    x_ref = x.detach().cpu().float()
    cache_indices = _normalize_cache_indices(
        cache_indices,
        batch=batch,
        num_cache_lines=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
    )
    num_accepted_tokens = _normalize_num_accepted_tokens(num_accepted_tokens, batch)

    y = torch.zeros_like(x)
    for seq_idx, cache_idx in enumerate(cache_indices):
        if cache_idx == pad_slot_id:
            continue
        x_seq = x_ref[seq_idx : seq_idx + 1]
        if num_accepted_tokens is None or num_accepted_tokens[seq_idx] == 0:
            y_seq = causal_conv1d_update_ref(
                x_seq,
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                activation=activation_from_mode(activation_mode),
                cache_seqlens=None,
            )
        else:
            y_seq = causal_conv1d_update_spec_ref(
                x_seq.unsqueeze(-1),
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                num_accepted_tokens=[num_accepted_tokens[seq_idx]],
                activation=activation_from_mode(activation_mode),
            ).squeeze(-1)
        y[seq_idx : seq_idx + 1] = y_seq.to(dtype=x.dtype)
    return _finalize_outputs(y, conv_states_ref)


def _run_update_varlen_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: list[int],
    cache_indices: list[int] | None,
    num_accepted_tokens: list[int] | None,
    activation_mode: int,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(query_start_loc) - 1
    conv_states_ref = op_conv_states_to_ref(conv_states.clone())
    weight_ref = op_weight_to_ref(weight)
    bias_ref = None if bias is None else bias.detach().cpu().float()
    x_ref = x.detach().cpu().float()
    cache_indices = _normalize_cache_indices(
        cache_indices,
        batch=batch,
        num_cache_lines=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
    )
    num_accepted_tokens = _normalize_num_accepted_tokens(num_accepted_tokens, batch)

    y = torch.zeros_like(x)
    for seq_idx, cache_idx in enumerate(cache_indices):
        start = query_start_loc[seq_idx]
        end = query_start_loc[seq_idx + 1]
        if cache_idx == pad_slot_id:
            continue
        x_seq = x_ref[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        if num_accepted_tokens is None or num_accepted_tokens[seq_idx] == 0:
            y_seq = causal_conv1d_update_ref(
                x_seq,
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                activation=activation_from_mode(activation_mode),
                cache_seqlens=None,
            )
        else:
            y_seq = causal_conv1d_update_spec_ref(
                x_seq,
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                num_accepted_tokens=[num_accepted_tokens[seq_idx]],
                activation=activation_from_mode(activation_mode),
            )
        y[start:end] = y_seq.squeeze(0).transpose(0, 1).contiguous().to(dtype=x.dtype)
    return _finalize_outputs(y, conv_states_ref)


def _run_update_batch_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    cache_indices: list[int] | None,
    num_accepted_tokens: list[int] | None,
    activation_mode: int,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    conv_states_ref = op_conv_states_to_ref(conv_states.clone())
    weight_ref = op_weight_to_ref(weight)
    bias_ref = None if bias is None else bias.detach().cpu().float()
    x_ref = op_batch_x_to_ref(x)
    cache_indices = _normalize_cache_indices(
        cache_indices,
        batch=batch,
        num_cache_lines=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
    )
    num_accepted_tokens = _normalize_num_accepted_tokens(num_accepted_tokens, batch)

    y = torch.zeros_like(x)
    for seq_idx, cache_idx in enumerate(cache_indices):
        if cache_idx == pad_slot_id:
            continue
        x_seq = x_ref[seq_idx : seq_idx + 1]
        if num_accepted_tokens is None or num_accepted_tokens[seq_idx] == 0:
            y_seq = causal_conv1d_update_ref(
                x_seq,
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                activation=activation_from_mode(activation_mode),
                cache_seqlens=None,
            )
        else:
            y_seq = causal_conv1d_update_spec_ref(
                x_seq,
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                num_accepted_tokens=[num_accepted_tokens[seq_idx]],
                activation=activation_from_mode(activation_mode),
            )
        y[seq_idx : seq_idx + 1] = y_seq.permute(0, 2, 1).contiguous().to(dtype=x.dtype)
    return _finalize_outputs(y, conv_states_ref)


def run_causal_conv1d_gpu(
    input_data: InputDataset, device_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal_conv1d on GPU using Triton kernels extracted from vLLM.

    The GPU path must preserve the same six-scenario dispatch contract as the
    CPU reference path. In particular:
    - Speculative varlen update forwards `query_start_loc` and `max_query_len`
      exactly as vLLM does. Plain varlen update is split into non-varlen calls,
      because vLLM's varlen state-length adjustment assumes speculative state.
    - Any branch that may receive `cache_indices == pad_slot_id` must also pass
      `null_block_id=pad_slot_id`, otherwise Triton may treat the pad sentinel
      as a real cache-line index.
    """
    gpu_causal_conv1d_fn, gpu_causal_conv1d_update = _get_gpu_triton_fns()
    device = f"cuda:{device_id}"

    x = input_data.kwargs["x"]
    weight = input_data.kwargs["weight"]
    bias = _optional_tensor(input_data.kwargs["bias"])
    conv_states = input_data.kwargs["conv_states"]
    query_start_loc = _optional_int_list(input_data.kwargs["query_start_loc_cpu"])
    cache_indices = _optional_int_list(input_data.kwargs["cache_indices_cpu"])
    initial_state_mode = _optional_int_list(input_data.kwargs["has_initial_state_cpu"])
    num_accepted_tokens = _optional_int_list(input_data.kwargs["num_accepted_tokens_cpu"])
    activation = str(input_data.kwargs["activation"])
    activation_mode = 0 if activation == "none" else 1
    pad_slot_id = -1
    run_mode = int(input_data.kwargs["run_mode"])
    head_num = int(input_data.kwargs.get("head_num", 0))

    activation = activation_from_mode(activation_mode)
    is_varlen_2d = _is_varlen_2d_input(x, query_start_loc, run_mode)

    # ATK op layout -> GPU (Triton) layout
    # weight: (width, dim) -> (dim, width)
    weight_gpu = weight.to(device).transpose(0, 1).contiguous()
    # conv_states: (N, state_len, dim) -> (N, dim, state_len)
    conv_states_gpu = conv_states.clone().to(device).permute(0, 2, 1).contiguous()
    bias_gpu = bias.to(device) if bias is not None else None

    if run_mode == 0:
        # ---- Prefill ----
        if x.dim() == 3:
            batch, seqlen, dim = x.shape
            # Flatten 3D batch to varlen 2D: (batch*seqlen, dim)
            # Then .T for channel-last: (dim, batch*seqlen) with stride(0)==1
            x_gpu = x.reshape(-1, dim).to(device).T
            qsl_gpu = torch.arange(
                0, batch * seqlen + 1, seqlen, dtype=torch.int32, device=device
            )
            ci_gpu = _to_int32_tensor(cache_indices, device)
            his_gpu = _to_bool_tensor(initial_state_mode, device)
            out_gpu = gpu_causal_conv1d_fn(
                x_gpu, weight_gpu, bias_gpu,
                conv_states=conv_states_gpu,
                query_start_loc=qsl_gpu,
                cache_indices=ci_gpu,
                has_initial_state=his_gpu,
                activation=activation,
                pad_slot_id=pad_slot_id,
                null_block_id=pad_slot_id,
            )
            # (dim, batch*seqlen) -> (batch, seqlen, dim)
            y = out_gpu.T.reshape(batch, seqlen, dim).contiguous().cpu().to(x.dtype)
        else:
            # Varlen prefill: x is (total_tokens, dim)
            # .T for channel-last: (dim, total_tokens) with stride(0)==1
            x_gpu = x.to(device).T
            qsl_gpu = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
            ci_gpu = _to_int32_tensor(cache_indices, device)
            his_gpu = _to_bool_tensor(initial_state_mode, device)
            out_gpu = gpu_causal_conv1d_fn(
                x_gpu, weight_gpu, bias_gpu,
                conv_states=conv_states_gpu,
                query_start_loc=qsl_gpu,
                cache_indices=ci_gpu,
                has_initial_state=his_gpu,
                activation=activation,
                pad_slot_id=pad_slot_id,
                null_block_id=pad_slot_id,
            )
            # (dim, total_tokens) -> (total_tokens, dim)
            y = out_gpu.T.contiguous().cpu().to(x.dtype)
    else:
        # ---- Update / Decode ----
        if x.dim() == 3:
            # Spec decode: x is (batch, seqlen, dim) -> (batch, dim, seqlen)
            x_gpu = x.to(device).permute(0, 2, 1).contiguous()
            batch = x.shape[0]
            ci_gpu = _to_int32_tensor(cache_indices, device)
            if ci_gpu is None:
                ci_gpu = torch.arange(batch, dtype=torch.int32, device=device)
            nat_values = _normalize_num_accepted_tokens(num_accepted_tokens, batch)
            if nat_values is not None and all(token == 0 for token in nat_values):
                nat_values = None
            elif nat_values is not None:
                for seq_idx, token in enumerate(nat_values):
                    if token <= 0 or token > x.shape[1]:
                        raise ValueError(
                            f"3D update GPU path expects 0 < num_accepted_tokens[{seq_idx}] <= {x.shape[1]}, "
                            f"got {token}"
                        )
            nat_gpu = _to_int32_tensor(nat_values, device)
            out_gpu = gpu_causal_conv1d_update(
                x_gpu, conv_states_gpu, weight_gpu,
                bias=bias_gpu, activation=activation,
                conv_state_indices=ci_gpu,
                num_accepted_tokens=nat_gpu,
                null_block_id=pad_slot_id,
            )
            # (batch, dim, seqlen) -> (batch, seqlen, dim)
            y = out_gpu.permute(0, 2, 1).contiguous().cpu().to(x.dtype)
        elif is_varlen_2d:
            x_gpu = x.to(device).contiguous()
            batch = len(query_start_loc) - 1
            ci_gpu = _to_int32_tensor(cache_indices, device)
            if ci_gpu is None:
                ci_gpu = torch.arange(batch, dtype=torch.int32, device=device)
            segment_lengths = [
                int(query_start_loc[seq_idx + 1] - query_start_loc[seq_idx])
                for seq_idx in range(batch)
            ]
            nat_values = _validate_gpu_varlen_num_accepted_tokens(
                _normalize_num_accepted_tokens(num_accepted_tokens, batch),
                segment_lengths=segment_lengths,
            )
            if nat_values is None:
                out_gpu = torch.zeros_like(x_gpu)
                cache_index_values = (
                    cache_indices if cache_indices is not None else list(range(batch))
                )
                for seq_idx, cache_idx in enumerate(cache_index_values):
                    if cache_idx == pad_slot_id:
                        continue
                    start = query_start_loc[seq_idx]
                    end = query_start_loc[seq_idx + 1]
                    if start == end:
                        continue
                    x_seq = (
                        x_gpu[start:end].transpose(0, 1).unsqueeze(0).contiguous()
                    )
                    out_seq = gpu_causal_conv1d_update(
                        x_seq,
                        conv_states_gpu,
                        weight_gpu,
                        bias=bias_gpu,
                        activation=activation,
                        conv_state_indices=ci_gpu[seq_idx : seq_idx + 1],
                        null_block_id=pad_slot_id,
                    )
                    out_gpu[start:end] = out_seq.squeeze(0).transpose(0, 1)
            else:
                qsl_gpu = torch.tensor(
                    query_start_loc, dtype=torch.int32, device=device
                )
                out_gpu = gpu_causal_conv1d_update(
                    x_gpu, conv_states_gpu, weight_gpu,
                    bias=bias_gpu, activation=activation,
                    conv_state_indices=ci_gpu,
                    num_accepted_tokens=_to_int32_tensor(nat_values, device),
                    query_start_loc=qsl_gpu,
                    max_query_len=max(segment_lengths),
                    null_block_id=pad_slot_id,
                )
            y = out_gpu.contiguous().cpu().to(x.dtype)
        else:
            # Decode 2D: x is (batch, dim) — already correct layout
            x_gpu = x.to(device)
            batch = x.shape[0]
            ci_gpu = _to_int32_tensor(cache_indices, device)
            if ci_gpu is None:
                ci_gpu = torch.arange(batch, dtype=torch.int32, device=device)
            nat_values = _normalize_num_accepted_tokens(num_accepted_tokens, batch)
            if nat_values is not None:
                for seq_idx, token in enumerate(nat_values):
                    if token not in (0, 1):
                        raise ValueError(
                            f"2D decode GPU path only supports num_accepted_tokens in {{0,1}}, "
                            f"got num_accepted_tokens[{seq_idx}]={token}"
                        )
            out_gpu = gpu_causal_conv1d_update(
                x_gpu, conv_states_gpu, weight_gpu,
                bias=bias_gpu, activation=activation,
                conv_state_indices=ci_gpu,
                null_block_id=pad_slot_id,
            )
            y = out_gpu.cpu().to(x.dtype)

    # conv_states back: (N, dim, state_len) -> (N, state_len, dim) (op layout)
    conv_states_out = conv_states_gpu.permute(0, 2, 1).contiguous().cpu().to(conv_states.dtype)
    return _finalize_backend_outputs(
        _to_head_major_output(y, head_num=head_num, run_mode=run_mode),
        conv_states_out,
        x=x,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        head_num=head_num,
    )


def run_causal_conv1d_reference(input_data: InputDataset) -> tuple[torch.Tensor, torch.Tensor]:
    x = input_data.kwargs["x"]
    weight = input_data.kwargs["weight"]
    bias = _optional_tensor(input_data.kwargs["bias"])
    conv_states = input_data.kwargs["conv_states"]
    query_start_loc = _optional_int_list(input_data.kwargs["query_start_loc_cpu"])
    cache_indices = _optional_int_list(input_data.kwargs["cache_indices_cpu"])
    initial_state_mode = _optional_int_list(input_data.kwargs["has_initial_state_cpu"])
    num_accepted_tokens = _optional_int_list(input_data.kwargs["num_accepted_tokens_cpu"])
    activation = str(input_data.kwargs["activation"])
    activation_mode = 0 if activation == "none" else 1
    pad_slot_id = -1
    run_mode = int(input_data.kwargs["run_mode"])
    head_num = int(input_data.kwargs.get("head_num", 0))

    outputs: tuple[torch.Tensor, torch.Tensor]
    if run_mode == 0:
        if x.dim() == 3:
            outputs = _run_prefill_batch(
                x,
                weight,
                bias,
                conv_states,
                cache_indices,
                initial_state_mode,
                activation_mode,
                pad_slot_id,
            )
        else:
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for 2D varlen prefill")
            outputs = _run_prefill_varlen(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                cache_indices,
                initial_state_mode,
                activation_mode,
                pad_slot_id,
            )
    elif x.dim() == 3:
        outputs = _run_update_batch_3d(
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
            pad_slot_id,
        )
    elif _is_varlen_2d_input(x, query_start_loc, run_mode):
        outputs = _run_update_varlen_2d(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
            pad_slot_id,
        )
    else:
        outputs = _run_update_decode_2d(
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
            pad_slot_id,
        )

    return _finalize_backend_outputs(
        _to_head_major_output(outputs[0], head_num=head_num, run_mode=run_mode),
        outputs[1],
        x=x,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        head_num=head_num,
    )


@register("npu_causal_conv1d")
class CausalConv1dApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            return run_causal_conv1d_gpu(input_data, device_id=self.device_id)
        return run_causal_conv1d_reference(input_data)


@register("pyaclnn_causal_conv1d")
class CausalConv1dAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["x"]
        input_data.kwargs.setdefault("head_num", 0)
        input_data.kwargs["bias"] = _optional_tensor(input_data.kwargs["bias"])
        query_start_loc = _optional_int_list(input_data.kwargs["query_start_loc_cpu"])
        cache_indices = _optional_int_list(input_data.kwargs["cache_indices_cpu"])
        has_initial_state = _optional_int_list(input_data.kwargs["has_initial_state_cpu"])
        num_accepted_tokens = _optional_int_list(input_data.kwargs["num_accepted_tokens_cpu"])
        input_data.kwargs["query_start_loc_cpu"] = query_start_loc
        input_data.kwargs["cache_indices_cpu"] = cache_indices
        input_data.kwargs["has_initial_state_cpu"] = has_initial_state
        input_data.kwargs["num_accepted_tokens_cpu"] = num_accepted_tokens
        max_query_len = -1
        if _is_varlen_2d_input(x, query_start_loc, int(input_data.kwargs["run_mode"])):
            max_query_len = max(
                (right - left for left, right in zip(query_start_loc, query_start_loc[1:])),
                default=0,
            )
        self._max_query_len = max_query_len
        self._pad_slot_mask_meta = {
            "x": x,
            "query_start_loc": query_start_loc,
            "cache_indices": cache_indices,
            "pad_slot_id": -1,
            "run_mode": int(input_data.kwargs["run_mode"]),
            "head_num": int(input_data.kwargs.get("head_num", 0)),
        }

        input_args = []
        for name in ("x", "weight", "bias", "conv_states"):
            value = input_data.kwargs[name]
            if name == "bias" and value is None:
                input_args.append(self._empty_acl_tensor())
                continue
            input_args.extend(self.backend.convert_input_data(value, name=name))

        # Preferred device metadata slots are absent in this compatibility
        # dataset; the four following *_cpu aclIntArray slots exercise the
        # deprecated host-array ABI in the same ACLNN entry point.
        input_args.extend(self._empty_acl_tensor() for _ in range(4))
        for name in (
            "query_start_loc_cpu",
            "cache_indices_cpu",
            "has_initial_state_cpu",
            "num_accepted_tokens_cpu",
        ):
            value = input_data.kwargs[name]
            if value is None:
                input_args.append(self._empty_acl_int_array())
                continue
            input_args.extend(self.backend.convert_input_data(value, name=name))

        activation = str(input_data.kwargs["activation"])
        self._activation_buffer = ctypes.create_string_buffer(activation.encode("utf-8"))
        input_args.append(ctypes.cast(self._activation_buffer, ctypes.c_char_p))
        # These vLLM-compatible sentinels are fixed by this legacy ATK
        # dataset, so they have no InputDataset metadata for
        # backend.convert_input_data() to consult.  Pass the generated ACLNN
        # ABI's int64_t values explicitly instead.
        input_args.extend((ctypes.c_int64(-1), ctypes.c_int64(-1)))
        for name, value in (
            ("run_mode", input_data.kwargs["run_mode"]),
            ("head_num", input_data.kwargs["head_num"]),
        ):
            input_args.extend(self.backend.convert_input_data(value, name=name))
        input_args.append(ctypes.c_int64(self._max_query_len))

        output_packages = []
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_packages.extend(self.backend.convert_output_data(output_data, index))
        if len(output_packages) != 2:
            raise ValueError(
                f"causal_conv1d expects benchmark outputs (y, conv_states), got {len(output_packages)} outputs"
            )

        input_args.append(output_packages[0])
        output_packages[:] = [output_packages[0], input_args[3]]
        return input_args, output_packages

    def after_call(self, output_packages):
        y, conv_states = (self.acl_tensor_to_torch(output_pack) for output_pack in output_packages)
        return _finalize_backend_outputs(
            y,
            conv_states,
            **self._pad_slot_mask_meta,
        )

    @staticmethod
    def _empty_acl_int_array():
        return ctypes.POINTER(AclIntArray)()

    @staticmethod
    def _empty_acl_tensor():
        return ctypes.POINTER(AclTensor)()

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnCausalConv1dGetWorkspaceSize("
            "const aclTensor *x, "
            "const aclTensor *weight, "
            "const aclTensor *biasOptional, "
            "const aclTensor *convStates, "
            "const aclTensor *queryStartLocOptional, "
            "const aclTensor *cacheIndicesOptional, "
            "const aclTensor *hasInitialStateOptional, "
            "const aclTensor *numAcceptedTokensOptional, "
            "const aclIntArray *queryStartLocCpuOptional, "
            "const aclIntArray *cacheIndicesCpuOptional, "
            "const aclIntArray *hasInitialStateCpuOptional, "
            "const aclIntArray *numAcceptedTokensCpuOptional, "
            "char *activationOptional, "
            "int64_t padSlotId, "
            "int64_t nullBlockId, "
            "int64_t runMode, "
            "int64_t headNum, "
            "int64_t maxQueryLen, "
            "const aclTensor *out, "
            "uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )


# 异常用例约束说明（对应 atk_causal_conv1d_abnormal.json）。
# 以下为各用例设计时违反的输入约束；固定输出标杆不执行约束校验。
# 实际 DUT 的拒绝行为及报错需由运行结果确认。
# case 00: causal_conv1d_abnormal_weight_width_1
#   变异输入：weight.shape = [1, 1536]
#   违反约束：W only supports 2/3/4
# case 01: causal_conv1d_abnormal_weight_width_5
#   变异输入：weight.shape = [5, 1536]
#   违反约束：W only supports 2/3/4
# case 02: causal_conv1d_abnormal_weight_rank_1
#   变异输入：weight.shape = [1536]
#   违反约束：weight must be rank 2
# case 03: causal_conv1d_abnormal_weight_dim_mismatch
#   变异输入：weight.shape = [4, 1520]
#   违反约束：weight dim must equal x dim
# case 04: causal_conv1d_abnormal_state_history_short
#   变异输入：conv_states.shape = [1, 2, 1536]
#   违反约束：state length must be at least W-1
# case 05: causal_conv1d_abnormal_state_dim_mismatch
#   变异输入：conv_states.shape = [1, 3, 1520]
#   违反约束：state dim must equal x dim
# case 06: causal_conv1d_abnormal_state_rank_2
#   变异输入：conv_states.shape = [3, 1536]
#   违反约束：conv_states must be rank 3
# case 07: causal_conv1d_abnormal_bias_dim_mismatch
#   变异输入：bias.shape = [1520]
#   违反约束：nonempty bias length must equal x dim
# case 08: causal_conv1d_abnormal_x_fp32_unsupported
#   变异输入：x.dtype = "fp32"
#   违反约束：x supports FP16/BF16 only
# case 09: causal_conv1d_abnormal_run_mode_2
#   变异输入：run_mode.range_values = 2
#   违反约束：run_mode supports 0/1 only
# case 10: causal_conv1d_abnormal_head_num_negative
#   变异输入：head_num.range_values = -1
#   违反约束：head_num must be nonnegative
# case 11: causal_conv1d_abnormal_head_num_not_divisor
#   变异输入：head_num.range_values = 5
#   违反约束：D must be divisible by positive head_num
# case 12: causal_conv1d_abnormal_initial_state_flag_2
#   变异输入：has_initial_state_cpu.array = [2]
#   违反约束：initial state flags support 0/1 only
# case 13: causal_conv1d_abnormal_initial_state_length_2
#   变异输入：has_initial_state_cpu.array = [0, 0]
#   违反约束：initial state array length must equal batch=1
# case 14: causal_conv1d_abnormal_query_start_nonzero
#   变异输入：query_start_loc_cpu.array = [1, 32768]
#   违反约束：query_start_loc must start at 0
@register("npu_causal_conv1d_abnormal")
class CausalConv1dAbnormalApi(CausalConv1dApi):
    """CPU uses fixed abnormal outputs; other devices use the original API.

    The existing ACLNN registration and DUT execution remain unchanged.
    """
    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "cpu":
            return None
        return super().init_by_input_data(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return super().__call__(input_data, with_output)
        device = torch.device("cpu")
        return (
            torch.zeros((32768, 1536), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 3, 1536), dtype=torch.bfloat16, device=device),
        )
