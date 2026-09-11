
"""Torch-based stub of tests/atk/common/_ascendc_common_executor.py (same semantics)."""
import torch

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
_RCP_LN2 = 1.4426950408889634


def _to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _case_spec(input_data, op_name):
    raw = _to_python(input_data.kwargs.get("case_spec"))
    if isinstance(raw, str):
        import json
        spec = json.loads(raw)
    else:
        spec = dict(raw or {})
    spec.setdefault("op", op_name)
    spec.setdefault("dtype", "bf16")
    return spec


def _marker_device(input_data):
    for value in input_data.kwargs.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _orig_dtype(name):
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)


def _calc_dtype(name, high_precision):
    if high_precision:
        return torch.float64
    return _orig_dtype(name)


def _randn(shape, dtype_name, calc_dtype, device, seed, scale=0.05):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.randn(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * float(scale)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _rand(shape, dtype_name, calc_dtype, device, seed, low=0.05, high=0.95):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32)
    data = data * (float(high) - float(low)) + float(low)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _zeros(shape, dtype_name, calc_dtype, device):
    data = torch.zeros(tuple(int(x) for x in shape), dtype=calc_dtype, device=device)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype)


def _gate(shape, calc_dtype, device, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-1)
    return data.to(calc_dtype).to(device)


def _kda_gate(shape, dtype_name, calc_dtype, device, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-2)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _int_tensor(values, device, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device=device)


def _chunks(total, chunk_size):
    for start in range(0, int(total), int(chunk_size)):
        yield start, min(start + int(chunk_size), int(total))


def _num_chunks(total, chunk_size):
    return (int(total) + int(chunk_size) - 1) // int(chunk_size)


def _finite_tuple(outputs, golden=False):
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    visible = []
    for output in outputs:
        if output is None or not isinstance(output, torch.Tensor):
            continue
        check = output.detach()
        if check.is_floating_point() and not torch.isfinite(check.float()).all().item():
            raise RuntimeError("NaN/Inf in output")
        if golden and output.dtype == torch.float64:
            output = output.to(torch.float32)
        visible.append(output)
    return tuple(visible)
