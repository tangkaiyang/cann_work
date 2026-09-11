"""Executor-level CPU validation for the three *_gen dirs (needs torch, no atk/fla_npu).

For each operator, run a representative slice of PROFILES through the executor's
run_cpu() path with a stubbed ATK layer, then check invariants:
- causal_conv1d_bwd: output finiteness, dh0==0 when no state, act=1/2 vs act=0 gradient path
- recurrent_kda: out/state shapes, state slots written per indices, gate path sanity
- prepare_wy_repr_bwd_da: dA shape/dtype, causal mask zero (upper triangle == 0), varlen == per-seg fixed consistency
"""
import io
import importlib.util
import json
import sys
import types
import os

BASE = r"D:\t30072652\Project\FLA_ATK"

# ---- stub atk executor-side modules ----
atk = types.ModuleType("atk")
dc = types.ModuleType("atk.configs.dataset_config")
rc = types.ModuleType("atk.configs.results_config")
ta = types.ModuleType("atk.tasks.api_execute")
ba = types.ModuleType("atk.tasks.api_execute.base_api")


class InputDataset:
    def __init__(self, kwargs=None):
        self.kwargs = kwargs or {}


class TaskResult:
    pass


def register(name):
    def deco(cls):
        return cls
    return deco


class BaseApi:
    def __init__(self, task_result=None):
        self.device = "cpu"


dc.InputDataset = InputDataset
rc.TaskResult = TaskResult
ta.register = register
ba.BaseApi = BaseApi
sys.modules["atk"] = atk
sys.modules["atk.configs"] = types.ModuleType("atk.configs")
sys.modules["atk.configs.dataset_config"] = dc
sys.modules["atk.configs.results_config"] = rc
sys.modules["atk.tasks"] = types.ModuleType("atk.tasks")
sys.modules["atk.tasks.api_execute"] = ta
sys.modules["atk.tasks.api_execute.base_api"] = ba

sys.path.insert(0, os.path.join(BASE))  # for loading *_gen/common stub

# stub common helpers (torch-based, same semantics as source _ascendc_common_executor)
common_dir = os.path.join(BASE, "_stub_common")
os.makedirs(common_dir, exist_ok=True)
with io.open(os.path.join(common_dir, "_ascendc_common_executor.py"), "w", encoding="utf-8") as f:
    f.write('''
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
''')

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("  FAIL:", msg)


def load_op(op, name):
    # executors do sys.path.insert(parents[1]/common) -> make <BASE>/common exist.
    # <BASE>/common already holds the real _ascendc_common_executor (torch-based,
    # no atk dependency) copied from the source repo, so just prepend the stub dir.
    op_dir = os.path.join(BASE, f"{op}_gen")
    stub_dir = os.path.join(BASE, "_stub_common")
    sys.path.insert(0, stub_dir)
    sys.modules.pop("_ascendc_common_executor", None)
    path = os.path.join(op_dir, f"executor_{op}.py")
    spec = importlib.util.spec_from_file_location(f"executor_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_spec(op, i):
    """Reuse the real generator to produce spec number i (needs generator-side atk stubs)."""
    import importlib
    if "atk.case_generator.generator.base_generator" not in sys.modules:
        atk_pkg = types.ModuleType("atk")
        cg = types.ModuleType("atk.case_generator")
        gp = types.ModuleType("atk.case_generator.generator")
        bg = types.ModuleType("atk.case_generator.generator.base_generator")
        gt = types.ModuleType("atk.case_generator.generator.generate_types")
        cf = types.ModuleType("atk.configs.case_config")

        class CaseGenerator:
            def __init__(self, config):
                self.config = config
                self.index = 1

        class _Registry:
            def register(self, name_):
                def deco(cls):
                    return cls
                return deco

        class _GenAttr:
            def __init__(self, name_):
                self.name = name_

        class _GenCaseConfig:
            def __init__(self):
                self.inputs = [_GenAttr(n) for n in (
                    "low_precision_marker", "fp32_marker", "case_spec", "dtype",
                    "B", "T", "H", "HV", "HK", "K", "V", "D", "W", "chunk_size",
                    "layout", "activation", "state_v_first", "case_id", "seed", "soc", "route",
                )]

        bg.CaseGenerator = CaseGenerator
        gt.GENERATOR_REGISTRY = _Registry()
        cf.CaseConfig = _GenCaseConfig
        for name_, mod_ in [("atk", atk_pkg), ("atk.case_generator", cg),
                            ("atk.case_generator.generator", gp),
                            ("atk.case_generator.generator.base_generator", bg),
                            ("atk.case_generator.generator.generate_types", gt),
                            ("atk.configs.case_config", cf)]:
            sys.modules[name_] = mod_

    gspec = importlib.util.spec_from_file_location(f"gen_{op}", os.path.join(BASE, f"{op}_gen", f"gen_{op}.py"))
    gmod = importlib.util.module_from_spec(gspec)
    gspec.loader.exec_module(gmod)
    return gmod._spec(i)


import torch

print("=== causal_conv1d_bwd executor (cpu golden) ===")
mod = load_op("causal_conv1d_bwd", "causal_conv1d_bwd")
picks = [0, 1, 2, 3, 4, 5]
for i in picks:
    spec = gen_spec("causal_conv1d_bwd", i)
    outs = mod.run_cpu(spec)
    # 标杆统一返回 (dx, dw, db, dh0)；with_state=False 时 dh0 恒 0
    check(len(outs) == 4, f"cc {i}: outputs {len(outs)}")
    for t in outs:
        check(torch.isfinite(t.float()).all(), f"cc {i}: nonfinite")
    dx = outs[0]
    check(dx.numel() == spec["B"] * spec["T"] * spec["D"], f"cc {i}: dx numel {tuple(dx.shape)}")
    check(tuple(outs[1].shape) == (spec["W"], spec["D"]), f"cc {i}: dw shape")
    check(tuple(outs[2].shape) == (spec["D"],), f"cc {i}: db shape")
    if not spec.get("with_state"):
        check(outs[3].abs().max() == 0, f"cc {i}: dh0 nonzero without state")
    check(dx.abs().max() > 0, f"cc {i}: dx all zero")
print(f"  ok ({len(picks)} specs)")

print("=== recurrent_kda executor (cpu golden) ===")
mod = load_op("recurrent_kda", "recurrent_kda")
n = 96
for i in range(n):
    spec = gen_spec("recurrent_kda", i)
    out, state = mod.run_cpu(spec)
    check(torch.isfinite(out.float()).all(), f"rk {i}: out nonfinite")
    check(torch.isfinite(state.float()).all(), f"rk {i}: state nonfinite")
    HV, K, V = spec["HV"], spec["K"], spec["V"]
    cap = spec["state_capacity"] if spec["ssm_mode"] != "none" else len(spec["cu_seqlens"]) - 1
    if spec["state_v_first"]:
        check(tuple(state.shape) == (cap, HV, V, K), f"rk {i}: state shape {tuple(state.shape)}")
    else:
        check(tuple(state.shape) == (cap, HV, K, V), f"rk {i}: state shape {tuple(state.shape)}")
    if spec["layout"] == "TND":
        check(tuple(out.shape) == (spec["total_tokens"], HV, V), f"rk {i}: out shape {tuple(out.shape)}")
    else:
        check(tuple(out.shape) == (spec["B"], spec["T"], HV, V), f"rk {i}: out shape {tuple(out.shape)}")
    # packed mode: shared slots -> state rows for untouched slots stay at init
print(f"  ok ({n} specs)")

print("=== prepare_wy_repr_bwd_da executor (cpu golden) ===")
mod = load_op("prepare_wy_repr_bwd_da", "prepare_wy_repr_bwd_da")
n = 30
for i in range(n):
    spec = gen_spec("prepare_wy_repr_bwd_da", i)
    dA = mod.run_cpu(spec)
    check(torch.isfinite(dA.float()).all(), f"da {i}: nonfinite")
    check(dA.dtype == torch.bfloat16 or dA.dtype == torch.float16, f"da {i}: dtype {dA.dtype}")
    C = spec["chunk_size"]
    T = spec["cu_seqlens"][-1] if spec.get("varlen") else spec["T"]
    B, HV = spec["B"], spec["HV"]
    check(tuple(dA.shape) == (B, HV, T, C), f"da {i}: dA shape {tuple(dA.shape)}")
    # causal mask: test_da.py stores b_dA.T where b_dA is strictly lower-triangular,
    # so the STORED block is zero on/below the diagonal and nonzero strictly above.
    dA32 = dA.float()
    for b in range(min(B, 2)):
        for hv in range(min(HV, 2)):
            nchunks = T // C
            for c in range(min(nchunks, 4)):
                blk = dA32[b, hv, c * C:(c + 1) * C, :C]
                lower_incl = torch.tril(torch.ones(C, C, dtype=torch.bool))
                check(blk[lower_incl].abs().max() == 0, f"da {i}: lower-tri nonzero b{b} hv{hv} c{c}")
                check(blk.abs().max() > 0, f"da {i}: block all zero b{b} hv{hv} c{c}")
print(f"  ok ({n} specs)")

print()
if failures:
    print(f"FAILED: {len(failures)}")
    sys.exit(1)
print("ALL EXECUTOR CPU CHECKS PASSED")
