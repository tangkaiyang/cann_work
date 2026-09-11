"""Offline sanity check for the three *_gen generators (no atk/torch/fla_npu needed)."""
import io
import importlib.util
import json
import sys
import types
import os

BASE = r"D:\t30072652\Project\FLA_ATK"

# ---- stub atk modules ----
atk = types.ModuleType("atk")
case_generator = types.ModuleType("atk.case_generator")
generator_pkg = types.ModuleType("atk.case_generator.generator")
base_gen = types.ModuleType("atk.case_generator.generator.base_generator")
gen_types = types.ModuleType("atk.case_generator.generator.generate_types")
configs_pkg = types.ModuleType("atk.configs")
case_config_mod = types.ModuleType("atk.configs.case_config")


class CaseGenerator:
    def __init__(self, config):
        self.config = config
        self.index = 1


class _Registry:
    def __init__(self):
        self._reg = {}

    def register(self, name):
        def deco(cls):
            self._reg[name] = cls
            return cls
        return deco


GENERATOR_REGISTRY = _Registry()


class _Attr:
    def __init__(self, name):
        self.name = name
        self.dtype = None
        self.range_values = None


class CaseConfig:
    def __init__(self):
        self.id = None
        self.default_seed = None
        self.name = None
        self.inputs = [_Attr(n) for n in (
            "low_precision_marker", "fp32_marker", "case_spec", "dtype",
            "B", "T", "H", "HV", "HK", "K", "V", "D", "W", "chunk_size",
            "layout", "activation", "state_v_first", "case_id", "seed", "soc", "route",
        )]


base_gen.CaseGenerator = CaseGenerator
gen_types.GENERATOR_REGISTRY = GENERATOR_REGISTRY
case_config_mod.CaseConfig = CaseConfig
sys.modules["atk"] = atk
sys.modules["atk.case_generator"] = case_generator
sys.modules["atk.case_generator.generator"] = generator_pkg
sys.modules["atk.case_generator.generator.base_generator"] = base_gen
sys.modules["atk.case_generator.generator.generate_types"] = gen_types
sys.modules["atk.configs"] = configs_pkg
sys.modules["atk.configs.case_config"] = case_config_mod

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def load_gen(op):
    path = os.path.join(BASE, f"{op}_gen", f"gen_{op}.py")
    spec = importlib.util.spec_from_file_location(f"gen_{op}_gen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def yaml_spec(op):
    """读取 yaml 里第一条 case_spec 做快照一致性参考（仅确认可 json 解析）。"""
    path = os.path.join(BASE, f"{op}_gen", f"{op}.yaml")
    src = io.open(path, encoding="utf-8").read()
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("- name: case_spec"):
            pass
        if s.startswith("values: [\"{"):
            raw = s[len("values: [\""):-2]
            raw = raw.encode().decode("unicode_escape")
            return json.loads(raw)
    return None


def run_generator(mod, op, n_cases=64):
    reg_name = f"generator_{op}"
    check(reg_name in GENERATOR_REGISTRY._reg, f"{op}: generator not registered")
    cls = GENERATOR_REGISTRY._reg[reg_name]
    for i in range(n_cases):
        gen = cls(None)
        gen.index = i + 1
        cc = CaseConfig()
        gen.after_case_config(cc)
        spec_attr = next(a for a in cc.inputs if a.name == "case_spec")
        spec = json.loads(spec_attr.range_values)
        check(spec["case_id"] == i, f"{op} case {i}: case_id mismatch")
        check(spec["seed"] == 20260817 + i, f"{op} case {i}: seed mismatch")
        # marker dtype
        marker = next(a for a in cc.inputs if a.name == "low_precision_marker")
        check(marker.dtype == spec["dtype"], f"{op} case {i}: marker dtype mismatch")
        # mirrored attrs present in spec are set
        for a in cc.inputs:
            if a.name in spec:
                check(a.range_values == spec[a.name],
                      f"{op} case {i}: attr {a.name}={a.range_values} != spec {spec[a.name]}")
        yield spec


# ================= causal_conv1d_bwd =================
mod = load_gen("causal_conv1d_bwd")
n_profiles = len(mod.PROFILES)
print(f"[causal_conv1d_bwd] profiles = {n_profiles}")
covered = {"layouts": set(), "acts": set(), "varlen_seg_counts": set(), "with_state": set()}
for i, spec in enumerate(run_generator(mod, "causal_conv1d_bwd", n_profiles)):
    B, T, D, W = spec["B"], spec["T"], spec["D"], spec["W"]
    layout, act = spec["layout"], spec["activation"]
    check(T >= W, f"cc case {i}: T<T W")
    if layout in ("BSND", "BSH", "BSH") and layout != "BNSD":
        check(D % 16 == 0, f"cc case {i}: D%16 for {layout}")
    covered["layouts"].add(layout)
    covered["acts"].add(act)
    covered["with_state"].add(spec["with_state"])
    qsl = spec.get("query_start_loc")
    if layout in ("TND", "NTD"):
        check(qsl is not None, f"cc case {i}: varlen missing query_start_loc")
        check(qsl[0] == 0 and qsl[-1] == T, f"cc case {i}: qsl endpoints {qsl} T={T}")
        segs = [qsl[j + 1] - qsl[j] for j in range(len(qsl) - 1)]
        check(all(s >= W for s in segs), f"cc case {i}: seg < W, {segs} W={W}")
        check(len(segs) == B, f"cc case {i}: seg count != B")
        covered["varlen_seg_counts"].add(len(segs))
    else:
        check(qsl is None, f"cc case {i}: fixed layout has qsl")
check(covered["layouts"] == {"BSND", "BSH", "BNSD", "TND", "NTD"}, f"cc layouts: {covered['layouts']}")
check(covered["acts"] == {0, 1, 2}, f"cc acts: {covered['acts']}")
check(covered["with_state"] == {False, True}, f"cc with_state: {covered['with_state']}")
snap = yaml_spec("causal_conv1d_bwd")
check(snap is not None and snap["layout"] == "BSND" and snap["activation"] == 0, f"cc yaml snapshot: {snap}")
print(f"  layouts={sorted(covered['layouts'])} acts={sorted(covered['acts'])} state={sorted(map(str, covered['with_state']))}")

# ================= recurrent_kda =================
GENERATOR_REGISTRY._reg.clear()
mod = load_gen("recurrent_kda")
n_profiles = len(mod.PROFILES)
print(f"[recurrent_kda] profiles = {n_profiles}")
covered = {"layouts": set(), "modes": set(), "dt": set(), "V": set(), "svf": set()}
for i, spec in enumerate(run_generator(mod, "recurrent_kda", min(n_profiles, 4096))):
    cu, mode = spec["cu_seqlens"], spec["ssm_mode"]
    layout = spec["layout"]
    H, HV, K, V = spec["H"], spec["HV"], spec["K"], spec["V"]
    check(K == 128 and V in (128, 256), f"rk case {i}: K/V {K}/{V}")
    check(HV % H == 0, f"rk case {i}: HV%H")
    check(cu[0] == 0, f"rk case {i}: cu[0]")
    check(all(b >= a for a, b in zip(cu, cu[1:])), f"rk case {i}: cu monotonic {cu}")
    check(all(x - a <= 8 for a, x in zip(cu, cu[1:])), f"rk case {i}: seg len > 8 {cu}")
    check(cu[-1] == spec["total_tokens"], f"rk case {i}: total_tokens mismatch")
    seg_n = len(cu) - 1
    if layout == "TND":
        check(spec["T"] == spec["total_tokens"], f"rk case {i}: TND T mismatch")
    else:
        # BSND：dense [B, T]，packed 容量 B*T == total_tokens == cu[-1]，
        # 段长可以超过单批 T（容量约束只要求 cu[-1] <= B*T）
        check(spec["B"] * spec["T"] == spec["total_tokens"], f"rk case {i}: BSND B*T != total_tokens")
        check(spec["B"] >= seg_n, f"rk case {i}: BSND B < segs")
    if mode == "none":
        check("ssm_state_indices" not in spec, f"rk case {i}: none has indices")
        check(spec["state_capacity"] == seg_n, f"rk case {i}: none capacity != seq_num")
    elif mode == "packed":
        idx = spec["ssm_state_indices"]
        check(len(idx) == spec["total_tokens"], f"rk case {i}: packed len")
        check(all(0 <= x < spec["state_capacity"] for x in idx), f"rk case {i}: packed range")
    else:
        mat = spec["ssm_state_indices"]
        acc = spec["num_accepted_tokens"]
        check(len(mat) == seg_n, f"rk case {i}: spec rows")
        check(all(len(r) == len(mat[0]) for r in mat), f"rk case {i}: spec ragged")
        check(all(0 <= x < spec["state_capacity"] for r in mat for x in r), f"rk case {i}: spec range")
        check(len(acc) == seg_n and all(1 <= a <= (cu[j + 1] - cu[j]) for j, a in enumerate(acc)),
              f"rk case {i}: accepted {acc} vs {cu}")
    covered["layouts"].add(layout)
    covered["modes"].add(mode)
    covered["dt"].add(spec["dt_bias_mode"])
    covered["V"].add(V)
    covered["svf"].add(spec["state_v_first"])
check(covered["layouts"] == {"BSND", "TND"}, f"rk layouts {covered['layouts']}")
check(covered["modes"] == {"none", "packed", "speculative"}, f"rk modes {covered['modes']}")
check(covered["dt"] == {"flat", "matrix"}, f"rk dt {covered['dt']}")
check(covered["V"] == {128, 256}, f"rk V {covered['V']}")
check(covered["svf"] == {True, False}, f"rk svf {covered['svf']}")
snap = yaml_spec("recurrent_kda")
check(snap is not None and snap["cu_seqlens"] == [0, 8], f"rk yaml snapshot cu: {snap.get('cu_seqlens')}")
print(f"  layouts={sorted(covered['layouts'])} modes={sorted(covered['modes'])} dt={sorted(covered['dt'])} V={sorted(covered['V'])}")

# ================= prepare_wy_repr_bwd_da =================
GENERATOR_REGISTRY._reg.clear()
mod = load_gen("prepare_wy_repr_bwd_da")
n_profiles = len(mod.PROFILES)
print(f"[prepare_wy_repr_bwd_da] profiles = {n_profiles}")
covered = {"modes": set(), "V": set(), "dtypes": set(), "gva": set(), "T": set()}
for i, spec in enumerate(run_generator(mod, "prepare_wy_repr_bwd_da", min(n_profiles, 4096))):
    B, HK, HV, T, K, V, C = (spec[x] for x in ("B", "HK", "HV", "T", "K", "V", "chunk_size"))
    check(K == 128 and V in (128, 256), f"da case {i}: K/V {K}/{V}")
    check(HV % HK == 0, f"da case {i}: HV%HK {HV}/{HK}")
    check(T % C == 0, f"da case {i}: T%C {T}")
    covered["T"].add(T)
    covered["V"].add(V)
    covered["dtypes"].add(spec["dtype"])
    covered["gva"].add((HK, HV))
    if spec["varlen"]:
        check(B == 1, f"da case {i}: varlen B!=1")
        cu, ci = spec["cu_seqlens"], spec["chunk_indices"]
        check(cu[0] == 0 and cu[-1] == T, f"da case {i}: cu endpoints {cu}")
        segs = [cu[j + 1] - cu[j] for j in range(len(cu) - 1)]
        check(all(s % C == 0 for s in segs), f"da case {i}: seg not chunk aligned {segs}")
        check(len(ci) % 2 == 0, f"da case {i}: ci odd len")
        # 重放 chunk_indices 校验
        expect = []
        for s in range(len(cu) - 1):
            ln = cu[s + 1] - cu[s]
            for ch in range((ln + C - 1) // C):
                expect.extend((s, ch))
        check(ci == expect, f"da case {i}: chunk_indices mismatch")
        covered["modes"].add("varlen")
    else:
        check("cu_seqlens" not in spec, f"da case {i}: fixed has cu")
        covered["modes"].add("fixed")
check(covered["modes"] == {"fixed", "varlen"}, f"da modes {covered['modes']}")
check(covered["V"] == {128, 256}, f"da V {covered['V']}")
check(covered["dtypes"] == {"bf16", "fp16"}, f"da dtypes {covered['dtypes']}")
check(any(hv > hk for hk, hv in covered["gva"]), f"da GVA pairs {covered['gva']}")
check(covered["T"] == {64, 128, 256}, f"da T {covered['T']}")
snap = yaml_spec("prepare_wy_repr_bwd_da")
check(snap is not None and snap["varlen"] is False and snap["B"] == 2, f"da yaml snapshot: {snap}")
print(f"  modes={sorted(covered['modes'])} GVA={sorted(covered['gva'])} T={sorted(covered['T'])}")

print()
if failures:
    print(f"FAILED: {len(failures)} problem(s)")
    for f in failures[:20]:
        print(" -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
