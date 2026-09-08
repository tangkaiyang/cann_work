"""causal_conv1d_bwd 的 ATK 泛化用例生成器。

ATK 注册入口与默认 CLI 均生成真实输入 TND、activation=0、带状态结构。
边界优先，支持三 dtype、T<W、非 2 次幂、模型通道与固定种子随机补充。
独立 CLI 可导出 ATK JSON；extended 仅导出待接入的多布局/激活/状态清单。
max-elements 仅限制单个 x 元素数，不代表 CPU 标杆峰值内存上限。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

OP_NAME = "causal_conv1d_bwd"
SEED_BASE = 20260817

# 内置用户 JSON 模板；YAML + 本文件即可使用 ATK 注册入口。
DIRECT_TEMPLATE = {'id': 0,
 'default_seed': None,
 'name': 'aclnn.causal_conv1d_bwd',
 'aclnn_name': 'CausalConv1dBwd',
 'triton_name': None,
 'kernel_name': None,
 'version': 'v2.1',
 'expected_error_msg': None,
 'api': 'pytorch',
 'api_type': 'executor_causal_conv1d_bwd',
 'aclnn_api_type': 'aclnn_causal_conv1d_bwd',
 'triton_api_type': 'triton_causal_conv1d_bwd',
 'fusion_api_type': 'fusion_function',
 'fusion_mode': None,
 'dist_api_type': 'dist_function',
 'kernel_api_type': 'kernel_function',
 'backward': False,
 'standard': {'acc': {'cv_fused_double_benchmark': {'max_re_ratio': 5,
                                                    'avg_re_ratio': 1.5,
                                                    'root_mean_squared_ratio': 1.5}},
              'perf': 'not_key',
              'mem': 1.1},
 'outputs': None,
 'inputs': [{'name': 'x',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [64, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'y',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [64, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'weight',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [2, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'dy',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [64, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'initial_state',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [1, 2, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'dht',
             'type': 'tensor',
             'required': True,
             'dtype': 'fp32',
             'shape': [1, 2, 16],
             'range_values': [-2, 2],
             'backward': True,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'queryStartLoc',
             'type': 'attr',
             'required': True,
             'dtype': 'int',
             'shape': None,
             'range_values': [0, 64],
             'backward': False,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'activation',
             'type': 'attr',
             'required': True,
             'dtype': 'int',
             'shape': None,
             'range_values': 0,
             'backward': False,
             'align_32B': None,
             'outlier_values': None},
            {'name': 'inputLayout',
             'type': 'attr',
             'required': True,
             'dtype': 'string',
             'shape': None,
             'range_values': 'TND',
             'backward': False,
             'align_32B': None,
             'outlier_values': None}],
 'acl_json': '',
 'method_inputs': None,
 'tensor_input': None,
 'compute_times': None,
 'save_name': None,
 'uuid': None,
 'downloaded': False,
 'is_boundary': False,
 'xrun_cs_name': None,
 'xrun_data': None,
 'strategy': None}

W_VALUES = (2, 3, 4)


DTYPES = ("bf16", "fp16", "fp32")
B_VALUES = (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128)
T_VALUES = (1, 2, 3, 4, 5, 16, 17, 32, 54, 55, 56, 63, 64, 65,
            103, 104, 105, 127, 128, 129, 151, 223, 224, 225,
            255, 256, 257, 372, 512, 784, 1024, 1134, 2048, 4096, 8192, 16384)
D_VALUES = (16, 32, 48, 64, 80, 112, 128, 144, 192, 256, 384, 768,
            1024, 1536, 2048, 2064, 2304, 3072, 4096, 4128, 8192)
T_VALUES = tuple(sorted(set(T_VALUES + (7, 8, 15, 31, 33, 109, 110, 111, 191, 192, 193, 207, 208, 209, 447, 448, 449, 511, 513, 1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193, 32768, 65536))))
D_VALUES = tuple(sorted(set(D_VALUES + (240, 272, 496, 512, 528, 1008, 1040, 1520, 1552, 2032, 2080, 3056, 3088, 4080, 4112, 6144, 8176, 8208, 12288, 16384))))
MAX_ELEMENTS = 4_194_304


def build_profiles(random_shapes=256, max_elements=MAX_ELEMENTS):
    """边界优先，三 dtype 相邻；限制单个 x 大小，不声称限制峰值内存。"""
    import itertools
    shapes, seen = [], set()

    def add(B, T, D, W, category):
        key = (B, T, D, W)
        if B*T*D <= max_elements and key not in seen:
            seen.add(key)
            shapes.append(dict(B=B, T=T, D=D, W=W, category=category))

    for W in W_VALUES:
        for T in (1, 2, 3, 4, 5, 63, 64, 65):
            add(1, T, 16, W, "short_and_chunk")
    for D in D_VALUES:

        for W in W_VALUES:
            add(2, 17, D, W, "channel")
    for B in B_VALUES:

        for W in W_VALUES:
            add(B, 65, 64, W, "batch")
    for T in T_VALUES:

        for W in W_VALUES:
            add(1, T, 64, W, "sequence")
    for D in (1536, 2048, 2304, 3072, 4128):
        for B, T in ((1, 512), (1, 1024), (1, 2048), (4, 512)):
            add(B, T, D, 4, "model_shape_no_activation")
    candidates = [s for s in itertools.product(B_VALUES, T_VALUES, D_VALUES, W_VALUES)
                  if s[0]*s[1]*s[2] <= max_elements and s not in seen]
    random.Random(SEED_BASE).shuffle(candidates)
    for shape in candidates[:random_shapes]:
        add(*shape, "random")
    profiles = [dict(p, dtype=dtype, input_layout="BSND", activation=0,
                 has_initial_state=False, has_dht=False,
                 name=f"{dtype}_bsnd_B{p['B']}_T{p['T']}_D{p['D']}_W{p['W']}")
            for p in shapes for dtype in DTYPES]
    # 不等长正长度序列：短长混排、块边界、极端倾斜和顺序变化。
    lengths_set = ((1, 2, 3, 4, 5), (7, 65, 130), (1, 4096, 2),
                   (4096, 1, 2), (63, 64, 65), (1, 65536, 1),
                   tuple([1]*31+[1024]), tuple([17]*63+[65]))
    for lengths in lengths_set:
        for D, W, dtype in itertools.product((16, 64, 128, 2048), W_VALUES, DTYPES):
            if sum(lengths)*D > max_elements:
                continue
            q = [0]
            for length in lengths:
                q.append(q[-1]+length)
            profiles.append(dict(B=len(lengths), T=max(lengths), D=D, W=W,
                dtype=dtype, input_layout="TND", activation=0,
                has_initial_state=True, has_dht=True, category="ragged",
                query_start_loc=q, total_tokens=q[-1],
                name=f"{dtype}_ragged{len(profiles)}_D{D}_W{W}"))
    return profiles


def extended_profiles():
    """仅设计清单：旧 executor 不支持这些字段，禁止直接导出为 ATK。"""
    import itertools
    profiles = []
    for layout, dtype, act, h0, dht in itertools.product(
            ("BSND", "BSH", "BNSD", "TND", "NTD"), DTYPES, (0, 1, 2), (False, True), (False, True)):
        p = dict(B=3, T=65, D=128, W=4, dtype=dtype, input_layout=layout,
                 activation=act, has_initial_state=h0, has_dht=dht, category="semantic_grid")
        if layout in ("BNSD", "NTD"):
            p.update(N=2, Dh=64)
        if layout in ("TND", "NTD"):
            p.update(query_start_loc=[0, 1, 4, 69], total_tokens=69)
        profiles.append(p)
    for dtype, D, T in itertools.product(("bf16", "fp16"), (64, 128),
                                        (54, 55, 56, 103, 104, 105, 223, 224, 225)):
        profiles.append(dict(B=1, T=T, D=D, W=4, dtype=dtype, input_layout="BSND",
                             activation=1, has_initial_state=False, has_dht=False,
                             category="fast_path_candidate"))
    for i, p in enumerate(profiles):
        p["name"] = f"extended_{i:04d}_{p['dtype']}_{p['input_layout'].lower()}"
        p["y_source"] = "independent_preactivation" if p["activation"] else "none"
    return profiles


PROFILES = build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index, profiles=None, soc="ascend910b", seed_base=SEED_BASE):
    profiles = PROFILES if profiles is None else profiles
    if index < 0 or not profiles:
        raise ValueError("index must be non-negative and profiles non-empty")
    profile = deepcopy(profiles[index % len(profiles)])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": seed_base + index // len(profiles),
            "repeat": index // len(profiles),
            "route": "ascendc",
            "soc": soc,
        }
    )
    return profile


def populate_case_config(case_config, index):
    """直接修改 ATK CaseConfig/InputConfig，由 ATK 负责 JSON 序列化。"""
    spec = _spec(index)
    B, T, D, W = (spec[key] for key in ("B", "T", "D", "W"))
    configs = [cfg for item in case_config.inputs
               for cfg in (item if isinstance(item, list) else [item])]
    names = ["x", "y", "weight", "dy", "initial_state", "dht",
             "queryStartLoc", "activation", "inputLayout"]
    if [cfg.name for cfg in configs] != names:
        raise ValueError("请配套使用真实输入版 causal_conv1d_bwd.yaml")

    case_config.id = index
    case_config.default_seed = spec["seed"]
    case_config.name = "aclnn.causal_conv1d_bwd"
    # tensor 的 range_values/outlier_values 由 YAML/ATK 决定，保持原值。
    # API 名称、standard 和其余辅助字段由 YAML/ATK 初始化，不覆盖对象类型。
    for cfg in configs:
        if cfg.name in ("x", "y", "dy"):
            cfg.dtype = spec["dtype"]
            cfg.shape = [spec.get("total_tokens", B * T), D]
        elif cfg.name == "weight":
            cfg.dtype = spec["dtype"]
            cfg.shape = [W, D]
        elif cfg.name in ("initial_state", "dht"):
            cfg.dtype = spec["dtype"]
            cfg.shape = [B, W, D]
        elif cfg.name == "queryStartLoc":
            cfg.dtype = "int"
            cfg.shape = None
            cfg.range_values = spec.get("query_start_loc", [i * T for i in range(B + 1)]).copy()
        elif cfg.name == "activation":
            cfg.dtype = "int"
            cfg.shape = None
            cfg.range_values = 0
        elif cfg.name == "inputLayout":
            cfg.dtype = "string"
            cfg.shape = None
            cfg.range_values = "TND"
    return case_config


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
    class CausalConv1dBwdGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return populate_case_config(case_config, max(int(self.index)-1, 0))


def export_marker_atk(specs, template):
    """保留现有 ATK JSON 结构；只改用例输入和标识。"""
    result = []
    for spec in specs:
        case = deepcopy(template)
        case.update(id=spec["case_id"], default_seed=spec["seed"],
                    name=f"{OP_NAME}_{spec['case_id']:05d}_{spec['name']}_s{spec['seed']}")
        for item in case["inputs"]:
            for cfg in item if isinstance(item, list) else [item]:
                if cfg["name"] == "low_precision_marker":
                    cfg["dtype"] = spec["dtype"]
                elif cfg["name"] == "case_spec":
                    cfg["range_values"] = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg["name"] in spec:
                    cfg["range_values"] = spec[cfg["name"]]
        result.append(case)
    return result


def export_atk(specs, template):
    """按用户真实输入模板导出。模板决定布局/激活/状态存在性，spec 泛化形状。"""
    expected = ["x", "y", "weight", "dy", "initial_state", "dht",
                "queryStartLoc", "activation", "inputLayout"]
    if [v["name"] for v in template["inputs"]] != expected:
        raise ValueError("真实输入模板必须包含 x/y/weight/dy/initial_state/dht/queryStartLoc/activation/inputLayout")
    values = {v["name"]: v for v in template["inputs"]}
    layout = values["inputLayout"]["range_values"]
    if layout not in ("BSND", "BSH", "TND"):
        raise ValueError("形状泛化模板当前支持 BSND/BSH/TND；多头布局需指定 N/Dh")
    result = []
    for spec in specs:
        case = deepcopy(template)
        # 保留模板 name 及所有辅助字段，只改变用例编号、随机种子和输入形状/dtype。
        case.update(id=spec["case_id"], default_seed=spec["seed"])
        B, T, D, W = (spec[k] for k in ("B", "T", "D", "W"))
        if "query_start_loc" in spec and layout != "TND":
            raise ValueError("不等长用例需要 TND 模板")
        logical = [spec.get("total_tokens", B*T), D] if layout == "TND" else [B, T, D]
        shapes = dict(x=logical, y=logical, weight=[W, D], dy=logical,
                      initial_state=[B, W, D], dht=[B, W, D])
        for cfg in case["inputs"]:
            if cfg["name"] in shapes:
                cfg["dtype"] = spec["dtype"]
                cfg["shape"] = shapes[cfg["name"]].copy()
            elif cfg["name"] == "queryStartLoc":
                cfg["range_values"] = spec.get("query_start_loc", [i*T for i in range(B+1)]).copy()
        result.append(case)
    return result


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="causal_conv1d_bwd 泛化用例导出")
    parser.add_argument("--suite", choices=("compatible", "extended"), default="compatible")
    parser.add_argument("--format", choices=("specs", "atk", "marker-atk"), default="atk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="自定义单条对象或数组模板")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--random-shapes", type=int, default=256)
    parser.add_argument("--max-elements", type=int, default=MAX_ELEMENTS)
    parser.add_argument("--soc", choices=("ascend910b", "ascend910_93", "ascend950"), default="ascend910b")
    args = parser.parse_args()
    if args.template is None:
        args.template = Path(__file__).with_name(
            "atk_causal_conv1d_bwd.json" if args.format == "marker-atk"
            else "causal_conv1d_bwd_direct_template.json")
    if args.seeds < 1 or args.random_shapes < 0 or args.max_elements < 1:
        parser.error("seeds/max-elements 必须为正数，random-shapes 必须非负")
    if args.suite == "extended" and args.format != "specs":
        parser.error("extended 只支持 specs：现有 executor 未实现布局/激活/状态参数")
    profiles = (build_profiles(args.random_shapes, args.max_elements) if args.suite == "compatible"
                else [p for p in extended_profiles()
                      if p.get("total_tokens", p["B"]*p["T"])*p["D"] <= args.max_elements])
    if not profiles:
        parser.error("内存上限过滤掉了全部用例")
    if args.format == "marker-atk":
        profiles = [p for p in profiles if "query_start_loc" not in p]
    specs = [_spec(i, profiles, args.soc, args.seed_base) for i in range(len(profiles)*args.seeds)]
    if args.output.resolve() == args.template.resolve():
        parser.error("请使用新输出路径，避免覆盖原始用例模板")
    if args.format in ("atk", "marker-atk"):
        template = (deepcopy(DIRECT_TEMPLATE) if args.format == "atk" and
                    args.template == Path(__file__).with_name("causal_conv1d_bwd_direct_template.json")
                    else json.loads(args.template.read_text(encoding="utf-8-sig")))
        if isinstance(template, list):
            template = template[0]
        exporter = export_atk if args.format == "atk" else export_marker_atk
        payload = exporter(specs, template)
    else:
        payload = dict(schema="causal_conv1d_bwd.case_specs.v1",
                   suite=args.suite, executor_compatible=args.suite == "compatible",
                   npu_verified=False, profile_count=len(profiles), case_count=len(specs),
                   seeds_per_profile=args.seeds, max_x_elements=args.max_elements, cases=specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"{args.suite}: {len(profiles)} profiles x {args.seeds} seeds = {len(specs)} cases -> {args.output}")


if __name__ == "__main__":
    main()
