"""Generate mutually valid inputs for the ACLNN RecurrentKda signature."""

from __future__ import annotations

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = GENERATOR_REGISTRY = CaseConfig = None


CASE_COUNT = 200


def _configs(case_config):
    result = {}
    for item in case_config.inputs:
        cfg = item[0] if isinstance(item, list) else item
        result[cfg.name] = cfg
    return result


def _profile(index):
    layout = ("BSND", "TND")[index % 2]
    batch, time, heads, value_heads = (
        (1, 1, 1, 1), (1, 2, 1, 2), (2, 2, 2, 4),
        (2, 4, 4, 8), (4, 2, 4, 16),
    )[(index // 2) % 5]
    if layout == "TND":
        batch = 1
    use_gate = index % 4 in (2, 3)
    use_beta = index % 5 in (3, 4)
    indexed = index % 6 in (4, 5)
    return {
        "layout": layout, "B": batch, "T": time, "H": heads,
        "HV": value_heads, "K": 128, "V": (128, 256)[(index // 7) % 2],
        "gate_dtype": ("fp32", "bf16", "fp16")[(index // 3) % 3],
        "beta_dtype": ("fp32", "bf16", "fp16")[(index // 5) % 3],
        "state_dtype": ("fp32", "bf16")[(index // 11) % 2],
        "stateVFirst": bool(index % 2), "useGateInKernel": use_gate,
        "useBetaSigmoidInKernel": use_beta,
        "allowNegEigval": use_beta and index % 2 == 0,
        "safeGate": use_gate and index % 3 == 0,
        "indexed": indexed, "outputFinalState": index % 3 != 0,
        "inplaceFinalState": bool(index % 2),
        "useQkL2normInKernel": index % 7 == 0,
    }


def configure_case(case_config, index):
    cfg = _configs(case_config)
    p = _profile(index % CASE_COUNT)
    b, t, h, hv, k, v = (p[n] for n in ("B", "T", "H", "HV", "K", "V"))
    token_count = b * t
    q_shape = [token_count, h, k] if p["layout"] == "TND" else [b, t, h, k]
    v_shape = [token_count, hv, v] if p["layout"] == "TND" else [b, t, hv, v]
    g_shape = [token_count, hv, k] if p["layout"] == "TND" else [b, t, hv, k]
    beta_shape = [token_count, hv] if p["layout"] == "TND" else [b, t, hv]
    state_shape = [b, hv, v, k] if p["stateVFirst"] else [b, hv, k, v]
    for name in ("query", "key"):
        cfg[name].shape, cfg[name].dtype = q_shape, "bf16"
    cfg["value"].shape, cfg["value"].dtype = v_shape, "bf16"
    cfg["gate"].shape, cfg["gate"].dtype = g_shape, p["gate_dtype"]
    cfg["beta"].shape, cfg["beta"].dtype = beta_shape, p["beta_dtype"]
    cfg["initialStateRef"].shape = state_shape
    cfg["initialStateRef"].dtype = p["state_dtype"]

    offsets = [i * t for i in range(b + 1)]
    cfg["cuSeqlensOptional"].required = True
    cfg["cuSeqlensOptional"].dtype = ("int32", "int64")[index % 2]
    cfg["cuSeqlensOptional"].shape = [len(offsets)]
    cfg["cuSeqlensOptional"].range_values = offsets
    if p["indexed"]:
        cfg["ssmStateIndicesOptional"].required = True
        cfg["ssmStateIndicesOptional"].shape = [token_count]
        cfg["ssmStateIndicesOptional"].range_values = [i // t for i in range(token_count)]
    else:
        cfg["ssmStateIndicesOptional"].required = False
        cfg["ssmStateIndicesOptional"].range_values = "null"
    cfg["numAcceptedTokensOptional"].required = False
    cfg["numAcceptedTokensOptional"].range_values = "null"

    cfg["aLogOptional"].shape = [hv]
    cfg["aLogOptional"].required = p["useGateInKernel"]
    cfg["aLogOptional"].range_values = [-1.0, -0.2] if p["useGateInKernel"] else "null"
    cfg["dtBiasOptional"].shape = [hv, k]
    cfg["dtBiasOptional"].required = False
    cfg["dtBiasOptional"].range_values = [-0.1, 0.1] if p["useGateInKernel"] and index % 2 else "null"
    attrs = {
        "layout": p["layout"], "scale": 128 ** -0.5,
        "outputFinalState": p["outputFinalState"],
        "inplaceFinalState": p["inplaceFinalState"],
        "useQkL2normInKernel": p["useQkL2normInKernel"],
        "useGateInKernel": p["useGateInKernel"],
        "useBetaSigmoidInKernel": p["useBetaSigmoidInKernel"],
        "allowNegEigval": p["allowNegEigval"], "safeGate": p["safeGate"],
        "lowerBound": -2.5 if p["safeGate"] else -5.0,
        "stateVFirst": p["stateVFirst"],
    }
    for name, value in attrs.items():
        cfg[name].range_values = value
    case_config.id = index
    case_config.default_seed = 20260817 + index
    case_config.name = f"recurrent_kda_{index:04d}"
    return case_config


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_recurrent_kda")
    class Generator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return configure_case(case_config, max(int(self.index) - 1, 0))
