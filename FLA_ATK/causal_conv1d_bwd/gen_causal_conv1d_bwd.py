"""Generate valid cases for the public ACLNN parameter list."""

from __future__ import annotations

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = GENERATOR_REGISTRY = CaseConfig = None


LAYOUTS = ("BSND", "BSH", "TND", "BNSD", "NTD")
FLOAT_DTYPES = {"bf16", "fp16", "fp32"}


def _configs(case_config):
    result = {}
    for item in case_config.inputs:
        config = item[0] if isinstance(item, list) else item
        result[config.name] = config
    return result


def _value(config, default):
    return default if config.range_values in (None, "null") else config.range_values


def _align16(value):
    return max(16, ((int(value) + 15) // 16) * 16)


def configure_case(case_config, index=0):
    cfg = _configs(case_config)
    expected = {
        "x", "yOptional", "weight", "dy", "initialStateOptional",
        "dhtOptional", "queryStartLocOptional", "activation",
        "inputLayoutOptional",
    }
    missing = expected - set(cfg)
    if missing:
        raise KeyError(f"missing ACLNN parameters: {sorted(missing)}")

    dtype = str(cfg["x"].dtype).lower()
    if dtype not in FLOAT_DTYPES:
        dtype = "bf16"
    for name in (
        "x", "yOptional", "weight", "dy", "initialStateOptional",
        "dhtOptional",
    ):
        cfg[name].dtype = dtype

    # Optional string attributes may arrive as null in every ATK seed. Cycle the
    # documented layouts explicitly so attr_tuple cases are actually exercised.
    layout = LAYOUTS[index % len(LAYOUTS)]
    cfg["inputLayoutOptional"].range_values = layout
    activation = int(_value(cfg["activation"], 0))
    activation = activation if activation in (0, 1, 2) else 0
    cfg["activation"].range_values = activation

    raw_shape = list(cfg["x"].shape or [1, 64, 16])
    if len(raw_shape) == 2:
        batch, time, dim = 1, raw_shape[0], raw_shape[1]
    else:
        batch, time, dim = raw_shape[:3]
    batch = min(8, max(1, int(batch)))
    time = min(32768, max(1, int(time)))
    dim = min(1536, _align16(dim))
    time = min(time, max(1, (4 * 1024 * 1024) // (batch * dim)))
    width = max(1, int((cfg["weight"].shape or [4])[0]))

    heads = 2 if dim >= 32 else 1
    head_dim = _align16((dim + heads - 1) // heads)
    if heads * head_dim > 1536:
        heads, head_dim = 1, dim
    dim = heads * head_dim
    varlen = layout in {"TND", "NTD"}
    x_shape = [time, dim] if varlen else [batch, time, dim]
    if layout == "BNSD":
        grad_shape = [batch, heads, time, head_dim]
    elif layout == "NTD":
        grad_shape = [heads, time, head_dim]
    else:
        grad_shape = list(x_shape)
    state_shape = [1 if varlen else batch, width, dim]

    cfg["x"].shape = x_shape
    cfg["weight"].shape = [width, dim]
    for name in ("yOptional", "dy"):
        cfg[name].shape = grad_shape
    for name in ("initialStateOptional", "dhtOptional"):
        cfg[name].shape = state_shape

    cfg["yOptional"].required = activation != 0
    query_values = [0, time]
    cfg["queryStartLocOptional"].required = varlen
    cfg["queryStartLocOptional"].range_values = (
        query_values if varlen else "null"
    )
    return case_config


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
    class CausalConv1dBwdGenerator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return configure_case(case_config, max(int(self.index) - 1, 0))
