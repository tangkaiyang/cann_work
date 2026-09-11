"""Generate directly coupled ACLNN inputs for PrepareWyReprBwdDa."""

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
DATA_DTYPES = ("bf16", "fp16")
GATE_DTYPES = ("fp32", "bf16", "fp16")
HEAD_PAIRS = ((1, 1), (2, 2), (2, 4), (4, 8), (4, 16), (8, 32))
TIMES = (16, 24, 64, 128, 196, 256, 512)
VALUE_DIMS = (128, 256)
CHUNK_SIZES = (64, 128)


def _profile(index):
    dtype = DATA_DTYPES[index % len(DATA_DTYPES)]
    gtype = GATE_DTYPES[(index // 2) % len(GATE_DTYPES)]
    hk, hv = HEAD_PAIRS[(index // 3) % len(HEAD_PAIRS)]
    time = TIMES[(index // 5) % len(TIMES)]
    varlen = index % 4 == 3
    return {
        "dtype": dtype,
        "gtype": gtype,
        "B": 1 if varlen else (1, 2, 4)[(index // 7) % 3],
        "HK": hk,
        "HV": hv,
        "T": time,
        "K": 128,
        "V": VALUE_DIMS[(index // 11) % 2],
        "chunkSize": CHUNK_SIZES[(index // 13) % 2],
        "varlen": varlen,
    }


def _offsets(total, seed):
    values = [0]
    while values[-1] < total:
        step = 1 + ((seed + len(values) * 3) % min(8, total - values[-1]))
        values.append(min(total, values[-1] + step))
    return values


def _chunk_indices(offsets, chunk_size):
    values = []
    for sequence, (start, end) in enumerate(zip(offsets, offsets[1:])):
        count = (end - start + chunk_size - 1) // chunk_size
        for chunk in range(count):
            values.extend((sequence, chunk))
    return values


def _configs(case_config):
    result = {}
    for item in case_config.inputs:
        cfg = item[0] if isinstance(item, list) else item
        result[cfg.name] = cfg
    return result


def configure_case(case_config, index):
    cfg = _configs(case_config)
    expected = {
        "k", "v", "beta", "a", "dw", "du", "g",
        "cuSeqlensOptional", "chunkIndicesOptional", "chunkSize",
    }
    missing = expected - set(cfg)
    if missing:
        raise KeyError(f"missing ACLNN parameters: {sorted(missing)}")
    p = _profile(index % CASE_COUNT)
    b, hk, hv, t, k, v = (p[name] for name in ("B", "HK", "HV", "T", "K", "V"))
    bt = p["chunkSize"]
    shapes = {
        "k": [b, hk, t, k], "v": [b, hv, t, v],
        "beta": [b, hv, t], "a": [b, hv, t, bt],
        "dw": [b, hv, t, k], "du": [b, hv, t, v],
        "g": [b, hv, t],
    }
    for name, shape in shapes.items():
        cfg[name].shape = shape
    for name in ("k", "v", "a", "dw", "du"):
        cfg[name].dtype = p["dtype"]
    cfg["beta"].dtype = "fp32"
    cfg["g"].dtype = p["gtype"]
    cfg["chunkSize"].range_values = bt
    if p["varlen"]:
        offsets = _offsets(t, 20260817 + index)
        cfg["cuSeqlensOptional"].required = True
        cfg["cuSeqlensOptional"].range_values = offsets
        cfg["chunkIndicesOptional"].required = True
        chunk_indices = _chunk_indices(offsets, bt)
        cfg["chunkIndicesOptional"].range_values = chunk_indices
    else:
        cfg["cuSeqlensOptional"].required = False
        cfg["cuSeqlensOptional"].range_values = "null"
        cfg["chunkIndicesOptional"].required = False
        cfg["chunkIndicesOptional"].range_values = "null"
    case_config.id = index
    case_config.default_seed = 20260817 + index
    case_config.name = f"prepare_wy_repr_bwd_da_{index:04d}"
    return case_config


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_prepare_wy_repr_bwd_da")
    class Generator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return configure_case(case_config, max(int(self.index) - 1, 0))
