"""Metadata contract tests; no ATK/NPU tensors are allocated."""
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parent


def load_generator():
    modules = {}
    for name in ("atk.case_generator.generator.base_generator",
                 "atk.case_generator.generator.generate_types", "atk.configs.case_config"):
        modules[name] = types.ModuleType(name)
    modules["atk.case_generator.generator.base_generator"].CaseGenerator = object
    modules["atk.configs.case_config"].CaseConfig = object
    modules["atk.case_generator.generator.generate_types"].GENERATOR_REGISTRY = types.SimpleNamespace(
        register=lambda name: lambda cls: cls)
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "mla_generator", ROOT / "ascend_generate_aclnn_mla_prolog_v4_a5.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class GeneratorContractTest(unittest.TestCase):
    def test_document_constraints_and_coverage(self):
        module = load_generator()
        metadata = yaml.safe_load((ROOT / "aclnnMlaPrologV4WeightNz_a5.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(metadata["inputs"]), 33)
        config = types.SimpleNamespace(inputs=[types.SimpleNamespace() for _ in range(33)])
        generator = module.MlaPrologV4A5Generator()
        for profile, expected_count in (("golden", 10), ("full", 16)):
            seen, layouts, rope, empty, smooth = set(), set(), set(), set(), set()
            random.seed(20260917)
            with patch.dict(os.environ, MLA_V4_PROFILE=profile):
                for _ in range(5000):
                    he = random.choice([1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192])
                    seq = random.choice([0, 1, 3, 16, 17, 127, 128, 129, 256])
                    shape = ([seq, he] if random.choice([True, False]) else
                             [random.choice([0, 1, 2, 4]), seq, he])
                    config.inputs[0].shape = shape[:]
                    generator.after_case_config(config)
                    self.assertEqual(config.inputs[0].shape, shape)
                    x = config.inputs
                    attr = lambda i: x[i].range_values
                    wq, kvq, qq = attr(24), attr(25), attr(26)
                    seen.add((wq, kvq, qq))
                    mode = attr(23)
                    leading, he = x[0].shape[:-1], x[0].shape[-1]
                    t = math.prod(leading)
                    layouts.add((mode, len(leading)))
                    rope.add(attr(32))
                    empty.add(t == 0)
                    smooth.add(x[18].shape != [])
                    self.assertIn((wq, kvq, qq), module.A5_SCENES)
                    self.assertEqual(qq, int(kvq == 1))
                    self.assertEqual((attr(27), attr(28)), (int(kvq == 3),) * 2)
                    self.assertEqual(x[9].shape[-1], 656 if kvq == 3 else 512)
                    if kvq == 3:
                        self.assertIn(mode, ("PA_BSND", "BSND", "TND"))
                        self.assertEqual(x[10].shape, [0])
                        self.assertEqual(x[10].dtype, "bf16")
                    if attr(32):
                        for idx in (7, 8):
                            self.assertEqual(x[idx].shape, leading + [64])
                            self.assertTrue(x[idx].required)
                    else:
                        self.assertEqual((x[7].shape, x[8].shape), ([], []))
                    if mode in ("BSND", "TND"):
                        self.assertEqual(len(leading), 2 if mode == "BSND" else 1)
                        self.assertEqual(x[11].shape, [])
                        self.assertEqual(x[9].shape[:-2], leading)
                    elif mode.startswith("PA_BLK"):
                        block_size = x[9].shape[1]
                        count = math.ceil(leading[-1] / block_size)
                        self.assertEqual(x[11].shape, [count] if len(leading) == 1 else [leading[0], count])
                        if len(leading) == 1:
                            self.assertEqual(x[19].shape, [1])
                            self.assertEqual(attr(19), t)
                    else:
                        self.assertEqual(x[11].shape, leading)
                        self.assertGreaterEqual(math.prod(x[9].shape[:2]), t)
                    if wq >= 2:
                        self.assertEqual(x[12].shape, [t, he // 32 if wq == 3 else 1])
                    else:
                        self.assertEqual(x[12].shape, [])
                    self.assertEqual(x[16].shape, [1] if kvq == 1 else [1, 512] if kvq == 2 else [])
                    self.assertEqual(x[17].shape, [1, 64] if kvq == 2 else [])
                    self.assertEqual(x[20].shape, [1] if kvq == 3 and wq in (1, 2) else [])
                    for idx, inp in enumerate(x):
                        definition = metadata["inputs"][idx]
                        if idx >= 21:
                            self.assertIn(inp.range_values, definition["ranges"]["valid"]["values"])
                        elif inp.shape:
                            self.assertIn(inp.dtype, definition["dtypes"]["values"])
                            if idx != 0:  # Exercise both supported ranks beyond the YAML default.
                                self.assertIn(len(inp.shape), definition["shapes"]["dim_numbers"]["values"])
            self.assertEqual(len(seen), expected_count)
            self.assertEqual(len(layouts), 10)
            self.assertEqual(rope, {True, False})
            self.assertEqual(empty, {True, False})
            self.assertEqual(smooth, {True, False})

    def test_bad_profile(self):
        with patch.dict(os.environ, MLA_V4_PROFILE="typo"):
            with self.assertRaises(ValueError):
                load_generator()._scenes()

    def test_shape_constraints(self):
        module = load_generator()
        generator = module.MlaPrologV4A5Generator()
        def config_for(shape):
            inputs = [types.SimpleNamespace() for _ in range(33)]
            inputs[0].shape = shape[:]
            return types.SimpleNamespace(inputs=inputs)
        with patch.dict(os.environ, MLA_V4_PROFILE="golden"):
            repairs = [([], [1, 1024]), ([7168], [1, 7168]),
                       ([2, 3, 4, 7168], [24, 7168]), ([-1, 7168], [0, 7168]),
                       ([1, -1, 7168], [1, 0, 7168]), ([1, 1000], [1, 1024]),
                       ([1, 9000], [1, 8192]), ([1, 7500], [1, 7680]),
                       ([65537, 1, 7168], [65536, 1, 7168]),
                       ([1.5, 7168], [1, 7168]), ([True, 7168], [1, 7168])]
            for shape, expected in repairs:
                with self.subTest(shape=shape):
                    config = config_for(shape)
                    generator.after_case_config(config)
                    self.assertEqual(config.inputs[0].shape, expected)
                    self.assertEqual(config.inputs[1].shape[0], expected[-1])
                    self.assertEqual(config.inputs[4].shape[0], expected[-1])
                    if config.inputs[32].range_values:
                        self.assertEqual(config.inputs[7].shape, expected[:-1] + [64])
            for shape in ([0, 1024], [0, 1, 8192], [65536, 0, 7168], [2**31, 7168]):
                config = config_for(shape)
                generator.after_case_config(config)
                self.assertEqual(config.inputs[0].shape, shape)
                if shape == [2**31, 7168]:
                    self.assertFalse(config.inputs[23].range_values.startswith("PA_BLK"))


if __name__ == "__main__":
    unittest.main()
