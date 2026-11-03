from __future__ import annotations

import subprocess
import sys
import unittest

from pv01_examples.export_utils import read_mqs, read_pv
from pv01_examples.paths import CONFIG_STR, PYTHON_DIR, SETTINGS_SUFFIXES

OUTPUT_DIR = PYTHON_DIR / "target" / "example-output-jax"
GOLDEN_DIR = PYTHON_DIR / "tests" / "golden"
EXAMPLE_SCRIPT = (
    PYTHON_DIR / "pv01_examples" / "jax" / "calibration_interpolation_example_jax.py"
)


class TestJaxCalibrationInterpolationExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            [sys.executable, str(EXAMPLE_SCRIPT)],
            cwd=str(PYTHON_DIR.parent),
            check=True,
        )

    def test_example_writes_outputs_to_jax_directory(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                self.assertTrue(
                    (OUTPUT_DIR / f"{CONFIG_STR}{suffix}-pv.csv").is_file()
                )
                self.assertTrue(
                    (OUTPUT_DIR / f"{CONFIG_STR}{suffix}-mqs.csv").is_file()
                )

    def test_jax_outputs_match_golden_pv_and_mqs(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                actual_pv = read_pv(
                    OUTPUT_DIR / f"{CONFIG_STR}{suffix}-pv.csv"
                )
                expected_pv = read_pv(
                    GOLDEN_DIR / f"{CONFIG_STR}{suffix}-pv.csv"
                )
                self.assertAlmostEqual(actual_pv, expected_pv, delta=0.01)

                actual_mqs = read_mqs(
                    OUTPUT_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
                )
                expected_mqs = read_mqs(
                    GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
                )
                self.assertEqual(tuple(actual_mqs), tuple(expected_mqs))
                for label, expected in expected_mqs.items():
                    self.assertAlmostEqual(
                        actual_mqs[label],
                        expected,
                        delta=0.01,
                        msg=f"{suffix}, {label}",
                    )


if __name__ == "__main__":
    unittest.main()
