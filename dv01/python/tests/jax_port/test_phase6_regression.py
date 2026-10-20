from __future__ import annotations

import csv
import unittest

from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_examples.jax.write_jax_goldens import (
    compute_values,
    golden_path,
)
from tests.jax_port.test_phase5_risk import _problem

_GOLDEN_WRITER = "pv01_examples/jax/write_jax_goldens.py"


class TestPhase6JaxRegression(unittest.TestCase):
    def test_jax_risk_values_match_frozen_goldens(self):
        for suffix in SETTINGS_SUFFIXES:
            path = golden_path(suffix)
            with self.subTest(settings_suffix=suffix):
                self.assertTrue(
                    path.is_file(),
                    f"Missing JAX golden file {path}; generate it with {_GOLDEN_WRITER}.",
                )

                with path.open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)

                problem = _problem(suffix)
                quote_rows = [row for row in rows if row["label"] != "PV"]
                self.assertEqual(
                    tuple(row["label"] for row in quote_rows),
                    problem.quote_labels,
                    f"Quote labels or order differ in {path}",
                )
                self.assertEqual(
                    {row["label"] for row in quote_rows},
                    set(problem.quote_labels),
                )
                pv_rows = [row for row in rows if row["label"] == "PV"]
                self.assertEqual(len(pv_rows), 1, f"Expected one PV row in {path}")

                computed_rows, computed_pv = compute_values(suffix)
                for row, (label, expected_pv01, expected_gamma) in zip(
                    quote_rows,
                    computed_rows,
                ):
                    self.assertAlmostEqual(
                        float(row["mq_pv01"]),
                        expected_pv01,
                        delta=1e-6,
                        msg=f"Market-quote PV01 mismatch for {suffix}, {label}",
                    )
                    self.assertAlmostEqual(
                        float(row["gamma_diag"]),
                        expected_gamma,
                        delta=1e-6,
                        msg=f"Cross-gamma diagonal mismatch for {suffix}, {label}",
                    )
                self.assertAlmostEqual(
                    float(pv_rows[0]["mq_pv01"]),
                    computed_pv,
                    delta=1e-6,
                    msg=f"PV mismatch for {suffix}",
                )


if __name__ == "__main__":
    unittest.main()
