import unittest

from pv01_examples.common import VALUATION_DATE, load_example_inputs
from pv01_examples.export_utils import read_mqs, read_pv
from pv01_examples.paths import CONFIG_STR, PROJECT_DIR, SETTINGS_SUFFIXES
from pv01_jax.workflow import price_and_risk

GOLDEN_DIR = PROJECT_DIR / "tests" / "golden"


class TestPhase7Workflow(unittest.TestCase):
    """price_and_risk against the goldens, at the delta=0.01 GBP used by test_phase5e_jax_example."""

    def test_price_and_risk_matches_golden_pv_and_mqs(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                result = price_and_risk(*load_example_inputs(suffix), VALUATION_DATE)

                expected_pv = read_pv(GOLDEN_DIR / f"{CONFIG_STR}{suffix}-pv.csv")
                self.assertAlmostEqual(result["pv"], expected_pv, delta=0.01)

                expected_mqs = read_mqs(GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv")
                actual_mqs = {
                    label: value
                    for buckets in result["pv01"].values()
                    for label, value in buckets.items()
                }
                self.assertEqual(tuple(actual_mqs), tuple(expected_mqs))
                for label, expected in expected_mqs.items():
                    self.assertAlmostEqual(
                        actual_mqs[label], expected, delta=0.01, msg=f"{suffix}, {label}"
                    )

    def test_pv01_is_grouped_by_curve(self):
        config, market_data, swap = load_example_inputs(SETTINGS_SUFFIXES[0])
        result = price_and_risk(config, market_data, swap, VALUATION_DATE)
        self.assertEqual(set(result["pv01"]), set(config.curves))
        for curve_name, buckets in result["pv01"].items():
            labels = tuple(node.label for node in config.curves[curve_name].nodes)
            self.assertEqual(tuple(buckets), labels)


if __name__ == "__main__":
    unittest.main()
