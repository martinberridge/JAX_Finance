"""Reference correctness checks for curve calibration and bucketed PV01.

Each check compares an analytic quantity produced by the pv01_* packages with an independent
finite-difference reference, for every interpolator configuration in SETTINGS_SUFFIXES.
The swap is the one priced by calibration_interpolation_example.main().
"""

import unittest
from functools import lru_cache
from typing import Dict, List

import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import (
    BP1,
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_loader.models import RatesCurveGroupDefinition
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.instrument_pricers import par_spread
from pv01_pricer.market_quote_sensitivity import MarketQuoteSensitivityCalculator
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.rates_provider import ImmutableRatesProvider
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import ResolvedTrade, create_test_swap, resolve_calibration_trade
from pv01_pricer.bump import bump_pv01, bumped_market_data

LINEAR = "-linear-settings"
CURVES = ("discount_curve", "libor_curve")
LONG_TENOR_NODES = (
    "GBP-OIS-15Y", "GBP-OIS-20Y", "GBP-OIS-30Y",
    "GBP-IRS6M-15Y", "GBP-IRS6M-20Y", "GBP-IRS6M-30Y",
)

# Finite difference
FD_CURVE_H = 1e-6
FD_QUOTE_H = 1e-6

TA_TOL = 1e-6
TB_PAR_SPREAD_TOL = 1e-12
TB_MAX_COND = 1e8
PV01_TOL = 0.01
TD_LONG_BUCKET_TOL = 0.1
TE_TOL = 1e-8

# ---------------------------------------------------------------------------
# Shared, cached inputs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(VALUATION_DATE, PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv")


@lru_cache(maxsize=None)
def _config(suffix: str) -> RatesCurveGroupDefinition:
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _swap():
    return create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )


@lru_cache(maxsize=None)
def _provider(suffix: str) -> ImmutableRatesProvider:
    return RatesCurveCalibrator.standard().calibrate(_config(suffix), _market_data(), VALUATION_DATE)


@lru_cache(maxsize=None)
def _bump_pv01(suffix: str) -> Dict[str, float]:
    return bump_pv01(_config(suffix), _market_data(), _swap(), VALUATION_DATE)


@lru_cache(maxsize=None)
def _mqs_buckets(suffix: str) -> Dict[str, float]:
    """Market quote sensitivity of the swap PV, in GBP per bp, keyed by node label."""
    provider = _provider(suffix)
    pricer = DiscountingSwapTradePricer.DEFAULT
    ps = provider.parameter_sensitivity(pricer.present_value_sensitivity(_swap(), provider))
    mqs = MarketQuoteSensitivityCalculator.DEFAULT.sensitivity(ps, provider)
    buckets: Dict[str, float] = {}
    for s in mqs.sensitivities:
        for label, value in zip(s.labels, s.sensitivity):
            buckets[label] = buckets.get(label, 0.0) + float(value) * BP1
    return buckets


def _nodes_in_curve_order(suffix: str, provider: ImmutableRatesProvider):
    """Curve nodes ordered as the provider's parameters: discount curve first, then LIBOR."""
    config = _config(suffix)
    return [node for curve in (provider.discount_curve, provider.libor_curve) for node in config.curves[curve.name].nodes]


def _calibration_trades(suffix: str, provider: ImmutableRatesProvider, market_data=None) -> List[ResolvedTrade]:
    market_data = market_data or _market_data()
    return [
        resolve_calibration_trade(node, VALUATION_DATE, market_data)
        for node in _nodes_in_curve_order(suffix, provider)
    ]


def _with_curve_y(provider: ImmutableRatesProvider, curve_attr: str, y: np.ndarray) -> ImmutableRatesProvider:
    """Rebuild the provider with one curve's y_values replaced and the other curve unchanged."""
    curves = {attr: getattr(provider, attr) for attr in CURVES}
    curves[curve_attr] = curves[curve_attr].with_y_values(y)
    return ImmutableRatesProvider(
        valuation_date=provider.valuation_date,
        discount_curve=curves["discount_curve"],
        libor_curve=curves["libor_curve"],
        curve_order=provider.curve_order,
    )


def _with_params(provider: ImmutableRatesProvider, params: np.ndarray) -> ImmutableRatesProvider:
    n_dsc = provider.discount_curve.parameter_count()
    bumped = _with_curve_y(provider, "discount_curve", params[:n_dsc])
    return _with_curve_y(bumped, "libor_curve", params[n_dsc:])


def _residuals(trades: List[ResolvedTrade], provider: ImmutableRatesProvider) -> np.ndarray:
    return np.array([par_spread(t, provider) for t in trades])


# ---------------------------------------------------------------------------
# T-A: analytic parameter sensitivity vs FD on curve y_values (curves fixed)
# ---------------------------------------------------------------------------

class TestReferenceCorrectness(unittest.TestCase):
    def _assert_buckets_close(
        self, actual: Dict[str, float], expected: Dict[str, float], atol: float, what: str
    ) -> None:
        self.assertLessEqual(
            set(expected) - set(actual),
            set(),
            f"{what}: missing labels {sorted(set(expected) - set(actual))}",
        )
        diffs = {label: actual[label] - expected[label] for label in expected}
        worst = max(diffs, key=lambda label: abs(diffs[label]))
        detail = ", ".join(
            f"{label}: {actual[label]:.6f} vs {expected[label]:.6f}"
            for label in expected
            if abs(diffs[label]) > atol
        )
        self.assertLessEqual(
            abs(diffs[worst]),
            atol,
            f"{what}: max |diff| {abs(diffs[worst]):.6g} at {worst}; {detail}",
        )

    def _check_ta_parameter_sensitivity_matches_curve_fd(self, suffix, curve_attr):
        provider = _provider(suffix)
        swap = _swap()
        pricer = DiscountingSwapTradePricer.DEFAULT
        curve = getattr(provider, curve_attr)

        ps = provider.parameter_sensitivity(pricer.present_value_sensitivity(swap, provider))
        analytic = sum(
            (s.sensitivity for s in ps.sensitivities if s.market_data_name == curve.name),
            np.zeros(curve.parameter_count()),
        )

        fd = np.zeros(curve.parameter_count())
        for i in range(curve.parameter_count()):
            y_up = curve.y_values.copy()
            y_dn = curve.y_values.copy()
            y_up[i] += FD_CURVE_H
            y_dn[i] -= FD_CURVE_H
            pv_up = pricer.present_value(swap, _with_curve_y(provider, curve_attr, y_up))["GBP"]
            pv_dn = pricer.present_value(swap, _with_curve_y(provider, curve_attr, y_dn))["GBP"]
            fd[i] = (pv_up - pv_dn) / (2.0 * FD_CURVE_H)

        self._assert_buckets_close(
            dict(zip(curve.node_labels, analytic * BP1)),
            dict(zip(curve.node_labels, fd * BP1)),
            TA_TOL,
            f"{curve.name} parameter sensitivity (GBP/bp)",
        )

    def _check_tb_calibration_par_spreads_are_zero(self, suffix):
        provider = _provider(suffix)
        trades = _calibration_trades(suffix, provider)
        residuals = _residuals(trades, provider)
        worst = int(np.argmax(np.abs(residuals)))
        self.assertLess(
            abs(residuals[worst]),
            TB_PAR_SPREAD_TOL,
            f"max |par spread| {abs(residuals[worst]):.3g} at {trades[worst].label}",
        )

    def _check_tb_calibration_jacobian_well_conditioned(self, suffix):
        provider = _provider(suffix)
        trades = _calibration_trades(suffix, provider)
        params = np.concatenate([provider.discount_curve.y_values, provider.libor_curve.y_values])

        jac = np.zeros((len(trades), len(params)))
        for j in range(len(params)):
            up = params.copy()
            dn = params.copy()
            up[j] += FD_CURVE_H
            dn[j] -= FD_CURVE_H
            jac[:, j] = (
                _residuals(trades, _with_params(provider, up)) - _residuals(trades, _with_params(provider, dn))
            ) / (2.0 * FD_CURVE_H)

        rank = np.linalg.matrix_rank(jac)
        cond = np.linalg.cond(jac)
        self.assertEqual(rank, len(params), f"rank {rank} < {len(params)} (cond {cond:.3g})")
        self.assertLess(cond, TB_MAX_COND, f"cond(J) {cond:.3g}")

    def _check_tc_market_quote_sensitivity_matches_bump(self, suffix):
        self._assert_buckets_close(
            _mqs_buckets(suffix), _bump_pv01(suffix), PV01_TOL, "MQS vs bump PV01 (GBP/bp)"
        )

    def _check_td_bucket_sum_matches_parallel_bump(self, suffix):
        config = _config(suffix)
        quote_ids = [node.quote_id for curve in config.curves.values() for node in curve.nodes]
        pvs = []
        for shift in (BP1, -BP1):
            md = bumped_market_data(_market_data(), {q: shift for q in quote_ids})
            provider = RatesCurveCalibrator.standard().calibrate(config, md, VALUATION_DATE)
            pvs.append(DiscountingSwapTradePricer.DEFAULT.present_value(_swap(), provider)["GBP"])
        parallel = (pvs[0] - pvs[1]) / 2.0

        bucket_sum = sum(_mqs_buckets(suffix).values())
        self.assertLessEqual(
            abs(bucket_sum - parallel),
            PV01_TOL,
            f"sum of buckets {bucket_sum:.6f} vs parallel bump {parallel:.6f}, "
            f"diff {bucket_sum - parallel:.6g}",
        )

    def _check_td_linear_long_tenor_buckets_negligible(self):
        buckets = _mqs_buckets(LINEAR)
        long_buckets = {label: buckets[label] for label in LONG_TENOR_NODES}
        worst = max(long_buckets, key=lambda label: abs(long_buckets[label]))
        self.assertLess(
            abs(long_buckets[worst]),
            TD_LONG_BUCKET_TOL,
            f"max |bucket| {abs(long_buckets[worst]):.6g} at {worst}; {long_buckets}",
        )

    def _check_te_residual_quote_jacobian_is_minus_identity(self, suffix):
        provider = _provider(suffix)
        nodes = _nodes_in_curve_order(suffix, provider)

        dr_dq = np.zeros((len(nodes), len(nodes)))
        for j, node in enumerate(nodes):
            md_up = bumped_market_data(_market_data(), {node.quote_id: FD_QUOTE_H})
            md_dn = bumped_market_data(_market_data(), {node.quote_id: -FD_QUOTE_H})
            r_up = _residuals(_calibration_trades(suffix, provider, md_up), provider)
            r_dn = _residuals(_calibration_trades(suffix, provider, md_dn), provider)
            dr_dq[:, j] = (r_up - r_dn) / (2.0 * FD_QUOTE_H)

        err = np.abs(dr_dq + np.eye(len(nodes)))
        i, j = np.unravel_index(int(np.argmax(err)), err.shape)
        self.assertLessEqual(
            err[i, j],
            TE_TOL,
            f"max |dR/dq + I| {err[i, j]:.3g} at R[{nodes[i].label}] / q[{nodes[j].label}] "
            f"(dR/dq = {dr_dq[i, j]:.6g})",
        )


def _register_test_case(name: str, check, args: tuple) -> None:
    def test(self):
        check(self, *args)

    test.__name__ = name
    setattr(TestReferenceCorrectness, name, test)


def _case_name(value: str) -> str:
    return value.strip("-").replace("-", "_")


for _suffix in SETTINGS_SUFFIXES:
    _suffix_name = _case_name(_suffix)
    for _curve_attr in CURVES:
        _register_test_case(
            f"test_ta_parameter_sensitivity_matches_curve_fd__{_suffix_name}__{_curve_attr}",
            TestReferenceCorrectness._check_ta_parameter_sensitivity_matches_curve_fd,
            (_suffix, _curve_attr),
        )

    _register_test_case(
        f"test_tb_calibration_par_spreads_are_zero__{_suffix_name}",
        TestReferenceCorrectness._check_tb_calibration_par_spreads_are_zero,
        (_suffix,),
    )
    _register_test_case(
        f"test_tb_calibration_jacobian_well_conditioned__{_suffix_name}",
        TestReferenceCorrectness._check_tb_calibration_jacobian_well_conditioned,
        (_suffix,),
    )
    _register_test_case(
        f"test_tc_market_quote_sensitivity_matches_bump__{_suffix_name}",
        TestReferenceCorrectness._check_tc_market_quote_sensitivity_matches_bump,
        (_suffix,),
    )
    _register_test_case(
        f"test_td_bucket_sum_matches_parallel_bump__{_suffix_name}",
        TestReferenceCorrectness._check_td_bucket_sum_matches_parallel_bump,
        (_suffix,),
    )
    _register_test_case(
        f"test_te_residual_quote_jacobian_is_minus_identity__{_suffix_name}",
        TestReferenceCorrectness._check_te_residual_quote_jacobian_is_minus_identity,
        (_suffix,),
    )


    _register_test_case(
        "test_td_linear_long_tenor_buckets_negligible",
        TestReferenceCorrectness._check_td_linear_long_tenor_buckets_negligible,
        (),
    )
