"""Synthetic two-curve calibration problems of any size.

Used by the scaling benchmark and by structural tests that need no reference
implementation. Everything is built directly from ``pv01_jax.arrays``; no
Strata-layer objects are involved.

The problem has an OIS discount curve calibrated to an ``OIS`` block and a
forward curve calibrated to an ``IRS`` block, each with N/2 nodes at tenors
evenly spaced out to 30 years. Each calibration swap runs from 0 to its node
tenor in equal periods of at most a year, with year fractions as plain
floats. The quotes lie on smooth upward-sloping curves (about 0.5% to 3%),
close enough to the zero rates that Newton converges from ``q0``.

The test swap is a 20-year annual fixed-versus-floating swap. Its payments
cover two-thirds of the curve and leave nodes beyond its last payment, whose
zero-rate PV01 must be exactly zero under linear interpolation.
"""

from __future__ import annotations

import math

import numpy as np

from pv01_jax.arrays import (
    CalibBlock,
    CalibrationProblem,
    CurveSpec,
    FloatLegArrays,
    LegArrays,
    SwapArrays,
)

MAX_TENOR_YEARS = 30.0
SWAP_TENOR_YEARS = 20
SWAP_COUPON = 0.025
SWAP_NOTIONAL = 10_000_000.0
DISCOUNT_CURVE = "SYN-OIS"
FORWARD_CURVE = "SYN-IRS"


def _ois_quote(t: np.ndarray) -> np.ndarray:
    return 0.005 + 0.025 * (1.0 - np.exp(-t / 7.0))


def _irs_quote(t: np.ndarray) -> np.ndarray:
    return _ois_quote(t) + 0.0015 + 0.0005 * (1.0 - np.exp(-t / 5.0))


def _periods(maturity: float) -> tuple[np.ndarray, np.ndarray]:
    n_periods = max(1, math.ceil(maturity - 1e-9))
    edges = np.linspace(0.0, maturity, n_periods + 1)
    return edges[:-1], edges[1:]


def _block(trade_type: str, tenors: np.ndarray, quote_offset: int) -> CalibBlock:
    starts, ends, trade_ids = [], [], []
    for trade_id, tenor in enumerate(tenors):
        start, end = _periods(float(tenor))
        starts.append(start)
        ends.append(end)
        trade_ids.append(np.full(len(start), trade_id, dtype=np.int64))
    start = np.concatenate(starts)
    end = np.concatenate(ends)
    tau = end - start
    zeros = np.zeros_like(start)
    ones = np.ones_like(start)
    return CalibBlock(
        trade_type=trade_type,
        quote_idx=np.arange(quote_offset, quote_offset + len(tenors), dtype=np.int64),
        trade_id=np.concatenate(trade_ids),
        float_leg=FloatLegArrays(start=start, end=end, rate=zeros, notional=ones),
        fixed_leg=LegArrays(start=start, end=end, rate=zeros, notional=ones),
        t_pay=end,
        tau=tau,
        t_fix_start=start,
        t_fix_end=end,
        tau_fix=tau,
        sign=np.ones(len(tenors), dtype=np.float64),
    )


def _test_swap() -> SwapArrays:
    start = np.arange(SWAP_TENOR_YEARS, dtype=np.float64)
    end = start + 1.0
    tau = end - start
    n = len(start)
    return SwapArrays(
        float_leg=FloatLegArrays(
            start=start, end=end, rate=np.zeros(n), notional=np.full(n, SWAP_NOTIONAL)
        ),
        spread=np.zeros(n),
        fixed_leg=LegArrays(
            start=start, end=end, rate=np.full(n, SWAP_COUPON), notional=np.full(n, SWAP_NOTIONAL)
        ),
        t_pay=end,
        trade_id=np.zeros(n, dtype=np.int64),
        sign=np.ones(n),
        fixed_rate=SWAP_COUPON,
        tau=tau,
        t_fix_start=start,
        t_fix_end=end,
        tau_fix=tau,
    )


def synthetic_problem(
    n_quotes: int,
    interpolator: str,
) -> tuple[CalibrationProblem, SwapArrays]:
    """Return a calibration problem with ``n_quotes`` quotes and a test swap.

    ``n_quotes`` must be even and at least 4; each curve gets half. The
    ``interpolator`` is a key of ``pv01_jax.interp.INTERPOLATORS``
    (``"Linear"``, ``"DoubleQuadratic"`` or ``"NaturalCubicSpline"``) and is
    used for both curves.
    """
    if n_quotes < 4 or n_quotes % 2:
        raise ValueError(f"n_quotes must be even and at least 4, got {n_quotes}")

    m = n_quotes // 2
    tenors = MAX_TENOR_YEARS * np.arange(1, m + 1, dtype=np.float64) / m
    curves = []
    for name, prefix, offset in ((DISCOUNT_CURVE, "OIS", 0), (FORWARD_CURVE, "IRS", m)):
        curves.append(
            CurveSpec(
                name=name,
                param_slice=(offset, offset + m),
                labels=tuple(f"{prefix}-{t:g}Y" for t in tenors),
                interpolator=interpolator,
                x_nodes=tenors,
            )
        )

    problem = CalibrationProblem(
        name=f"SYNTHETIC-{n_quotes}",
        quote_labels=tuple(label for curve in curves for label in curve.labels),
        discount_curve=DISCOUNT_CURVE,
        forward_curve=FORWARD_CURVE,
        curves=tuple(curves),
        blocks=(_block("OIS", tenors, 0), _block("IRS", tenors, m)),
        q0=np.concatenate([_ois_quote(tenors), _irs_quote(tenors)]),
    )
    return problem, _test_swap()


__all__ = ["synthetic_problem"]
