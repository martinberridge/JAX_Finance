"""DiscountingSwapTradePricer."""

from pv01_basics.day_count import act_365f_year_fraction
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import ResolvedSwapTrade
from pv01_pricer.instrument_pricers import _pv_fixed_leg, _pv_float_leg_ibor
from pv01_pricer.rates_provider import ImmutableRatesProvider, PointSensitivities, PointSensitivity


def _sign(buy_sell: BuySell) -> float:
    return 1.0 if buy_sell == BuySell.BUY else -1.0


class DiscountingSwapTradePricer:
    DEFAULT = None

    @staticmethod
    def present_value(swap: ResolvedSwapTrade, provider: ImmutableRatesProvider) -> dict:
        pv_fixed = _pv_fixed_leg(swap.fixed_leg, provider, swap.fixed_rate)
        pv_float = _pv_float_leg_ibor(swap.float_leg, provider)
        total = (pv_float - pv_fixed) * _sign(swap.buy_sell)
        return {"GBP": total}

    @staticmethod
    def present_value_sensitivity(swap: ResolvedSwapTrade, provider: ImmutableRatesProvider) -> PointSensitivities:
        sens = PointSensitivities()
        sign = _sign(swap.buy_sell)

        for leg, is_fixed in [(swap.fixed_leg, True), (swap.float_leg, False)]:
            for p in leg:
                accrual = act_365f_year_fraction(p.start, p.end)
                df = provider.discount_factor(p.payment)
                yf_pay = provider.year_fraction(p.payment)

                # Discount factor sensitivity
                if is_fixed:
                    forecast = swap.notional * swap.fixed_rate * accrual
                else:
                    fwd = provider.forward_rate(p.start, p.end)
                    forecast = swap.notional * fwd * accrual

                df_sens = PointSensitivity("ZeroRate", yf_pay, "GBP", sign * forecast * (-df * yf_pay))

                if not is_fixed:
                    yf_end = provider.year_fraction(p.end)
                    ibor_sens = PointSensitivity("IborRate", yf_end, "GBP", sign * swap.notional * accrual * df)
                    sens = sens.combined_with(PointSensitivities([ibor_sens]))

                sens = sens.combined_with(PointSensitivities([df_sens]))

        return sens


DiscountingSwapTradePricer.DEFAULT = DiscountingSwapTradePricer()
