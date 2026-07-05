"""MarketQuoteSensitivityCalculator."""

from collections import defaultdict

import numpy as np

from pv01_pricer.rates_provider import CurrencyParameterSensitivities, CurrencyParameterSensitivity, ImmutableRatesProvider


class MarketQuoteSensitivityCalculator:
    DEFAULT = None

    @staticmethod
    def sensitivity(
        param_sensitivities: CurrencyParameterSensitivities,
        provider: ImmutableRatesProvider,
    ) -> CurrencyParameterSensitivities:
        # Accumulate market quote sensitivities per curve name
        accumulated: dict[str, dict] = defaultdict(lambda: {"currency": "GBP", "values": None, "labels": []})

        for ps in param_sensitivities.sensitivities:
            curve = provider.find_curve(ps.market_data_name)
            if curve.jacobian is None:
                raise ValueError(f"Jacobian required for curve {ps.market_data_name}")

            jacobian = curve.jacobian.jacobian_matrix
            market_quote_sens = ps.sensitivity @ jacobian
            split = curve.jacobian.split_values(market_quote_sens)

            for curve_name, values in split.items():
                c = provider.find_curve(curve_name)
                labels = c.node_labels
                if accumulated[curve_name]["values"] is None:
                    accumulated[curve_name]["values"] = np.zeros(len(values))
                    accumulated[curve_name]["labels"] = labels
                    accumulated[curve_name]["currency"] = ps.currency
                accumulated[curve_name]["values"] += values

        result = CurrencyParameterSensitivities.empty()
        for curve_name in provider.curve_order:
            name = curve_name.name
            if accumulated[name]["values"] is not None:
                result = result.combined_with(CurrencyParameterSensitivities([
                    CurrencyParameterSensitivity(
                        market_data_name=name,
                        currency=accumulated[name]["currency"],
                        sensitivity=accumulated[name]["values"],
                        labels=accumulated[name]["labels"],
                    )
                ]))
        return result


MarketQuoteSensitivityCalculator.DEFAULT = MarketQuoteSensitivityCalculator()
