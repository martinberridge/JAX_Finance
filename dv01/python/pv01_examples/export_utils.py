"""ExportUtils — CSV export matching Strata example format."""

from pathlib import Path

from pv01_pricer.rates_provider import CurrencyParameterSensitivities


def export_pv(pv: dict, file_name: str) -> None:
    parts = []
    for ccy, amount in pv.items():
        parts.append(f"{ccy},{amount},")
    export("".join(parts), file_name)


def export_mqs(sensitivity: CurrencyParameterSensitivities, scale: float, file_name: str) -> None:
    output = "Label, Value\n"
    for s in sensitivity.sensitivities:
        output += f"{s.market_data_name}, {s.currency}\n"
        for i, label in enumerate(s.labels):
            sens = s.sensitivity[i] * scale
            output += f"{label}, {sens}\n"
    export(output, file_name)


def export(content: str, file_name: str) -> None:
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
