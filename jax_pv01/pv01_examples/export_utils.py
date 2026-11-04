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


def read_pv(path) -> float:
    """Read the first amount from a file written by ``export_pv``."""
    return float(Path(path).read_text(encoding="utf-8").strip().split(",")[1])


def read_mqs(path) -> dict[str, float]:
    """Read node sensitivities keyed by label from a file written by ``export_mqs``.

    The header row and each curve's ``name, currency`` row have no numeric
    value and are skipped.
    """
    result = {}
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines()[1:]:
        if not line.strip():
            continue
        label, value = [p.strip() for p in line.split(",", 1)]
        try:
            result[label] = float(value)
        except ValueError:
            pass
    return result


def export(content: str, file_name: str) -> None:
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
