"""RatesCalibrationCsvLoader — load curve group, settings, and nodes."""

from pathlib import Path
from typing import Dict, Union

import pandas as pd

from pv01_loader.models import (
    CurveDefinition,
    CurveGroupEntry,
    CurveNode,
    CurveSettings,
    RatesCurveGroupDefinition,
)
from pv01_market.quote_id import QuoteId

INTERPOLATOR_MAP = {
    "Linear": "Linear",
    "DoubleQuadratic": "DoubleQuadratic",
    "NaturalCubicSpline": "NaturalCubicSpline",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_group(path: Path) -> list[CurveGroupEntry]:
    df = _read_csv(path)
    entries = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Group Name")):
            continue
        entries.append(CurveGroupEntry(
            group_name=str(row["Group Name"]).strip(),
            curve_type=str(row["Curve Type"]).strip(),
            reference=str(row["Reference"]).strip(),
            curve_name=str(row["Curve Name"]).strip(),
        ))
    return entries


def _parse_settings(path: Path) -> Dict[str, CurveSettings]:
    df = _read_csv(path)
    settings: Dict[str, CurveSettings] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Curve Name")):
            continue
        name = str(row["Curve Name"]).strip()
        settings[name] = CurveSettings(
            curve_name=name,
            value_type=str(row["Value Type"]).strip(),
            day_count=str(row["Day Count"]).strip(),
            interpolator=INTERPOLATOR_MAP.get(str(row["Interpolator"]).strip(), str(row["Interpolator"]).strip()),
            left_extrapolator=str(row["Left Extrapolator"]).strip(),
            right_extrapolator=str(row["Right Extrapolator"]).strip(),
        )
    return settings


def _parse_nodes(path: Path) -> Dict[str, list[CurveNode]]:
    df = _read_csv(path)
    nodes_by_curve: Dict[str, list[CurveNode]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Curve Name")) or str(row.get("Curve Name", "")).strip() == "":
            continue
        curve_name = str(row["Curve Name"]).strip()
        time_val = row.get("Time")
        time_str = None if pd.isna(time_val) or str(time_val).strip() == "" else str(time_val).strip()
        node = CurveNode(
            curve_name=curve_name,
            label=str(row["Label"]).strip(),
            quote_id=QuoteId(
                str(row["Symbology"]).strip(),
                str(row["Ticker"]).strip(),
                str(row.get("Field Name", "MarketValue")).strip() or "MarketValue",
            ),
            node_type=str(row["Type"]).strip(),
            convention=str(row["Convention"]).strip(),
            time=time_str,
        )
        nodes_by_curve.setdefault(curve_name, []).append(node)
    return nodes_by_curve


def load(
    groups_resource: Union[str, Path],
    settings_resource: Union[str, Path],
    *curve_node_resources: Union[str, Path],
) -> Dict[str, RatesCurveGroupDefinition]:
    """Load and merge group, settings, and node CSV files."""
    entries = _parse_group(Path(groups_resource))
    settings = _parse_settings(Path(settings_resource))
    nodes: Dict[str, list[CurveNode]] = {}
    for resource in curve_node_resources:
        for curve_name, curve_nodes in _parse_nodes(Path(resource)).items():
            nodes.setdefault(curve_name, []).extend(curve_nodes)

    if not entries:
        raise ValueError("No group entries found")

    group_name = entries[0].group_name
    curves: Dict[str, CurveDefinition] = {}
    for curve_name, curve_settings in settings.items():
        curves[curve_name] = CurveDefinition(
            name=curve_name,
            settings=curve_settings,
            nodes=nodes.get(curve_name, []),
        )

    definition = RatesCurveGroupDefinition(
        group_name=group_name,
        entries=entries,
        curves=curves,
    )
    return {group_name: definition}
