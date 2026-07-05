"""Shared curve type definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurveParameterSize:
    name: str
    parameter_count: int
