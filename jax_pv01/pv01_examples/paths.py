# Derived in part from OpenGamma Strata (https://github.com/OpenGamma/Strata):
#   CalibrationInterpolationExample
# Copyright (C) 2017 - present by OpenGamma Inc. and the OpenGamma group of companies
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.txt.
# Changed by Martin Berridge, 2026: example configuration name and file suffixes
#   translated to Python.

"""Paths within the jax_pv01 project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
PATH_CONFIG = DATA_DIR / "curves"
PATH_QUOTES = DATA_DIR / "quotes"
PATH_RESULTS = PROJECT_DIR / "target" / "example-output"

CONFIG_STR = "GBP-DSCONOIS-L6MIRS-FRTB"
SETTINGS_SUFFIXES = ("-linear-settings", "-dq-settings", "-ncs-settings")
