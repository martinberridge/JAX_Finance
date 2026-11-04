"""Paths within the pv01 project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
PATH_CONFIG = DATA_DIR / "curves"
PATH_QUOTES = DATA_DIR / "quotes"
PATH_RESULTS = PROJECT_DIR / "target" / "example-output"

CONFIG_STR = "GBP-DSCONOIS-L6MIRS-FRTB"
SETTINGS_SUFFIXES = ("-linear-settings", "-dq-settings", "-ncs-settings")
