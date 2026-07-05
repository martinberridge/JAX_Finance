"""Path constants relative to the Strata repository root."""

from pathlib import Path

# python/pv01_examples/paths.py -> repo root is two levels up from python/
PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent

EXAMPLE_CALIBRATION = REPO_ROOT / "examples" / "src" / "main" / "resources" / "example-calibration"
PATH_CONFIG = EXAMPLE_CALIBRATION / "curves"
PATH_QUOTES = EXAMPLE_CALIBRATION / "quotes"
PATH_RESULTS = PYTHON_DIR / "target" / "example-output"

CONFIG_STR = "GBP-DSCONOIS-L6MIRS-FRTB"
SETTINGS_SUFFIXES = ("-linear-settings", "-dq-settings", "-ncs-settings")
