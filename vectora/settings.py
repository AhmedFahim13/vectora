"""Central paths and constants. No logic here."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
DB_PATH = DATA_DIR / "vectora.duckdb"
HOLIDAYS_CSV = REFERENCE_DIR / "holidays.csv"

DSE_BASE = "https://www.dsebd.org"
USER_AGENT = "VectoraResearch/0.1 (personal academic research)"
REQUEST_DELAY_S = 1.5
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 3

# Data-quality alert floor (spec §5.3)
MIN_QUALITY_SCORE = 80

BACKFILL_PARQUET = REFERENCE_DIR / "backfill_2012_2026.parquet"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

# Signal admission (spec §9.3). g10_h30 excluded: overconfident tail
# (training report 2026-07-16) until Phase 5 recalibration.
SIGNAL_THRESHOLDS = {"g5_h10": 0.55}
ANALOG_K = 20
POSITION_TAKA = 500_000  # assumed position size for exit-days liquidity risk
