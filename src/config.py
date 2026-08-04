"""Central configuration for the appliance-energy forecasting pipeline.

Keeping the knobs in one place makes the whole study reproducible: change a
value here and every part of the pipeline picks it up.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "outputs" / "figures"
METRIC_DIR = ROOT / "outputs" / "metrics"
for _d in (DATA_DIR, FIG_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data ------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)
# A couple of mirrors are tried in order if the primary UCI host is down.
DATA_MIRRORS = [
    DATA_URL,
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv",
]
RAW_CSV = DATA_DIR / "energydata_complete.csv"

# --- Forecasting problem ---------------------------------------------------
TARGET = "Appliances"          # appliance energy use in Wh (10-min original)
FREQ = "h"                     # we model at hourly resolution
HORIZON = 24                   # forecast the next 24 hours
DAILY_SEASON = 24              # hours in a day
WEEKLY_SEASON = 24 * 7         # hours in a week
ML_TEST_DAYS = 14              # Part 6: last 14 days used as the test period

# --- SARIMAX grid search ---------------------------------------------------
# Brief requires looping p in [0,6], d in [0,2], q in [0,6].
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)
# Seasonal search is kept small because a daily period (m=24) is expensive.
SEASONAL_ORDER_CANDIDATES = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
SARIMAX_TRAIN_DAYS = 45        # cap history for tractable m=24 SARIMA fits

RANDOM_STATE = 42
