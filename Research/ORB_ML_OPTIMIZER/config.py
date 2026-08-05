import os
from pathlib import Path

# Project Directory Structure
PROJECT_ROOT = Path(r"C:\dhan algo\Research\ORB_ML_OPTIMIZER")
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = PROJECT_ROOT / "charts"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data Source Configuration
DATA_FILE = Path(r"C:\dhan algo\dhan_historical_data_180days.csv")

# Column Name Mappings for Dataset
COL_DATETIME = "datetime"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"

# Minimum records required per session to qualify for analysis
MIN_MINUTES_PER_DAY = 305  # Standard full trading session minutes

# Financial & Risk Parameters (Indian Market Standards)
INITIAL_CAPITAL = 100000.0
BROKERAGE_PCT = 0.0003      # 0.03% or flat fee structure proxy
EXCHANGE_CHARGES_PCT = 0.0000325
GST_PCT = 0.18              # 18% on (brokerage + exchange charges)
STT_PCT = 0.001             # Securities Transaction Tax on sell side (delivery/intraday composite estimate)
STAMP_DUTY_PCT = 0.00003    # Stamp duty on buy side
SLIPPAGE_PCT = 0.0005       # 0.05% estimated execution slippage

# Research Grid Search Parameters
ORB_DURATIONS = [5, 10, 15, 20, 30, 45, 60]                  # Minutes
STOP_LOSS_TICKS_OR_PCT = [4, 5, 6, 7, 8, 10, 12, 15, 20]     # Multipliers / Points / ATR fractions
TRAILING_STOPS = [2, 3, 4, 5, 6, 8, 10, 12]                  # R-multiples or points
FIXED_TARGETS = ["1R", "1.5R", "2R", "3R", "4R", "No Target"]
TIME_FILTERS = ["09:20", "09:25", "09:30", "09:35", "09:40", "09:45", "10:00", "10:15"]
DAY_FILTERS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CONFIDENCE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Machine Learning & Validation Parameters
TRAIN_SPLIT = 0.70
TEST_SPLIT = 0.30
MONTE_CARLO_SIMULATIONS = 5000