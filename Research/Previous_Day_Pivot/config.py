import os
from pathlib import Path

# Project Directories
PROJECT_ROOT = Path(r"C:\dhan algo\Research\Previous_Day_Pivot")
DATA_FILE = Path(r"C:\dhan algo\dhan_historical_data_180days.csv")

REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = PROJECT_ROOT / "charts"

# Dataset Columns Mapping
COL_DATETIME = "datetime"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"

# Realistic Trading Assumptions (Frictions & Costs)
INITIAL_CAPITAL = 100000.0
BROKERAGE_PCT = 0.0003          # 0.03% per executed order
EXCHANGE_CHARGES_PCT = 0.0000325 # 0.00325% typical NSE exchange transaction charge
GST_PCT = 0.18                  # 18% applied to (Brokerage + Exchange Charges)
STT_PCT = 0.00025               # 0.025% Securities Transaction Tax (on sell side for intraday)
STAMP_DUTY_PCT = 0.00003        # 0.003% applied on buy side
SLIPPAGE_PCT = 0.0005           # 0.05% assumed slippage per leg
EXECUTION_DELAY_SEC = 0.5       # 500ms execution latency

# Analysis Parameters
MIN_MINUTES_PER_DAY = 30        # Minimum candles required to consider a day valid for analysis