import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUTS_DIR = OUTPUT_DIR  # Alias to fix report.py
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

REAL_HISTORICAL_PATH = r"C:\dhan algo\dhan_historical_data_180days.csv"
DATA_PATH = REAL_HISTORICAL_PATH
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "Market_DNA.csv")

def init_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
