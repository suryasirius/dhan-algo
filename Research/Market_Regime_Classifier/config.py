import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_PATH = r"C:\dhan algo\Research\Market_DNA\outputs\Market_DNA.csv"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
def init_directories():
    for path in [REPORTS_DIR, CHARTS_DIR, OUTPUT_DIR, LOGS_DIR]:
        os.makedirs(path, exist_ok=True)

