import os

PROJECT_ROOT = r"C:\dhan algo\Research\AI_Trade_Decision_Engine"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

HISTORICAL_DATA_PATH = r"C:\dhan algo\dhan_historical_data_180days.csv"
MARKET_DNA_PATH = r"C:\dhan algo\Research\Market_DNA\outputs\Market_DNA.csv"
MARKET_REGIMES_PATH = r"C:\dhan algo\Research\Market_Regime_Classifier\output\Market_Regimes.csv"
STRATEGY_SELECTOR_PATH = r"C:\dhan algo\Research\Strategy_Selector_AI\outputs\reports\Strategy_Selector_AI_Report.csv"

def init_directories():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
