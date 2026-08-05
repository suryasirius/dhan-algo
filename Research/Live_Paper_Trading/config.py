import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from token_manager import get_access_token, get_client_id  # noqa: E402

PROJECT_ROOT = _THIS_DIR
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# --- AUTHENTICATED DHAN CREDENTIALS (loaded via Research/token_manager.py) ---
DHAN_CLIENT_ID = get_client_id()
DHAN_ACCESS_TOKEN = get_access_token()

# --- SAFETY GUARDRAIL ---
PAPER_TRADING_ONLY = True
INITIAL_CAPITAL = 25000.0
LOT_SIZE = 25
SLIPPAGE_POINTS = 0.5
BROKERAGE_PER_ORDER = 20.0
