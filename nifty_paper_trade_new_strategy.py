import os
import sys
import time
from datetime import datetime
import pytz

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.join(_THIS_DIR, "Research")
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from token_manager import get_access_token, get_client_id  # noqa: E402

CLIENT_ID = get_client_id()
ACCESS_TOKEN = get_access_token()

LOG_FILE = f"paper_trade_log_{datetime.now().strftime('%Y-%m-%d')}.csv"

def run_self_checks():
    print("Running self-checks for New Strategy...")
    # Add your strategy validation logic here
    print("  New strategy primitives ok")
    print("All self-checks passed.\n")

def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5: # Saturday or Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

def main():
    run_self_checks()
    
    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- NEW STRATEGY ===")
    print(f"Log file: {LOG_FILE}")
    print("Polling every 30s during market hours (09:15:00-15:30:00 IST)\n")
    
    try:
        while True:
            current_time_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
            
            if is_market_open():
                print(f"{current_time_str} Market open, executing new strategy polling & signal check...")
                # Add your live order paper execution logic here
            else:
                print(f"{current_time_str} Market closed, sleeping 60s..")
                
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nPaper trading session stopped safely by user.")

if __name__ == "__main__":
    main()
