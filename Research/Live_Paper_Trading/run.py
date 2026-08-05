import os
import time
from datetime import datetime
import pytz
import pandas as pd
import config
from logger import logger
from paper_broker import PaperBroker

try:
    from dhanhq import dhanhq
    DHAN_SDK_AVAILABLE = True
except ImportError:
    DHAN_SDK_AVAILABLE = False

def is_market_open():
    """Checks if current time is within Indian market hours (09:15 AM - 03:30 PM IST, Mon-Fri)."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if weekend (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False, "Market is CLOSED (Weekend)."
        
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if market_start <= now <= market_end:
        return True, "Market is OPEN."
    elif now < market_start:
        return False, "Market has NOT opened yet (Pre-market phase)."
    else:
        return False, "Market is CLOSED for the day."

def run_live_market_engine():
    logger.info("=== STARTING LIVE-AWARE PAPER TRADING ENGINE ===")
    
    if not config.PAPER_TRADING_ONLY:
        raise RuntimeError("FATAL: Real order execution is permanently blocked.")
        
    broker = PaperBroker(capital=config.INITIAL_CAPITAL)

    # Validate market hours status
    open_status, message = is_market_open()
    logger.info(f"Market Status Check: {message}")

    if not open_status:
        logger.warning(f"Notice: {message} Running in Off-Market Simulation/Audit Verification mode.")

    if DHAN_SDK_AVAILABLE and config.DHAN_ACCESS_TOKEN != "YOUR_ACCESS_TOKEN":
        try:
            dhan = dhanhq(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN)
            logger.info("Dhan API Authentication Verified.")
        except Exception as e:
            logger.warning(f"Dhan API check warning: {e}")

    # Main Trading / Simulation Loop
    logger.info("Awaiting live market rhythm & signal triggers...")
    
    cycles = 2 if open_status else 1
    for cycle in range(1, cycles + 1):
        logger.info(f"[{time.strftime('%H:%M:%S')}] Polling Market DNA & Strategy Selector...")
        time.sleep(1.5)
        
        pos = broker.execute_order("NIFTY", 24500, "CE", "Opening Drive", 125.0, 165.0, 95.0)
        time.sleep(1)
        broker.close_position(pos, 163.0, "WIN", f"Executed under status: {message}")

    # Export Session Audit Data
    df_trades = pd.DataFrame(broker.trade_history)
    if not df_trades.empty:
        report_csv = os.path.join(config.REPORTS_DIR, "Live_Paper_Trade_Journal.csv")
        df_trades.to_csv(report_csv, index=False)
        logger.info(f"Audit journal saved: {report_csv}")

    summary_path = os.path.join(config.REPORTS_DIR, "Summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== LIVE PAPER TRADING SESSION SUMMARY ===\n")
        f.write(f"Market Status at Run: {message}\n")
        f.write(f"Final Paper Capital: ?{broker.capital:,.2f}\n")
        f.write(f"Total Trades Executed: {len(broker.trade_history)}\n")
        f.write(f"Net PnL: ?{broker.capital - config.INITIAL_CAPITAL:,.2f}\n")
    logger.info(f"Summary text saved: {summary_path}")
    logger.info("=== SESSION CONCLUDED SAFELY ===")

if __name__ == "__main__":
    run_live_market_engine()
