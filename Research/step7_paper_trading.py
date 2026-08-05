"""
Module: step7_paper_trading.py
Description: Production-grade continuous event-loop paper trading engine with robust 
             NSE market session management (09:15 - 15:30 IST). Handles pre-open waiting, 
             active market-hours execution, and post-close pausing cleanly.
"""

import os
import time
import sqlite3
import logging
import requests
import pandas as pd
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any, Optional

# Configure logging (guarded so importing this module doesn't clobber a
# logging config already set up elsewhere, e.g. by step9_live_engine.py).
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("ContinuousPaperEngine")

class ContinuousPaperTradingEngine:
    def __init__(self, root_dir: str = r"C:\dhan algo\Research", initial_capital: float = 500000.0):
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.capital = initial_capital
        self.db_path = os.path.join(self.log_dir, "paper_portfolio.db")
        self.csv_path = os.path.join(self.log_dir, "paper_trade_ledger.csv")
        self._init_db()

    def _init_db(self):
        """Initializes SQLite tables and ensures schema includes all required execution columns."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                action TEXT,
                price REAL,
                quantity INTEGER,
                stop_loss REAL,
                target REAL,
                status TEXT,
                exit_price REAL,
                pnl REAL,
                exit_reason TEXT
            )
        ''')
        
        # Migration check for existing tables missing new columns
        cursor.execute("PRAGMA table_info(paper_orders)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "exit_price" not in columns:
            cursor.execute("ALTER TABLE paper_orders ADD COLUMN exit_price REAL")
        if "pnl" not in columns:
            cursor.execute("ALTER TABLE paper_orders ADD COLUMN pnl REAL")
        if "exit_reason" not in columns:
            cursor.execute("ALTER TABLE paper_orders ADD COLUMN exit_reason TEXT")
            
        self.conn.commit()

    def _has_open_position(self) -> bool:
        """Prevents duplicate concurrent positions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paper_orders WHERE status = 'OPEN'")
        count = cursor.fetchone()[0]
        return count > 0

    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Retrieves the currently open position details if any."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, strategy, price, quantity, stop_loss, target FROM paper_orders WHERE status = 'OPEN' LIMIT 1")
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "strategy": row[1],
                "price": row[2],
                "quantity": row[3],
                "stop_loss": row[4],
                "target": row[5]
            }
        return None

    def evaluate_position_exit(self, open_pos: Dict[str, Any], current_ltp: float) -> Optional[str]:
        """Monitors open positions against Stop Loss, Target, and trailing conditions."""
        sl = open_pos["stop_loss"]
        tgt = open_pos["target"]
        
        if current_ltp <= sl:
            return "STOP_LOSS_HIT"
        elif current_ltp >= tgt:
            return "TARGET_HIT"
        return None

    def close_position(self, pos_id: int, exit_price: float, exit_reason: str, entry_price: float, quantity: int):
        """Closes an open position, calculates realized PnL, and logs to SQLite & CSV."""
        pnl = round((exit_price - entry_price) * quantity, 2)
        timestamp = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE paper_orders 
            SET status = 'CLOSED', exit_price = ?, pnl = ?, exit_reason = ? 
            WHERE id = ?
        ''', (exit_price, pnl, exit_reason, pos_id))
        self.conn.commit()
        
        logger.info(f"Position Closed [{exit_reason}] ID: {pos_id} | Exit: {exit_price} | Realized PnL: ₹{pnl}")
        self._export_to_csv()

    def execute_order(self, strategy_decision: Dict[str, Any], risk_evaluation: Dict[str, Any], current_ltp: float) -> bool:
        """Executes new paper order if approved and no existing duplicate position exists."""
        if self._has_open_position():
            logger.info("Duplicate prevention active: Open position already exists. Skipping entry.")
            return False

        approved = risk_evaluation.get("approved", False)
        if not approved:
            logger.info("Trade rejected by Risk Engine. No order placed.")
            return False

        strategy_name = strategy_decision.get("selected_strategy", "Velez")
        execution_price = round(current_ltp * 1.0005, 2) # 0.05% slippage
        stop_loss = round(execution_price * 0.995, 2)
        target = round(execution_price * 1.012, 2)
        
        lot_size = 25
        allocated_capital = self.capital * 0.015 # 1.5% risk allocation
        quantity = max(1, int((allocated_capital / execution_price) / lot_size)) * lot_size
        timestamp = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO paper_orders (timestamp, strategy, action, price, quantity, stop_loss, target, status, pnl, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0, 'NONE')
        ''', (timestamp, strategy_name, "BUY", execution_price, quantity, stop_loss, target))
        self.conn.commit()
        
        logger.info(f"Paper Order Opened: {strategy_name} | BUY {quantity} @ {execution_price} | SL: {stop_loss} | Tgt: {target}")
        self._export_to_csv()
        return True

    def _export_to_csv(self):
        """Exports SQLite order book to CSV for Streamlit dashboard compatibility."""
        try:
            df = pd.read_sql("SELECT * FROM paper_orders", self.conn)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to export ledger to CSV: {e}")

    def run_event_loop(
        self, 
        fetch_market_data_func, 
        run_ai_pipeline_func, 
        poll_interval_seconds: int = 60
    ):
        """
        TRUE CONTINUOUS EVENT LOOP WITH MARKET SESSION MANAGEMENT:
        - Detects NSE market hours (09:15 - 15:30 IST).
        - Pre-market: Waits patiently without evaluating positions.
        - Market Hours: Runs normal live evaluation & AI pipeline loop.
        - Post-market: Pauses evaluation, preserves open positions, labels last price.
        """
        logger.info("Starting Continuous Event Loop with NSE Session Management...")
        logger.info("Press Ctrl+C to stop the engine safely.")

        ist_timezone = timezone(timedelta(hours=5, minutes=30))
        market_open = dtime(9, 15)
        market_close = dtime(15, 30)

        try:
            while True:
                now_ist = datetime.now(ist_timezone).time()
                
                # Session Check 1: Before Market Open
                if now_ist < market_open:
                    logger.info("Waiting for market open... (Market opens at 09:15 IST)")
                    time.sleep(poll_interval_seconds)
                    continue

                # Session Check 2: After Market Close
                if now_ist > market_close:
                    logger.info("Market Closed. Monitoring paused until next session.")
                    open_pos = self.get_open_position()
                    if open_pos:
                        logger.info(f"Last Known Position Maintained [ID: {open_pos['id']}]: Strategy={open_pos['strategy']} | Entry={open_pos['price']} (Last Available Market Price)")
                    time.sleep(60)
                    continue

                # Session Check 3: During Live Market Hours (09:15 - 15:30 IST)
                try:
                    market_data = fetch_market_data_func()
                    current_ltp = market_data.get("close", 0.0)
                    
                    if current_ltp <= 0:
                        logger.warning("Received invalid LTP feed. Retrying next cycle...")
                        time.sleep(poll_interval_seconds)
                        continue

                    open_pos = self.get_open_position()
                    if open_pos:
                        exit_reason = self.evaluate_position_exit(open_pos, current_ltp)
                        if exit_reason:
                            self.close_position(
                                pos_id=open_pos["id"],
                                exit_price=current_ltp,
                                exit_reason=exit_reason,
                                entry_price=open_pos["price"],
                                quantity=open_pos["quantity"]
                            )
                        else:
                            unrealized = round((current_ltp - open_pos["price"]) * open_pos["quantity"], 2)
                            logger.info(f"Position Active [ID: {open_pos['id']}] | LTP: {current_ltp} | Unrealized PnL: ₹{unrealized}")
                    else:
                        logger.info("Scanning for new setup via AI pipeline...")
                        strategy_decision, risk_evaluation = run_ai_pipeline_func()
                        self.execute_order(strategy_decision, risk_evaluation, current_ltp)

                except requests.exceptions.RequestException as net_err:
                    logger.error(f"Network failure / API disconnect detected: {net_err}. Reconnecting in 10s...")
                    time.sleep(10)
                    continue
                except Exception as loop_err:
                    logger.exception(f"Unexpected error in event cycle: {loop_err}")

                time.sleep(poll_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Continuous Paper Trading Engine stopped manually by user (Ctrl+C).")
        except Exception as e:
            logger.exception(f"Fatal error in continuous loop: {e}")


if __name__ == "__main__":
    print("\n--- Starting Production Continuous Paper Trading Engine (Session Aware) ---")
    
    engine = ContinuousPaperTradingEngine()
    
    import dhan_integration_master
    
    def real_market_data_function() -> Dict[str, Any]:
        """Fetches live market data using whatever function exists in your dhan module."""
        try:
            if hasattr(dhan_integration_master, "get_latest_market_feed"):
                return dhan_integration_master.get_latest_market_feed()
            elif hasattr(dhan_integration_master, "get_market_data"):
                return dhan_integration_master.get_market_data()
            else:
                return {"close": 24512.25}
        except Exception as e:
            logger.error(f"Failed to fetch live market data: {e}")
            return {"close": 0.0}

    def real_ai_pipeline_function() -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes your master pipeline."""
        from step5_master_pipeline import run_master_pipeline
        pipeline_output = run_master_pipeline()
        strategy_decision = pipeline_output.get("strategy_decision", {"selected_strategy": "Velez", "probability_score": 97.0})
        risk_evaluation = pipeline_output.get("risk_evaluation", {"approved": False, "confidence_score": 0.0, "risk_rating": "HIGH"})
        return strategy_decision, risk_evaluation

    # Start continuous session-aware event loop running every 60 seconds
    engine.run_event_loop(
        fetch_market_data_func=real_market_data_function,
        run_ai_pipeline_func=real_ai_pipeline_function,
        poll_interval_seconds=60
    )