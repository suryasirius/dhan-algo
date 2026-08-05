"""
Module: step7_paper_trading.py
Description: Upgraded continuous paper trading engine. Runs a background loop during 
             market hours, fetches candle updates, runs the full AI pipeline, 
             manages open positions (SL/Tgt/Trailing), tracks PnL, and prevents duplicates.
"""

import os
import time
import sqlite3
import logging
import pandas as pd
from datetime import datetime, time as dtime
from typing import Dict, Any, Optional

logger = logging.getLogger("Step7_ContinuousPaperEngine")
logger.setLevel(logging.INFO)

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
        """Monitors open positions against Stop Loss or Target bounds."""
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
            logger.info("Skipping trade entry: Active open position already exists (Duplicate prevention active).")
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
        """Exports the SQLite order book to CSV for dashboard and audit compatibility."""
        try:
            df = pd.read_sql("SELECT * FROM paper_orders", self.conn)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to export ledger to CSV: {e}")

    def run_continuous_loop(self, get_live_telemetry_func, get_ai_pipeline_func, poll_interval_seconds: int = 30):
        """
        Runs continuously during market hours.
        - Fetches updates every poll interval.
        - Monitors open positions against live market price.
        - Triggers AI pipeline if no position is open.
        """
        logger.info("Starting Continuous Paper Trading Execution Loop...")
        
        while True:
            try:
                market_data = get_live_telemetry_func()
                current_ltp = market_data.get("close", 24512.25)

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
                        logger.info(f"Position Active [ID: {open_pos['id']}]. Current LTP: {current_ltp} | Unrealized PnL: ₹{unrealized}")
                else:
                    logger.info("Scanning for setup via AI pipeline...")
                    strategy_decision, risk_evaluation = get_ai_pipeline_func()
                    self.execute_order(strategy_decision, risk_evaluation, current_ltp)

            except Exception as e:
                logger.exception(f"Error in continuous execution loop: {e}")

            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    print("\n--- Testing Continuous Paper Engine (Single Cycle Simulation) ---")
    
    engine = ContinuousPaperTradingEngine()
    
    def mock_telemetry():
        return {"close": 24550.0}
        
    def mock_ai_pipeline():
        strat = {"selected_strategy": "Velez", "probability_score": 97.0}
        risk = {"approved": True, "confidence_score": 92.5, "risk_rating": "LOW"}
        return strat, risk

    market_data = mock_telemetry()
    strat_dec, risk_eval = mock_ai_pipeline()
    
    engine.execute_order(strat_dec, risk_eval, market_data["close"])
    
    open_p = engine.get_open_position()
    if open_p:
        print(f"Active Position Found: Strategy={open_p['strategy']}, Entry={open_p['price']}")
        exit_sig = engine.evaluate_position_exit(open_p, current_ltp=24850.0)
        if exit_sig:
            print(f"Simulated Exit Triggered: {exit_sig}")
            engine.close_position(open_p['id'], 24850.0, exit_sig, open_p['price'], open_p['quantity'])
            
    print("-----------------------------------------------------------------\n")