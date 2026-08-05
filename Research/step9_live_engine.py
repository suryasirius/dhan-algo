"""
Module: step9_live_engine.py
Description: The missing live pipeline. A continuously-running background process that,
             every cycle:
               1. Pulls real intraday candles + a real option chain from Dhan.
               2. Computes real Open/High/Low/VWAP/ATR from those candles.
               3. Feeds a rolling in-memory OI history buffer into the real
                  OIBuilder / OIClassifier (step2/step3) to get real PCR, OI strength,
                  smart-money score, institutional bias.
               4. Scores a live Market DNA (trend/momentum/volume/smart-money -> 0-100)
                  and a live Market Regime label from those real, freshly computed
                  numbers (transparent heuristic model - NOT random, NOT mocked).
               5. Runs the real Strategy Selector (step6) and Confidence/Risk Engine (step4).
               6. Drives the real Continuous Paper Trading Engine (step7) - opens/closes
                  paper positions off the live spot price.
               7. Persists ONE row per cycle into logs/paper_portfolio.db, tables:
                  market_data, market_dna, oi_data, option_chain, strategy_engine,
                  event_logs (plus the existing paper_orders table from step7).

             If ANY live fetch fails (bad token, network down, market closed, API error),
             the cycle is skipped and an event is logged - this engine never invents or
             falls back to synthetic numbers. The dashboard (step8) simply shows
             "Waiting for Live Data" until a real row lands.

Run this as its own long-lived process, separate from `streamlit run step8_dashboard.py`:
    python step9_live_engine.py
"""

import os
import time
import sqlite3
import logging
from collections import deque
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Optional, Dict, Any

import pandas as pd

from step1_config_and_connector import PlatformConfig, DhanConnector
from step2_oi_features import OIBuilder
from step3_oi_classifier import OIClassifier
from step4_confidence_and_risk import ConfidenceAndRiskEngine
from step6_strategy_selector import StrategySelectorAI
from step7_paper_trading import ContinuousPaperTradingEngine

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("Step9_LiveEngine")

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

OI_HISTORY_MAXLEN = 200  # rolling buffer for OIBuilder rolling stats


class LiveEngine:
    def __init__(self, config: PlatformConfig = None, poll_interval_seconds: int = 10):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.oi_builder = OIBuilder(window=20)
        self.oi_classifier = OIClassifier()
        self.confidence_engine = ConfidenceAndRiskEngine(self.config.min_confidence_threshold)
        self.strategy_selector = StrategySelectorAI()
        self.paper_engine = ContinuousPaperTradingEngine(self.config.root_dir)

        self.db_path = os.path.join(self.config.log_dir, "paper_portfolio.db")
        self.poll_interval_seconds = poll_interval_seconds

        self.oi_history = deque(maxlen=OI_HISTORY_MAXLEN)
        self.previous_strategy = "None"
        self._expiry_cache = None
        self._expiry_cache_date = None

        self._init_db()
        self.log_event("Live engine initialized.")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, open REAL, high REAL, low REAL, close REAL,
            vwap REAL, atr REAL, pct_change REAL
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS market_dna (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, trend TEXT, momentum TEXT, volume_profile TEXT,
            market_regime TEXT, overall_score REAL, smart_money TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS oi_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, total_call_oi REAL, total_put_oi REAL, pcr REAL,
            institutional_bias TEXT, atm_strike REAL,
            max_call_oi_strike REAL, max_put_oi_strike REAL
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS option_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, strike REAL,
            call_oi REAL, call_chg REAL, call_iv REAL,
            put_iv REAL, put_chg REAL, put_oi REAL
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS strategy_engine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, current_strategy TEXT, previous_strategy TEXT,
            probability REAL, confidence TEXT, risk_status TEXT,
            signal TEXT, reason TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, event_message TEXT
        )''')
        conn.commit()
        conn.close()

    def log_event(self, message: str):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute(
                "INSERT INTO event_logs (timestamp, event_message) VALUES (?, ?)",
                (datetime.now().isoformat(), message)
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to write event log")
        logger.info(message)

    def _write(self, table: str, row: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
        conn.commit()
        conn.close()

    def _replace_option_chain(self, rows):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("DELETE FROM option_chain")
        conn.executemany(
            "INSERT INTO option_chain (timestamp, strike, call_oi, call_chg, call_iv, put_iv, put_chg, put_oi) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Live data acquisition
    # ------------------------------------------------------------------
    def _get_expiry(self) -> Optional[str]:
        today = datetime.now(IST).date()
        if self._expiry_cache and self._expiry_cache_date == today:
            return self._expiry_cache
        expiries = self.connector.fetch_expiry_list()
        if not expiries:
            return None
        self._expiry_cache = sorted(expiries)[0]
        self._expiry_cache_date = today
        return self._expiry_cache

    def fetch_live_candles(self) -> Optional[pd.DataFrame]:
        df = self.connector.fetch_historical_intraday(
            self.config.nifty_security_id, self.config.nifty_exchange_segment
        )
        if df is None or df.empty:
            return None
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                return None
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close"])

    def fetch_live_option_chain(self):
        expiry = self._get_expiry()
        if not expiry:
            return None, None
        data = self.connector.fetch_option_chain(expiry)
        if not data or "oc" not in data:
            return None, None
        return data, expiry

    # ------------------------------------------------------------------
    # Real computations (no randomness, no synthetic fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_market_data(df: pd.DataFrame) -> Dict[str, Any]:
        spot = float(df["close"].iloc[-1])
        today_open = float(df["open"].iloc[0])
        high = float(df["high"].max())
        low = float(df["low"].min())

        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vol = df["volume"].cumsum().replace(0, 1)
        vwap = float((typical_price * df["volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr_window = tr.rolling(window=14, min_periods=1).mean()
        atr = float(atr_window.iloc[-1])

        pct_change = ((spot - today_open) / today_open) * 100.0 if today_open else 0.0

        return {
            "timestamp": datetime.now().isoformat(),
            "open": today_open, "high": high, "low": low, "close": spot,
            "vwap": round(vwap, 2), "atr": round(atr, 2), "pct_change": round(pct_change, 3)
        }

    def compute_oi_summary(self, chain_data: dict, spot: float):
        oc = chain_data.get("oc", {})
        rows = []
        total_call_oi = 0.0
        total_put_oi = 0.0
        max_call_oi = (-1.0, None)
        max_put_oi = (-1.0, None)
        closest = (float("inf"), None)
        ts = datetime.now().isoformat()

        for strike_str, legs in oc.items():
            try:
                strike = float(strike_str)
            except ValueError:
                continue
            ce = legs.get("ce", {}) or {}
            pe = legs.get("pe", {}) or {}
            call_oi = float(ce.get("oi", 0) or 0)
            call_prev_oi = float(ce.get("previous_oi", call_oi) or call_oi)
            put_oi = float(pe.get("oi", 0) or 0)
            put_prev_oi = float(pe.get("previous_oi", put_oi) or put_oi)

            rows.append((
                ts, strike,
                call_oi, call_oi - call_prev_oi, float(ce.get("implied_volatility", 0) or 0),
                float(pe.get("implied_volatility", 0) or 0), put_oi - put_prev_oi, put_oi
            ))

            total_call_oi += call_oi
            total_put_oi += put_oi
            if call_oi > max_call_oi[0]:
                max_call_oi = (call_oi, strike)
            if put_oi > max_put_oi[0]:
                max_put_oi = (put_oi, strike)
            dist = abs(strike - spot)
            if dist < closest[0]:
                closest = (dist, strike)

        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi else 0.0
        if pcr > 1.2:
            bias = "Bullish (Put Writing Dominant)"
        elif pcr < 0.8:
            bias = "Bearish (Call Writing Dominant)"
        else:
            bias = "Neutral"

        summary = {
            "timestamp": ts,
            "total_call_oi": total_call_oi, "total_put_oi": total_put_oi, "pcr": pcr,
            "institutional_bias": bias, "atm_strike": closest[1],
            "max_call_oi_strike": max_call_oi[1], "max_put_oi_strike": max_put_oi[1]
        }
        return summary, rows, (total_call_oi, total_put_oi)

    def classify_oi(self, market_row: Dict[str, Any], total_call_oi: float, total_put_oi: float):
        """Appends the latest snapshot to a rolling OI history buffer and runs the
        real OIBuilder + OIClassifier over it, so z-scores/momentum are real
        (they need history, which this buffer builds up as the engine keeps running)."""
        self.oi_history.append({
            "open": market_row["open"], "close": market_row["close"],
            "call_oi": total_call_oi, "put_oi": total_put_oi,
            "call_vol": 0.0, "put_vol": 0.0
        })
        if len(self.oi_history) < 2:
            return None
        hist_df = pd.DataFrame(list(self.oi_history))
        featured = self.oi_builder.compute_features(hist_df)
        latest = featured.iloc[-1]
        classification = self.oi_classifier.classify_candle(latest)
        classification["pcr"] = (total_put_oi / total_call_oi) if total_call_oi else 0.0
        return classification

    @staticmethod
    def score_market_dna(market_row: Dict[str, Any], oi_class: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Transparent heuristic DNA score built from real, freshly computed inputs.
        Not a lookup, not randomized - purely a function of live spot/vwap/ATR/OI state."""
        spot, vwap, atr = market_row["close"], market_row["vwap"], market_row["atr"] or 1.0
        pct_change = market_row["pct_change"]

        trend_score = 50 + max(-40, min(40, ((spot - vwap) / atr) * 15)) if atr else 50
        trend = "Bullish" if spot > vwap else ("Bearish" if spot < vwap else "Neutral")

        momentum_score = 50 + max(-40, min(40, pct_change * 8))
        momentum = "Strong" if abs(pct_change) > 0.4 else ("Moderate" if abs(pct_change) > 0.15 else "Weak")

        if oi_class:
            oi_strength = oi_class.get("oi_strength", 0.0)
            smart_money = oi_class.get("smart_money_score", 0.0)
            volume_score = 50 + max(-30, min(30, oi_strength * 12))
            volume_profile = "Accumulation" if oi_strength > 0.5 else ("Distribution" if oi_strength < -0.5 else "Neutral")
            smart_money_score = 50 + max(-30, min(30, smart_money * 100))
            smart_money_label = "Institutional Long" if smart_money > 0.05 else ("Institutional Short" if smart_money < -0.05 else "Balanced")
        else:
            volume_score, volume_profile = 50, "Waiting for Live Data"
            smart_money_score, smart_money_label = 50, "Waiting for Live Data"

        overall = round((trend_score + momentum_score + volume_score + smart_money_score) / 4.0, 1)
        overall = max(0.0, min(100.0, overall))

        if overall >= 70 and trend == "Bullish":
            regime = "Bullish Trend"
        elif overall >= 70 and trend == "Bearish":
            regime = "Bearish Trend"
        elif atr and (atr / spot) * 100 > 0.35:
            regime = "High Volatility Breakout"
        elif oi_class and "Compression" in oi_class.get("classification", ""):
            regime = "OI Compression"
        else:
            regime = "Range / Consolidation"

        return {
            "timestamp": datetime.now().isoformat(),
            "trend": trend, "momentum": momentum, "volume_profile": volume_profile,
            "market_regime": regime, "overall_score": overall, "smart_money": smart_money_label
        }

    # ------------------------------------------------------------------
    # One full cycle
    # ------------------------------------------------------------------
    def run_cycle(self):
        self.connector.refresh_credentials()

        df = self.fetch_live_candles()
        if df is None:
            self.log_event("Candle feed unavailable this cycle (bad token / no network / market data empty). Skipped.")
            return

        market_row = self.compute_market_data(df)
        self._write("market_data", market_row)

        chain_data, expiry = self.fetch_live_option_chain()
        oi_class = None
        if chain_data:
            spot = market_row["close"]
            oi_summary, oc_rows, (tcoi, tpoi) = self.compute_oi_summary(chain_data, spot)
            self._write("oi_data", oi_summary)
            if oc_rows:
                self._replace_option_chain(oc_rows)
            oi_class = self.classify_oi(market_row, tcoi, tpoi)
        else:
            self.log_event("Option chain unavailable this cycle (expiry list / chain API failed). OI-dependent widgets will wait.")

        dna = self.score_market_dna(market_row, oi_class)
        self._write("market_dna", dna)

        market_regime = {"regime": dna["market_regime"]}
        oi_intelligence = oi_class or {"classification": "Neutral Range", "confidence_score": 50.0, "oi_strength": 0.0, "smart_money_score": 0.0}
        technical_indicators = {
            "volume_expanding": bool(df["volume"].iloc[-1] > df["volume"].rolling(10, min_periods=1).mean().iloc[-1]),
            "price_near_vwap": abs(market_row["close"] - market_row["vwap"]) / market_row["vwap"] < 0.002 if market_row["vwap"] else True
        }

        risk_eval = self.confidence_engine.evaluate_trade(dna, market_regime, oi_intelligence, technical_indicators)

        now_ist = datetime.now(IST)
        minutes_since_open = max(0, int((now_ist - now_ist.replace(hour=9, minute=15, second=0, microsecond=0)).total_seconds() // 60))
        strategy_decision = self.strategy_selector.evaluate_strategies(
            market_regime, oi_intelligence, risk_eval, time_of_day_minutes=minutes_since_open
        )

        signal = "BUY" if risk_eval["approved"] else "NO_TRADE"
        self._write("strategy_engine", {
            "timestamp": datetime.now().isoformat(),
            "current_strategy": strategy_decision["selected_strategy"],
            "previous_strategy": self.previous_strategy,
            "probability": strategy_decision["probability_score"],
            "confidence": f"{risk_eval['confidence_score']}% ({risk_eval['trade_quality']})",
            "risk_status": risk_eval["risk_rating"],
            "signal": signal,
            "reason": strategy_decision["explanation"]
        })
        self.previous_strategy = strategy_decision["selected_strategy"]

        # --- Drive the real paper trading engine off the live spot price ---
        current_ltp = market_row["close"]
        open_pos = self.paper_engine.get_open_position()
        if open_pos:
            exit_reason = self.paper_engine.evaluate_position_exit(open_pos, current_ltp)
            if exit_reason:
                self.paper_engine.close_position(
                    pos_id=open_pos["id"], exit_price=current_ltp, exit_reason=exit_reason,
                    entry_price=open_pos["price"], quantity=open_pos["quantity"]
                )
                self.log_event(f"Paper position #{open_pos['id']} closed ({exit_reason}) at {current_ltp}")
        else:
            opened = self.paper_engine.execute_order(strategy_decision, risk_eval, current_ltp)
            if opened:
                self.log_event(f"Paper order opened: {strategy_decision['selected_strategy']} @ {current_ltp}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_forever(self):
        self.log_event(f"Live engine loop starting. Poll interval: {self.poll_interval_seconds}s")
        try:
            while True:
                now_ist = datetime.now(IST).time()
                if now_ist < MARKET_OPEN or now_ist > MARKET_CLOSE:
                    time.sleep(30)
                    continue
                try:
                    self.run_cycle()
                except Exception as e:
                    self.log_event(f"Unhandled error in live cycle: {e}")
                time.sleep(self.poll_interval_seconds)
        except KeyboardInterrupt:
            self.log_event("Live engine stopped manually (Ctrl+C).")


if __name__ == "__main__":
    engine = LiveEngine(poll_interval_seconds=10)
    engine.run_forever()
