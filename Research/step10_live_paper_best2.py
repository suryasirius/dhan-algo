"""
Module: step10_live_paper_best2.py
Description: Live paper-trading engine for ONLY the two strategies picked on
             5/8/26 (see best_two_of_claude_5_8_26.py):
               1. Opening-Drive Momentum, stop=20pts / trail=1pts
                  (logic reused verbatim from nifty_open_momentum_stoptrail_sweep.py)
               2. Six Concepts -- RBI/GBI relaxed
                  (logic reused verbatim from nifty_six_concepts_screening.py)

             This is DELIBERATELY separate from step9_live_engine.py:
               - It does NOT import or use step6_strategy_selector.py (the
                 rule-based Opening Drive/Velez/ORB "AI" picker) or step9's
                 DNA/regime/OI heuristics. There is no strategy-selection
                 filter here at all - both strategies always run, every day,
                 unconditionally (subject only to their own entry rules).
               - The only two decisions this engine ever makes are: (a) has
                 either strategy's own entry condition fired, and (b) the
                 capital-sharing priority rule below.

             CAPITAL SHARING: both strategies trade off ONE shared paper
             capital pool (Rs 25,000 by default), not one pool each. If
             open_momentum already has an open position, six_concepts will
             NOT open a new one until it closes (open_momentum has priority -
             this mirrors the same rule used in the offline combined backtest,
             where the two conflicted on the same day 15 times out of 990
             trades and open_momentum won every time).

             ENTRY-TIMING FAITHFULNESS TO THE BACKTEST:
               - open_momentum enters at the NEXT bar's open after its
                 opening-drive signal bar (1-bar execution lag) - same as
                 nifty_open_momentum_stoptrail_sweep.py.
               - six_concepts enters at the signal bar's own close - same as
                 nifty_six_concepts_screening.py (this is not lookahead: in
                 live polling, by the time a completed 2-min bar is observed,
                 its close is already a real, tradeable price).
               - Both exit rules (hard stop -> trailing-after-profit, EOD
                 square-off) are reused via small per-bar "step" functions
                 that replay the EXACT same conservative intrabar ordering as
                 each source script's simulate_trade() - just called once per
                 newly-arrived bar instead of looping over a whole known
                 future array, since live data arrives one bar at a time.

             CAUTIONS CARRIED OVER FROM THE SOURCE SCRIPTS (do not lose these):
               - open_momentum's sl20_trail1 combo sits at the EDGE of its
                 tested parameter grid (widest stop, tightest trail tested) -
                 flagged in its own docstring as a possible curve-fit risk.
               - six_concepts RBI/GBI relaxed was flagged by its own script's
                 diagnostics as one of three modules whose suspiciously high,
                 near-identical SQN numbers "smells like an entry-frequency
                 artifact, not a shared edge."
             Neither caution has been resolved - this is paper trading so the
             live track record can start answering that question with real
             (simulated) money on the line, safely.

Run as its own long-lived process, separate from step9_live_engine.py and
streamlit run step8_dashboard.py:
    python step10_live_paper_best2.py
"""

import os
import sys
import time
import sqlite3
import logging
from datetime import datetime, time as dtime, timezone, timedelta

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # C:\dhan algo
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from step1_config_and_connector import PlatformConfig, DhanConnector
import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("Step10_LivePaperBest2")

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

STARTING_CAPITAL = 25_000.0
OM_STOP_PTS = 20
OM_TRAIL_PTS = 1
WARMUP_DAYS = 10          # calendar days of history fetched at startup for six_concepts' SMA200
POLL_SECONDS = 30         # how often we check for a newly-completed 2-min bar


class Best2LivePaperEngine:
    def __init__(self, config: PlatformConfig = None, security_id: str = "13",
                 exchange_segment: str = "IDX_I"):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.security_id = security_id
        self.exchange_segment = exchange_segment

        self.db_path = os.path.join(self.config.log_dir, "best2_paper_portfolio.db")
        self._init_db()

        self.capital = STARTING_CAPITAL
        self._load_capital_from_db()

        self.history_df = pd.DataFrame()     # prior days' 2-min bars, fetched once at startup
        self.last_seen_bar_time = None        # last 2-min bar timestamp already processed today

        # per-strategy live state (reset daily)
        self.today = None
        self.om_traded_today = False
        self.om_pending_signal_idx = None     # opening-drive found, waiting for next bar's open
        self.om_position = None               # dict: entry_price, stop_level, peak, trailing_active, ...
        self.sc_traded_today = False
        self.sc_position = None

        self.log_event("Best-2 live paper engine initialized "
                        f"(open_momentum sl{OM_STOP_PTS}_trail{OM_TRAIL_PTS} + six_concepts RBI_GBI_relaxed, "
                        f"shared capital Rs {STARTING_CAPITAL:,.0f}, open_momentum has priority).")

    # ------------------------------------------------------------------
    # Schema / persistence
    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, strategy TEXT, direction TEXT, side TEXT,
            entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL, exit_reason TEXT,
            pnl_pts REAL, rupee_pnl REAL, lot_size INTEGER,
            capital_after REAL, status TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, event_message TEXT
        )''')
        conn.commit()
        conn.close()

    def _load_capital_from_db(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            row = conn.execute(
                "SELECT capital_after FROM paper_orders WHERE status='CLOSED' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                self.capital = float(row[0])
        except Exception:
            pass

    def log_event(self, message: str):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("INSERT INTO event_logs (timestamp, event_message) VALUES (?, ?)",
                         (datetime.now().isoformat(), message))
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to write event log")
        logger.info(message)

    def _record_open(self, strategy, direction, side, entry_time, entry_price):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''INSERT INTO paper_orders
            (date, strategy, direction, side, entry_time, entry_price, status, capital_after)
            VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)''',
            (str(entry_time.date()), strategy, direction, side, entry_time.isoformat(),
             entry_price, self.capital))
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    def _record_close(self, row_id, exit_time, exit_price, exit_reason, pnl_pts, rupee_pnl, lot_size):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('''UPDATE paper_orders SET
            exit_time=?, exit_price=?, exit_reason=?, pnl_pts=?, rupee_pnl=?, lot_size=?,
            capital_after=?, status='CLOSED' WHERE id=?''',
            (exit_time.isoformat(), exit_price, exit_reason, pnl_pts, rupee_pnl, lot_size,
             self.capital, row_id))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def _fetch_and_resample(self, from_date=None, to_date=None):
        self.connector.refresh_credentials()
        raw = self.connector.fetch_historical_intraday(
            self.security_id, self.exchange_segment,
            from_date=from_date, to_date=to_date, interval="1"
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce") if raw["timestamp"].notna().any() else None
        if raw["timestamp"].isna().all():
            # fall back to "now" indexing if the feed didn't supply real timestamps this cycle
            return pd.DataFrame()
        df = raw.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            return df
        return om.resample_2min_per_day(df)

    def bootstrap_history(self):
        """Fetches WARMUP_DAYS of prior-day 2-min bars once at startup so
        six_concepts' SMA200 has enough real history to compute from day one,
        instead of silently sitting idle for its first ~1-2 trading days."""
        to_date = (datetime.now(IST).date() - timedelta(days=1))
        from_date = to_date - timedelta(days=WARMUP_DAYS)
        self.log_event(f"Bootstrapping warmup history: {from_date} to {to_date} ...")
        hist = self._fetch_and_resample(from_date=from_date.strftime("%Y-%m-%d"),
                                         to_date=to_date.strftime("%Y-%m-%d"))
        if hist.empty:
            self.log_event("Warmup history fetch returned no data - six_concepts will wait for "
                            "enough live bars to accumulate its own SMA200 warmup instead.")
            self.history_df = pd.DataFrame()
        else:
            self.history_df = hist
            self.log_event(f"Warmup history loaded: {len(hist)} bars across "
                            f"{len(set(hist.index.date))} prior trading day(s).")

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------
    def _maybe_reset_for_new_day(self, today):
        if self.today != today:
            self.today = today
            self.om_traded_today = False
            self.om_pending_signal_idx = None
            self.sc_traded_today = False
            if self.om_position is not None:
                self.log_event(f"New day started with an unexpectedly still-open open_momentum "
                                f"position from {self.today} - this should not happen (EOD square-off "
                                f"should have closed it). Leaving state as-is; investigate.")
            if self.sc_position is not None:
                self.log_event(f"New day started with an unexpectedly still-open six_concepts "
                                f"position - investigate.")
            self.log_event(f"New trading day: {today}. Both strategies re-armed.")

    # ------------------------------------------------------------------
    # open_momentum: entry + incremental exit step (faithful to om.simulate_trade)
    # ------------------------------------------------------------------
    def _check_om_entry(self, today_df):
        if self.om_traded_today or self.om_position is not None or self.om_pending_signal_idx is not None:
            return
        local_i, direction = om.find_opening_drive(today_df)
        if local_i is None:
            return
        # 1-bar execution lag: remember the signal, enter on the NEXT bar's open
        self.om_pending_signal_idx = local_i
        self.log_event(f"open_momentum: opening-drive signal detected ({direction}) at "
                        f"{today_df.index[local_i]} - will enter at the next bar's open.")
        self._om_pending_direction = direction

    def _try_om_execute_pending(self, today_df, day_open_spot):
        if self.om_pending_signal_idx is None or self.om_position is not None:
            return
        exec_idx = self.om_pending_signal_idx + 1
        if exec_idx >= len(today_df):
            return  # next bar hasn't arrived yet
        direction = self._om_pending_direction
        side = "CE" if direction == "LONG" else "PE"
        strike = om.itm_call_strike(day_open_spot) if side == "CE" else om.itm_put_strike(day_open_spot)
        premium_df = om.build_premium_series(today_df, strike, day_open_spot, side)
        entry_time = today_df.index[exec_idx]
        entry_price = float(premium_df["open"].iloc[exec_idx])
        lot = om.lot_size_for_date(entry_time.date())

        row_id = self._record_open("open_momentum", direction, side, entry_time, entry_price)
        self.om_position = {
            "row_id": row_id, "direction": direction, "side": side, "strike": strike,
            "entry_time": entry_time, "entry_price": entry_price, "lot": lot,
            "stop_level": entry_price - OM_STOP_PTS, "peak": entry_price, "trailing_active": False,
        }
        self.om_pending_signal_idx = None
        self.om_traded_today = True
        self.log_event(f"open_momentum: OPENED {side} {direction} @ {entry_price:.2f} "
                        f"(strike {strike}, lot {lot}) at {entry_time}")

    def _step_om_position(self, today_df, day_open_spot, bar_time):
        pos = self.om_position
        if pos is None:
            return
        premium_df = om.build_premium_series(today_df, pos["strike"], day_open_spot, pos["side"])
        bar = premium_df.loc[bar_time]

        # Faithful to om.simulate_trade's conservative ordering: check this
        # bar's low against the stop level AS IT STOOD BEFORE this bar first.
        if bar["low"] <= pos["stop_level"]:
            reason = "TRAIL" if pos["trailing_active"] else "STOP"
            self._close_om_position(pos["stop_level"], reason, bar_time)
            return
        if bar_time.time() >= om.EOD_SQUAREOFF:
            self._close_om_position(float(bar["close"]), "EOD", bar_time)
            return
        if bar["high"] > pos["peak"]:
            pos["peak"] = float(bar["high"])
        if pos["peak"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], pos["peak"] - OM_TRAIL_PTS)

    def _close_om_position(self, exit_price, reason, exit_time):
        pos = self.om_position
        pnl_pts = exit_price - pos["entry_price"]
        rupee_pnl = pnl_pts * pos["lot"] - om.BROKERAGE_PER_TRADE
        self.capital += rupee_pnl
        self._record_close(pos["row_id"], exit_time, exit_price, reason, pnl_pts, rupee_pnl, pos["lot"])
        self.log_event(f"open_momentum: CLOSED {pos['side']} @ {exit_price:.2f} ({reason}) | "
                        f"PnL Rs {rupee_pnl:,.2f} | capital now Rs {self.capital:,.2f}")
        self.om_position = None

    # ------------------------------------------------------------------
    # six_concepts RBI/GBI relaxed: entry + incremental exit step (faithful to sc.simulate_trade)
    # ------------------------------------------------------------------
    def _check_sc_entry(self, full_df, day_open_spot):
        if self.sc_traded_today or self.sc_position is not None:
            return
        # Priority rule: open_momentum wins any capital conflict.
        if self.om_position is not None:
            return
        if len(full_df) < sc.MODULE_WARMUP:
            return  # not enough bars yet for a reliable SMA200
        i = len(full_df) - 1  # latest completed bar
        direction = sc.rbi_gbi_signal(full_df, i, relaxed=True)
        if direction is None:
            return
        side = "CE" if direction == "LONG" else "PE"
        strike = om.itm_call_strike(day_open_spot) if side == "CE" else om.itm_put_strike(day_open_spot)
        today_df = full_df[full_df.index.date == full_df.index[i].date()]
        premium_df = om.build_premium_series(today_df, strike, day_open_spot, side)
        # entry at this signal bar's own close (matches sc.run_module - no extra lag)
        local_i = today_df.index.get_loc(full_df.index[i])
        entry_time = today_df.index[local_i]
        entry_price = float(premium_df["close"].iloc[local_i])
        lot = sc.lot_size_for_date(entry_time.date())

        row_id = self._record_open("six_concepts", direction, side, entry_time, entry_price)
        self.sc_position = {
            "row_id": row_id, "direction": direction, "side": side, "strike": strike,
            "entry_time": entry_time, "entry_price": entry_price, "lot": lot,
            "stop_level": entry_price - sc.STOP_PTS, "peak": entry_price, "trailing_active": False,
        }
        self.sc_traded_today = True
        self.log_event(f"six_concepts: OPENED {side} {direction} (RBI/GBI relaxed) @ {entry_price:.2f} "
                        f"(strike {strike}, lot {lot}) at {entry_time}")

    def _step_sc_position(self, today_df, day_open_spot, bar_time):
        pos = self.sc_position
        if pos is None:
            return
        premium_df = om.build_premium_series(today_df, pos["strike"], day_open_spot, pos["side"])
        bar = premium_df.loc[bar_time]

        # Faithful to sc.simulate_trade's ordering: update peak/trail from
        # THIS bar first, THEN check whether this same bar breaches the stop.
        if bar["high"] > pos["peak"]:
            pos["peak"] = float(bar["high"])
        if pos["peak"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], pos["peak"] - sc.TRAIL_PTS)
        if bar["low"] <= pos["stop_level"]:
            reason = "TRAIL" if pos["trailing_active"] else "STOP"
            self._close_sc_position(pos["stop_level"], reason, bar_time)
            return
        if bar_time.time() >= sc.EOD_SQUAREOFF:
            self._close_sc_position(float(bar["close"]), "EOD", bar_time)
            return

    def _close_sc_position(self, exit_price, reason, exit_time):
        pos = self.sc_position
        pnl_pts = exit_price - pos["entry_price"]
        rupee_pnl = pnl_pts * pos["lot"] - sc.BROKERAGE_PER_TRADE
        self.capital += rupee_pnl
        self._record_close(pos["row_id"], exit_time, exit_price, reason, pnl_pts, rupee_pnl, pos["lot"])
        self.log_event(f"six_concepts: CLOSED {pos['side']} @ {exit_price:.2f} ({reason}) | "
                        f"PnL Rs {rupee_pnl:,.2f} | capital now Rs {self.capital:,.2f}")
        self.sc_position = None

    # ------------------------------------------------------------------
    # One polling cycle
    # ------------------------------------------------------------------
    def run_cycle(self):
        now_ist = datetime.now(IST)
        today = now_ist.date()
        self._maybe_reset_for_new_day(today)

        today_df = self._fetch_and_resample()  # today-only, default params
        if today_df.empty:
            self.log_event("No live bar data this cycle (feed empty/bad token/market data unavailable). Skipped.")
            return

        full_df = pd.concat([self.history_df, today_df]) if not self.history_df.empty else today_df
        full_df_sc = sc.compute_indicators(full_df)
        full_df_sc = sc.compute_slope_stage(full_df_sc)

        latest_bar_time = today_df.index[-1]
        if self.last_seen_bar_time is not None and latest_bar_time <= self.last_seen_bar_time:
            return  # no new completed 2-min bar yet this cycle

        # Process every newly-completed bar in order (handles the rare case of
        # more than one new bar appearing between polls without skipping any).
        new_bar_times = today_df.index[today_df.index > (self.last_seen_bar_time or today_df.index[0] - timedelta(seconds=1))]
        day_open_spot = today_df["open"].iloc[0]

        for bar_time in new_bar_times:
            local_today_df = today_df.loc[:bar_time]
            local_full_df_sc = full_df_sc.loc[:bar_time]

            # 1) manage open positions first (exit checks use this bar)
            self._step_om_position(local_today_df, day_open_spot, bar_time)
            self._step_sc_position(local_today_df, day_open_spot, bar_time)

            # 2) execute any pending open_momentum signal from the prior bar
            self._try_om_execute_pending(local_today_df, day_open_spot)

            # 3) look for fresh entries
            if now_ist.time() < om.ENTRY_CUTOFF or self.om_pending_signal_idx is not None:
                self._check_om_entry(local_today_df)
            self._check_sc_entry(local_full_df_sc, day_open_spot)

        self.last_seen_bar_time = latest_bar_time

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_forever(self):
        self.bootstrap_history()
        self.log_event(f"Live loop starting. Poll interval: {POLL_SECONDS}s. "
                        f"Capital: Rs {self.capital:,.2f}")
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
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            self.log_event("Best-2 live paper engine stopped manually (Ctrl+C).")


if __name__ == "__main__":
    engine = Best2LivePaperEngine()
    engine.run_forever()
