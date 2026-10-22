"""
LIVE-PAPER ENGINE -- Progressive 3pt Entry, Cap=1, 09:45 CUTOFF, 2pt trail ONLY.
step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py

NEW, separate engine copied from the cap=2 / 10:15-cutoff 2pt-only engine
(step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py) with exactly
two changes: MAX_POSITIONS 2 -> 1 (one position at a time) and ENTRY_CUTOFF
10:15 -> 09:45. Everything else is unchanged: progressive-3pt index entry
(20% max retracement), 20pt stop, 2pt trail, 3-min time-stop, 20s poll,
real per-trade cost model (trade_cost_model.compute_trade_cost).

THIS FILE       : step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py
THIS DB         : logs/om_progressive3pt_cap1_2ptonly_945cutoff_paper_portfolio.db
                  (starts EMPTY)
THIS CONTROL DIR: logs/control_progressive3pt_cap1_2ptonly_945cutoff/
THIS LOGGER NAME: Step10_LivePaper_OM_Progressive3pt_Cap1_2ptOnly_945Cutoff
THIS PORT       : 8523 (next free slot after 8522; no collisions)

DOES NOT TOUCH, READ, OR WRITE ANYTHING FROM ANY OTHER LIVE ENGINE.
Run only ONE live engine at a time against the Dhan account (rate limits).
PAPER TRADING ONLY.
"""

import os
import sys
import time
import random
import sqlite3
import logging
from datetime import datetime, time as dtime, timezone, timedelta

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from step1_config_and_connector import PlatformConfig, DhanConnector
import nifty_open_momentum_stoptrail_sweep as om  # itm_call_strike/itm_put_strike/lot_size_for_date/resample_2min_per_day/EOD_SQUAREOFF
from trade_cost_model import compute_trade_cost  # real per-trade brokerage+charges formula (replaces the old flat BROKERAGE_PER_TRADE=55 estimate)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Step10_LivePaper_OM_Progressive3pt_Cap1_2ptOnly_945Cutoff")

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

STARTING_CAPITAL = 25_000.0
OM_STOP_PTS = 20
ENTRY_CUTOFF = dtime(9, 45)  # entries only accepted before 09:45, not all session.
POLL_SECONDS = 20  # same reduced interval as the no-cutoff 2pt-only engine this was copied from --
                    # smaller gaps mean stop/trail exits fill closer to their configured level.
STALL_WARN_MINUTES = 5
BROKERAGE_PER_TRADE = 55.0  # this project's canonical brokerage+charges figure (local override --
                             # om.BROKERAGE_PER_TRADE defaults to Rs40, used by older engines)

# Same 3-min time-stop rule validated on the 945cutoff engine (122-day sweep:
# PF 4.44->4.53, net +Rs4,058, maxDD -9.5%, full STOP count 31->22): if a
# position has never ticked favorable (trailing_active still False) within
# TIME_LIMIT_MIN minutes of entry, close it at market instead of waiting for
# the full 20pt stop. Applied here by analogy, not independently re-proven
# for this specific cutoff/variant combination.
TIME_LIMIT_MIN = 3

TRIGGER_PTS = 3.0
RETRACE_FRACTION = 0.20
MAX_POSITIONS = 1  # cap=1: one position at a time

VARIANTS = [
    {"label": "trail2pt", "trail_pts": 2},   # ONLY variant in this copy -- trail1pt and trail3pt removed
]

DB_FILENAME = "om_progressive3pt_cap1_2ptonly_945cutoff_paper_portfolio.db"
CONTROL_SUBDIR = "control_progressive3pt_cap1_2ptonly_945cutoff"


def get_ltp_from_chain(chain_data, strike, side):
    if not chain_data or not isinstance(chain_data, dict):
        return None
    oc = chain_data.get("oc", {}) or {}
    side_key = "ce" if side == "CE" else "pe"
    for strike_str, sides in oc.items():
        try:
            s = float(strike_str)
        except (TypeError, ValueError):
            continue
        if abs(s - strike) < 0.01:
            block = sides.get(side_key) if isinstance(sides, dict) else None
            if isinstance(block, dict):
                ltp = block.get("last_price")
                try:
                    return float(ltp) if ltp is not None else None
                except (TypeError, ValueError):
                    return None
    return None


class SideTracker:
    """Identical to the validated single-position engine's tracker."""

    def __init__(self, start_value):
        self.ref = start_value
        self.peak = start_value

    def update_and_check(self, value):
        if value > self.peak:
            self.peak = value
        else:
            preceding_move = self.peak - self.ref
            retracement = self.peak - value
            if preceding_move > 0 and retracement > RETRACE_FRACTION * preceding_move:
                self.ref = value
                self.peak = value
        return (self.peak - self.ref) >= TRIGGER_PTS

    def reset(self, new_value):
        self.ref = new_value
        self.peak = new_value


def run_self_checks():
    print("Running self-checks (synthetic data, no network)...")
    sample = {"last_price": 24343.5, "oc": {"24300.000000": {"ce": {"last_price": 75.2}, "pe": {"last_price": 40.1}}}}
    assert get_ltp_from_chain(sample, 24300, "CE") == 75.2
    assert get_ltp_from_chain(sample, 24400, "CE") is None
    assert get_ltp_from_chain(None, 24300, "CE") is None
    print("  get_ltp_from_chain: OK")

    assert len(VARIANTS) == 1, "this copy must run exactly the trail2pt variant, nothing else"
    assert VARIANTS[0]["label"] == "trail2pt" and VARIANTS[0]["trail_pts"] == 2
    print(f"  variant config: OK (2pt-only copy, {len(VARIANTS)} variant: {[v['label'] for v in VARIANTS]})")

    assert ENTRY_CUTOFF == dtime(9, 45), f"this engine must cut off entries at 09:45, got {ENTRY_CUTOFF}"
    assert MAX_POSITIONS == 1, "this engine is specifically the cap=1 variant"
    assert TIME_LIMIT_MIN == 3
    assert POLL_SECONDS == 20
    print(f"  cutoff check: OK (ENTRY_CUTOFF = {ENTRY_CUTOFF}), MAX_POSITIONS = {MAX_POSITIONS}, "
          f"TIME_LIMIT_MIN = {TIME_LIMIT_MIN}, POLL_SECONDS = {POLL_SECONDS}")

    tr = SideTracker(24200.0)
    choppy = [24201, 24200.2, 24201.5, 24200.5, 24202, 24201.2, 24203]
    assert not any(tr.update_and_check(p) for p in choppy), "choppy path must be rejected"
    tr2 = SideTracker(24200.0)
    clean = [24201, 24200.85, 24202, 24201.85, 24203]
    assert any(tr2.update_and_check(p) for p in clean), "clean ascent must fire"
    print("  progressive-entry rule (reject choppy, accept clean ascent) check ok")

    down_tr = SideTracker(-24200.0)
    clean_down = [-24199.0, -24199.15, -24198.0, -24198.15, -24197.0]
    assert any(down_tr.update_and_check(p) for p in clean_down), "sign-flipped down-tracker must fire"
    print("  progressive-entry rule, SHORT (sign-flipped) direction check ok")

    print("This engine's DB is separate:", DB_FILENAME)
    print("This engine's control folder is separate:", CONTROL_SUBDIR)
    print("All self-checks passed.\n")


class TrailVariantEngine:
    """Owns its own capital/tracker state and UP TO MAX_POSITIONS
    simultaneously open positions (list, not a single dict) -- this is
    the substantive change from the single-position sibling engine."""

    def __init__(self, db_path, label, trail_pts, control_dir):
        self.db_path = db_path
        self.label = label
        self.trail_pts = trail_pts
        self.control_dir = control_dir

        self.capital = STARTING_CAPITAL
        self._load_capital_from_db()

        self.today = None
        self.day_open_spot = None
        self.up_tracker = None
        self.down_tracker = None
        self.pending_signals = []  # list of {"time":..., "direction":...} -- can have >1 pending
        self.positions = []        # list of open-position dicts, len <= MAX_POSITIONS
        self.trades_today = 0

    def log_event(self, message: str):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("INSERT INTO event_logs (timestamp, variant, event_message) VALUES (?, ?, ?)",
                         (datetime.now().isoformat(), self.label, message))
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to write event log")
        logger.info(f"[{self.label}] {message}")

    def _load_capital_from_db(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            row = conn.execute(
                "SELECT capital_after FROM paper_orders WHERE variant=? AND status='CLOSED' "
                "ORDER BY id DESC LIMIT 1", (self.label,)
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                self.capital = float(row[0])
        except Exception:
            pass

    def _record_open(self, direction, side, entry_time, entry_price, reason, lot_size, strike, expiry):
        initial_stop = entry_price - OM_STOP_PTS
        target_note = f"No fixed target - exits via {self.trail_pts}pt trailing stop or 15:20 EOD square-off"
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''INSERT INTO paper_orders
            (date, variant, strategy, direction, side, entry_time, entry_price, status, capital_after,
             initial_stop, current_stop, peak_price, last_price, target_note, reason, lot_size,
             strike, trailing_active, expiry)
            VALUES (?, ?, 'om_progressive3pt_cap1_2ptonly_945cutoff', ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)''',
            (str(entry_time.date()), self.label, direction, side, entry_time.isoformat(),
             entry_price, self.capital, initial_stop, initial_stop, entry_price, entry_price,
             target_note, reason, lot_size, strike, expiry))
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    def _update_live_state(self, row_id, current_stop, peak_price, last_price, trailing_active):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('''UPDATE paper_orders SET current_stop=?, peak_price=?, last_price=?, trailing_active=?
                WHERE id=? AND variant=? AND status='OPEN' ''',
                (current_stop, peak_price, last_price, int(trailing_active), row_id, self.label))
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to update live position state")

    def _record_close(self, row_id, exit_time, exit_price, exit_reason, pnl_pts, rupee_pnl, lot_size):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('''UPDATE paper_orders SET
            exit_time=?, exit_price=?, exit_reason=?, pnl_pts=?, rupee_pnl=?, lot_size=?,
            capital_after=?, status='CLOSED' WHERE id=? AND variant=?''',
            (exit_time.isoformat(), exit_price, exit_reason, pnl_pts, rupee_pnl, lot_size,
             self.capital, row_id, self.label))
        conn.commit()
        conn.close()

    def _close_stale_orphaned_positions(self, today):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            orphans = conn.execute(
                "SELECT * FROM paper_orders WHERE variant=? AND status='OPEN' AND date < ?",
                (self.label, str(today))
            ).fetchall()
            for row in orphans:
                exit_price = row["last_price"] if row["last_price"] is not None else row["entry_price"]
                pnl_pts = exit_price - row["entry_price"]
                rupee_pnl = round(pnl_pts * row["lot_size"]
                                   - compute_trade_cost(row["entry_price"], exit_price, row["lot_size"]), 2)
                conn.execute('''UPDATE paper_orders SET
                    exit_time=?, exit_price=?, exit_reason='STALE_AUTOCLOSED_ON_RESTART',
                    pnl_pts=?, rupee_pnl=?, status='CLOSED' WHERE id=? AND variant=?''',
                    (str(row["date"]) + "T15:20:00", exit_price, pnl_pts, rupee_pnl, row["id"], self.label))
                self.capital += rupee_pnl
                conn.commit()
                self.log_event(f"WARNING: found an orphaned OPEN position from {row['date']} (id={row['id']}) "
                                f"-- auto-closed at {exit_price} (STALE_AUTOCLOSED_ON_RESTART, "
                                f"PnL Rs {rupee_pnl:,.2f}).")
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to check for/close stale orphaned positions")

    def _restore_strategy_state(self, today):
        self._close_stale_orphaned_positions(today)
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM paper_orders WHERE variant=? AND date=? ORDER BY id ASC",
                (self.label, str(today))
            ).fetchall()
            conn.close()
        except Exception:
            logger.exception("Failed to check DB for existing state today - assuming none yet.")
            return [], 0

        if not rows:
            return [], 0
        positions = []
        trades_today = 0
        for row in rows:
            if row["status"] == "OPEN":
                positions.append({
                    "row_id": row["id"], "direction": row["direction"], "side": row["side"],
                    "strike": row["strike"], "expiry": row["expiry"],
                    "entry_time": pd.to_datetime(row["entry_time"]),
                    "entry_price": row["entry_price"], "lot": row["lot_size"],
                    "stop_level": row["current_stop"], "peak": row["peak_price"],
                    "trailing_active": bool(row["trailing_active"]),
                })
            elif row["status"] == "CLOSED":
                trades_today += 1
        if positions:
            self.log_event(f"RESTORED {len(positions)} OPEN position(s) after restart -- resuming live tracking.")
        return positions, trades_today

    def maybe_reset_for_new_day(self, today):
        if self.today != today:
            self.today = today
            self.day_open_spot = None
            restored_positions, trades_today = self._restore_strategy_state(today)
            self.positions = restored_positions
            self.trades_today = trades_today
            self.up_tracker = None
            self.down_tracker = None
            self.pending_signals = []
            self.log_event(f"New trading day: {today}. trail={self.trail_pts}pt. PROGRESSIVE 3pt entry "
                            f"(index-based, 20% max retracement), CAP={MAX_POSITIONS} concurrent positions. "
                            f"ENTRY CUTOFF 09:45, 2pt-TRAIL-ONLY. {trades_today} trade(s) already closed "
                            f"today (if any, from a restart). {len(self.positions)} open position(s) resumed.")

    def check_entry(self, today_df, bar_time):
        if len(self.positions) >= MAX_POSITIONS:
            return
        if bar_time.time() >= ENTRY_CUTOFF:
            return
        latest_close = float(today_df["close"].iloc[-1])
        long_fired = self.up_tracker.update_and_check(latest_close)
        short_fired = self.down_tracker.update_and_check(-latest_close)
        direction = None
        if long_fired:
            direction = "LONG"
        elif short_fired:
            direction = "SHORT"
        if direction is None:
            return
        # reset the tracker that fired immediately (matching the backtest's
        # LOCKED logic: each tracker resets to the index level at signal
        # time, regardless of whether the trade actually gets taken -- if
        # the cap is already full when execution is attempted, the signal
        # is dropped but the tracker has already moved on)
        if direction == "LONG":
            self.up_tracker.reset(latest_close)
        else:
            self.down_tracker.reset(-latest_close)
        self.pending_signals.append({"time": bar_time, "direction": direction})
        self.log_event(f"PROGRESSIVE signal detected ({direction}) at {bar_time} -- "
                        f"{len(self.positions)}/{MAX_POSITIONS} slots currently open. "
                        f"Will attempt entry next cycle (dropped if slots still full by execution time).")

    def try_execute_pending(self, bar_time, chain, expiry):
        if not self.pending_signals or chain is None:
            return
        still_pending = []
        for sig in self.pending_signals:
            if bar_time <= sig["time"]:
                still_pending.append(sig)
                continue
            if len(self.positions) >= MAX_POSITIONS:
                self.log_event(f"Signal from {sig['time']} ({sig['direction']}) DROPPED -- "
                                f"{MAX_POSITIONS}/{MAX_POSITIONS} slots already full by execution time.")
                continue  # drop it, do not requeue
            direction = sig["direction"]
            side = "CE" if direction == "LONG" else "PE"
            strike = om.itm_call_strike(self.day_open_spot) if side == "CE" else om.itm_put_strike(self.day_open_spot)
            ltp = get_ltp_from_chain(chain, strike, side)
            if ltp is None:
                self.log_event(f"WARNING -- could not get a real LTP for strike {strike} {side} this cycle. "
                                f"Will retry next cycle -- signal NOT dropped.")
                still_pending.append(sig)
                continue
            entry_time = bar_time
            lot = om.lot_size_for_date(entry_time.date())
            reason = (f"Progressive-3pt Opening-Drive Momentum (CONCURRENT cap={MAX_POSITIONS}, "
                      f"trail={self.trail_pts}pt, REAL premium, 2pt-ONLY, 09:45-CUTOFF COPY): index made a CLEAN "
                      f"{TRIGGER_PTS}pt {direction.lower()} move -- trade #{self.trades_today + 1} today, "
                      f"{len(self.positions) + 1}/{MAX_POSITIONS} slots after this entry.")
            row_id = self._record_open(direction, side, entry_time, ltp, reason, lot, strike, expiry)
            self.positions.append({
                "row_id": row_id, "direction": direction, "side": side, "strike": strike, "expiry": expiry,
                "entry_time": entry_time, "entry_price": ltp, "lot": lot,
                "stop_level": ltp - OM_STOP_PTS, "peak": ltp, "trailing_active": False,
            })
            self.log_event(f"OPENED {side} {direction} @ {ltp:.2f} (real LTP, strike {strike}, lot {lot}) at "
                            f"{entry_time} [trade #{self.trades_today + 1} today, "
                            f"{len(self.positions)}/{MAX_POSITIONS} slots now open]")
        self.pending_signals = still_pending

    def step_positions_live(self, chain):
        if not self.positions:
            return
        still_open = []
        now_ist = datetime.now(IST)
        for pos in self.positions:
            ltp = get_ltp_from_chain(chain, pos["strike"], pos["side"])
            if ltp is None:
                self.log_event(f"WARNING -- could not get a real LTP for open position id={pos['row_id']} "
                                f"strike {pos['strike']} {pos['side']} this cycle. Skipping management check.")
                still_open.append(pos)
                continue
            self._update_live_state(pos["row_id"], pos["stop_level"], pos["peak"], ltp, pos["trailing_active"])

            if ltp <= pos["stop_level"]:
                reason = "TRAIL" if pos["trailing_active"] else "STOP"
                self._close_position(pos, ltp, reason, now_ist)
                continue
            if not pos["trailing_active"]:
                # pos["entry_time"] is a naive IST wall-clock timestamp (from
                # the resampled index bar time), while now_ist is tz-aware --
                # strip tzinfo before subtracting so this can't crash on a
                # naive/aware mismatch.
                entry_time_naive = pos["entry_time"]
                if getattr(entry_time_naive, "tzinfo", None) is not None:
                    entry_time_naive = entry_time_naive.replace(tzinfo=None)
                elapsed_min = (now_ist.replace(tzinfo=None) - entry_time_naive).total_seconds() / 60.0
                if elapsed_min >= TIME_LIMIT_MIN:
                    self._close_position(pos, ltp, "TIME_STOP", now_ist)
                    continue
            if now_ist.time() >= om.EOD_SQUAREOFF:
                self._close_position(pos, ltp, "EOD", now_ist)
                continue
            if ltp > pos["peak"]:
                pos["peak"] = ltp
            if pos["peak"] > pos["entry_price"]:
                pos["trailing_active"] = True
            if pos["trailing_active"]:
                pos["stop_level"] = max(pos["stop_level"], pos["peak"] - self.trail_pts)
            self._update_live_state(pos["row_id"], pos["stop_level"], pos["peak"], ltp, pos["trailing_active"])
            still_open.append(pos)
        self.positions = still_open

    def _close_position(self, pos, exit_price, reason, exit_time):
        pnl_pts = exit_price - pos["entry_price"]
        rupee_pnl = pnl_pts * pos["lot"] - compute_trade_cost(pos["entry_price"], exit_price, pos["lot"])
        self.capital += rupee_pnl
        self._record_close(pos["row_id"], exit_time, exit_price, reason, pnl_pts, rupee_pnl, pos["lot"])
        self.trades_today += 1
        self.log_event(f"CLOSED {pos['side']} @ {exit_price:.2f} (real LTP, {reason}) | "
                        f"PnL Rs {rupee_pnl:,.2f} | capital now Rs {self.capital:,.2f} | "
                        f"trades today: {self.trades_today} | {len(self.positions) - 1}/{MAX_POSITIONS} slots remain open")

    def needs_chain_this_cycle(self):
        return bool(self.pending_signals) or bool(self.positions)

    def manual_exit_flag_path(self):
        return os.path.join(self.control_dir, f"MANUAL_EXIT_{self.label}.flag")

    def check_manual_exit_requested(self):
        all_path = os.path.join(self.control_dir, "MANUAL_EXIT_ALL.flag")
        return os.path.exists(self.manual_exit_flag_path()) or os.path.exists(all_path)

    def clear_manual_exit_flag(self):
        path = self.manual_exit_flag_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                logger.exception("Failed to remove manual-exit flag file")

    def manual_close_all_if_requested(self, chain):
        if not self.positions or not self.check_manual_exit_requested():
            return False
        if chain is None:
            self.log_event("MANUAL EXIT requested but no chain data this cycle -- will retry next cycle.")
            return True
        now_ist = datetime.now(IST)
        remaining = []
        for pos in self.positions:
            ltp = get_ltp_from_chain(chain, pos["strike"], pos["side"])
            if ltp is None:
                self.log_event(f"MANUAL EXIT: could not get real LTP for id={pos['row_id']} this cycle -- retrying next cycle.")
                remaining.append(pos)
                continue
            self.log_event(f"MANUAL EXIT requested -- closing position id={pos['row_id']} now at real LTP.")
            self._close_position(pos, ltp, "MANUAL", now_ist)
        self.positions = remaining
        if not self.positions:
            self.clear_manual_exit_flag()
        return True


class Cap1Runner:
    def __init__(self, config: PlatformConfig = None, security_id: str = "13", exchange_segment: str = "IDX_I"):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.security_id = security_id
        self.exchange_segment = exchange_segment

        self.db_path = os.path.join(self.config.log_dir, DB_FILENAME)
        self._init_db()

        self.control_dir = os.path.join(self.config.log_dir, CONTROL_SUBDIR)
        os.makedirs(self.control_dir, exist_ok=True)

        self.variants = [TrailVariantEngine(self.db_path, v["label"], v["trail_pts"], self.control_dir)
                          for v in VARIANTS]

        self.last_seen_bar_time = None
        self._last_new_bar_wall_time = datetime.now(IST)
        self._stall_warned_at = None

        self.current_expiry = None
        self._expiry_fetched_date = None
        self._chain_cooldown_until = None
        self._consecutive_fail_count = 0

        for v in self.variants:
            v.log_event(f"[PROGRESSIVE-3pt/CAP={MAX_POSITIONS}/09:45-CUTOFF/2pt-ONLY] live paper engine "
                        f"initialized (stop={OM_STOP_PTS}pt, trail={v.trail_pts}pt, entry=progressive {TRIGGER_PTS}pt "
                        f"index-based/20% max retracement, cutoff={ENTRY_CUTOFF}, concurrent cap={MAX_POSITIONS}, "
                        f"independent capital Rs {STARTING_CAPITAL:,.0f}). Real Dhan option-chain LTP. Own DB, own "
                        f"control folder -- fully separate from every other engine in this project, INCLUDING the "
                        f"no-cutoff 2pt-only version this was copied from.")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, variant TEXT, strategy TEXT, direction TEXT, side TEXT,
            entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL, exit_reason TEXT,
            pnl_pts REAL, rupee_pnl REAL, lot_size INTEGER,
            capital_after REAL, status TEXT,
            initial_stop REAL, current_stop REAL, peak_price REAL,
            last_price REAL, target_note TEXT, reason TEXT,
            strike REAL, trailing_active INTEGER, expiry TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, variant TEXT, event_message TEXT
        )''')
        conn.commit()
        conn.close()

    def _fetch_and_resample_index(self):
        self.connector.refresh_credentials()
        raw = self.connector.fetch_historical_intraday(self.security_id, self.exchange_segment, interval="1")
        if raw is None or raw.empty:
            return pd.DataFrame()
        ts_col = raw["timestamp"]
        if ts_col.notna().any():
            if pd.api.types.is_numeric_dtype(ts_col):
                raw["timestamp"] = pd.to_datetime(ts_col, unit="s", errors="coerce") + pd.Timedelta(hours=5, minutes=30)
            else:
                raw["timestamp"] = pd.to_datetime(ts_col, errors="coerce")
        else:
            raw["timestamp"] = None
        if raw["timestamp"].isna().all():
            return pd.DataFrame()
        df = raw.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            return df
        return om.resample_2min_per_day(df)

    def _ensure_expiry(self, today):
        if self.current_expiry is not None and self._expiry_fetched_date == today:
            return self.current_expiry
        expiries = self.connector.fetch_expiry_list(self.security_id, self.exchange_segment)
        if not expiries:
            logger.warning("Could not fetch expiry list this cycle -- will retry next cycle.")
            return self.current_expiry
        try:
            expiry = sorted(expiries)[0]
        except Exception:
            expiry = expiries[0]
        self.current_expiry = expiry
        self._expiry_fetched_date = today
        return expiry

    def _fetch_live_chain(self, today):
        now_ist = datetime.now(IST)
        if self._chain_cooldown_until is not None and now_ist < self._chain_cooldown_until:
            remaining = (self._chain_cooldown_until - now_ist).total_seconds()
            logger.info(f"Option-chain call skipped -- in cooldown ({remaining:.0f}s remaining).")
            return None
        expiry = self._ensure_expiry(today)
        if expiry is None:
            backoff_s = self._next_backoff_seconds()
            self._chain_cooldown_until = now_ist + timedelta(seconds=backoff_s)
            logger.warning(f"Could not resolve an expiry this cycle -- backing off {backoff_s:.0f}s.")
            return None
        chain = self.connector.fetch_option_chain(expiry, self.security_id, self.exchange_segment)
        if chain is None:
            backoff_s = self._next_backoff_seconds()
            self._chain_cooldown_until = now_ist + timedelta(seconds=backoff_s)
            logger.warning(f"Option-chain fetch failed this cycle (likely rate-limited) -- backing off "
                            f"{backoff_s:.0f}s (consecutive failures: {self._consecutive_fail_count}).")
        else:
            self._consecutive_fail_count = 0
            self._chain_cooldown_until = None
        return chain

    def _next_backoff_seconds(self):
        self._consecutive_fail_count += 1
        base = min(90 * (2 ** (self._consecutive_fail_count - 1)), 300)
        jitter = random.uniform(-5, 10)
        return max(60, base + jitter)

    def run_cycle(self):
        now_ist = datetime.now(IST)
        today = now_ist.date()
        for v in self.variants:
            v.maybe_reset_for_new_day(today)

        today_df = self._fetch_and_resample_index()
        if today_df.empty:
            logger.info("No live index bar data this cycle. Skipped.")
            return

        day_open_spot = float(today_df["open"].iloc[0])
        for v in self.variants:
            if v.day_open_spot is None:
                v.day_open_spot = day_open_spot
            if v.up_tracker is None:
                v.up_tracker = SideTracker(day_open_spot)
            if v.down_tracker is None:
                v.down_tracker = SideTracker(-day_open_spot)

        latest_bar_time = today_df.index[-1]
        if self.last_seen_bar_time is not None and latest_bar_time <= self.last_seen_bar_time:
            stalled_minutes = (now_ist - self._last_new_bar_wall_time).total_seconds() / 60.0
            if stalled_minutes >= STALL_WARN_MINUTES:
                already_warned = (self._stall_warned_at is not None and
                                   (now_ist - self._stall_warned_at).total_seconds() < STALL_WARN_MINUTES * 60)
                if not already_warned:
                    logger.warning(f"Index feed appears STALLED -- latest bar still {latest_bar_time} "
                                    f"({stalled_minutes:.1f} min).")
                    self._stall_warned_at = now_ist
            new_bars = False
        else:
            self._last_new_bar_wall_time = now_ist
            self._stall_warned_at = None
            new_bars = True

        manual_exit_pending = any(v.positions and v.check_manual_exit_requested() for v in self.variants)
        need_chain = any(v.needs_chain_this_cycle() for v in self.variants) or manual_exit_pending
        chain = self._fetch_live_chain(today) if need_chain else None
        expiry = self.current_expiry

        if new_bars:
            new_bar_times = today_df.index[
                today_df.index > (self.last_seen_bar_time or today_df.index[0] - timedelta(seconds=1))]
            for bar_time in new_bar_times:
                local_today_df = today_df.loc[:bar_time]
                for v in self.variants:
                    v.check_entry(local_today_df, bar_time)
                for v in self.variants:
                    v.try_execute_pending(bar_time, chain, expiry)  # FIXED: was latest_bar_time, called once per cycle outside this loop -- now once per newly-closed bar
            self.last_seen_bar_time = latest_bar_time

        for v in self.variants:
            if v.positions:
                if v.manual_close_all_if_requested(chain):
                    continue
                v.step_positions_live(chain)

        all_flag = os.path.join(self.control_dir, "MANUAL_EXIT_ALL.flag")
        if os.path.exists(all_flag) and all(not v.positions for v in self.variants):
            try:
                os.remove(all_flag)
            except Exception:
                logger.exception("Failed to remove MANUAL_EXIT_ALL.flag after all positions closed")

    def run_forever(self):
        for v in self.variants:
            v.log_event(f"Live loop starting. Poll interval: {POLL_SECONDS}s. Capital: Rs {v.capital:,.2f}")
        try:
            while True:
                now_ist = datetime.now(IST).time()
                if now_ist < MARKET_OPEN or now_ist > MARKET_CLOSE:
                    time.sleep(30)
                    continue
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.exception(f"Unhandled error in live cycle: {e}")
                time.sleep(POLL_SECONDS + random.uniform(0, 4))
        except KeyboardInterrupt:
            for v in self.variants:
                v.log_event(f"[PROGRESSIVE-3pt/CAP={MAX_POSITIONS}/09:45-CUTOFF/2pt-ONLY] live paper engine stopped manually (Ctrl+C).")


if __name__ == "__main__":
    run_self_checks()
    runner = Cap1Runner()
    runner.run_forever()
