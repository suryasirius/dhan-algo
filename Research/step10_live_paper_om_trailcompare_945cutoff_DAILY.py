"""
DAILY LIVE-PAPER ENGINE, 09:45 CUTOFF -- step10_live_paper_om_trailcompare_945cutoff_DAILY.py

STATUS (2026-08-20): starting TOMORROW, this is the script to run daily --
NOT step10_live_paper_om_trailcompare_NOCUTOFF_TEMP.py. Per explicit
instruction: "todays script leave... from tomorrow it should be 9.45
cutoff.. dont mix script save this as new file and path separate log
also." Today's no-cutoff run keeps its own file and DB completely
untouched -- this is a BRAND NEW file with its OWN separate DB, so
tomorrow's before945-only trades never mix with today's all-day data.

THREE DIFFERENT DBs now exist for this strategy family -- know which is which:
    1. openmomentum_before945_trailcompare_realpremium_paper_portfolio.db
       -- the original validated before945 engine, from earlier today's
       rate-limit troubleshooting. Not in daily use.
    2. openmomentum_trailcompare_nocutoff_temp_paper_portfolio.db
       -- TODAY ONLY, all-day entries (cutoff 15:15). Left running as-is,
       do not touch. Has today's real trade history in it.
    3. openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db
       -- THIS FILE's DB. Starts EMPTY. This is what accumulates your
       real daily-use history from tomorrow onward, with the 09:45 cutoff
       restored.

Description: Three independent trail-width variants (1pt, 2pt, 3pt),
             sharing ONE option-chain fetch per cycle, one process.
             Entries only fire before 09:45 (ENTRY_CUTOFF = 09:45, same
             as the originally validated before945 rule) -- NOT all day.
             Everything else (stop=20pt, 1-bar lag, re-anchor-on-exit,
             strike selection, shared-fetch pattern, exponential backoff)
             is identical to the no-cutoff daily engine.

DAILY USAGE: run the OI/premium collector during 09:14-09:45 as usual,
close it manually at 09:45, THEN start this script. Do not run this
alongside the collector, the nocutoff-temp engine, or the older
before945 script -- same Dhan account, same rate limit, same collision
as before. Only ONE of these three engines should ever be running at a
time.

TRADE LOGS / DATA LOCATION:
    C:\\dhan algo\\Research\\logs\\openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db
    (SQLite -- paper_orders + event_logs, partitioned by `variant`:
    trail1pt/trail2pt/trail3pt (stale, pre-cooldown-fix) plus trail1pt_v2
    (current, validated 1.0pt trail -- see VARIANTS below for the numbers).)

Run as its own process:
    cd "C:\\dhan algo\\Research"
    python step10_live_paper_om_trailcompare_945cutoff_DAILY.py
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
import nifty_open_momentum_stoptrail_sweep as om
from trade_cost_model import compute_trade_cost  # real per-trade brokerage+STT+exchange+GST+stamp
# duty cost model -- same one used by step10_live_paper_vwap20ema_trail2act10.py. Replaces the old
# flat om.BROKERAGE_PER_TRADE deduction below, which only accounted for brokerage and silently
# understated real costs (missing STT, exchange transaction charges, GST, SEBI fees, stamp duty).

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Step10_LivePaper_OM_TrailCompare_945Cutoff_DAILY")

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

STARTING_CAPITAL = 25_000.0
OM_STOP_PTS = 20
MIN_MOVE_PTS = om.MIN_MOVE_PTS
# 09:45 CUTOFF RESTORED -- this is the whole point of this file (as
# opposed to the nocutoff-temp engine, which allows entries all day).
ENTRY_CUTOFF = om.ENTRY_CUTOFF
POLL_SECONDS = 20  # reduced from 45s per explicit request -- tighter poll means
                    # stop/trail exits fill closer to their configured level.
STALL_WARN_MINUTES = 5

VARIANTS = [
    # trail1pt and trail3pt REMOVED per explicit request -- 2pt-trail ONLY now.
    # Their historical rows in the shared DB (variant='trail1pt'/'trail3pt')
    # are untouched, just no longer written to going forward.
    #
    # TRAIL WIDTH TIGHTENED (2.0pt -> 1.0pt): a trail-width sweep (1.0 to
    # 4.0pt) run on top of the validated cooldown fix found tighter trail
    # helps THIS strategy -- the opposite direction from OM Best, where
    # wider trail won. All three datasets agreed (not just a real9 fluke):
    # real9 PF 2.40->2.64 (net +Rs2,514->+Rs2,694), plain122 PF 2.43->2.63,
    # noise122 PF 2.51->2.63. OOS-checked: plain122 IS 2.78->OOS 2.25,
    # noise122 IS 2.78->OOS 2.32 (no collapse, both ahead of 2.0pt's own
    # OOS numbers at every point). A finer sweep down to 0.25-0.75pt kept
    # nudging the backtest number higher still, but that range was NOT
    # adopted -- a trail that tight sits inside a typical option's bid-ask
    # spread, which this backtest doesn't model, so those numbers aren't
    # trustworthy. 1.0pt is the tightest value still comfortably outside
    # that concern.
    #
    # Relabeled from "trail2pt" to "trail1pt_v2" (NOT reusing the old
    # "trail1pt" label) -- the original trail1pt variant's historical DB
    # rows predate the 09:45-cutoff restore AND the cooldown fix, so
    # reusing that label would silently mix two incomparable strategy
    # versions under one key. "trail1pt_v2" starts a clean history.
    {"label": "trail1pt_v2", "trail_pts": 1.0},
]

# EXHAUSTION-TOP FIX (same underlying problem the user flagged for the OM
# Best/SideTracker strategy, but this engine's re-anchoring mechanism is
# different, so the fix that worked there -- an ATR extension cap -- was
# tested here and did NOT help (real 9-day PF stayed ~0.7-0.8 at every
# threshold tried). The real mechanism here: `reference` re-anchors to
# whatever the index closed at the INSTANT a position exits, then the
# engine immediately starts hunting for another MIN_MOVE_PTS move from
# that fresh anchor -- so a signal right after a stop-out is very often
# just chasing the tail end of the SAME already-exhausted move. The fix is
# a cooldown: require COOLDOWN_BARS bars (COOLDOWN_BARS * 2min) to pass
# after any exit before a new signal is allowed to fire again. Validated:
# real 9-day PF 0.79->2.40, net -Rs1,017.50->+Rs2,513.75 at 5 bars; stayed
# on a clean plateau across 4-6 bars; both 122-day synthetic sets stayed
# comparably profitable (not curve-fit to real data alone); OOS split held
# up (PLAIN_122 2.64->1.90 OOS, NOISE_122 2.67->2.18 OOS).
# COOLDOWN REMOVED (set to 0) per explicit user request. The 5-bar cooldown
# above was the validated setting (real 9-day PF 0.79->2.40 with it); with 0
# the engine can re-enter on the very bar after an exit, so the
# exhaustion-top behavior described above is back in play. Documented as a
# conscious tradeoff. Set back to 5 to restore the validated behavior.
COOLDOWN_BARS = 0


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


def run_self_checks():
    print("Running self-checks (synthetic data, no network)...")
    sample = {
        "last_price": 24343.5,
        "oc": {
            "24300.000000": {
                "ce": {"last_price": 75.2}, "pe": {"last_price": 40.1},
            },
        },
    }
    assert get_ltp_from_chain(sample, 24300, "CE") == 75.2
    assert get_ltp_from_chain(sample, 24400, "CE") is None
    assert get_ltp_from_chain(None, 24300, "CE") is None
    print("  get_ltp_from_chain: OK")

    assert len(VARIANTS) == 1, f"this copy must run ONLY one trail variant, got {len(VARIANTS)}"
    assert VARIANTS[0]["label"] == "trail1pt_v2" and VARIANTS[0]["trail_pts"] == 1.0
    print(f"  variant config: OK (1.0pt-trail ONLY [tightened from 2pt, validated], {[v['label'] for v in VARIANTS]})")

    assert ENTRY_CUTOFF == dtime(9, 45), f"this DAILY variant must use the 09:45 cutoff, got {ENTRY_CUTOFF}"
    print(f"  cutoff check: OK (ENTRY_CUTOFF = {ENTRY_CUTOFF}, restored to 09:45 -- NOT the all-day "
          f"nocutoff-temp behavior)")
    assert POLL_SECONDS == 20, f"expected 20s poll interval, got {POLL_SECONDS}"
    print(f"  poll interval check: OK (POLL_SECONDS = {POLL_SECONDS})")

    assert COOLDOWN_BARS == 0, f"cooldown was removed by request, expected 0, got {COOLDOWN_BARS}"
    import tempfile
    _selftest_db = os.path.join(tempfile.gettempdir(), "trailcompare_945cutoff_daily_selftest_scratch.db")
    eng = TrailVariantEngine(_selftest_db, "selftest", 2)
    base_time = pd.Timestamp("2026-01-01 09:20:00")
    eng.last_exit_bar_time = base_time
    eng.reference = 24200.0
    # with the cooldown removed, a genuine move right after an exit (2 minutes,
    # i.e. the very next bar) must be allowed through
    still_cooling_df = pd.DataFrame({"close": [24210.0]})
    eng.check_entry(still_cooling_df, base_time + timedelta(minutes=2))
    assert eng.pending_signal_time is not None, "with COOLDOWN_BARS=0 a signal right after an exit must NOT be blocked"
    print(f"  cooldown check: OK (COOLDOWN_BARS={COOLDOWN_BARS} -- cooldown removed, signal allowed on the "
          f"bar right after an exit)")
    print("All self-checks passed.\n")


class TrailVariantEngine:
    """One independent multi-entry strategy instance for a single trail
    width, with the 09:45 entry cutoff restored. Owns its own capital/
    position/reference state. Shares a SQLite DB FILE with its sibling
    variants, but every query is scoped to its own `variant` label, so
    they never see or affect each other's rows."""

    def __init__(self, db_path, label, trail_pts):
        self.db_path = db_path
        self.label = label
        self.trail_pts = trail_pts

        self.capital = STARTING_CAPITAL
        self._load_capital_from_db()

        self.today = None
        self.day_open_spot = None
        self.reference = None
        self.pending_signal_time = None
        self._pending_direction = None
        self.position = None
        self.trades_today = 0
        self.last_exit_bar_time = None  # cooldown state -- reset each new day

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
            VALUES (?, ?, 'open_momentum_trailcompare_945cutoff_daily', ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)''',
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
                conn.commit()   # commit BEFORE log_event opens its own connection -- avoids "database is locked"
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
            return None, 0

        if not rows:
            return None, 0
        position = None
        trades_today = 0
        for row in rows:
            if row["status"] == "OPEN":
                position = {
                    "row_id": row["id"], "direction": row["direction"], "side": row["side"],
                    "strike": row["strike"], "expiry": row["expiry"],
                    "entry_time": pd.to_datetime(row["entry_time"]),
                    "entry_price": row["entry_price"], "lot": row["lot_size"],
                    "stop_level": row["current_stop"], "peak": row["peak_price"],
                    "trailing_active": bool(row["trailing_active"]),
                }
            elif row["status"] == "CLOSED":
                trades_today += 1
        if position is not None:
            self.log_event(f"RESTORED an OPEN position after restart (entry {position['entry_price']} at "
                            f"{position['entry_time']}) - resuming live tracking.")
        return position, trades_today

    def maybe_reset_for_new_day(self, today):
        if self.today != today:
            self.today = today
            self.day_open_spot = None
            restored_position, trades_today = self._restore_strategy_state(today)
            self.position = restored_position
            self.trades_today = trades_today
            self.reference = None
            self.pending_signal_time = None
            self.last_exit_bar_time = None
            self.log_event(f"New trading day: {today}. trail={self.trail_pts}pt. cutoff=09:45. "
                            f"{trades_today} trade(s) already closed today (if any, from a restart). "
                            f"{'Resuming an OPEN position.' if self.position else 'No open position.'}")

    def check_entry(self, today_df, bar_time):
        if self.position is not None or self.pending_signal_time is not None:
            return
        if bar_time.time() >= ENTRY_CUTOFF:
            return
        if self.last_exit_bar_time is not None:
            cooldown_until = self.last_exit_bar_time + timedelta(minutes=2 * COOLDOWN_BARS)
            if bar_time < cooldown_until:
                return  # exhaustion-top fix: still cooling down after the last exit
        latest_close = float(today_df["close"].iloc[-1])
        move = latest_close - self.reference
        direction = None
        if move >= MIN_MOVE_PTS:
            direction = "LONG"
        elif move <= -MIN_MOVE_PTS:
            direction = "SHORT"
        if direction is None:
            return
        self.pending_signal_time = bar_time
        self._pending_direction = direction
        self.log_event(f"signal #{self.trades_today + 1} today detected ({direction}, {move:+.1f}pts from "
                        f"reference {self.reference:.1f}) at {bar_time} - will enter next bar's open.")

    def try_execute_pending(self, bar_time, chain, expiry):
        if self.pending_signal_time is None or self.position is not None:
            return
        if bar_time <= self.pending_signal_time:
            return
        direction = self._pending_direction
        side = "CE" if direction == "LONG" else "PE"
        strike = om.itm_call_strike(self.day_open_spot) if side == "CE" else om.itm_put_strike(self.day_open_spot)
        ltp = get_ltp_from_chain(chain, strike, side)
        if ltp is None:
            self.log_event(f"WARNING -- could not get a real LTP for strike {strike} {side} this cycle. "
                            f"Will retry next cycle -- pending signal NOT dropped.")
            return
        entry_time = bar_time
        lot = om.lot_size_for_date(entry_time.date())
        reason = (f"Opening-Drive Momentum (multi-entry before 09:45, trail={self.trail_pts}pt, REAL "
                  f"premium): Nifty moved {MIN_MOVE_PTS}+ points {direction.lower()} from the re-anchored "
                  f"reference ({self.reference:.1f}) -- trade #{self.trades_today + 1} today.")
        row_id = self._record_open(direction, side, entry_time, ltp, reason, lot, strike, expiry)
        self.position = {
            "row_id": row_id, "direction": direction, "side": side, "strike": strike, "expiry": expiry,
            "entry_time": entry_time, "entry_price": ltp, "lot": lot,
            "stop_level": ltp - OM_STOP_PTS, "peak": ltp, "trailing_active": False,
        }
        self.pending_signal_time = None
        self.log_event(f"OPENED {side} {direction} @ {ltp:.2f} (real LTP, strike {strike}, lot {lot}) at "
                        f"{entry_time} [trade #{self.trades_today + 1} today]")

    def step_position_live(self, index_close_now, chain):
        pos = self.position
        if pos is None:
            return
        ltp = get_ltp_from_chain(chain, pos["strike"], pos["side"])
        if ltp is None:
            self.log_event(f"WARNING -- could not get a real LTP for open position's strike {pos['strike']} "
                            f"{pos['side']} this cycle. Skipping management check.")
            return
        now_ist = datetime.now(IST)
        self._update_live_state(pos["row_id"], pos["stop_level"], pos["peak"], ltp, pos["trailing_active"])

        if ltp <= pos["stop_level"]:
            reason = "TRAIL" if pos["trailing_active"] else "STOP"
            self._close_position(ltp, reason, now_ist, index_close_now)
            return
        if now_ist.time() >= om.EOD_SQUAREOFF:
            self._close_position(ltp, "EOD", now_ist, index_close_now)
            return
        if ltp > pos["peak"]:
            pos["peak"] = ltp
        if pos["peak"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], pos["peak"] - self.trail_pts)
        self._update_live_state(pos["row_id"], pos["stop_level"], pos["peak"], ltp, pos["trailing_active"])

    def _close_position(self, exit_price, reason, exit_time, index_close_at_exit):
        pos = self.position
        pnl_pts = exit_price - pos["entry_price"]
        rupee_pnl = pnl_pts * pos["lot"] - compute_trade_cost(pos["entry_price"], exit_price, pos["lot"])
        self.capital += rupee_pnl
        self._record_close(pos["row_id"], exit_time, exit_price, reason, pnl_pts, rupee_pnl, pos["lot"])
        self.trades_today += 1
        self.log_event(f"CLOSED {pos['side']} @ {exit_price:.2f} (real LTP, {reason}) | "
                        f"PnL Rs {rupee_pnl:,.2f} | capital now Rs {self.capital:,.2f} | "
                        f"trades today: {self.trades_today}")
        self.position = None
        self.reference = index_close_at_exit
        self.last_exit_bar_time = exit_time.replace(tzinfo=None) if getattr(exit_time, "tzinfo", None) else exit_time
        self.log_event(f"reference re-anchored to {index_close_at_exit:.1f} -- cooling down for "
                        f"{COOLDOWN_BARS} bars ({2 * COOLDOWN_BARS}min) before scanning for the next opportunity.")

    def needs_chain_this_cycle(self, new_bars, latest_bar_time):
        pending_ready = (self.pending_signal_time is not None and self.position is None
                          and new_bars and latest_bar_time > self.pending_signal_time)
        return pending_ready or self.position is not None


class TrailCompareRunner:
    """Owns the shared index/chain fetch and drives every TrailVariantEngine
    instance off the SAME data each cycle -- one option-chain call per
    cycle total, regardless of how many variants need a price."""

    def __init__(self, config: PlatformConfig = None, security_id: str = "13", exchange_segment: str = "IDX_I"):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.security_id = security_id
        self.exchange_segment = exchange_segment

        # Brand new, separate DB -- distinct from both the earlier
        # before945 engine's DB and today's nocutoff-temp DB. Starts
        # empty; this is what accumulates the daily-use history from
        # tomorrow onward.
        self.db_path = os.path.join(
            self.config.log_dir, "openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db")
        self._init_db()

        self.variants = [TrailVariantEngine(self.db_path, v["label"], v["trail_pts"]) for v in VARIANTS]

        self.last_seen_bar_time = None
        self._last_new_bar_wall_time = datetime.now(IST)
        self._stall_warned_at = None

        self.current_expiry = None
        self._expiry_fetched_date = None
        self._chain_cooldown_until = None
        self._consecutive_fail_count = 0

        for v in self.variants:
            v.log_event(f"[DAILY/09:45-CUTOFF] trail-compare live paper engine initialized (stop={OM_STOP_PTS}pt, "
                        f"trail={v.trail_pts}pt, cutoff={ENTRY_CUTOFF}, independent capital "
                        f"Rs {STARTING_CAPITAL:,.0f}). Real Dhan option-chain LTP, shared fetch per cycle "
                        f"with sibling variant(s). Separate DB from both the earlier before945 engine and "
                        f"today's nocutoff-temp engine.")

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
            if v.reference is None:
                v.reference = day_open_spot

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

        need_chain = any(v.needs_chain_this_cycle(new_bars, latest_bar_time) for v in self.variants) or \
                     any(v.position is not None for v in self.variants)
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

        index_close_now = float(today_df["close"].iloc[-1])
        for v in self.variants:
            if v.position is not None:
                v.step_position_live(index_close_now, chain)

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
                v.log_event("[DAILY/09:45-CUTOFF] Trail-compare live paper engine stopped manually (Ctrl+C).")


if __name__ == "__main__":
    run_self_checks()
    runner = TrailCompareRunner()
    runner.run_forever()
