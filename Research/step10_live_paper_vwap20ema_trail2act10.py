"""
VWAP 20EMA -- TRAIL2/ACT10 -- LIVE PAPER ENGINE
step10_live_paper_vwap20ema_trail2act10.py

WHAT THIS IS: live paper-trading engine for the VWAP+20EMA crossover
strategy, frozen with the config validated on real data + two independent
122-day synthetic datasets + an out-of-sample split + a neighborhood
robustness sweep, ALL using a corrected, look-ahead-free simulation (an
earlier "30-min swing exit" version of this strategy was found and fixed
to have a look-ahead bug in its exit logic before this simpler design was
adopted instead):

    Real 9 OI days:  n=8,  PF=3.37, net +Rs3,452.54
    31 Aug 2026 (live check): n=3, PF=1.86, net +Rs1,349.50
    PLAIN_122 (full):  n=321, PF=1.80, net +Rs99,301.37
    NOISE_122 (full):  n=321, PF=1.85, net +Rs1,19,466.54
    PLAIN_122 OOS (last 42 days, unseen): PF=2.11, net +Rs43,211.60
    NOISE_122 OOS (last 42 days, unseen): PF=2.59, net +Rs62,177.83
    Neighborhood sweep (SL 15/20/25, activation 8/10/12, trail 1/2/3pt):
        every combination stayed solidly profitable on both OOS sets and
        real data -- this config sits inside a stable plateau, not on a
        lucky edge.

ENTRY RULE (identical to the validated backtest -- the "old rule"
    crossover, NOT the Setup A/B/C + SideTracker progressive-confirmation
    gate used by the sibling engine step10_live_paper_vwap20ema_stagedstop_v2.py):
        On 2-min NIFTY index bars:
          1. "Armed" state is sticky/monotonic: CE-armed whenever price
             AND EMA20 are both below VWAP; PE-armed whenever price AND
             EMA20 are both above VWAP (checked on the PREVIOUS bar).
          2. CE signal fires when: was CE-armed, EMA20 was below VWAP on
             the previous bar, EMA20 is now >= VWAP, AND price > EMA20
             on this SAME bar (same-bar confirmation -- no separate
             persistence/pullback/retest logic).
          3. PE signal fires on the mirror condition.
        Execution happens on the NEXT 2-min bar's real option premium
        (1-bar lag, no look-ahead). Single position at a time (cap=1).
        Entry cutoff and EOD square-off are both 15:20 (ENTRY_CUTOFF /
        om.EOD_SQUAREOFF) -- entries allowed all session up to that point,
        which in practice coincides with the EOD square-off itself, so no
        bar after 15:20 would open a fresh position anyway. Also capped at
        MAX_TRADES_PER_DAY entries per day.

        VWAP CAVEAT (unchanged from every backtest run this project): no
        dataset here has real index volume, so VWAP is an UNWEIGHTED
        cumulative average of typical price (H+L+C)/3 from day start --
        NOT a true volume-weighted VWAP. This was directly confirmed to
        cause some early-session (first ~30-45 min) false signals during
        this session's research; accepted as a known, unfixable data
        limitation (no real volume feed available).

EXIT RULE (in OPTION PREMIUM points, not index points -- see the
    INITIAL_STOP_PTS / VWAP_DIST_MAX_FILTER / MAX_TRADES_PER_DAY constants
    below and their inline comments for the current values and the
    tightening history; this docstring intentionally does not repeat the
    live numbers so it can't go stale again the way the old "20pt SL"
    wording did):
        Initial stop  : entry_price - INITIAL_STOP_PTS  (premium points)
        Trail activate: once (peak_premium - entry_price) >= TRAIL_ACTIVATE_PTS
        Once active   : staged widening trail per STAGED_TRAIL_TABLE,
                         monotonic, never loosens.
        Else          : EOD square-off at 15:20. Entry cutoff is also
                         15:20 (ENTRY_CUTOFF) -- effectively coincides with
                         EOD square-off, so no bar after 15:20 would open
                         a fresh position anyway.
        Max trades/day: MAX_TRADES_PER_DAY (enforced in _process_new_bar --
                         no new pending_signal is set once trades_today
                         reaches this cap; an already-open position still
                         manages/exits normally).
    Both the MFE/peak tracking and the stop-hit check are evaluated
    using the real option LTP each poll cycle (not bar close/open --
    this is a live engine, there is no "bar" once inside a live position,
    just the latest real premium tick each cycle, matching every other
    live engine in this project).

    NOTE ON PERFORMANCE FIGURES ABOVE: the backtest numbers quoted at the
    top of this docstring (Real9/PLAIN_122/NOISE_122/OOS) were measured
    against an earlier 20pt-stop, no-VWAP-filter configuration. They do
    NOT describe the current INITIAL_STOP_PTS=10 + VWAP_DIST_MAX_FILTER=40
    configuration this file now runs -- see the inline comments next to
    those two constants for the results that actually apply today.

IMPLEMENTATION NOTE ON RESTART BEHAVIOR: EMA20 and VWAP are maintained
    INCREMENTALLY (running EMA update, running typical-price cumulative
    sum/count) bar-by-bar as new 2-min bars arrive, NOT recomputed from
    the full day's history every cycle. On the first cycle after a fresh
    start/restart, historical bars are fast-forwarded WITHOUT generating
    signals (restart-replay guard, same class as every other engine in
    this project) -- EMA20/VWAP begin accumulating fresh from the restart
    point. Already-OPEN positions are restored from the DB and managed
    normally regardless.

THIS FILE       : step10_live_paper_vwap20ema_trail2act10.py
THIS DB         : logs/vwap20ema_trail2act10_paper_portfolio.db
THIS CONTROL DIR: logs/control_vwap20ema_trail2act10/
THIS LOGGER NAME: Step10_LivePaper_VWAP20EMA_Trail2Act10
THIS PORT       : 8520 (8503-8519 = every other engine in this project,
                  documented in their own files -- this is the next free
                  slot, does not collide with any of them)

DOES NOT TOUCH, READ, OR WRITE ANYTHING FROM ANY OTHER LIVE ENGINE in
this project -- separate DB, separate control folder, separate port.

PAPER TRADING ONLY. No real orders are ever placed by this file.

Run as its own process:
    cd "C:\\dhan algo\\Research"
    python step10_live_paper_vwap20ema_trail2act10.py

Dashboard (separate process, separate port):
    python -m streamlit run step14_dashboard_vwap20ema_trail2act10.py --server.port 8520
"""

import os
import sys
import json
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
from trade_cost_model import compute_trade_cost  # real per-trade brokerage+charges formula

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Step10_LivePaper_VWAP20EMA_Trail2Act10")

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

STARTING_CAPITAL = 25_000.0
ENTRY_CUTOFF = dtime(15, 20)  # effectively no cutoff -- entries allowed all session up to EOD square-off
POLL_SECONDS = 20
STALL_WARN_MINUTES = 5

EMA_SPAN = 20
EMA_ALPHA = 2.0 / (EMA_SPAN + 1.0)

INITIAL_STOP_PTS = 10.0   # option premium points (tightened again from 15->10 per
# explicit request on 2026-09-18. Backtest evidence continues to point the other
# way at every step tested: 20pt outperformed 15pt (PLAIN_122 PF 1.81->1.70,
# NOISE_122 PF 1.77->1.60), and the fuller sweep down to 10pt was worse still
# (PLAIN_122 PF 1.56->1.45 net +794.76->+600.45pt; NOISE_122 PF 1.51->1.38
# net +876.69->+622.61pt, vs the original 20pt baseline). Applied anyway per
# explicit user request -- documented here so this is a known, conscious
# tradeoff, not an oversight.
TRAIL_ACTIVATE_PTS = 10.0  # option premium points of profit before trail engages

MAX_TRADES_PER_DAY = 5   # ADDED 2026-09-25 per code-review finding: trades_today
# was already tracked and logged everywhere ("trade #N today"), but nothing
# actually compared it against a cap -- the docstring/logs said "maxTrades=5"
# while the code would have kept taking a 6th, 7th, etc. signal indefinitely.
# Enforced in _process_new_bar(): once trades_today >= 5, no new pending_signal
# is set for the rest of that trading day (existing open position, if any,
# still manages/exits normally -- this only blocks new entries).

# VWAP-DISTANCE ENTRY FILTER -- skip a crossover signal if the index price
# has already closed more than this many INDEX points away from VWAP at
# the signal bar. Validated this session: tested candle-range momentum
# (no effect), EMA-slope steepness (weak negative), and this VWAP-distance
# measure (clearest signal) against real9 + PLAIN_122 + NOISE_122 --
# signals close to VWAP win far more often (78.4% win%, 15.9% full-stop
# rate) than signals far from VWAP (59.6% win%, 32.6% full-stop rate).
# A <=40pt cutoff raised PF ~16-17% on both 122-day sets (PLAIN 1.56->1.81,
# NOISE 1.51->1.77) while keeping net points flat/better, and barely
# touched real9 (lost 1 of 8 trades, -3pt). Tighter cutoffs (25/20pt) push
# PF higher still but start giving up real net points -- 40pt was the
# validated sweet spot, not cherry-picked from a single dataset.
VWAP_DIST_MAX_FILTER = 40.0  # index points

# STAGED (monotonic-widening) TRAIL -- replaces the old flat TRAIL_PTS=2.0.
# Validated this session: backtested against real9 + PLAIN_122 + NOISE_122,
# beat the flat-2pt baseline on ALL THREE datasets with the same trade count
# and win% (only the exit changed, not entries), so the improvement is
# purely from letting genuinely large moves keep more room instead of
# capping them at a tight 2pt giveback:
#   Real9:      PF 4.02->4.51, net +53.45->+62.10pt
#   PLAIN_122:  PF 1.45->1.56, net +633.80->+794.76pt
#   NOISE_122:  PF 1.47->1.51, net +805.56->+876.69pt
# Two riskier variants were tested and REJECTED: a "cyclic" version that
# resets the trail tight every 5pt (worse on all 3 datasets -- yanks the
# trail tight right as big moves are extending) and an "extended" version
# that keeps widening past 18pt to a 5pt cap (mixed: better on NOISE_122,
# worse on PLAIN_122, flat on real9 -- not a consistent signal). A further
# sweep of the beyond-20pt trail width showed classic grid-edge/artifact
# behavior (PF kept climbing without turning over as the trail widened
# toward "effectively no trail", non-monotonic on NOISE_122) -- rejected,
# NOT adopted. This table (widen 10->15, then hold flat at 4pt) is the
# clean, consistent result with no red flags.
STAGED_TRAIL_TABLE = [(15.0, 4.0), (14.0, 3.5), (13.0, 3.0), (12.0, 2.5), (10.0, 2.0)]


def trail_distance_for_mfe(mfe_pts):
    """Returns the trail distance (option premium points) for a given
    max-favorable-excursion (peak - entry). Monotonic step table, checked
    in descending threshold order; falls through to the base 2.0pt if mfe
    hasn't reached even the first activation-adjacent threshold (shouldn't
    happen in practice since this is only called once trailing_active)."""
    for threshold, dist in STAGED_TRAIL_TABLE:
        if mfe_pts >= threshold:
            return dist
    return 2.0

DB_FILENAME = "vwap20ema_trail2act10_paper_portfolio.db"
CONTROL_SUBDIR = "control_vwap20ema_trail2act10"


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
    sample = {"last_price": 24343.5, "oc": {"24300.000000": {"ce": {"last_price": 75.2}, "pe": {"last_price": 40.1}}}}
    assert get_ltp_from_chain(sample, 24300, "CE") == 75.2
    assert get_ltp_from_chain(sample, 24400, "CE") is None
    assert get_ltp_from_chain(None, 24300, "CE") is None
    print("  get_ltp_from_chain: OK")

    # EMA incremental-update sanity check against pandas ewm on a fixed series
    series = pd.Series([100, 101, 102.5, 101.8, 103, 104.2, 103.5, 105])
    pandas_ema = series.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1]
    running = float(series.iloc[0])
    for v in series.iloc[1:]:
        running = EMA_ALPHA * float(v) + (1 - EMA_ALPHA) * running
    assert abs(running - pandas_ema) < 1e-9, f"incremental EMA mismatch: {running} vs {pandas_ema}"
    print("  incremental EMA20 matches pandas ewm(adjust=False): OK")

    # Crossover detection self-check (must match the offline backtest's
    # "old rule": armed sticky, EMA crosses VWAP, same-bar price confirm)
    class _T:
        def __init__(s):
            s.ce_armed = s.pe_armed = False
            s.prev_price = s.prev_ema = s.prev_vwap = None

        def step(s, price, ema, vwap):
            fired = None
            if s.prev_price is not None:
                if s.prev_price < s.prev_vwap and s.prev_ema < s.prev_vwap:
                    s.ce_armed = True
                if s.prev_price > s.prev_vwap and s.prev_ema > s.prev_vwap:
                    s.pe_armed = True
                if s.ce_armed and s.prev_ema < s.prev_vwap and ema >= vwap and price > ema:
                    fired = "CE"; s.ce_armed = s.pe_armed = False
                elif s.pe_armed and s.prev_ema > s.prev_vwap and ema <= vwap and price < ema:
                    fired = "PE"; s.ce_armed = s.pe_armed = False
            s.prev_price, s.prev_ema, s.prev_vwap = price, ema, vwap
            return fired

        def steps(s, triples):
            out = []
            for p, e, v in triples:
                r = s.step(p, e, v)
                if r:
                    out.append(r)
            return out

    t = _T()
    # below VWAP, EMA below VWAP, then EMA catches up and crosses, price confirms above EMA
    seq = [(99, 98, 100), (99.5, 98.5, 100), (101.5, 100.2, 100.0), (102, 101, 100.1)]
    fires = t.steps(seq)
    assert fires == ["CE"], f"expected a single CE fire, got {fires}"
    print("  crossover state-machine (old rule, matches offline backtest): OK")

    # Staged-trail table self-check: verify the widening steps directly.
    assert trail_distance_for_mfe(9.9) == 2.0   # below activation zone, base distance
    assert trail_distance_for_mfe(10.0) == 2.0
    assert trail_distance_for_mfe(11.9) == 2.0
    assert trail_distance_for_mfe(12.0) == 2.5
    assert trail_distance_for_mfe(13.0) == 3.0
    assert trail_distance_for_mfe(14.0) == 3.5
    assert trail_distance_for_mfe(15.0) == 4.0
    assert trail_distance_for_mfe(50.0) == 4.0  # caps at 4.0, does not keep widening
    print("  staged trail-distance table (2/2.5/3/3.5/4pt @ mfe 10/12/13/14/15, caps at 4pt): OK")

    # Trail mechanics self-check, using the STAGED (monotonic-widening) trail
    def simulate_trail(prices, entry_price):
        stop = entry_price - INITIAL_STOP_PTS
        peak = entry_price
        trailing_active = False
        for p in prices:
            if p > peak:
                peak = p
            mfe = peak - entry_price
            if not trailing_active and mfe >= TRAIL_ACTIVATE_PTS:
                trailing_active = True
            if trailing_active:
                stop = max(stop, peak - trail_distance_for_mfe(mfe))
            if p <= stop:
                return stop, "TRAIL" if trailing_active else "HARD_STOP"
        return None, "STILL_OPEN"

    exit_price, reason = simulate_trail([100, 105, 110, 112, 115, 113, 111], 100.0)
    # peak climbs 110->112->115 (mfe 10->12->15), trail widens 2.0->2.5->4.0
    # alongside it (stop 108->109.5->111.0), then price falls back to 111
    # which hits the now-111.0 stop -> exit TRAIL at 111.0, not the tighter
    # flat-2pt stop of 113 that the OLD rule would have hit two bars earlier.
    assert reason == "TRAIL" and abs(exit_price - 111.0) < 1e-9, f"unexpected trail result: {exit_price}, {reason}"
    print("  staged trail activation/widening/ratchet mechanics: OK (exit at 111.0, not the old flat-2pt 113.0)")

    # Max-trades/day self-check (ADDED 2026-09-25 per code-review finding:
    # trades_today was tracked/logged but nothing actually enforced the
    # documented 5/day cap). Drives a real VwapEmaTrail2Act10Engine against
    # a throwaway tempdir DB/control folder, forces trades_today to the cap,
    # then feeds it the exact same crossover bar sequence proven to fire a
    # signal above -- asserts NO pending_signal is set once at the cap, and
    # that it works normally below the cap.
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _tmpdir:
        _db = os.path.join(_tmpdir, "selfcheck.db")
        _ctrl = _tmpdir
        conn = sqlite3.connect(_db)
        conn.execute('''CREATE TABLE IF NOT EXISTS paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, variant TEXT, strategy TEXT,
            direction TEXT, side TEXT, entry_time TEXT, exit_time TEXT, entry_price REAL,
            exit_price REAL, exit_reason TEXT, pnl_pts REAL, rupee_pnl REAL, lot_size INTEGER,
            capital_after REAL, status TEXT, initial_stop REAL, current_stop REAL,
            peak_price REAL, last_price REAL, target_note TEXT, reason TEXT, strike REAL,
            trailing_active INTEGER, expiry TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, variant TEXT, event_message TEXT)''')
        conn.commit()
        conn.close()

        bar_time = pd.Timestamp.now(tz=IST).normalize() + pd.Timedelta(hours=10)
        # Directly prime the pre-fire state so the ONE bar fed below is
        # guaranteed to satisfy the real CE crossover condition through the
        # actual incremental EMA/VWAP formulas (not hand-fed ema/vwap
        # values like the synthetic crossover self-check above uses) --
        # solved so that after this bar: ema_now=100.2, vwap_now=100.0,
        # price=101.5, matching the exact fire point already proven by the
        # crossover state-machine self-check above (prev_price=99.5 <
        # prev_vwap=100, prev_ema=98.5 < prev_vwap=100 -> ce_armed; then
        # ema_now(100.2) >= vwap_now(100.0) and price(101.5) > ema_now
        # (100.2) -> fires).
        high, low, close = 101.5, 97.0, 101.5  # typical=(101.5+97.0+101.5)/3=100.0 -> vwap_now=100.0 (count 0->1)
        prior_ema_needed = (100.2 - EMA_ALPHA * close) / (1 - EMA_ALPHA)  # solves for ema_now == 100.2

        def _primed_engine(trades_today):
            # each gets its OWN control subdir -- otherwise the first
            # engine's saved pending_signal JSON would leak into the
            # second via _load_pending_signal() in __init__ (same label,
            # same day, same folder), which would invalidate this test
            # regardless of whether the max-trades cap actually works.
            own_ctrl = os.path.join(_ctrl, f"engine_{trades_today}")
            os.makedirs(own_ctrl, exist_ok=True)
            e = VwapEmaTrail2Act10Engine(_db, own_ctrl)
            e.today = datetime.now(IST).date()  # skip maybe_reset_for_new_day's DB restore path
            e.trades_today = trades_today
            e.ce_armed = True
            e.prev_price, e.prev_ema, e.prev_vwap = 99.5, 98.5, 100.0
            e.ema20 = prior_ema_needed
            e.vwap_sum, e.vwap_count = 0.0, 0
            return e

        # Below the cap (4 of 5 used): the primed firing bar must set a
        # pending_signal, confirming the harness itself is valid.
        eng = _primed_engine(MAX_TRADES_PER_DAY - 1)
        eng._process_new_bar(bar_time, high, low, close)
        assert eng.pending_signal is not None, "test harness error: expected a signal below the max-trades cap"
        assert abs(eng.prev_ema - 100.2) < 1e-9 and abs(eng.prev_vwap - 100.0) < 1e-9, "test harness math error"
        print(f"  max-trades/day cap: signal fires normally at {MAX_TRADES_PER_DAY - 1}/{MAX_TRADES_PER_DAY} trades: OK")

        # At the cap (5 of 5 used): the IDENTICAL primed firing bar must NOT
        # set a pending_signal -- this is the exact gap the code review found.
        eng2 = _primed_engine(MAX_TRADES_PER_DAY)
        eng2._process_new_bar(bar_time, high, low, close)
        assert eng2.pending_signal is None, "trade #6 should have been BLOCKED at the max-trades cap but was not"
        print(f"  max-trades/day cap: signal correctly BLOCKED at {MAX_TRADES_PER_DAY}/{MAX_TRADES_PER_DAY} trades (trade #6 rejected): OK")

        # Kill switch: same proven firing bar must NOT queue a signal while the flag exists
        eng3 = _primed_engine(0)
        open(eng3.kill_switch_path(), "w").close()
        assert eng3.check_manual_exit_requested(), "kill switch must also request exit of an open position"
        eng3._process_new_bar(bar_time, high, low, close)
        assert eng3.pending_signal is None, "kill switch must block new signals"
        os.remove(eng3.kill_switch_path())
        print("  kill switch (blocks new signals, requests exit): OK")

    print("This engine's DB is separate:", DB_FILENAME)
    print("This engine's control folder is separate:", CONTROL_SUBDIR)
    print("All self-checks passed.\n")


class VwapEmaTrail2Act10Engine:
    """Single-position (cap=1) VWAP+20EMA 'old rule' crossover entry engine
    with a fixed 20pt initial stop and a 2pt trail that activates once
    profit reaches +10pt. Owns its own capital/EMA/VWAP state and DB."""

    def __init__(self, db_path, control_dir):
        self.db_path = db_path
        self.control_dir = control_dir
        self.label = "vwap20ema_trail2act10"

        self.capital = STARTING_CAPITAL
        self._load_capital_from_db()

        self.today = None
        self.ce_armed = False
        self.pe_armed = False
        self.prev_price = None
        self.prev_ema = None
        self.prev_vwap = None
        self.ema20 = None
        self.vwap_sum = 0.0
        self.vwap_count = 0
        self.day_open_spot = None

        self.pending_signal = None  # {"time":..., "direction":...}
        self.position = None
        self.trades_today = 0
        self._max_trades_logged_today = False

        # PENDING-SIGNAL RESTART-LOSS FIX (2026-09-21): a crossover signal
        # detected in the seconds before a process restart used to be
        # silently dropped -- pending_signal only ever lived in memory,
        # never in the DB, so a restart-replay guard that correctly rebuilt
        # EMA20/VWAP state still lost any signal that hadn't executed yet.
        # Confirmed directly: 2026-09-21 logged "VWAP20EMA signal detected
        # (SHORT... vwap_dist=10.8pt) at 10:07:00 -- will attempt entry next
        # bar", then the next event was a 10:40:24 restart with zero trades
        # recorded that day -- the signal vanished with no entry, no log,
        # nothing. Fix: persist pending_signal to a small JSON file in the
        # control folder the moment it's set, delete it the moment it's
        # cleared (executed or reset for a new day), and reload it here on
        # startup so a restart resumes the wait instead of losing it.
        self._load_pending_signal()

    def _pending_signal_path(self):
        return os.path.join(self.control_dir, f"pending_signal_{self.label}.json")

    def _save_pending_signal(self):
        path = self._pending_signal_path()
        try:
            if self.pending_signal is None:
                if os.path.exists(path):
                    os.remove(path)
                return
            payload = {"time": self.pending_signal["time"].isoformat(),
                       "direction": self.pending_signal["direction"]}
            with open(path, "w") as f:
                json.dump(payload, f)
        except Exception:
            logger.exception("Failed to persist pending_signal to disk")

    def _load_pending_signal(self):
        path = self._pending_signal_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            sig_time = pd.Timestamp(payload["time"])
            # only restore if it's from today -- a stale signal from a
            # previous day (engine left running overnight, crashed, etc.)
            # should not fire into a brand new session
            if sig_time.date() == datetime.now(IST).date():
                self.pending_signal = {"time": sig_time, "direction": payload["direction"]}
                self.log_event(f"Restored pending signal from disk after restart: "
                                f"{payload['direction']} detected at {sig_time} -- "
                                f"will still attempt entry on the next new bar.")
            else:
                os.remove(path)
        except Exception:
            logger.exception("Failed to reload pending_signal from disk")

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
        initial_stop = entry_price - INITIAL_STOP_PTS
        target_note = (f"Initial stop entry-{INITIAL_STOP_PTS:.0f}pt. No fixed target - trail activates at "
                        f"+{TRAIL_ACTIVATE_PTS}pt, then STAGED trail (2/2.5/3/3.5/4pt behind peak, widening at "
                        f"mfe 10/12/13/14/15), else 15:20 EOD square-off. Max {MAX_TRADES_PER_DAY} trades/day.")
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''INSERT INTO paper_orders
            (date, variant, strategy, direction, side, entry_time, entry_price, status, capital_after,
             initial_stop, current_stop, peak_price, last_price, target_note, reason, lot_size,
             strike, trailing_active, expiry)
            VALUES (?, ?, 'vwap20ema_trail2act10', ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)''',
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
            conn.execute('''UPDATE paper_orders SET current_stop=?, peak_price=?, last_price=?,
                trailing_active=? WHERE id=? AND variant=? AND status='OPEN' ''',
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
                    "peak_price": row["peak_price"], "current_stop": row["current_stop"],
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
            self.ce_armed = False
            self.pe_armed = False
            self.prev_price = None
            self.prev_ema = None
            self.prev_vwap = None
            self.ema20 = None
            self.vwap_sum = 0.0
            self.vwap_count = 0
            self.pending_signal = None
            self._save_pending_signal()
            self._max_trades_logged_today = self.trades_today >= MAX_TRADES_PER_DAY
            self.log_event(f"New trading day: {today}. VWAP20EMA 'old rule' crossover entry, "
                            f"{INITIAL_STOP_PTS:.0f}pt SL / activate@+{TRAIL_ACTIVATE_PTS}pt / staged trail "
                            f"(2/2.5/3/3.5/4pt) exit, entry cutoff {ENTRY_CUTOFF}, cap=1, "
                            f"max {MAX_TRADES_PER_DAY} trades/day. {trades_today} trade(s) already closed today "
                            f"(if any, from a restart). "
                            f"{'Resuming an OPEN position.' if self.position else 'No open position.'}")

    def _process_new_bar(self, bar_time, idx_high, idx_low, idx_close, warmup=False):
        """Update EMA20/VWAP incrementally and run the crossover-detection
        state machine for exactly one new 2-min bar.

        warmup=True (fix for the restart-EMA/VWAP-reset issue: confirmed
        directly by comparing a live day's actual trade against an
        independent, continuous-since-9:15 replay -- 2026-09-03 signal was
        10:31 in the true continuous history, but the live engine restarted
        at 09:47 and its cold-reset VWAP/EMA produced a different, earlier
        signal at 10:17 instead, changing entry price, exit price, and P&L.
        Same root cause and same fix already applied to
        step10_live_paper_ema20_50_crossover_cooldown.py.) still updates
        vwap_sum/vwap_count/ema20/prev_price/prev_ema/prev_vwap exactly as
        normal, but SKIPS all armed-state and signal-detection logic. Used
        by run_cycle()'s restart-replay guard to rebuild true,
        continuous-since-day-open VWAP/EMA20 history after a restart,
        without generating any signals from historical bars."""
        typical = (idx_high + idx_low + idx_close) / 3.0
        self.vwap_sum += typical
        self.vwap_count += 1
        vwap_now = self.vwap_sum / self.vwap_count

        if self.ema20 is None:
            self.ema20 = idx_close
        else:
            self.ema20 = EMA_ALPHA * idx_close + (1 - EMA_ALPHA) * self.ema20
        ema_now = self.ema20

        if warmup:
            self.prev_price, self.prev_ema, self.prev_vwap = idx_close, ema_now, vwap_now
            return

        if self.position is not None or self.pending_signal is not None:
            # keep VWAP/EMA state current, but don't hunt for new signals
            # while a trade is open or already pending (matches backtest:
            # cap=1, only one signal chased at a time)
            self.prev_price, self.prev_ema, self.prev_vwap = idx_close, ema_now, vwap_now
            return

        if bar_time.time() >= ENTRY_CUTOFF:
            self.prev_price, self.prev_ema, self.prev_vwap = idx_close, ema_now, vwap_now
            return

        if self.kill_switch_on():
            # kill switch: keep EMA/VWAP state current, but no signal detection
            self.prev_price, self.prev_ema, self.prev_vwap = idx_close, ema_now, vwap_now
            return

        if self.trades_today >= MAX_TRADES_PER_DAY:
            if not self._max_trades_logged_today:
                self._max_trades_logged_today = True
                self.log_event(f"Max trades/day reached ({self.trades_today}/{MAX_TRADES_PER_DAY}) -- "
                                f"no further entries today, existing state otherwise unaffected.")
            self.prev_price, self.prev_ema, self.prev_vwap = idx_close, ema_now, vwap_now
            return

        price = idx_close
        if self.prev_price is not None:
            if self.prev_price < self.prev_vwap and self.prev_ema < self.prev_vwap:
                self.ce_armed = True
            if self.prev_price > self.prev_vwap and self.prev_ema > self.prev_vwap:
                self.pe_armed = True

            vwap_dist = abs(price - vwap_now)

            if self.ce_armed and self.prev_ema < self.prev_vwap and ema_now >= vwap_now and price > ema_now:
                self.ce_armed = self.pe_armed = False
                if vwap_dist <= VWAP_DIST_MAX_FILTER:
                    self.pending_signal = {"time": bar_time, "direction": "LONG"}
                    self._save_pending_signal()
                    self.log_event(f"VWAP20EMA signal detected (LONG, EMA20 crossed VWAP, price confirmed, "
                                    f"vwap_dist={vwap_dist:.1f}pt) at {bar_time} -- will attempt entry next bar.")
                else:
                    self.log_event(f"VWAP20EMA crossover (LONG) SKIPPED -- vwap_dist={vwap_dist:.1f}pt exceeds "
                                    f"filter of {VWAP_DIST_MAX_FILTER}pt (signal too far extended from VWAP, "
                                    f"validated to underperform) at {bar_time}.")
            elif self.pe_armed and self.prev_ema > self.prev_vwap and ema_now <= vwap_now and price < ema_now:
                self.ce_armed = self.pe_armed = False
                if vwap_dist <= VWAP_DIST_MAX_FILTER:
                    self.pending_signal = {"time": bar_time, "direction": "SHORT"}
                    self._save_pending_signal()
                    self.log_event(f"VWAP20EMA signal detected (SHORT, EMA20 crossed VWAP, price confirmed, "
                                    f"vwap_dist={vwap_dist:.1f}pt) at {bar_time} -- will attempt entry next bar.")
                else:
                    self.log_event(f"VWAP20EMA crossover (SHORT) SKIPPED -- vwap_dist={vwap_dist:.1f}pt exceeds "
                                    f"filter of {VWAP_DIST_MAX_FILTER}pt (signal too far extended from VWAP, "
                                    f"validated to underperform) at {bar_time}.")

        self.prev_price, self.prev_ema, self.prev_vwap = price, ema_now, vwap_now

    def try_execute_pending(self, bar_time, chain, expiry):
        if self.pending_signal is not None and self.kill_switch_on():
            self.log_event(f"KILL SWITCH ON -- dropped pending {self.pending_signal['direction']} signal.")
            self.pending_signal = None
            self._save_pending_signal()
            return
        if self.pending_signal is None or self.position is not None:
            return
        if bar_time <= self.pending_signal["time"]:
            return
        direction = self.pending_signal["direction"]
        side = "CE" if direction == "LONG" else "PE"
        strike = om.itm_call_strike(self.day_open_spot) if side == "CE" else om.itm_put_strike(self.day_open_spot)
        ltp = get_ltp_from_chain(chain, strike, side)
        if ltp is None:
            self.log_event(f"WARNING -- could not get a real LTP for strike {strike} {side} this cycle. "
                            f"Will retry next cycle -- pending signal NOT dropped.")
            return
        entry_time = bar_time
        lot = om.lot_size_for_date(entry_time.date())
        reason = (f"VWAP20EMA old-rule crossover ({direction}), {INITIAL_STOP_PTS:.0f}pt SL / "
                  f"activate@+{TRAIL_ACTIVATE_PTS}pt / staged trail (2/2.5/3/3.5/4pt @ 10/12/13/14/15) exit, "
                  f"REAL premium: trade #{self.trades_today + 1} of max {MAX_TRADES_PER_DAY} today.")
        row_id = self._record_open(direction, side, entry_time, ltp, reason, lot, strike, expiry)
        self.position = {
            "row_id": row_id, "direction": direction, "side": side, "strike": strike, "expiry": expiry,
            "entry_time": entry_time, "entry_price": ltp, "lot": lot,
            "peak_price": ltp, "current_stop": ltp - INITIAL_STOP_PTS, "trailing_active": False,
        }
        self.pending_signal = None
        self._save_pending_signal()
        self.log_event(f"OPENED {side} {direction} @ {ltp:.2f} (real LTP, strike {strike}, lot {lot}) at "
                        f"{entry_time} [trade #{self.trades_today + 1} today]")

    def step_position_live(self, chain):
        pos = self.position
        if pos is None:
            return
        ltp = get_ltp_from_chain(chain, pos["strike"], pos["side"])
        if ltp is None:
            self.log_event(f"WARNING -- could not get a real LTP for open position's strike {pos['strike']} "
                            f"{pos['side']} this cycle. Skipping management check.")
            return
        now_ist = datetime.now(IST)

        if ltp > pos["peak_price"]:
            pos["peak_price"] = ltp
        mfe = pos["peak_price"] - pos["entry_price"]
        if not pos["trailing_active"] and mfe >= TRAIL_ACTIVATE_PTS:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["current_stop"] = max(pos["current_stop"], pos["peak_price"] - trail_distance_for_mfe(mfe))

        self._update_live_state(pos["row_id"], pos["current_stop"], pos["peak_price"], ltp, pos["trailing_active"])

        exit_reason = None
        if ltp <= pos["current_stop"]:
            exit_reason = "TRAIL" if pos["trailing_active"] else "HARD_STOP"
        elif now_ist.time() >= om.EOD_SQUAREOFF:
            exit_reason = "EOD"
        if exit_reason is not None:
            self._close_position(ltp, exit_reason, now_ist)

    def _close_position(self, exit_price, reason, exit_time):
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

    def needs_chain_this_cycle(self, new_bars, latest_bar_time):
        pending_ready = (self.pending_signal is not None and self.position is None
                          and new_bars and latest_bar_time > self.pending_signal["time"])
        return pending_ready or self.position is not None

    def manual_exit_flag_path(self):
        return os.path.join(self.control_dir, f"MANUAL_EXIT_{self.label}.flag")

    def check_manual_exit_requested(self):
        all_path = os.path.join(self.control_dir, "MANUAL_EXIT_ALL.flag")
        # the kill switch also counts as an exit request, so an open position is closed
        return (os.path.exists(self.manual_exit_flag_path()) or os.path.exists(all_path)
                or self.kill_switch_on())

    # KILL SWITCH (added 2026-10-16, same design as the progressive-3pt cap-1
    # engine): KILL_SWITCH.flag in this engine's control folder (written/removed
    # by the dashboard) = close any open position next cycle AND take no new
    # entries until the file is removed. EMA20/VWAP keep updating meanwhile, so
    # nothing is stale when it's turned off. Paper trading only.
    def kill_switch_path(self):
        return os.path.join(self.control_dir, "KILL_SWITCH.flag")

    def kill_switch_on(self):
        return os.path.exists(self.kill_switch_path())

    def note_kill_switch_state(self):
        state = self.kill_switch_on()
        if state != getattr(self, "_kill_state", False):
            self._kill_state = state
            self.log_event("KILL SWITCH ON -- closing any open position, no new entries until it is turned off."
                            if state else "KILL SWITCH OFF -- entries allowed again (subject to cutoff and daily cap).")

    def clear_manual_exit_flag(self):
        path = self.manual_exit_flag_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                logger.exception("Failed to remove manual-exit flag file")

    def manual_close_if_requested(self, chain):
        if self.position is None or not self.check_manual_exit_requested():
            return False
        if chain is None:
            self.log_event("MANUAL EXIT requested but no chain data this cycle -- will retry next cycle.")
            return True
        ltp = get_ltp_from_chain(chain, self.position["strike"], self.position["side"])
        if ltp is None:
            self.log_event("MANUAL EXIT: could not get real LTP this cycle -- retrying next cycle.")
            return True
        reason = "KILL_SWITCH" if self.kill_switch_on() else "MANUAL"
        self.log_event(f"{reason} -- closing position now at real LTP.")
        self._close_position(ltp, reason, datetime.now(IST))
        self.clear_manual_exit_flag()
        return True


class Runner:
    def __init__(self, config: PlatformConfig = None, security_id: str = "13", exchange_segment: str = "IDX_I"):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.security_id = security_id
        self.exchange_segment = exchange_segment

        self.db_path = os.path.join(self.config.log_dir, DB_FILENAME)
        self._init_db()

        self.control_dir = os.path.join(self.config.log_dir, CONTROL_SUBDIR)
        os.makedirs(self.control_dir, exist_ok=True)

        self.engine = VwapEmaTrail2Act10Engine(self.db_path, self.control_dir)

        self.last_seen_bar_time = None
        self._last_new_bar_wall_time = datetime.now(IST)
        self._stall_warned_at = None

        self.current_expiry = None
        self._expiry_fetched_date = None
        self._chain_cooldown_until = None
        self._consecutive_fail_count = 0

        self.engine.log_event(f"[VWAP20EMA/TRAIL2-ACT10] live paper engine initialized (entry=old-rule EMA20xVWAP "
                               f"crossover, exit={INITIAL_STOP_PTS:.0f}pt SL / activate@+{TRAIL_ACTIVATE_PTS}pt / "
                               f"staged trail (2/2.5/3/3.5/4pt), "
                               f"cutoff={ENTRY_CUTOFF}, cap=1, max {MAX_TRADES_PER_DAY} trades/day, "
                               f"capital Rs {STARTING_CAPITAL:,.0f}). Real Dhan "
                               f"option-chain LTP. Own DB, own control folder -- fully separate from every "
                               f"other engine in this project. PAPER TRADING ONLY.")

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
        self.engine.maybe_reset_for_new_day(today)
        self.engine.note_kill_switch_state()

        today_df = self._fetch_and_resample_index()
        if today_df.empty:
            logger.info("No live index bar data this cycle. Skipped.")
            return

        if self.engine.day_open_spot is None:
            self.engine.day_open_spot = float(today_df["open"].iloc[0])

        latest_bar_time = today_df.index[-1]
        is_first_cycle_after_start = self.last_seen_bar_time is None
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

        need_chain = self.engine.needs_chain_this_cycle(new_bars, latest_bar_time) or self.engine.position is not None
        chain = self._fetch_live_chain(today) if need_chain else None
        expiry = self.current_expiry

        if new_bars:
            if is_first_cycle_after_start:
                # RESTART-REPLAY GUARD, VWAP/EMA-WARMUP FIX (same fix as
                # already applied to step10_live_paper_ema20_50_crossover_
                # cooldown.py, same root cause): replay every bar already
                # seen today through _process_new_bar(..., warmup=True) --
                # rebuilds true, continuous-since-day-open VWAP/EMA20
                # values but generates NO signals and touches no
                # pending_signal/position state, since warmup=True skips
                # all armed-state/signal-detection logic. Only genuinely
                # new bars AFTER this point are checked for entries, so
                # this still fully avoids the original look-ahead concern
                # the restart-replay guard existed for.
                for bar_time, row in today_df.iterrows():
                    self.engine._process_new_bar(bar_time, float(row["high"]), float(row["low"]), float(row["close"]), warmup=True)
                self.last_seen_bar_time = latest_bar_time
                self.engine.log_event(f"Restart-replay guard: replayed {len(today_df)} historical bar(s) in "
                                       f"WARMUP mode (bars up to {latest_bar_time}) -- VWAP/EMA20 now reflect "
                                       f"the continuous day-open history (ema20={self.engine.ema20:.2f}, "
                                       f"vwap={(self.engine.vwap_sum/self.engine.vwap_count):.2f}), matching a "
                                       "never-restarted engine. No signals were generated from this replay. "
                                       "Only genuinely new bars from here on are checked for entries.")
            else:
                new_bar_times = today_df.index[
                    today_df.index > (self.last_seen_bar_time or today_df.index[0] - timedelta(seconds=1))]
                for bar_time in new_bar_times:
                    row = today_df.loc[bar_time]
                    self.engine._process_new_bar(bar_time, float(row["high"]), float(row["low"]), float(row["close"]))
                    self.engine.try_execute_pending(bar_time, chain, expiry)
                self.last_seen_bar_time = latest_bar_time

        if self.engine.position is not None:
            if self.engine.manual_close_if_requested(chain):
                pass
            else:
                self.engine.step_position_live(chain)

    def run_forever(self):
        self.engine.log_event(f"Live loop starting. Poll interval: {POLL_SECONDS}s. Capital: Rs {self.engine.capital:,.2f}")
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
            self.engine.log_event("[VWAP20EMA/TRAIL2-ACT10] live paper engine stopped manually (Ctrl+C).")


if __name__ == "__main__":
    run_self_checks()
    runner = Runner()
    runner.run_forever()
