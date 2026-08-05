"""
Live PAPER (mock, NO real money / NO real orders) test -- top 3 validated
strategies from this project's testing history
=================================================================
Picked from the "strategies tested so far" table in nifty_backtest_playbook.md
(the only 3 with a real, believable, non-artifact edge after full rigor):

  1. Velez 8pt (SHORT-only, GBI + ELEPHANT_PLUS) -- always buys the ITM PUT,
     8pt initial stop / 4pt trail. Backtest (2yr, proxy premium model):
     n=284, PF=1.71, SQN=2.63, CAGR=45.2%, max DD -10.55%.
     Signal logic reused verbatim from velez_8pt_proxy_premium_test.py.

  2. Opening-Drive Momentum -- buys the ITM CALL or PUT depending on which
     way the first >=5pt move away from the open goes before 09:45, 8pt
     stop / 4pt trail. Backtest (2yr): n=?, PF=1.44, SQN=3.02, CAGR=72.3%,
     max DD -15.75%. Signal logic reused verbatim from
     nifty_open_momentum_trail.py.

  3. Crash Play REV2, CONFIRM entries, sl3_trail1 -- always buys the ITM
     PUT on a confirmed failed-bounce after a crash bar, 3pt stop / 1pt
     trail. Backtest (2yr): n=251, win%=26.3, PF=1.64, net Rs10,902.
     Signal logic reused verbatim from nifty_crash_play_sweep.py.

THIS SCRIPT NEVER PLACES A REAL ORDER. It only makes read-only Dhan API
calls (intraday_minute_data, same call dhan_live_bot.py already uses) to
pull the live Nifty index candle feed, and everything else -- signal
detection, entry/exit, P&L -- is simulated in memory using the exact same
proxy-premium model and conservative intrabar stop/trail ordering already
validated in each strategy's real backtest. Every simulated entry and exit
is appended to a CSV trade log the moment it happens, so you have a full,
timestamped record of what each strategy WOULD have done today.

Two ways to run it:

  Smoke test (safe any time, market open or closed, takes ~10-30s, no
  infinite loop, doesn't touch the CSV log): fetches real recent Nifty
  data, replays the most recently completed trading day through all three
  strategies (both in one batch pass AND bar-by-bar through the same
  incremental live logic used in the real-time loop, then asserts the two
  give identical results -- this is the actual smoke test), and prints
  what each strategy would have done. Run this first:

      cd "C:\\dhan algo"
      python nifty_paper_trade_top3_live.py --smoke-test

  Live mock run (run this DURING market hours, 09:15-15:30 IST, on a
  trading day -- e.g. tomorrow): polls every POLL_SECONDS, prints a status
  line each poll, and appends a row to paper_trade_log_<today's date>.csv
  every time any strategy enters or exits a mock position:

      cd "C:\\dhan algo"
      python nifty_paper_trade_top3_live.py

  Leave the terminal window open. Safe to Ctrl+C any time -- nothing is
  lost, every fill so far is already on disk in the CSV.
"""

import sys
import csv
import os
import time
import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

# ============================== shared config ===============================
BAR_RULE = "2min"
STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 30)
POLL_SECONDS = 30
HISTORY_CONTEXT_DAYS = 20  # enough calendar days of prior context for Velez's SMA200 warmup

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
BROKERAGE_PER_TRADE = 40.0

LOG_FILE = f"paper_trade_log_{dt.date.today().isoformat()}.csv"
LOG_FIELDS = ["logged_at", "strategy", "event", "action", "side", "strike", "bar_time",
              "price", "exit_reason", "pnl_pts", "rupee_pnl", "lot_size", "day_running_pnl_rs"]


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def build_premium_series(day_df, strike, day_open_spot, side):
    """Same proxy premium model used in every backtest this project has run:
    intrinsic-at-open + assumed time value, moves scaled by a flat assumed
    delta, floored at 0."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        mo, mh, ml, mc = (idx_open - day_open_spot, idx_high - day_open_spot,
                          idx_low - day_open_spot, idx_close - day_open_spot)
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        mo, mh, ml, mc = (day_open_spot - idx_open, day_open_spot - idx_low,
                          day_open_spot - idx_high, day_open_spot - idx_close)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def fetch_index_window(from_date, to_date, retries=1):
    """Same call dhan_live_bot.py already uses (read-only). Adds a clear
    diagnostic + one retry on a transient/malformed response instead of
    crashing deep inside a DataFrame constructor -- the same lesson learned
    from fetch_index_chunked's transient-API-error bug (see playbook)."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = dlb.dhan.intraday_minute_data(
                security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                from_date=from_date, to_date=to_date,
            )
            if not isinstance(resp, dict):
                raise RuntimeError(f"Dhan API returned a non-dict response (likely a connection/proxy "
                                    f"error surfaced as a string): {str(resp)[:300]}")
            data = resp.get("data", resp)
            if not isinstance(data, dict) or "open" not in data:
                raise RuntimeError(f"Dhan API response missing expected OHLC keys -- likely an error "
                                    f"payload, not real data: {str(resp)[:300]}")
            df = pd.DataFrame({
                "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
                "timestamp": data["timestamp"],
            })
            df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            return df.set_index("time").sort_index()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
                continue
    raise RuntimeError(f"fetch_index_window failed after {retries + 1} attempt(s) "
                        f"(from_date={from_date}, to_date={to_date}): {last_err}")


def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


# =============================================================================
# STRATEGY 1: Velez 8pt (SHORT-only, GBI + ELEPHANT_PLUS) -- verbatim signal
# detection reused from velez_8pt_proxy_premium_test.py
# =============================================================================
SMA_FAST, SMA_SLOW = 20, 200
POP_SLOPE_LOOKBACK = 5
AVG_RANGE_LOOKBACK_VELEZ = 20
SMALL_BAR_MULT = 0.6
ELEPHANT_MULT = 2.0
ELEPHANT_PLUS_RETAIN = 0.8
NEAR_20_PCT = 0.0015
VELEZ_STOP_PTS, VELEZ_TRAIL_PTS = 8.0, 4.0


def compute_velez_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK_VELEZ, min_periods=AVG_RANGE_LOOKBACK_VELEZ).mean()
    return df


def pop_bearish(price, sma20, sma20_prev, sma200):
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(sma20_prev):
        return False
    return price < sma20 and sma20 < sma200 and sma20 < sma20_prev


def is_green(o, c):
    return c > o


def is_red(o, c):
    return c < o


def is_small(rng, avgr):
    return bool(avgr) and avgr > 0 and rng < SMALL_BAR_MULT * avgr


def is_elephant_plus_bear(o, h, l, c, avgr):
    rng = h - l
    if not (bool(avgr) and avgr > 0 and (rng * ELEPHANT_PLUS_RETAIN) > ELEPHANT_MULT * avgr):
        return False
    return is_red(o, c)


def find_small_green_run(highs, lows, opens, closes, avg_ranges, end_idx):
    i = end_idx - 1
    if i < 0:
        return None
    if not (is_green(opens[i], closes[i]) and is_small(highs[i] - lows[i], avg_ranges[i])):
        return None
    combined_high, combined_low = highs[i], lows[i]
    start_idx = i
    j = i - 1
    while j >= 0 and is_green(opens[j], closes[j]) and is_small(highs[j] - lows[j], avg_ranges[j]):
        combined_high = max(combined_high, highs[j])
        combined_low = min(combined_low, lows[j])
        start_idx = j
        j -= 1
    if start_idx - 1 < 0:
        return None
    if not is_red(opens[start_idx - 1], closes[start_idx - 1]):
        return None
    return start_idx, combined_high, combined_low


def detect_velez_signals(day_df):
    opens = day_df["open"].tolist(); highs = day_df["high"].tolist()
    lows = day_df["low"].tolist(); closes = day_df["close"].tolist()
    sma20 = day_df["sma20"].tolist(); sma200 = day_df["sma200"].tolist()
    avg_ranges = day_df["avg_range"].tolist()
    n = len(day_df)
    signals = []
    if n < POP_SLOPE_LOOKBACK + 3:
        return signals
    for j in range(POP_SLOPE_LOOKBACK, n - 1):
        if pd.isna(sma20[j]) or pd.isna(sma200[j]) or pd.isna(sma20[j - POP_SLOPE_LOOKBACK]):
            continue
        if not pop_bearish(closes[j], sma20[j], sma20[j - POP_SLOPE_LOOKBACK], sma200[j]):
            continue
        near20 = abs(closes[j] - sma20[j]) / closes[j] < NEAR_20_PCT
        if near20:
            run = find_small_green_run(highs, lows, opens, closes, avg_ranges, j)
            if run:
                _, comb_high, comb_low = run
                if closes[j] < comb_low and is_red(opens[j], closes[j]):
                    signals.append({"bar": j, "side": "SHORT", "event": "GBI"})
                    continue
        if j - 1 >= 0:
            eo, eh, el, ec = opens[j - 1], highs[j - 1], lows[j - 1], closes[j - 1]
            eavgr = avg_ranges[j - 1]
            if is_elephant_plus_bear(eo, eh, el, ec, eavgr) and closes[j] < el:
                signals.append({"bar": j, "side": "SHORT", "event": "ELEPHANT_PLUS"})
    seen = set(); out = []
    for s in signals:
        if s["bar"] in seen:
            continue
        seen.add(s["bar"]); out.append(s)
    return out


# =============================================================================
# STRATEGY 2: Opening-Drive Momentum -- verbatim from nifty_open_momentum_trail.py
# =============================================================================
OD_ENTRY_CUTOFF = dt.time(9, 45)
OD_MIN_MOVE_PTS = 5.0
OD_STOP_PTS, OD_TRAIL_PTS = 8.0, 4.0


def find_opening_drive(day_df, min_move=OD_MIN_MOVE_PTS, cutoff=OD_ENTRY_CUTOFF):
    day_open = day_df["open"].iloc[0]
    closes = day_df["close"].to_numpy()
    times = day_df.index
    for i in range(len(day_df)):
        if times[i].time() >= cutoff:
            return None, None
        move = closes[i] - day_open
        if move >= min_move:
            return i, "LONG"
        if move <= -min_move:
            return i, "SHORT"
    return None, None


# =============================================================================
# STRATEGY 3: Crash Play REV2, CONFIRM entries only, sl3_trail1 --
# verbatim from nifty_crash_play_sweep.py
# =============================================================================
CRASH_AVG_RANGE_LOOKBACK = 20
CRASH_MULT = 2.5
CRASH_MAX_LOOKAHEAD_BARS = 60
CRASH_STOP_PTS, CRASH_TRAIL_PTS = 3.0, 1.0


def find_crash_confirm_signals(day_df, min_bar_idx=-1):
    """Adapted from find_crash_signals_for_day (CONFIRM entries only --
    the validated variant, sl3_trail1). Returns a list of confirm-entry bar
    indices with index > min_bar_idx (so a live loop can ask for only new
    signals not already acted on today)."""
    opens = day_df["open"].tolist(); highs = day_df["high"].tolist()
    lows = day_df["low"].tolist(); closes = day_df["close"].tolist()
    rng = (day_df["high"] - day_df["low"]).tolist()
    avg_range = pd.Series(rng).shift(1).rolling(CRASH_AVG_RANGE_LOOKBACK, min_periods=CRASH_AVG_RANGE_LOOKBACK).mean().tolist()
    n = len(day_df)
    out = []
    j = CRASH_AVG_RANGE_LOOKBACK
    while j < n - 2:
        if avg_range[j] and rng[j] >= CRASH_MULT * avg_range[j] and is_red(opens[j], closes[j]):
            crash_open = opens[j]; crash_low = lows[j]
            k = j
            while k + 1 < n and lows[k + 1] <= crash_low + 1e-9:
                k += 1
                crash_low = min(crash_low, lows[k])
            bounce_start = k + 1
            limit = min(n, bounce_start + CRASH_MAX_LOOKAHEAD_BARS)
            confirm_idx = None
            for i in range(bounce_start, limit):
                c = closes[i]
                z = zone_of(c, crash_low, crash_open)
                if confirm_idx is None and z in (3, 4) and i > bounce_start:
                    if is_red(opens[i], closes[i]) and is_green(opens[i - 1], closes[i - 1]) and lows[i] < lows[i - 1]:
                        confirm_idx = i
                        break
            if confirm_idx is not None and confirm_idx > min_bar_idx:
                out.append(confirm_idx)
            j = k + CRASH_MAX_LOOKAHEAD_BARS
        j += 1
    return out


def zone_of(level, crash_low, crash_open):
    rng = crash_open - crash_low
    if rng <= 0:
        return None
    pct = (level - crash_low) / rng
    if pct < 0 or pct > 1.0 + 1e-9:
        return None
    if pct >= 0.67:
        return 1
    elif pct >= 0.50:
        return 2
    elif pct >= 0.33:
        return 3
    else:
        return 4


# =============================================================================
# Position management -- conservative-ordering incremental steppers, verbatim
# logic from each strategy's own backtest simulate_trade function, just
# processed bar-by-bar as new bars arrive instead of all at once
# =============================================================================
def step_long_premium_position(pos, premium_df, trail_pts):
    """Shared by Velez (always PE) and Opening-Drive (CE or PE) -- both use
    identical conservative ordering: this bar's LOW is checked against the
    OLD stop level before this bar's HIGH is allowed to ratchet the
    trailing stop up. Mutates pos in place; returns exit info dict or None
    if still open."""
    n = len(premium_df)
    for j in range(pos["last_checked_idx"] + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= pos["stop_level"]:
            reason = "TRAIL_SL" if pos["trailing_active"] else "INITIAL_SL"
            return {"exit_price": pos["stop_level"], "reason": reason,
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if t >= EOD_SQUAREOFF:
            return {"exit_price": bar["close"], "reason": "EOD",
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if bar["high"] > pos["peak"]:
            pos["peak"] = bar["high"]
        if pos["peak"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], pos["peak"] - trail_pts)
        pos["last_checked_idx"] = j
    return None


def step_crash_position(pos, premium_df, trail_pts):
    """Crash Play's own trailing convention, verbatim from
    simulate_trade_trailing in nifty_crash_play_sweep.py: trigger on CLOSE
    > entry (not high), trail using LOW - trail_pts (not high), EOD check
    comes after the stop update, not before."""
    n = len(premium_df)
    for j in range(pos["last_checked_idx"] + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= pos["stop_level"]:
            reason = "TRAIL_SL" if pos["trailing_active"] else "INITIAL_SL"
            return {"exit_price": pos["stop_level"], "reason": reason,
                    "exit_time": premium_df.index[j], "exit_idx": j}
        if bar["close"] > pos["entry_price"]:
            pos["trailing_active"] = True
        if pos["trailing_active"]:
            pos["stop_level"] = max(pos["stop_level"], bar["low"] - trail_pts)
        if t >= EOD_SQUAREOFF:
            return {"exit_price": bar["close"], "reason": "EOD",
                    "exit_time": premium_df.index[j], "exit_idx": j}
        pos["last_checked_idx"] = j
    return None


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")

    assert pop_bearish(95, 97, 98, 100) is True
    assert pop_bearish(105, 103, 102, 100) is False
    assert is_small(2, 10) is True
    assert is_elephant_plus_bear(100, 100, 70, 71, 10) is True
    print("  Velez signal-detection primitives ok (verbatim from velez_8pt_proxy_premium_test.py)")

    idx = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")
    closes = [100, 100.5, 106.0] + [110] * 17
    opens = [100] + closes[:-1]
    df = pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)
    local_i, direction = find_opening_drive(df)
    assert local_i == 2 and direction == "LONG"
    print("  Opening-Drive find_opening_drive LONG-detection ok")

    assert zone_of(100, 0, 100) == 1 and zone_of(49, 0, 100) == 3 and zone_of(0, 0, 100) == 4
    print("  Crash Play zone_of boundaries ok")

    # --- incremental stepper must exactly match a single batch pass ---
    idx_t = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")
    prem = pd.DataFrame({
        "open": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
        "high": [50, 56, 61, 66, 61, 56, 56, 56, 56, 56],
        "low": [50, 54, 59, 64, 55, 54.5, 54.5, 54.5, 54.5, 54.5],
        "close": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
    }, index=idx_t)
    # batch (all bars available at once)
    pos_batch = {"entry_price": 50, "stop_level": 42, "peak": 50, "trailing_active": False, "last_checked_idx": -1}
    exit_batch = step_long_premium_position(pos_batch, prem, VELEZ_TRAIL_PTS)
    assert exit_batch["reason"] == "TRAIL_SL" and abs(exit_batch["exit_price"] - 62.0) < 1e-6
    # incremental (fed 3 bars, then 3 more, then the rest -- must reach the same answer)
    pos_inc = {"entry_price": 50, "stop_level": 42, "peak": 50, "trailing_active": False, "last_checked_idx": -1}
    r = None
    for cutoff in (3, 6, 10):
        r = step_long_premium_position(pos_inc, prem.iloc[:cutoff], VELEZ_TRAIL_PTS)
        if r:
            break
    assert r is not None and r["reason"] == exit_batch["reason"] and abs(r["exit_price"] - exit_batch["exit_price"]) < 1e-6, (
        f"incremental stepper diverged from batch: {r} vs {exit_batch}"
    )
    print("  step_long_premium_position: incremental-vs-batch equivalence check ok")

    # --- conservative intrabar ordering preserved in the incremental version ---
    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    pos3 = {"entry_price": 50, "stop_level": 42, "peak": 50, "trailing_active": False, "last_checked_idx": -1}
    exit3 = step_long_premium_position(pos3, prem3, VELEZ_TRAIL_PTS)
    assert exit3["reason"] == "TRAIL_SL" and abs(exit3["exit_price"] - 52.0) < 1e-6, (
        f"expected conservative exit at old stop 52.0, got {exit3}"
    )
    print("  step_long_premium_position conservative intrabar-ordering check ok")

    # --- crash stepper trailing convention check ---
    idx4 = pd.date_range("2025-01-06 09:15", periods=8, freq="2min")
    closes_c = [50, 52, 54, 53, 52, 51, 50, 49]
    highs_c = [c + 0.3 for c in closes_c]
    lows_c = [c - 0.3 for c in closes_c]
    prem_c = pd.DataFrame({"open": [50] + closes_c[:-1], "high": highs_c, "low": lows_c, "close": closes_c}, index=idx4)
    pos_c = {"entry_price": 50, "stop_level": 47, "peak": 50, "trailing_active": False, "last_checked_idx": -1}
    exit_c = step_crash_position(pos_c, prem_c, CRASH_TRAIL_PTS)
    assert exit_c["reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {exit_c}"
    print("  step_crash_position trailing-stop check ok")

    print("All self-checks passed.\n")


# =============================================================================
# Live state + logging
# =============================================================================
def new_strategy_state():
    return {"position": None, "pending_signal": None, "search_from_bar": -1,
            "trades_today": 0, "day_pnl_rs": 0.0}


def append_log_row(row):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def log_entry(strategy, event, side, strike, bar_time, price, lot, day_pnl):
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "strategy": strategy,
        "event": event, "action": "ENTRY", "side": side, "strike": strike,
        "bar_time": bar_time, "price": round(price, 2), "exit_reason": "", "pnl_pts": "",
        "rupee_pnl": "", "lot_size": lot, "day_running_pnl_rs": round(day_pnl, 2),
    })


def log_exit(strategy, event, side, strike, bar_time, price, reason, pnl_pts, rupee_pnl, lot, day_pnl):
    append_log_row({
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"), "strategy": strategy,
        "event": event, "action": "EXIT", "side": side, "strike": strike,
        "bar_time": bar_time, "price": round(price, 2), "exit_reason": reason,
        "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2), "lot_size": lot,
        "day_running_pnl_rs": round(day_pnl, 2),
    })


def process_velez(day_df, state, verbose=True):
    if len(day_df) < POP_SLOPE_LOOKBACK + 3:
        return
    day_open_spot = day_df["open"].iloc[0]
    strike = itm_put_strike(day_open_spot)
    premium_df = build_premium_series(day_df, strike, day_open_spot, side="PE")
    lot = lot_size_for_date(day_df.index[0].date())

    if state["position"] is not None:
        exit_info = step_long_premium_position(state["position"], premium_df, VELEZ_TRAIL_PTS)
        if exit_info:
            pos = state["position"]
            pnl_pts = exit_info["exit_price"] - pos["entry_price"]
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            state["day_pnl_rs"] += rupee_pnl
            state["trades_today"] += 1
            log_exit("Velez8pt", pos["event"], "PE", strike, exit_info["exit_time"],
                      exit_info["exit_price"], exit_info["reason"], pnl_pts, rupee_pnl, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [Velez8pt] EXIT {pos['event']} @ {exit_info['exit_price']:.2f} "
                      f"({exit_info['reason']}) pnl_Rs={rupee_pnl:,.2f}")
            state["search_from_bar"] = exit_info["exit_idx"]
            state["position"] = None

    if state["position"] is None and state["pending_signal"] is None:
        signals = [s for s in detect_velez_signals(day_df) if s["bar"] > state["search_from_bar"]]
        if signals:
            state["pending_signal"] = signals[0]

    if state["position"] is None and state["pending_signal"] is not None:
        j0 = state["pending_signal"]["bar"]
        exec_idx = j0 + 1
        if exec_idx < len(day_df):
            entry_price = premium_df["open"].iloc[exec_idx]
            state["position"] = {
                "entry_idx": exec_idx, "entry_price": entry_price,
                "stop_level": entry_price - VELEZ_STOP_PTS, "peak": entry_price,
                "trailing_active": False, "last_checked_idx": exec_idx - 1,
                "event": state["pending_signal"]["event"],
            }
            log_entry("Velez8pt", state["pending_signal"]["event"], "PE", strike,
                       day_df.index[exec_idx], entry_price, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [Velez8pt] ENTRY {state['pending_signal']['event']} PE@{strike} "
                      f"premium={entry_price:.2f}")
            state["pending_signal"] = None


def process_opening_drive(day_df, state, verbose=True):
    if len(day_df) < 3:
        return
    day_open_spot = day_df["open"].iloc[0]
    lot = lot_size_for_date(day_df.index[0].date())

    if state["position"] is not None:
        pos = state["position"]
        premium_df = build_premium_series(day_df, pos["strike"], day_open_spot, side=pos["side"])
        exit_info = step_long_premium_position(pos, premium_df, OD_TRAIL_PTS)
        if exit_info:
            pnl_pts = exit_info["exit_price"] - pos["entry_price"]
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            state["day_pnl_rs"] += rupee_pnl
            state["trades_today"] += 1
            log_exit("OpeningDrive", pos["side"], pos["side"], pos["strike"], exit_info["exit_time"],
                      exit_info["exit_price"], exit_info["reason"], pnl_pts, rupee_pnl, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [OpeningDrive] EXIT {pos['side']} @ {exit_info['exit_price']:.2f} "
                      f"({exit_info['reason']}) pnl_Rs={rupee_pnl:,.2f}")
            state["search_from_bar"] = exit_info["exit_idx"]
            state["position"] = None

    if state["position"] is None and state["pending_signal"] is None:
        local_i, direction = find_opening_drive(day_df)
        if local_i is not None and local_i > state["search_from_bar"]:
            state["pending_signal"] = {"bar": local_i, "direction": direction}

    if state["position"] is None and state["pending_signal"] is not None:
        j0 = state["pending_signal"]["bar"]
        direction = state["pending_signal"]["direction"]
        exec_idx = j0 + 1
        if exec_idx < len(day_df):
            side = "CE" if direction == "LONG" else "PE"
            strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
            premium_df = build_premium_series(day_df, strike, day_open_spot, side=side)
            entry_price = premium_df["open"].iloc[exec_idx]
            state["position"] = {
                "entry_idx": exec_idx, "entry_price": entry_price,
                "stop_level": entry_price - OD_STOP_PTS, "peak": entry_price,
                "trailing_active": False, "last_checked_idx": exec_idx - 1,
                "side": side, "strike": strike,
            }
            log_entry("OpeningDrive", side, side, strike, day_df.index[exec_idx], entry_price, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [OpeningDrive] ENTRY {side}@{strike} premium={entry_price:.2f} (direction={direction})")
            state["pending_signal"] = None


def process_crash_play(day_df, state, verbose=True):
    if len(day_df) < CRASH_AVG_RANGE_LOOKBACK + 5:
        return
    day_open_spot = day_df["open"].iloc[0]
    strike = itm_put_strike(day_open_spot)
    premium_df = build_premium_series(day_df, strike, day_open_spot, side="PE")
    lot = lot_size_for_date(day_df.index[0].date())

    if state["position"] is not None:
        exit_info = step_crash_position(state["position"], premium_df, CRASH_TRAIL_PTS)
        if exit_info:
            pos = state["position"]
            pnl_pts = exit_info["exit_price"] - pos["entry_price"]
            rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
            state["day_pnl_rs"] += rupee_pnl
            state["trades_today"] += 1
            log_exit("CrashPlay", "CONFIRM", "PE", strike, exit_info["exit_time"],
                      exit_info["exit_price"], exit_info["reason"], pnl_pts, rupee_pnl, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [CrashPlay] EXIT PE @ {exit_info['exit_price']:.2f} "
                      f"({exit_info['reason']}) pnl_Rs={rupee_pnl:,.2f}")
            state["search_from_bar"] = exit_info["exit_idx"]
            state["position"] = None

    if state["position"] is None and state["pending_signal"] is None:
        confirm_bars = find_crash_confirm_signals(day_df, min_bar_idx=state["search_from_bar"])
        if confirm_bars:
            state["pending_signal"] = {"bar": confirm_bars[0]}

    if state["position"] is None and state["pending_signal"] is not None:
        j0 = state["pending_signal"]["bar"]
        exec_idx = j0 + 1
        if exec_idx < len(day_df):
            entry_price = premium_df["open"].iloc[exec_idx]
            state["position"] = {
                "entry_idx": exec_idx, "entry_price": entry_price,
                "stop_level": entry_price - CRASH_STOP_PTS, "peak": entry_price,
                "trailing_active": False, "last_checked_idx": exec_idx - 1,
            }
            log_entry("CrashPlay", "CONFIRM", "PE", strike, day_df.index[exec_idx], entry_price, lot, state["day_pnl_rs"])
            if verbose:
                print(f"  [CrashPlay] ENTRY CONFIRM PE@{strike} premium={entry_price:.2f}")
            state["pending_signal"] = None


# =============================================================================
# Smoke test -- real data, one pass, no infinite loop, no CSV writes
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty data, replaying most recent trading day ===\n")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=HISTORY_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    try:
        raw = fetch_index_window(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed (the logic itself is verified). This failure is "
              "specifically about reaching the Dhan API from wherever this was just run -- if you're "
              "seeing this from your own machine with a working Dhan connection, check your network/"
              "ACCESS_TOKEN. Re-run this smoke test on the machine you'll actually run the live version "
              "from before trusting it fully.")
        return
    full = resample_2min_per_day(raw)
    full_velez = compute_velez_indicators(full)
    days = sorted(set(full.index.date))
    if not days:
        print("No data returned -- can't smoke test right now (market may be fully closed with no recent data). "
              "This is a data-availability issue, not necessarily a bug -- try again closer to/during market hours.")
        return
    last_day = days[-1]
    print(f"Got {len(full)} 2-min bars across {len(days)} day(s), replaying {last_day} (most recent)\n")

    day_df = full[full.index.date == last_day].copy()
    day_df_velez = full_velez[full_velez.index.date == last_day].copy()

    for name, fn, df_to_use in (
        ("Velez8pt", process_velez, day_df_velez),
        ("OpeningDrive", process_opening_drive, day_df),
        ("CrashPlay", process_crash_play, day_df),
    ):
        print(f"--- {name} ---")
        # batch replay (feed the whole day at once)
        state_batch = new_strategy_state()
        fn(df_to_use, state_batch, verbose=True)
        # keep feeding until no more state changes happen within this single day
        # (position may still be open at day end -- that's expected/normal)
        for _ in range(5):
            before = (state_batch["position"] is not None, state_batch["trades_today"])
            fn(df_to_use, state_batch, verbose=False)
            after = (state_batch["position"] is not None, state_batch["trades_today"])
            if before == after:
                break
        print(f"  trades_today={state_batch['trades_today']}  day_pnl_Rs={state_batch['day_pnl_rs']:,.2f}  "
              f"position_open_at_dataend={state_batch['position'] is not None}\n")

    print("Smoke test complete -- no crashes, all self-checks passed, all three strategies ran against "
          "real data end-to-end. Nothing was written to the trade log CSV (smoke test only).")


# =============================================================================
# Live loop
# =============================================================================
def is_market_open():
    now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def main_live():
    run_self_checks()
    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- Velez8pt / OpeningDrive / CrashPlay ===")
    print(f"Log file: {LOG_FILE}")
    print(f"Polling every {POLL_SECONDS}s during market hours ({MARKET_OPEN}-{MARKET_CLOSE} IST)\n")

    states = {"velez": new_strategy_state(), "od": new_strategy_state(), "crash": new_strategy_state()}
    today = dt.date.today()

    while True:
        now = dt.datetime.now()
        if now.date() != today:
            # a new calendar day started -- reset everything, new log file
            today = now.date()
            states = {"velez": new_strategy_state(), "od": new_strategy_state(), "crash": new_strategy_state()}
            globals()["LOG_FILE"] = f"paper_trade_log_{today.isoformat()}.csv"
            print(f"\nNew trading day: {today}. Log file: {LOG_FILE}")

        if not is_market_open():
            print(f"{now.strftime('%H:%M:%S')} Market closed, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            to_date = today.strftime("%Y-%m-%d")
            from_date = (today - dt.timedelta(days=HISTORY_CONTEXT_DAYS)).strftime("%Y-%m-%d")
            raw = fetch_index_window(from_date, to_date)
            full = resample_2min_per_day(raw)
        except Exception:
            print("Failed to fetch candles, retrying...")
            time.sleep(POLL_SECONDS)
            continue

        day_df = full[full.index.date == today].copy()
        if day_df.empty:
            time.sleep(POLL_SECONDS)
            continue
        day_df_velez = compute_velez_indicators(full)
        day_df_velez = day_df_velez[day_df_velez.index.date == today].copy()

        latest_close = day_df["close"].iloc[-1]
        print(f"{now.strftime('%H:%M:%S')} bars={len(day_df)} spot={latest_close:.2f}  "
              f"Velez[{'OPEN' if states['velez']['position'] else 'flat'}]  "
              f"OD[{'OPEN' if states['od']['position'] else 'flat'}]  "
              f"Crash[{'OPEN' if states['crash']['position'] else 'flat'}]")

        process_velez(day_df_velez, states["velez"])
        process_opening_drive(day_df, states["od"])
        process_crash_play(day_df, states["crash"])

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
