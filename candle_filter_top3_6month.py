"""
Applies the candle-prediction analysis (the 9-concept Price Action & Market
Structure engine) as an EXTRA CONFIRMATION FILTER on top of the project's
top 3 strategies (Velez 8pt, Crash Play, Opening-Drive Momentum), over the
same 6-month local Nifty 1-min dataset used for the candle-analysis test.

For each strategy: a baseline run (its own entry logic, unchanged) vs a
candle-filtered run (same entries, but ONLY kept if the candle-analysis
engine also fired an agreeing signal -- final_sell for SHORT/PE trades,
final_buy for LONG/CE trades -- within the 10 minutes before/at that entry).

Each strategy's own money-model convention is kept as-is (Velez = raw index
points x lot size, no delta; Crash Play / Opening-Drive = delta-scaled
option-premium proxy) so baseline-vs-filtered comparisons stay apples to
apples within each strategy, consistent with how each was originally tested
in this project.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from candle_pa_6month_offline_backtest import (   # noqa: E402
    compute_all_signals_for_day, attach_htf_trend,
)
from candle_pa_6month_run import load_data as load_1min_data   # noqa: E402

INPUT_FILE = Path(__file__).with_name("dhan_historical_data_180days_v5.csv")
CONFIRM_WINDOW_MIN = 10   # candle-analysis signal must fire within this many minutes before/at entry

# ---- shared money-model constants (match each strategy's own script) ----
LOT_SIZE_2026 = 65
BROKERAGE_PER_TRADE = 40.0
ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)


def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample("2min", origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


# ======================= Candle-analysis signal index =======================
def build_candle_signal_index(df_1min):
    """Runs the unrestricted 9-concept engine (no day-of-week / HTF filter,
    confluence=1) on every 1-min bar, returns a DataFrame indexed by time
    with final_buy/final_sell booleans, for use as an external confirmation
    filter on the other three strategies' entries."""
    rows = []
    for day, day_df in df_1min.groupby(df_1min.index.date):
        if len(day_df) < 40:
            continue
        sigs = compute_all_signals_for_day(day_df, disabled_concepts=set(),
                                            allowed_weekdays=None, require_htf=False,
                                            confluence_required=1)
        for s in sigs:
            rows.append((s["time"], s["final_buy"], s["final_sell"]))
    sig_df = pd.DataFrame(rows, columns=["time", "final_buy", "final_sell"]).set_index("time")
    return sig_df


def candle_confirms(sig_df, entry_time, direction, window_min=CONFIRM_WINDOW_MIN):
    """direction: 'LONG' or 'SHORT'. Looks for an agreeing candle-analysis
    signal in [entry_time - window_min, entry_time]."""
    start = entry_time - pd.Timedelta(minutes=window_min)
    window = sig_df.loc[start:entry_time]
    if window.empty:
        return False
    return bool(window["final_buy"].any()) if direction == "LONG" else bool(window["final_sell"].any())


# ============================= Velez 8pt (SHORT) =============================
SMA_FAST, SMA_SLOW = 20, 200
POP_SLOPE_LOOKBACK, AVG_RANGE_LOOKBACK = 5, 20
SMALL_BAR_MULT, ELEPHANT_MULT, ELEPHANT_PLUS_RETAIN, NEAR_20_PCT = 0.6, 2.0, 0.8, 0.0015
VELEZ_SL, VELEZ_TRAIL = 8, 4


def velez_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean()
    return df


def pop_bearish(price, sma20, sma20_prev, sma200):
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(sma20_prev):
        return False
    return price < sma20 and sma20 < sma200 and sma20 < sma20_prev


def is_green(o, c): return c > o
def is_red(o, c): return c < o
def is_small(rng, avgr): return bool(avgr) and avgr > 0 and rng < SMALL_BAR_MULT * avgr


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
        combined_high = max(combined_high, highs[j]); combined_low = min(combined_low, lows[j])
        start_idx = j; j -= 1
    if start_idx - 1 < 0:
        return None
    if not is_red(opens[start_idx - 1], closes[start_idx - 1]):
        return None
    return start_idx, combined_high, combined_low


def velez_detect_signals(day_df):
    opens, highs, lows, closes = (day_df["open"].tolist(), day_df["high"].tolist(),
                                    day_df["low"].tolist(), day_df["close"].tolist())
    sma20, sma200, avg_ranges = day_df["sma20"].tolist(), day_df["sma200"].tolist(), day_df["avg_range"].tolist()
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
                    signals.append({"bar": j, "event": "GBI"}); continue
        if j - 1 >= 0:
            eo, eh, el, ec, eavgr = opens[j-1], highs[j-1], lows[j-1], closes[j-1], avg_ranges[j-1]
            if is_elephant_plus_bear(eo, eh, el, ec, eavgr) and closes[j] < el:
                signals.append({"bar": j, "event": "ELEPHANT_PLUS"})
    seen, out = set(), []
    for s in signals:
        if s["bar"] in seen: continue
        seen.add(s["bar"]); out.append(s)
    return out


def velez_simulate(day_df, bar_idx, lot):
    closes, highs = day_df["close"].tolist(), day_df["high"].tolist()
    idx = day_df.index
    n = len(day_df)
    entry_price = closes[bar_idx]
    stop = entry_price + VELEZ_SL
    trailing_active = False
    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(bar_idx + 1, n):
        t = idx[j]; h, c = highs[j], closes[j]
        if h >= stop:
            exit_j, exit_price, reason = j, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL"); break
        if c < entry_price:
            trailing_active = True
        if trailing_active:
            stop = min(stop, h + VELEZ_TRAIL)
        if t.time() >= EOD_SQUAREOFF:
            exit_j, exit_price, reason = j, c, "EOD"; break
    points = entry_price - exit_price
    rupee = points * lot - BROKERAGE_PER_TRADE
    return {"entry_time": idx[bar_idx], "exit_time": idx[exit_j], "direction": "SHORT",
            "points": points, "rupee_pnl": rupee, "reason": reason}


def run_velez(df2m, sig_df):
    df2m = velez_indicators(df2m)
    baseline, filtered = [], []
    for day, day_df in df2m.groupby(df2m.index.date):
        signals = velez_detect_signals(day_df)
        open_trade_bar = -1
        for sig in signals:
            if sig["bar"] <= open_trade_bar:
                continue
            t = velez_simulate(day_df, sig["bar"], LOT_SIZE_2026)
            baseline.append(t)
            if candle_confirms(sig_df, t["entry_time"], "SHORT"):
                filtered.append(t)
            exit_bar_pos = day_df.index.get_loc(t["exit_time"])
            open_trade_bar = exit_bar_pos
    return pd.DataFrame(baseline), pd.DataFrame(filtered)


# ============================= Crash Play (PE / SHORT) =============================
CRASH_MULT = 2.5
MAX_LOOKAHEAD_BARS = 60
CRASH_SL, CRASH_TRAIL = 8.0, 4.0


def itm_put_strike(spot): return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def build_pe_premium_series(day_df, strike, day_open_spot):
    idx_open, idx_high, idx_low, idx_close = (day_df["open"].to_numpy(), day_df["high"].to_numpy(),
                                                day_df["low"].to_numpy(), day_df["close"].to_numpy())
    intrinsic0 = max(strike - day_open_spot, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + (day_open_spot - idx_open) * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + (day_open_spot - idx_close) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (day_open_spot - idx_low) * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + (day_open_spot - idx_high) * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def zone_of(level, crash_low, crash_open):
    rng = crash_open - crash_low
    if rng <= 0: return None
    pct = (level - crash_low) / rng
    if pct < 0 or pct > 1.0 + 1e-9: return None
    if pct >= 0.67: return 1
    elif pct >= 0.50: return 2
    elif pct >= 0.33: return 3
    else: return 4


def find_crash_signals(day_df):
    opens, highs, lows, closes = (day_df["open"].tolist(), day_df["high"].tolist(),
                                    day_df["low"].tolist(), day_df["close"].tolist())
    rng = (day_df["high"] - day_df["low"]).tolist()
    avg_range = pd.Series(rng).shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean().tolist()
    n = len(day_df)
    events = []
    j = AVG_RANGE_LOOKBACK
    while j < n - 2:
        if avg_range[j] and rng[j] >= CRASH_MULT * avg_range[j] and is_red(opens[j], closes[j]):
            crash_open, crash_low = opens[j], lows[j]
            k = j
            while k + 1 < n and lows[k + 1] <= crash_low + 1e-9:
                k += 1; crash_low = min(crash_low, lows[k])
            confirm_idx, zone_confirm = None, None
            bounce_start = k + 1
            limit = min(n, bounce_start + MAX_LOOKAHEAD_BARS)
            for i in range(bounce_start, limit):
                c = closes[i]; z = zone_of(c, crash_low, crash_open)
                if confirm_idx is None and z in (3, 4) and i > bounce_start:
                    if is_red(opens[i], closes[i]) and is_green(opens[i-1], closes[i-1]) and lows[i] < lows[i-1]:
                        confirm_idx, zone_confirm = i, z
            events.append({"crash_bar_idx": j, "confirm_entry_idx": confirm_idx})
            j = k + MAX_LOOKAHEAD_BARS
        j += 1
    return events


def crash_simulate(prem_df, entry_idx, lot):
    closes, highs, lows = prem_df["close"].tolist(), prem_df["high"].tolist(), prem_df["low"].tolist()
    idx = prem_df.index
    n = len(prem_df)
    entry_premium = closes[entry_idx]
    stop = entry_premium - CRASH_SL
    trailing_active = False
    exit_i, exit_premium, reason = n - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, n):
        t = idx[i]
        if lows[i] <= stop:
            exit_i, exit_premium, reason = i, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL"); break
        if closes[i] > entry_premium:
            trailing_active = True
        if trailing_active:
            stop = max(stop, lows[i] - CRASH_TRAIL)
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, closes[i], "EOD"; break
    points = exit_premium - entry_premium
    rupee = points * lot - BROKERAGE_PER_TRADE
    return {"entry_time": idx[entry_idx], "exit_time": idx[exit_i], "direction": "SHORT",
            "points": points, "rupee_pnl": rupee, "reason": reason}


def run_crash_play(df2m, sig_df):
    baseline, filtered = [], []
    for day, day_df in df2m.groupby(df2m.index.date):
        if len(day_df) < AVG_RANGE_LOOKBACK + 5:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_put_strike(day_open_spot)
        prem_df = build_pe_premium_series(day_df, strike, day_open_spot)
        events = find_crash_signals(day_df)
        for ev in events:
            if ev["confirm_entry_idx"] is None:
                continue
            t = crash_simulate(prem_df, ev["confirm_entry_idx"], LOT_SIZE_2026)
            baseline.append(t)
            if candle_confirms(sig_df, t["entry_time"], "SHORT"):
                filtered.append(t)
    return pd.DataFrame(baseline), pd.DataFrame(filtered)


# ============================= Opening-Drive Momentum =============================
ENTRY_CUTOFF = dt.time(9, 45)
MIN_MOVE_PTS = 5.0
ODM_SL, ODM_TRAIL = 8.0, 4.0


def itm_call_strike(spot): return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def build_premium_series(day_df, strike, day_open_spot, side):
    idx_open, idx_high, idx_low, idx_close = (day_df["open"].to_numpy(), day_df["high"].to_numpy(),
                                                day_df["low"].to_numpy(), day_df["close"].to_numpy())
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        mo, mh, ml, mc = (idx_open - day_open_spot, idx_high - day_open_spot, idx_low - day_open_spot, idx_close - day_open_spot)
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        mo, mh, ml, mc = (day_open_spot - idx_open, day_open_spot - idx_low, day_open_spot - idx_high, day_open_spot - idx_close)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def find_opening_drive(day_df):
    day_open = day_df["open"].iloc[0]
    closes = day_df["close"].to_numpy()
    times = day_df.index
    for i in range(len(day_df)):
        if times[i].time() >= ENTRY_CUTOFF:
            return None, None
        move = closes[i] - day_open
        if move >= MIN_MOVE_PTS:
            return i, "LONG"
        if move <= -MIN_MOVE_PTS:
            return i, "SHORT"
    return None, None


def odm_simulate(premium_df, entry_idx, entry_price):
    stop_level = entry_price - ODM_SL
    peak = entry_price
    trailing_active = False
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= stop_level:
            reason = "TRAIL" if trailing_active else "STOP"
            return stop_level, reason, premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
        if bar["high"] > peak:
            peak = bar["high"]
        if peak > entry_price:
            trailing_active = True
        if trailing_active:
            stop_level = max(stop_level, peak - ODM_TRAIL)
    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


def run_opening_drive(df2m, sig_df):
    baseline, filtered = [], []
    for day, day_df in df2m.groupby(df2m.index.date):
        if len(day_df) < 3:
            continue
        day_open_spot = day_df["open"].iloc[0]
        local_i, direction = find_opening_drive(day_df)
        if local_i is None:
            continue
        exec_idx = local_i + 1
        if exec_idx >= len(day_df):
            continue
        side = "CE" if direction == "LONG" else "PE"
        strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
        premium_df = build_premium_series(day_df, strike, day_open_spot, side)
        entry_price = premium_df["open"].iloc[exec_idx]
        exit_price, reason, exit_time = odm_simulate(premium_df, exec_idx, entry_price)
        lot = LOT_SIZE_2026
        pnl_pts = exit_price - entry_price
        rupee = pnl_pts * lot - BROKERAGE_PER_TRADE
        t = {"entry_time": day_df.index[exec_idx], "exit_time": exit_time, "direction": direction,
             "points": pnl_pts, "rupee_pnl": rupee, "reason": reason}
        baseline.append(t)
        if candle_confirms(sig_df, t["entry_time"], direction):
            filtered.append(t)
    return pd.DataFrame(baseline), pd.DataFrame(filtered)


# ================================ Metrics / report ================================
def profit_factor(pnl):
    gains = pnl[pnl > 0].sum(); losses = pnl[pnl < 0].sum()
    if losses == 0: return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def sqn(pnl):
    if len(pnl) < 2 or pnl.std(ddof=1) == 0: return 0.0
    return (pnl.mean() / pnl.std(ddof=1)) * (len(pnl) ** 0.5)


def summarize(tdf):
    if tdf.empty:
        return dict(trades=0, win_pct=0.0, pf=0.0, sqn=0.0, net_rs=0.0, max_dd=0.0)
    pnl = tdf["rupee_pnl"]
    wins = pnl > 0
    cum = pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    return dict(trades=len(tdf), win_pct=100 * wins.mean(), pf=profit_factor(pnl),
                sqn=sqn(pnl), net_rs=pnl.sum(), max_dd=dd)
