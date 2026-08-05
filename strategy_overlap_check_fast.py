"""
Overlap Check -- Velez (final, 8pt) vs Concepts 2&4 vs Gap Fade LONG-only
-- FAST VERSION
=================================================================
Same test as strategy_overlap_check.py, weekday-filtered + parallelized
for Concepts 2&4. Velez and Gap Fade unchanged (already fast).

*** CONTEXT-WINDOW FIX (this version) ***
Same fix just applied to concepts_2_4_sl_sensitivity_fast.py: earlier
versions sliced each day in isolation, giving compute_latest_signal no
prior history. Your live bot's HTF_CONTEXT_DAYS=5 config means it
likely needs several prior trading days to compute its 5m/15m trend --
without them it silently returns "no signal" every time (no error,
which is exactly what the last run showed: 0 trades, no error). Fixed
by building a rolling HTF_CONTEXT_DAYS-day window ending on each target
day, same as the sensitivity-check script.

Standalone -- does NOT modify any other file.
"""

import datetime as dt
import os
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

V_SMA_FAST, V_SMA_SLOW = 20, 200
V_POP_SLOPE_LOOKBACK = 5
V_AVG_RANGE_LOOKBACK = 20
V_SMALL_BAR_MULT = 0.6
V_ELEPHANT_MULT = 2.0
V_ELEPHANT_PLUS_RETAIN = 0.8
V_NEAR_20_PCT = 0.0015
V_INITIAL_SL_PTS = 8
V_TRAIL_PTS = 4
V_EOD_SQUAREOFF = dt.time(15, 20)
V_BROKERAGE = 40.0

C_WARMUP_BARS = 30
C_ASSUMED_DELTA = 0.6
C_LOT_SIZE = 65
C_BROKERAGE = 40.0

G_GAP_ATR_MIN_MULT = 0.3
G_GAP_ATR_MAX_MULT = 2.0
G_DAILY_ATR_PERIOD = 14
G_REJECTION_WINDOW_MIN = 15
G_TARGET_GAP_PCT = 0.15
G_TRAIL_PTS = 4
G_STOP_BUFFER_PTS = 1.0
G_PINBAR_WICK_RATIO = 2.0
G_ASSUMED_DELTA = 0.6
G_LOT_SIZE = 65
G_BROKERAGE = 40.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp)
        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
            "volume": data["volume"], "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


# ===== VELEZ FINAL (8pt) -- unchanged, already fast =====
def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample("2min", origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def v_compute_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(V_SMA_FAST, min_periods=V_SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(V_SMA_SLOW, min_periods=V_SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(V_AVG_RANGE_LOOKBACK, min_periods=V_AVG_RANGE_LOOKBACK).mean()
    return df


def v_pop_bearish(price, sma20, sma20_prev, sma200):
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(sma20_prev):
        return False
    return price < sma20 and sma20 < sma200 and sma20 < sma20_prev


def v_is_green(o, c):
    return c > o


def v_is_red(o, c):
    return c < o


def v_is_small(rng, avgr):
    return bool(avgr) and avgr > 0 and rng < V_SMALL_BAR_MULT * avgr


def v_is_elephant_plus_bear(o, h, l, c, avgr):
    rng = h - l
    if not (bool(avgr) and avgr > 0 and (rng * V_ELEPHANT_PLUS_RETAIN) > V_ELEPHANT_MULT * avgr):
        return False
    return v_is_red(o, c)


def v_find_small_green_run(highs, lows, opens, closes, avg_ranges, end_idx):
    i = end_idx - 1
    if i < 0:
        return None
    if not (v_is_green(opens[i], closes[i]) and v_is_small(highs[i] - lows[i], avg_ranges[i])):
        return None
    combined_high, combined_low = highs[i], lows[i]
    start_idx = i
    j = i - 1
    while j >= 0 and v_is_green(opens[j], closes[j]) and v_is_small(highs[j] - lows[j], avg_ranges[j]):
        combined_high = max(combined_high, highs[j])
        combined_low = min(combined_low, lows[j])
        start_idx = j
        j -= 1
    if start_idx - 1 < 0:
        return None
    if not v_is_red(opens[start_idx - 1], closes[start_idx - 1]):
        return None
    return start_idx, combined_high, combined_low


def v_detect_signals_for_day(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    sma20 = day_df["sma20"].tolist()
    sma200 = day_df["sma200"].tolist()
    avg_ranges = day_df["avg_range"].tolist()
    n = len(day_df)
    signals = []
    if n < V_POP_SLOPE_LOOKBACK + 3:
        return signals
    for j in range(V_POP_SLOPE_LOOKBACK, n - 1):
        if pd.isna(sma20[j]) or pd.isna(sma200[j]) or pd.isna(sma20[j - V_POP_SLOPE_LOOKBACK]):
            continue
        if not v_pop_bearish(closes[j], sma20[j], sma20[j - V_POP_SLOPE_LOOKBACK], sma200[j]):
            continue
        near20 = abs(closes[j] - sma20[j]) / closes[j] < V_NEAR_20_PCT
        if near20:
            run = v_find_small_green_run(highs, lows, opens, closes, avg_ranges, j)
            if run:
                _, comb_high, comb_low = run
                if closes[j] < comb_low and v_is_red(opens[j], closes[j]):
                    signals.append({"bar": j, "event": "GBI"})
                    continue
        if j - 1 >= 0:
            eo, eh, el, ec = opens[j - 1], highs[j - 1], lows[j - 1], closes[j - 1]
            eavgr = avg_ranges[j - 1]
            if v_is_elephant_plus_bear(eo, eh, el, ec, eavgr) and closes[j] < el:
                signals.append({"bar": j, "event": "ELEPHANT_PLUS"})
    seen = set()
    deduped = []
    for s in signals:
        if s["bar"] in seen:
            continue
        seen.add(s["bar"])
        deduped.append(s)
    return deduped


def v_simulate_trade(day_df, sig, day):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = sig["bar"]
    entry_price = closes[j0]
    lot = lot_size_for_date(day)
    stop = entry_price + V_INITIAL_SL_PTS
    trailing_active = False
    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        h, l, c = highs[j], lows[j], closes[j]
        if h >= stop:
            exit_j, exit_price, reason = j, stop, "TRAIL_SL" if trailing_active else "INITIAL_SL"
            break
        if c < entry_price:
            trailing_active = True
        if trailing_active:
            stop = min(stop, h + V_TRAIL_PTS)
        if t.time() >= V_EOD_SQUAREOFF:
            exit_j, exit_price, reason = j, c, "EOD"
            break
    points = entry_price - exit_price
    rupee_pnl = points * lot - V_BROKERAGE
    return {"strategy": "Velez", "date": day, "entry_time": idx[j0], "rupee_pnl": rupee_pnl}


def run_velez(df1m):
    df2m = resample_2min_per_day(df1m)
    df2m = v_compute_indicators(df2m)
    days = sorted(set(df2m.index.date))
    trades = []
    for day in days:
        day_df = df2m[df2m.index.date == day].copy()
        sigs = v_detect_signals_for_day(day_df)
        open_bar = -1
        for sig in sigs:
            if sig["bar"] <= open_bar:
                continue
            t = v_simulate_trade(day_df, sig, day)
            trades.append(t)
            open_bar = day_df.index.get_loc(t["entry_time"])
    return pd.DataFrame(trades)


# ===== CONCEPTS 2&4 -- weekday-filtered + parallelized + context window =====
def get_allowed_weekdays():
    wd = getattr(dlb, "ALLOWED_WEEKDAYS", None)
    if wd is None:
        print("  NOTE: dlb.ALLOWED_WEEKDAYS not found -- falling back to {1,2} (Tue/Wed).")
        return {1, 2}
    return set(wd)


def get_htf_context_days():
    cd = getattr(dlb, "HTF_CONTEXT_DAYS", None)
    if cd is None:
        print("  NOTE: dlb.HTF_CONTEXT_DAYS not found -- falling back to 5.")
        return 5
    return cd


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def c_extract_signal(sig):
    if sig.get("final_buy"):
        return "LONG"
    if sig.get("final_sell"):
        return "SHORT"
    return None


def _c24_scan_one_day(args):
    """Worker function -- args is (target_day, window_df) where window_df
    spans HTF_CONTEXT_DAYS trading days ending on target_day. Returns
    (trades_list, first_error_string_or_None)."""
    target_day, window_df = args
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return [], None
    n = len(window_df)
    first_bar_today = today_positions[0]
    last_bar_today = today_positions[-1]
    if (last_bar_today - first_bar_today) <= C_WARMUP_BARS + 2:
        return [], None

    closes = window_df["close"].tolist()
    highs = window_df["high"].tolist()
    lows = window_df["low"].tolist()
    idx = window_df.index

    trades = []
    first_error = None
    open_bar = -1
    for i in today_positions:
        if i - first_bar_today < C_WARMUP_BARS:
            continue
        if i <= open_bar:
            continue
        try:
            sig = dlb.compute_latest_signal(window_df.iloc[:i + 1])
        except Exception:
            if first_error is None:
                first_error = traceback.format_exc()
            continue
        if sig is None:
            continue
        side = c_extract_signal(sig)
        if side is None:
            continue
        entry_price = closes[i]
        if side == "LONG":
            stop = entry_price - dlb.SL_POINTS
            target = entry_price + dlb.TP_POINTS
        else:
            stop = entry_price + dlb.SL_POINTS
            target = entry_price - dlb.TP_POINTS
        exit_price, reason = entry_price, "EOD"
        for j in range(i + 1, min(i + 1 + dlb.MAX_BARS_IN_TRADE, n)):
            h, l, c = highs[j], lows[j], closes[j]
            if side == "LONG":
                if l <= stop:
                    exit_price, reason = stop, "STOP"; break
                if h >= target:
                    exit_price, reason = target, "TARGET"; break
            else:
                if h >= stop:
                    exit_price, reason = stop, "STOP"; break
                if l <= target:
                    exit_price, reason = target, "TARGET"; break
        else:
            last_idx = min(i + dlb.MAX_BARS_IN_TRADE, n - 1)
            exit_price, reason = closes[last_idx], "TIMEOUT"
        points = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        rupee_pnl = points * C_ASSUMED_DELTA * C_LOT_SIZE - C_BROKERAGE
        trades.append({"strategy": "Concepts2n4", "date": target_day, "entry_time": idx[i], "rupee_pnl": rupee_pnl})
        open_bar = i + dlb.MAX_BARS_IN_TRADE
    return trades, first_error


def run_concepts_2_4(df1m):
    days_covered = sorted(set(df1m.index.date))
    allowed_weekdays = get_allowed_weekdays()
    context_days = get_htf_context_days()
    target_days = [d for d in days_covered if d.weekday() in allowed_weekdays]
    print(f"  Filtering to allowed weekdays: {len(target_days)} of {len(days_covered)} days to scan.")
    print(f"  Each day gets a {context_days}-trading-day rolling context window.")

    jobs = []
    for target_day in target_days:
        window_days = build_context_window_days(days_covered, target_day, context_days)
        window_df = df1m[(df1m.index.date >= window_days[0]) & (df1m.index.date <= window_days[-1])]
        jobs.append((target_day, window_df))

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"  Scanning across {n_workers} worker processes...")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_c24_scan_one_day, jobs)

    all_trades = []
    errors = []
    for (target_day, window_df), (day_trades, err) in zip(jobs, results):
        all_trades.extend(day_trades)
        if err:
            errors.append((target_day, err))

    if errors:
        print(f"\n  {len(errors)} of {len(target_days)} days hit an error calling compute_latest_signal.")
        print(f"  First error (from {errors[0][0]}):\n{errors[0][1]}")
        print("  Paste this back to me.\n")

    return pd.DataFrame(all_trades)


# ===== GAP FADE LONG-ONLY -- unchanged, already fast =====
def build_daily_bars(df):
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def compute_daily_atr(daily_df, period=G_DAILY_ATR_PERIOD):
    h, l, c = daily_df["high"].tolist(), daily_df["low"].tolist(), daily_df["close"].tolist()
    trs = [h[0] - l[0]]
    for i in range(1, len(h)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    atr = []
    for i in range(len(trs)):
        w = trs[max(0, i-period+1):i+1]
        atr.append(sum(w)/len(w))
    return atr


def g_classify_daily_trend(daily_df, i):
    if i < 2:
        return None
    prev = daily_df.iloc[i-1]
    prev2 = daily_df.iloc[i-2]
    if prev["close"] > prev["open"] and prev["high"] > prev2["high"]:
        return "UP"
    if prev["close"] < prev["open"] and prev["low"] < prev2["low"]:
        return "DOWN"
    return "NEUTRAL"


def g_is_bullish_engulfing(o, c, j):
    if j < 1:
        return False
    return (c[j] > o[j] and c[j-1] < o[j-1] and min(o[j], c[j]) <= min(o[j-1], c[j-1])
            and max(o[j], c[j]) >= max(o[j-1], c[j-1]))


def g_is_bullish_pinbar(o, h, l, c, j, ratio=G_PINBAR_WICK_RATIO):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and lw >= ratio*body and lw > uw


def g_detect_long_signal_for_day(day_df, prev_trend, gap, gap_ok):
    if prev_trend != "UP" or not gap_ok or gap >= 0:
        return None
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = min(len(day_df), G_REJECTION_WINDOW_MIN)
    for j in range(1, n):
        if g_is_bullish_engulfing(o, c, j) or g_is_bullish_pinbar(o, h, l, c, j):
            return {"bar": j, "rej_low": l[j]}
    return None


def run_gap_fade(df1m):
    daily = build_daily_bars(df1m)
    daily_atr = compute_daily_atr(daily)
    daily_dates = daily.index.date.tolist()
    days = sorted(set(df1m.index.date))
    trades = []
    for day in days:
        if day not in daily_dates:
            continue
        i = daily_dates.index(day)
        prev_trend = g_classify_daily_trend(daily, i)
        if prev_trend != "UP" or i == 0:
            continue
        prev_close = daily["close"].iloc[i-1]
        today_open = daily["open"].iloc[i]
        gap = today_open - prev_close
        atr_i = daily_atr[i] if i < len(daily_atr) else None
        if not atr_i or atr_i <= 0:
            continue
        gap_ok = G_GAP_ATR_MIN_MULT * atr_i <= abs(gap) <= G_GAP_ATR_MAX_MULT * atr_i
        day_df = df1m[df1m.index.date == day]
        if len(day_df) < G_REJECTION_WINDOW_MIN + 2:
            continue
        sig = g_detect_long_signal_for_day(day_df, prev_trend, gap, gap_ok)
        if sig is None:
            continue
        closes = day_df["close"].tolist()
        highs = day_df["high"].tolist()
        lows = day_df["low"].tolist()
        idx = day_df.index
        n = len(day_df)
        j0 = sig["bar"]
        entry = closes[j0]
        target_pts = abs(gap) * G_TARGET_GAP_PCT
        stop = sig["rej_low"] - G_STOP_BUFFER_PTS
        target = entry + target_pts
        trailing_active = False
        stop_level = None
        exit_price = closes[-1]
        for j in range(j0 + 1, n):
            c, h, l = closes[j], highs[j], lows[j]
            if trailing_active:
                if c <= stop_level:
                    exit_price = c; break
                stop_level = max(stop_level, l - G_TRAIL_PTS)
            else:
                if c <= stop:
                    exit_price = c; break
                elif c >= target:
                    trailing_active = True
                    stop_level = l - G_TRAIL_PTS
        points = exit_price - entry
        rupee_pnl = points * G_ASSUMED_DELTA * G_LOT_SIZE - G_BROKERAGE
        trades.append({"strategy": "GapFadeLong", "date": day, "entry_time": idx[j0], "rupee_pnl": rupee_pnl})
    return pd.DataFrame(trades)


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert v_pop_bearish(95, 97, 98, 100) is True
    assert v_pop_bearish(105, 103, 102, 100) is False
    assert c_extract_signal({"final_buy": True, "final_sell": False}) == "LONG"
    assert c_extract_signal({"final_buy": False, "final_sell": False}) is None
    daily = pd.DataFrame({
        "open":  [100, 101, 105, 103],
        "high":  [102, 108, 106, 107],
        "low":   [99,  100, 102, 101],
        "close": [101, 107, 104, 106],
    })
    assert g_classify_daily_trend(daily, 2) == "UP"
    allowed = get_allowed_weekdays()
    assert isinstance(allowed, set) and len(allowed) > 0
    ctx = get_htf_context_days()
    assert isinstance(ctx, int) and ctx > 0
    days = [dt.date(2025, 1, d) for d in range(1, 11)]
    w = build_context_window_days(days, dt.date(2025, 1, 10), 5)
    assert w == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8),
                 dt.date(2025, 1, 9), dt.date(2025, 1, 10)]
    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Overlap Check (FAST): Velez vs Concepts 2&4 vs Gap Fade LONG-only ===\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    print("Running Velez final (8pt)...")
    velez_trades = run_velez(df1m)
    print(f"  {len(velez_trades)} trades.")

    print("Running Concepts 2&4 (real dlb.compute_latest_signal, weekday-filtered + parallelized + context window)...")
    c24_trades = run_concepts_2_4(df1m)
    print(f"  {len(c24_trades)} trades.")

    print("Running Gap Fade LONG-only...")
    gap_trades = run_gap_fade(df1m)
    print(f"  {len(gap_trades)} trades.\n")

    all_trades = pd.concat([velez_trades, c24_trades, gap_trades], ignore_index=True) if len(velez_trades) or len(c24_trades) or len(gap_trades) else pd.DataFrame()
    if all_trades.empty:
        print("No trades from any strategy -- cannot compute overlap.")
        return

    all_trades["date"] = pd.to_datetime(all_trades["date"])
    daily_pnl = all_trades.pivot_table(index="date", columns="strategy", values="rupee_pnl", aggfunc="sum")
    full_index = pd.date_range(days_covered[0], days_covered[-1], freq="D")
    daily_pnl = daily_pnl.reindex(full_index).fillna(0.0)

    strategies = [c for c in ["Velez", "Concepts2n4", "GapFadeLong"] if c in daily_pnl.columns]
    print("--- Trading frequency ---")
    for s in strategies:
        traded_days = (daily_pnl[s] != 0).sum()
        print(f"{s}: traded on {traded_days} of {len(daily_pnl)} calendar days ({100*traded_days/len(daily_pnl):.1f}%)")
    print()

    print("--- Day overlap (both strategies trade on the same day) ---")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a, b = strategies[i], strategies[j]
            both = ((daily_pnl[a] != 0) & (daily_pnl[b] != 0)).sum()
            a_days = (daily_pnl[a] != 0).sum()
            b_days = (daily_pnl[b] != 0).sum()
            if a_days and b_days:
                print(f"{a} & {b}: {both} overlapping days "
                      f"({100*both/a_days:.1f}% of {a}'s days, {100*both/b_days:.1f}% of {b}'s days)")
            else:
                print(f"{a} & {b}: {both} overlapping days")
    print()

    print("--- Daily P&L correlation (all calendar days, non-trading days = 0) ---")
    corr = daily_pnl[strategies].corr()
    print(corr.round(3).to_string())
    print()

    print("--- On days both strategies trade: do they agree in direction? ---")
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a, b = strategies[i], strategies[j]
            mask = (daily_pnl[a] != 0) & (daily_pnl[b] != 0)
            sub = daily_pnl.loc[mask, [a, b]]
            if len(sub) == 0:
                print(f"{a} & {b}: no overlapping days to compare.")
                continue
            same_sign = (np.sign(sub[a]) == np.sign(sub[b])).sum()
            print(f"{a} & {b}: on {len(sub)} overlapping days, same-direction (both win or both lose) "
                  f"on {same_sign} ({100*same_sign/len(sub):.1f}%)")
    print()

    print("Interpretation guide: low overlap % and low/negative correlation = safer to combine "
          "(bad days don't stack). High overlap % and positive correlation = combining risks the "
          "same hidden-drawdown problem seen in the earlier OI-combined test.")


if __name__ == "__main__":
    main()
