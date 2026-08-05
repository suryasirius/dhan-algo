"""
Oliver Velez Method (GBI + ELEPHANT_PLUS, SHORT-only) -- Initial Stop
Sensitivity Check
=================================================================
Answers your question directly: is the fixed 10-point initial stop
cutting into profit, and what does the data say the better number is?

Two things reported:

1. RAW MAE DISTRIBUTION -- for every signal, before any stop logic is
   applied at all, how far did price move against us (in index points)
   at its worst, from entry until end of day? This shows the natural
   "how far do these trades dip before turning" distribution, independent
   of any stop choice.

2. STOP-SIZE SWEEP -- the exact same signals (GBI + ELEPHANT_PLUS,
   SHORT-only, same 4pt trailing-once-in-profit mechanic) re-simulated
   with several different initial stop sizes (5/8/10/12/15/20/25 points),
   so you can directly compare win%, profit factor, net Rs, SQN, max
   drawdown, and whether it wipes out a Rs 25,000 account for each.

Standalone -- does not modify any other file in this project.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "2min"

SMA_FAST = 20
SMA_SLOW = 200
POP_SLOPE_LOOKBACK = 5
AVG_RANGE_LOOKBACK = 20

SMALL_BAR_MULT = 0.6
ELEPHANT_MULT = 2.0
ELEPHANT_PLUS_RETAIN = 0.8
NEAR_20_PCT = 0.0015

TRAIL_PTS = 4
EOD_SQUAREOFF = dt.time(15, 20)
SL_SWEEP = [5, 8, 10, 12, 15, 20, 25]

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def expiry_weekday_for_date(d):
    return 3 if d < dt.date(2025, 9, 1) else 1


def next_expiry(d):
    target_wd = expiry_weekday_for_date(d)
    days_ahead = (target_wd - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


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


def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def compute_indicators(df):
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


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert pop_bearish(95, 97, 98, 100) is True
    assert pop_bearish(105, 103, 102, 100) is False
    assert is_small(2, 10) is True
    assert is_elephant_plus_bear(100, 100, 70, 71, 10) is True
    assert is_elephant_plus_bear(100, 101, 99, 100.5, 10) is False

    # verify simulate_trade_with_sl and compute_mae agree on a known path
    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    closes_t = [100, 105, 99, 92, 97, 104]
    highs_t = [100, 106, 100.5, 93, 98, 105]
    lows_t = [100, 104, 98.5, 91, 96, 103]
    opens_t = [100, 100, 105, 99, 92, 97]
    day_df = pd.DataFrame({"open": opens_t, "high": highs_t, "low": lows_t, "close": closes_t}, index=idx)
    mae = compute_mae(day_df, 0)
    # worst adverse move for a SHORT from entry=100: highest high reached after entry is 106 (bar1) -> MAE=6
    assert abs(mae - 6.0) < 0.01
    trade = simulate_trade_with_sl(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6), 10)
    assert trade["exit_reason"] == "TRAIL_SL"
    assert abs(trade["points_gained"] - 3.0) < 0.01
    print("All self-checks passed.\n")


def detect_signals_for_day(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    sma20 = day_df["sma20"].tolist()
    sma200 = day_df["sma200"].tolist()
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

    seen_bars = set()
    deduped = []
    for s in signals:
        if s["bar"] in seen_bars:
            continue
        seen_bars.add(s["bar"])
        deduped.append(s)
    return deduped


def compute_mae(day_df, entry_bar):
    """Raw max adverse excursion (index points against a SHORT) from
    entry to end of day, with NO stop applied at all -- shows how far
    these trades naturally dip before recovering (or not)."""
    highs = day_df["high"].tolist()
    entry_price = day_df["close"].tolist()[entry_bar]
    n = len(day_df)
    if entry_bar + 1 >= n:
        return 0.0
    worst_high = max(highs[entry_bar + 1:])
    return max(0.0, worst_high - entry_price)


def simulate_trade_with_sl(day_df, sig, day, initial_sl_pts):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = sig["bar"]
    entry_price = closes[j0]

    lot = lot_size_for_date(day)
    stop = entry_price + initial_sl_pts
    trailing_active = False

    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        h, l, c = highs[j], lows[j], closes[j]
        if h >= stop:
            exit_j, exit_price, reason = j, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
            break
        if c < entry_price:
            trailing_active = True
        if trailing_active:
            stop = min(stop, h + TRAIL_PTS)
        if t.time() >= EOD_SQUAREOFF:
            exit_j, exit_price, reason = j, c, "EOD"
            break

    points_gained = entry_price - exit_price
    rupee_pnl = points_gained * lot - BROKERAGE_PER_TRADE

    return {
        "date": day, "event": sig["event"], "entry_time": idx[j0], "exit_time": idx[exit_j],
        "exit_reason": reason, "points_gained": points_gained, "rupee_pnl": rupee_pnl, "lot_size": lot,
    }


def summarize(tdf, sl_label):
    if tdf.empty:
        print(f"SL={sl_label}pts: no trades")
        return
    wins = tdf[tdf["points_gained"] > 0]
    losses = tdf[tdf["points_gained"] <= 0]
    win_rate = len(wins) / len(tdf)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    equity = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    running_peak = equity.cummax()
    max_dd = (equity - running_peak).min()
    wipes_out = (equity <= 0).any()
    r_multiple = tdf["rupee_pnl"] / (sl_label * tdf["lot_size"])
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"SL={sl_label:>3}pts | n={len(tdf):>4} | win%={100*win_rate:5.1f} | PF={pf:5.2f} | "
          f"net_Rs={tdf['rupee_pnl'].sum():>12,.2f} | end_Rs={equity.iloc[-1]:>12,.2f} | "
          f"max_dd_Rs={max_dd:>12,.2f} | SQN={sqn:5.2f} | {'WIPES OUT' if wipes_out else 'ok'}")


def main():
    run_self_checks()
    print("=== Initial Stop-Loss Sensitivity Check -- GBI + ELEPHANT_PLUS (SHORT-only) ===\n")

    print("Fetching Nifty index 1-min data (2 years) and resampling to 2-min bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    df2m = compute_indicators(df2m)
    days_covered = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    # collect signals once (they don't depend on stop size)
    day_signals = []
    for day in days_covered:
        day_df = df2m[df2m.index.date == day].copy()
        sigs = detect_signals_for_day(day_df)
        if sigs:
            day_signals.append((day, day_df, sigs))

    total_signals = sum(len(s[2]) for s in day_signals)
    print(f"Total raw signals found: {total_signals}\n")

    # --- 1. Raw MAE distribution (no stop applied at all) ---
    print("--- Raw MAE distribution (how far price moves against us before EOD, no stop applied) ---")
    maes = []
    for day, day_df, sigs in day_signals:
        open_trade_bar = -1
        for sig in sigs:
            if sig["bar"] <= open_trade_bar:
                continue
            maes.append(compute_mae(day_df, sig["bar"]))
            open_trade_bar = sig["bar"] + 1  # rough non-overlap approximation for this diagnostic pass
    maes = np.array(maes)
    buckets = [(0, 5), (5, 8), (8, 10), (10, 12), (10, 15), (15, 20), (20, 25), (25, 1e9)]
    for lo, hi in buckets:
        cnt = ((maes >= lo) & (maes < hi)).sum()
        print(f"  MAE in [{lo:>3},{hi if hi < 1e9 else 'inf':>4}) pts: {cnt:>4} signals ({100*cnt/len(maes):5.1f}%)")
    print(f"  Median MAE: {np.median(maes):.2f} pts   Mean MAE: {maes.mean():.2f} pts   "
          f"90th pct: {np.percentile(maes,90):.2f} pts\n")

    # --- 2. Stop-size sweep, same signals, same trailing mechanic, only SL changes ---
    print("--- Stop-size sweep (same signals, same 4pt trailing-once-in-profit mechanic) ---")
    for sl in SL_SWEEP:
        all_trades = []
        for day, day_df, sigs in day_signals:
            open_trade_bar = -1
            for sig in sigs:
                if sig["bar"] <= open_trade_bar:
                    continue
                t = simulate_trade_with_sl(day_df, sig, day, sl)
                if t:
                    all_trades.append(t)
                    exit_bar_pos = day_df.index.get_loc(t["exit_time"])
                    open_trade_bar = exit_bar_pos
        tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True) if all_trades else pd.DataFrame()
        summarize(tdf, sl)

    print(f"\n(Starting capital assumed: Rs {STARTING_CAPITAL:,} for the wipeout check above.)")


if __name__ == "__main__":
    main()
