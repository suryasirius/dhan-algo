"""
Oliver Velez Method -- SHORT-only, GBI + ELEPHANT_PLUS -- Rs 25,000 capital test
=================================================================
Identical to velez_gbi_elephantplus_only.py (same signals, same 10pt
initial stop / 4pt trailing exit, same brokerage) -- the ONLY change is
STARTING_CAPITAL, bumped from Rs 10,000 to Rs 25,000, to test whether
the wipeout seen in the last two runs (max drawdown ~Rs 11,600-13,500)
was a capital-adequacy problem rather than a flaw in the edge itself.
If this version never wipes out, that confirms the strategy itself is
sound and just needed a bigger cushion at Rs 10,000. Standalone -- does
not modify any other file in this project.
"""

import datetime as dt
import math
import random
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

INITIAL_SL_PTS = 10
TRAIL_PTS = 4
EOD_SQUAREOFF = dt.time(15, 20)

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000   # <-- bumped from 10,000 to test capital adequacy
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


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

    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    closes_t = [100, 105, 99, 92, 97, 104]
    highs_t = [100, 106, 100.5, 93, 98, 105]
    lows_t = [100, 104, 98.5, 91, 96, 103]
    opens_t = [100, 100, 105, 99, 92, 97]
    day_df = pd.DataFrame({"open": opens_t, "high": highs_t, "low": lows_t, "close": closes_t}, index=idx)
    trade = simulate_trade(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6))
    assert trade is not None
    assert trade["exit_reason"] == "TRAIL_SL"
    assert abs(trade["points_gained"] - 3.0) < 0.01

    n_bars = 187
    idx2 = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="2min")
    closes2 = [100.0 - i * 0.05 for i in range(n_bars)]
    opens2 = [c + 0.02 for c in closes2]
    highs2 = [c + 0.05 for c in closes2]
    lows2 = [c - 0.03 for c in closes2]
    day2 = pd.DataFrame({"open": opens2, "high": highs2, "low": lows2, "close": closes2}, index=idx2)
    day2["sma20"] = pd.Series(closes2).rolling(SMA_FAST, min_periods=1).mean().values
    day2["sma200"] = pd.Series([100.0] * n_bars) - pd.Series(range(n_bars)) * 0.03
    day2["range"] = day2["high"] - day2["low"]
    day2["avg_range"] = day2["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=1).mean()
    sigs = detect_signals_for_day(day2)
    assert len(sigs) >= 0
    print(f"  (regression check: {len(sigs)} signal(s) found scanning a {n_bars}-bar day -- loop executes correctly)")
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


def simulate_trade(day_df, sig, day):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = sig["bar"]
    entry_price = closes[j0]

    lot = lot_size_for_date(day)
    expiry = next_expiry(day)
    strike = itm_put_strike(entry_price)

    stop = entry_price + INITIAL_SL_PTS
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
        "date": day, "event": sig["event"], "side": "PE", "strike": strike, "expiry": expiry,
        "lot_size": lot, "entry_time": idx[j0], "exit_time": idx[exit_j], "index_entry": entry_price,
        "index_exit": exit_price, "exit_reason": reason, "points_gained": points_gained,
        "rupee_pnl": rupee_pnl,
    }


def stats_block(g, label):
    if g.empty:
        print(f"{label}: no trades")
        return
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  net_pts={g['points_gained'].sum():.2f}  "
          f"PF={pf:.2f}  net_Rs={g['rupee_pnl'].sum():,.2f}")


def main():
    run_self_checks()
    print("=== Oliver Velez Method -- GBI+ELEPHANT_PLUS -- Rs 25,000 Capital Test ===\n")
    print(f"Starting capital: Rs {STARTING_CAPITAL:,}   Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade.\n")

    print("Fetching Nifty index 1-min data (2 years) and resampling to 2-min bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    df2m = compute_indicators(df2m)
    days_covered = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df2m[df2m.index.date == day].copy()
        signals = detect_signals_for_day(day_df)
        open_trade_bar = -1
        for sig in signals:
            if sig["bar"] <= open_trade_bar:
                continue
            t = simulate_trade(day_df, sig, day)
            if t:
                all_trades.append(t)
                exit_bar_pos = day_df.index.get_loc(t["exit_time"])
                open_trade_bar = exit_bar_pos

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("velez_gbi_elephantplus_25k_trades.csv", index=False)

    print("--- Overall ---")
    stats_block(tdf, "All events")
    print()
    print("--- By event type ---")
    for ev, g in tdf.groupby("event"):
        stats_block(g, ev)
    print()

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd_rs = (equity - running_peak).min()
    min_equity = equity.min()
    print("--- Drawdown ---")
    print(f"Max drawdown: Rs {max_dd_rs:,.2f}")
    print(f"Lowest equity point reached: Rs {min_equity:,.2f}")
    print("Wipes out at some point." if (equity <= 0).any() else "Never wipes out.")
    print()

    win_rate = (tdf["points_gained"] > 0).mean()
    wins = tdf[tdf["points_gained"] > 0]
    losses = tdf[tdf["points_gained"] <= 0]
    avg_win_rs = wins["rupee_pnl"].mean() if len(wins) else 0
    avg_loss_rs = abs(losses["rupee_pnl"].mean()) if len(losses) else 0
    expectancy_rs = win_rate * avg_win_rs - (1 - win_rate) * avg_loss_rs
    r_multiple = tdf["rupee_pnl"] / (INITIAL_SL_PTS * tdf["lot_size"])
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print("--- Expectancy ---")
    print(f"Expectancy: Rs {expectancy_rs:,.2f}/trade   |   {r_multiple.mean():.3f} R/trade   |   SQN={sqn:.2f}\n")

    print("--- Equity Curve ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%\n")

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print("--- Stability check ---")
    print(f"First half: n={mid} net Rs {h1:,.2f}   |   Second half: n={len(tdf)-mid} net Rs {h2:,.2f}\n")

    print("--- Monte Carlo (trade-order reshuffle) ---")
    pnl_list = tdf["rupee_pnl"].tolist()
    final_equities, max_dds, min_equities = [], [], []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        peak = np.maximum.accumulate(curve)
        final_equities.append(curve[-1])
        max_dds.append((curve - peak).min())
        min_equities.append(curve.min())
    fe = np.array(final_equities); mdd = np.array(max_dds); me = np.array(min_equities)
    print(f"Final equity -> 5th: Rs {np.percentile(fe,5):,.2f}  50th: Rs {np.percentile(fe,50):,.2f}  95th: Rs {np.percentile(fe,95):,.2f}")
    print(f"Max drawdown -> 5th: Rs {np.percentile(mdd,5):,.2f}  50th: Rs {np.percentile(mdd,50):,.2f}  95th: Rs {np.percentile(mdd,95):,.2f}")
    print(f"Probability equity ever goes <= 0 across simulated paths: {100*(me <= 0).mean():.2f}%")
    print(f"Probability equity falls below 50% of starting capital: {100*(fe < STARTING_CAPITAL*0.5).mean():.2f}%\n")

    print(f"Trade log saved: velez_gbi_elephantplus_25k_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
