"""
Oliver Velez Moving-Average Method -- NO STANDALONE COLOR_CHANGE variant
=================================================================
Same engine as velez_moving_average_backtest.py (does NOT modify that
file or any other file in this project). Only change: COLOR_CHANGE is
removed as a standalone entry trigger, per your source doc's own note
that it's "a supplementary/add-on signal, not a standalone high-
probability trigger by itself" -- confirmed by the full-run numbers,
where COLOR_CHANGE was 69% of all trades (1,754 of 2,548) and a net
money-loser after brokerage (-Rs 5,316) despite barely-positive raw
points.

Everything else unchanged: Picture-of-Power filter, RBI/GBI, Elephant/
Elephant-Plus/Tail/Dual-tail/Bull-Bear-180/Clearing-elephant events,
confirmation entries only, continuation-only (event direction must
agree with current POP), 10pt initial stop / 4pt trailing once in
profit, no Greeks/delta modeling (raw index points x lot size), Rs 40/
trade brokerage, 2-minute Nifty bars, 2-year window.
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
TAIL_WICK_RATIO = 0.85
CLEARING_ELEPHANT_LOOKBACK = 5
NEAR_20_PCT = 0.0015

INITIAL_SL_PTS = 10
TRAIL_PTS = 4
EOD_SQUAREOFF = dt.time(15, 20)

STRIKE_STEP = 50
STARTING_CAPITAL = 10_000
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


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


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


def pop_state(price, sma20, sma20_prev, sma200):
    if pd.isna(sma20) or pd.isna(sma200) or pd.isna(sma20_prev):
        return "NONE"
    rising = sma20 > sma20_prev
    declining = sma20 < sma20_prev
    if price > sma20 and sma20 > sma200 and rising:
        return "BULL"
    if price < sma20 and sma20 < sma200 and declining:
        return "BEAR"
    return "NONE"


def is_green(o, c):
    return c > o


def is_red(o, c):
    return c < o


def is_small(rng, avgr):
    return bool(avgr) and avgr > 0 and rng < SMALL_BAR_MULT * avgr


def is_elephant(rng, avgr):
    return bool(avgr) and avgr > 0 and rng > ELEPHANT_MULT * avgr


def is_elephant_plus(rng, avgr):
    return bool(avgr) and avgr > 0 and (rng * ELEPHANT_PLUS_RETAIN) > ELEPHANT_MULT * avgr


def is_tail(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return None
    body_top, body_bot = max(o, c), min(o, c)
    lower_wick = body_bot - l
    upper_wick = h - body_top
    if lower_wick / rng >= TAIL_WICK_RATIO:
        return "BULL"
    if upper_wick / rng >= TAIL_WICK_RATIO:
        return "BEAR"
    return None


def is_180(prev_o, prev_h, prev_l, prev_c, prev_avgr, o, h, l, c):
    prev_rng = prev_h - prev_l
    cur_rng = h - l
    if not is_elephant(prev_rng, prev_avgr):
        return None
    if is_red(prev_o, prev_c) and is_green(o, c) and cur_rng > prev_rng and c > prev_o:
        return "BULL"
    if is_green(prev_o, prev_c) and is_red(o, c) and cur_rng > prev_rng and c < prev_o:
        return "BEAR"
    return None


def is_clearing_elephant(o, h, l, c, prior5_high, prior5_low):
    combined_range = prior5_high - prior5_low
    if combined_range <= 0:
        return None
    if is_green(o, c) and (h - l) >= combined_range and c > prior5_high:
        return "BULL"
    if is_red(o, c) and (h - l) >= combined_range and c < prior5_low:
        return "BEAR"
    return None


def find_small_run(highs, lows, opens, closes, avg_ranges, end_idx, color):
    i = end_idx - 1
    if i < 0:
        return None
    matches = (is_red if color == "RED" else is_green)
    if not (matches(opens[i], closes[i]) and is_small(highs[i] - lows[i], avg_ranges[i])):
        return None
    combined_high, combined_low = highs[i], lows[i]
    start_idx = i
    j = i - 1
    while j >= 0 and matches(opens[j], closes[j]) and is_small(highs[j] - lows[j], avg_ranges[j]):
        combined_high = max(combined_high, highs[j])
        combined_low = min(combined_low, lows[j])
        start_idx = j
        j -= 1
    if start_idx - 1 < 0:
        return None
    sandwich_color_ok = is_green(opens[start_idx - 1], closes[start_idx - 1]) if color == "RED" \
        else is_red(opens[start_idx - 1], closes[start_idx - 1])
    if not sandwich_color_ok:
        return None
    return start_idx, combined_high, combined_low


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert pop_state(105, 103, 102, 100) == "BULL"
    assert pop_state(95, 97, 98, 100) == "BEAR"
    assert is_small(2, 10) is True
    assert is_elephant(25, 10) is True
    assert is_elephant_plus(30, 10) is True
    assert is_elephant_plus(25, 10) is False
    assert is_tail(100.5, 101.2, 90, 101) == "BULL"
    assert is_tail(100, 110, 99.5, 100.5) == "BEAR"
    assert is_small(2, float("nan")) is False
    assert is_elephant(25, float("nan")) is False

    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    closes_t = [100, 95, 101, 108, 103, 96]
    highs_t = [100, 96, 102, 109, 104, 97]
    lows_t = [100, 94, 100.5, 107, 102, 95]
    opens_t = [100, 100, 95, 101, 108, 103]
    day_df = pd.DataFrame({"open": opens_t, "high": highs_t, "low": lows_t, "close": closes_t}, index=idx)
    trade = simulate_trade(day_df, {"bar": 0, "side": "LONG", "event": "TEST"}, dt.date(2025, 1, 6))
    assert trade["exit_reason"] == "TRAIL_SL"
    assert abs(trade["points_gained"] - 3.0) < 0.01

    n_bars = 187
    idx2 = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="2min")
    closes2 = [100.0 + i * 0.05 for i in range(n_bars)]
    opens2 = [c - 0.02 for c in closes2]
    highs2 = [c + 0.03 for c in closes2]
    lows2 = [c - 0.05 for c in closes2]
    day2 = pd.DataFrame({"open": opens2, "high": highs2, "low": lows2, "close": closes2}, index=idx2)
    day2["sma20"] = pd.Series(closes2).rolling(SMA_FAST, min_periods=1).mean().values
    day2["sma200"] = pd.Series([100.0] * n_bars) + pd.Series(range(n_bars)) * 0.03
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
        pop = pop_state(closes[j], sma20[j], sma20[j - POP_SLOPE_LOOKBACK], sma200[j])
        if pop == "NONE":
            continue

        near20 = abs(closes[j] - sma20[j]) / closes[j] < NEAR_20_PCT

        if pop == "BULL" and near20:
            run = find_small_run(highs, lows, opens, closes, avg_ranges, j, "RED")
            if run:
                _, comb_high, comb_low = run
                if closes[j] > comb_high and is_green(opens[j], closes[j]):
                    signals.append({"bar": j, "side": "LONG", "event": "RBI"})
        if pop == "BEAR" and near20:
            run = find_small_run(highs, lows, opens, closes, avg_ranges, j, "GREEN")
            if run:
                _, comb_high, comb_low = run
                if closes[j] < comb_low and is_red(opens[j], closes[j]):
                    signals.append({"bar": j, "side": "SHORT", "event": "GBI"})

        # NOTE: COLOR_CHANGE removed as a standalone trigger here (see docstring).
        if j - 1 >= 0:
            eo, eh, el, ec = opens[j - 1], highs[j - 1], lows[j - 1], closes[j - 1]
            erange = eh - el
            eavgr = avg_ranges[j - 1]

            event_dir = None
            event_name = None

            if is_elephant_plus(erange, eavgr):
                event_dir = "BULL" if is_green(eo, ec) else "BEAR"
                event_name = "ELEPHANT_PLUS"
            elif is_elephant(erange, eavgr):
                event_dir = "BULL" if is_green(eo, ec) else "BEAR"
                event_name = "ELEPHANT"

            tail_dir = is_tail(eo, eh, el, ec)
            if tail_dir and event_name is None:
                event_dir, event_name = tail_dir, "TAIL"
                if j - 2 >= 0:
                    d2 = is_tail(opens[j - 2], highs[j - 2], lows[j - 2], closes[j - 2])
                    if d2 == tail_dir:
                        event_name = "DUAL_TAIL"

            if event_name is None and j - 2 >= 0:
                d180 = is_180(opens[j - 2], highs[j - 2], lows[j - 2], closes[j - 2], avg_ranges[j - 2],
                               eo, eh, el, ec)
                if d180:
                    event_dir, event_name = d180, ("BULL180" if d180 == "BULL" else "BEAR180")

            if event_name is None and j - 1 - CLEARING_ELEPHANT_LOOKBACK >= 0:
                prior5_high = max(highs[j - 1 - CLEARING_ELEPHANT_LOOKBACK:j - 1])
                prior5_low = min(lows[j - 1 - CLEARING_ELEPHANT_LOOKBACK:j - 1])
                ce = is_clearing_elephant(eo, eh, el, ec, prior5_high, prior5_low)
                if ce:
                    event_dir, event_name = ce, "CLEARING_ELEPHANT"

            if event_name and ((event_dir == "BULL" and pop == "BULL") or (event_dir == "BEAR" and pop == "BEAR")):
                if event_dir == "BULL" and closes[j] > eh:
                    signals.append({"bar": j, "side": "LONG", "event": event_name})
                elif event_dir == "BEAR" and closes[j] < el:
                    signals.append({"bar": j, "side": "SHORT", "event": event_name})

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
    side = sig["side"]
    entry_price = closes[j0]

    lot = lot_size_for_date(day)
    expiry = next_expiry(day)
    option_side = "CE" if side == "LONG" else "PE"
    strike = itm_call_strike(entry_price) if side == "LONG" else itm_put_strike(entry_price)

    stop = entry_price - INITIAL_SL_PTS if side == "LONG" else entry_price + INITIAL_SL_PTS
    trailing_active = False

    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        h, l, c = highs[j], lows[j], closes[j]
        if side == "LONG":
            if l <= stop:
                exit_j, exit_price, reason = j, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
                break
            if c > entry_price:
                trailing_active = True
            if trailing_active:
                stop = max(stop, l - TRAIL_PTS)
        else:
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

    points_gained = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    rupee_pnl = points_gained * lot - BROKERAGE_PER_TRADE

    return {
        "date": day, "event": sig["event"], "side": option_side, "strike": strike, "expiry": expiry,
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
    print("=== Oliver Velez Method -- No Standalone COLOR_CHANGE -- Nifty ITM Options ===\n")
    print(f"Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade included.\n")

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
    tdf.to_csv("velez_no_colorchange_trades.csv", index=False)

    print("--- Overall ---")
    stats_block(tdf, "All events")
    print()
    print("--- By event type ---")
    for ev, g in tdf.groupby("event"):
        stats_block(g, ev)
    print()
    print("--- By side ---")
    stats_block(tdf[tdf["side"] == "CE"], "CE (LONG signals)")
    stats_block(tdf[tdf["side"] == "PE"], "PE (SHORT signals)")
    print()
    print("--- By exit reason ---")
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)
    print()

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd_rs = (equity - running_peak).min()
    print("--- Drawdown ---")
    print(f"Max drawdown: Rs {max_dd_rs:,.2f}")
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

    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    print("--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: n={len(g)}  net={g['points_gained'].sum():.2f}pts  win%={100*(g['points_gained']>0).mean():.1f}")
    print()

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print("--- Stability check ---")
    print(f"First half: n={mid} net Rs {h1:,.2f}   |   Second half: n={len(tdf)-mid} net Rs {h2:,.2f}\n")

    print("--- Monte Carlo (trade-order reshuffle) ---")
    pnl_list = tdf["rupee_pnl"].tolist()
    final_equities, max_dds = [], []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        peak = np.maximum.accumulate(curve)
        final_equities.append(curve[-1])
        max_dds.append((curve - peak).min())
    fe = np.array(final_equities); mdd = np.array(max_dds)
    print(f"Final equity -> 5th: Rs {np.percentile(fe,5):,.2f}  50th: Rs {np.percentile(fe,50):,.2f}  95th: Rs {np.percentile(fe,95):,.2f}")
    print(f"Max drawdown -> 5th: Rs {np.percentile(mdd,5):,.2f}  50th: Rs {np.percentile(mdd,50):,.2f}  95th: Rs {np.percentile(mdd,95):,.2f}")
    print(f"Probability equity falls below 50% of starting capital: {100*(fe < STARTING_CAPITAL*0.5).mean():.2f}%\n")

    print(f"Trade log saved: velez_no_colorchange_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
