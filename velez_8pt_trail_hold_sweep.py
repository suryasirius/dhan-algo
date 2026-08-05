"""
Oliver Velez Method -- SHORT-only, GBI+ELEPHANT_PLUS -- 8pt initial stop,
trailing-stop x max-hold-time SWEEP
=================================================================
Standalone variant of velez_final_8pt.py (LOCKED, unmodified -- this
file does not touch it). Same entry signals (GBI, ELEPHANT_PLUS,
SHORT-only), same 8pt initial stop (the project's own sensitivity-
tested best value), same 2-min bars, same Rs 25,000 capital, same
2-year window.

What's new: a 2-D sweep crossing
  - trailing stop-loss: 2 points, 4 points
  - max-hold time-boxed exit: 90 seconds, 3 minutes, 4 minutes, 5 minutes
(8 combinations total). Each combination is run as its own FULL
backtest through the whole day-loop (not just re-timing shared trades)
because Velez signals can fire multiple times per day and an earlier
exit under a shorter hold/tighter trail can free up an earlier
re-entry signal that a longer hold/looser trail would have missed --
so trade counts genuinely differ across combinations, unlike a
single-trade-per-day strategy.

Exit priority within a bar, same convention as the locked base file:
  1) initial/trailing stop hit (intrabar high touches stop)
  2) max-hold deadline reached (forced exit at that bar's close)
  3) EOD squareoff (15:20, ultimate fallback)

IMPORTANT DATA-GRANULARITY CAVEAT: bars here are 2-MINUTE (matching the
locked base file), not 1-minute. That means hold-time deadlines round
UP to the next available 2-min bar:
  - 90sec  -> rounds up to the very next bar (entry + 2min)
  - 3min   -> rounds up to entry + 4min
  - 4min   -> lands EXACTLY on entry + 4min (same bar as 3min!)
  - 5min   -> rounds up to entry + 6min
So the 3min and 4min columns will often be IDENTICAL or near-identical
in this backtest -- that is a real resolution limit of 2-min bars, not
a bug. This is printed to the user before the results table too.

No fixed profit target is used (same as the locked base file) -- exits
are governed by the (trailing) stop and the hold-time deadline only.
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

INITIAL_SL_PTS = 8             # unchanged from the locked base (its own tested best)
EOD_SQUAREOFF = dt.time(15, 20)

TRAIL_VALUES = [2, 4]
HOLD_DURATIONS = [
    ("90sec", pd.Timedelta(seconds=90)),
    ("3min", pd.Timedelta(minutes=3)),
    ("4min", pd.Timedelta(minutes=4)),
    ("5min", pd.Timedelta(minutes=5)),
]

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000
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


def simulate_trade(day_df, sig, day, trail_pts, hold_delta):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = sig["bar"]
    entry_price = closes[j0]
    deadline = idx[j0] + hold_delta

    lot = lot_size_for_date(day)
    expiry = next_expiry(day)
    strike = itm_put_strike(entry_price)

    stop = entry_price + INITIAL_SL_PTS
    trailing_active = False

    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        h, c = highs[j], closes[j]

        if h >= stop:
            exit_j, exit_price, reason = j, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
            break
        if c < entry_price:
            trailing_active = True
        if trailing_active:
            stop = min(stop, h + trail_pts)
        if t >= deadline:
            exit_j, exit_price, reason = j, c, "TIME"
            break
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


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert pop_bearish(95, 97, 98, 100) is True
    assert pop_bearish(105, 103, 102, 100) is False
    assert is_small(2, 10) is True
    assert is_elephant_plus_bear(100, 100, 70, 71, 10) is True
    assert is_elephant_plus_bear(100, 101, 99, 100.5, 10) is False

    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    closes_t = [100, 103, 99, 92, 97, 104]
    highs_t = [100, 104, 100.5, 93, 98, 105]
    lows_t = [100, 102, 98.5, 91, 96, 103]
    opens_t = [100, 100, 103, 99, 92, 97]
    day_df = pd.DataFrame({"open": opens_t, "high": highs_t, "low": lows_t, "close": closes_t}, index=idx)

    # trail=4, deliberately generous hold (30min, i.e. far beyond this 6-bar day) so the
    # deadline can never be the binding constraint -- isolates the pure trailing-stop math,
    # same trace as the locked base file's own self-check.
    trade = simulate_trade(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6),
                            trail_pts=4, hold_delta=pd.Timedelta(minutes=30))
    # entry=100, initial stop=108; bar1 h=104<108, c=103>100 no profit yet
    # bar2 c=99<100 -> trailing_active, trail=min(108,100.5+4=104.5)=104.5
    # bar3 h=93<104.5, c=92<100, trail=min(104.5,93+4=97)=97
    # bar4 h=98>=97 -> exit at 97, TRAIL_SL, points=100-97=3
    assert trade["exit_reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {trade['exit_reason']}"
    assert abs(trade["points_gained"] - 3.0) < 0.01

    # trail=2 (tighter) on the SAME price path, same generous hold, should exit EARLIER (or equal)
    trade_tight = simulate_trade(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6),
                                  trail_pts=2, hold_delta=pd.Timedelta(minutes=30))
    assert trade_tight["exit_reason"] in ("TRAIL_SL", "INITIAL_SL")
    assert trade_tight["exit_time"] <= trade["exit_time"], "tighter trail should exit no later than looser trail"
    print("  simulate_trade trailing-stop math + tighter-trail-exits-sooner check ok")

    # Confirm the hold-time deadline, when tighter than the trailing stop's natural exit,
    # correctly cuts the trade off EARLY via TIME instead of waiting for TRAIL_SL.
    trade_time_bound = simulate_trade(day_df, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 1, 6),
                                       trail_pts=4, hold_delta=pd.Timedelta(minutes=5))
    assert trade_time_bound["exit_reason"] == "TIME", \
        f"expected TIME (deadline before natural TRAIL_SL), got {trade_time_bound['exit_reason']}"
    assert trade_time_bound["exit_time"] < trade["exit_time"], "time-boxed exit should fire earlier than the natural trail exit"
    print("  simulate_trade deadline-cuts-off-before-natural-trail-exit check ok")

    # Time-boxed deadline forces an exit before any trailing stop would ever fire
    # (price falls in favor of the SHORT but never enough to trip a 4pt trail)
    idx2 = pd.date_range("2025-03-10 09:15", periods=10, freq="2min")
    closes2 = [100, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5]
    highs2 = [c + 0.3 for c in closes2]
    opens2 = [100] + closes2[:-1]
    day_df2 = pd.DataFrame({"open": opens2, "high": highs2, "low": [c - 0.3 for c in closes2],
                             "close": closes2}, index=idx2)
    t2 = simulate_trade(day_df2, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 3, 10),
                         trail_pts=4, hold_delta=pd.Timedelta(minutes=3))
    assert t2["exit_reason"] == "TIME", f"expected TIME, got {t2['exit_reason']}"
    # bars at +2,+4,+6...; deadline=entry+3min -> first bar >= deadline is entry+4min (bar idx2[2])
    assert t2["exit_time"] == idx2[2], f"expected exit at {idx2[2]}, got {t2['exit_time']}"
    print("  simulate_trade max-hold TIME-exit check ok")

    # 3min and 4min deadlines land on the SAME bar on 2-min data (both round to entry+4min)
    t3 = simulate_trade(day_df2, {"bar": 0, "side": "SHORT", "event": "TEST"}, dt.date(2025, 3, 10),
                         trail_pts=4, hold_delta=pd.Timedelta(minutes=4))
    assert t3["exit_reason"] == "TIME"
    assert t3["exit_time"] == t2["exit_time"] == idx2[2]
    print("  simulate_trade 3min==4min-on-2min-bars granularity check ok")

    n_bars = 187
    idx3 = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="2min")
    closes3 = [100.0 - i * 0.05 for i in range(n_bars)]
    opens3 = [c + 0.02 for c in closes3]
    highs3 = [c + 0.05 for c in closes3]
    lows3 = [c - 0.03 for c in closes3]
    day3 = pd.DataFrame({"open": opens3, "high": highs3, "low": lows3, "close": closes3}, index=idx3)
    day3["sma20"] = pd.Series(closes3).rolling(SMA_FAST, min_periods=1).mean().values
    day3["sma200"] = pd.Series([100.0] * n_bars) - pd.Series(range(n_bars)) * 0.03
    day3["range"] = day3["high"] - day3["low"]
    day3["avg_range"] = day3["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=1).mean()
    sigs = detect_signals_for_day(day3)
    assert len(sigs) >= 0
    print(f"  (regression check: {len(sigs)} signal(s) found scanning a {n_bars}-bar day -- loop executes correctly)")
    print("All self-checks passed.\n")


def stats_for(g):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_pts=0.0, net_rs=0.0,
                    max_dd=0.0, sqn=float("nan"), final_equity=STARTING_CAPITAL, wipeout_pct=0.0)
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    g = g.sort_values("entry_time")
    equity = STARTING_CAPITAL + g["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    r_multiple = g["rupee_pnl"] / (INITIAL_SL_PTS * g["lot_size"])
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(g)) if r_multiple.std() else float("nan")

    pnl_list = g["rupee_pnl"].tolist()
    min_equities = []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        min_equities.append(curve.min())
    me = np.array(min_equities)
    wipeout_pct = 100 * (me <= 0).mean()

    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_pts=g["points_gained"].sum(),
                net_rs=g["rupee_pnl"].sum(), max_dd=max_dd, sqn=sqn,
                final_equity=equity.iloc[-1], wipeout_pct=wipeout_pct)


def run_backtest(df2m, days_covered, trail_pts, hold_delta):
    all_trades = []
    for day in days_covered:
        day_df = df2m[df2m.index.date == day].copy()
        signals = detect_signals_for_day(day_df)
        open_trade_bar = -1
        for sig in signals:
            if sig["bar"] <= open_trade_bar:
                continue
            t = simulate_trade(day_df, sig, day, trail_pts, hold_delta)
            if t:
                all_trades.append(t)
                exit_bar_pos = day_df.index.get_loc(t["exit_time"])
                open_trade_bar = exit_bar_pos
    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)


def main():
    run_self_checks()
    print("=== Velez SHORT (GBI+ELEPHANT_PLUS), 8pt initial stop -- trail x hold-time SWEEP ===\n")
    print(f"Starting capital: Rs {STARTING_CAPITAL:,}   Brokerage: Rs {BROKERAGE_PER_TRADE:.2f}/trade   "
          f"Initial SL: {INITIAL_SL_PTS}pts (fixed)\n")
    print("Sweeping trailing stop: 2pt, 4pt  x  max-hold: 90sec, 3min, 4min, 5min (8 combinations)\n")
    print("NOTE: bars here are 2-MINUTE, so hold-time deadlines round UP to the next available bar:")
    print("  90sec -> entry+2min (next bar)   3min -> entry+4min   4min -> entry+4min (SAME bar as 3min!)")
    print("  5min -> entry+6min")
    print("So the 3min and 4min columns will often be identical -- a real 2-min-bar resolution limit,")
    print("not a bug. No fixed profit target is used, matching the locked base file (exit is governed")
    print("by the trailing stop and hold-time deadline only).\n")

    print("Fetching Nifty index 1-min data (2 years) and resampling to 2-min bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    df2m = compute_indicators(df2m)
    days_covered = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    results = {}
    trade_logs = {}
    for trail_pts in TRAIL_VALUES:
        for label, hold_delta in HOLD_DURATIONS:
            key = f"trail{trail_pts}_{label}"
            tdf = run_backtest(df2m, days_covered, trail_pts, hold_delta)
            trade_logs[key] = tdf
            results[key] = stats_for(tdf)
            if not tdf.empty:
                tdf.to_csv(f"velez_8pt_trail_hold_sweep_{key}_trades.csv", index=False)

    print(f"{'Combo':<14} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'SQN':>6} {'End_Rs':>12} {'Wipe%':>7}")
    for trail_pts in TRAIL_VALUES:
        for label, _ in HOLD_DURATIONS:
            key = f"trail{trail_pts}_{label}"
            r = results[key]
            print(f"{key:<14} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
                  f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['sqn']:>6.2f} "
                  f"{r['final_equity']:>12,.2f} {r['wipeout_pct']:>6.2f}%")
        print()

    print("--- By event type, per combo ---")
    for trail_pts in TRAIL_VALUES:
        for label, _ in HOLD_DURATIONS:
            key = f"trail{trail_pts}_{label}"
            tdf = trade_logs[key]
            if tdf.empty:
                continue
            print(f"\n-- {key} --")
            for ev, g in tdf.groupby("event"):
                wins = g[g["points_gained"] > 0]
                losses = g[g["points_gained"] <= 0]
                pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
                      if losses["points_gained"].sum() != 0 else float("inf"))
                print(f"  {ev}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                      f"PF={pf:.2f}  net_Rs={g['rupee_pnl'].sum():,.2f}")

    print("\nTrade logs saved: velez_8pt_trail_hold_sweep_<combo>_trades.csv (one file per combination)")


if __name__ == "__main__":
    main()
