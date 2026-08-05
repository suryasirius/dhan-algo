"""
Fixed-Strike ORB -- Target x Time-Exit GRID sweep, 2-year backtest
=================================================================
Same base strategy throughout this whole family of scripts: one fixed
nearest-ITM CE strike per day, mark the opening 5-minute (09:15-09:20)
premium high/low, enter on the first confirmed 1-min close above that
high, one trade per day, no re-entry.

This script runs the ENTIRE grid you asked for in one pass, using the
SAME entries across every combo (the entry signal doesn't depend on
the exit rule, so this is apples-to-apples -- same convention as the
original hold-duration timesweep script):

  Time exit:   2 minutes, 3 minutes
  Target:      3.5, 4, 5, 6 points (on the premium)

  -> 2 x 4 = 8 combinations, each exits at its own target OR its own
     time deadline, whichever comes first. NO stop-loss in any combo
     (matches the doc as written -- the time clock is the only
     downside brake).

PROXY PREMIUM MODE: same convention as every other option script in
this project (delta=0.6, floored at 0) -- real Dhan option-premium
data is still blocked (DH-901).

2-year window (this project's default backtest length -- say the word
if you want this re-run on 1 year or 6 months instead, same as the
earlier single-combo scripts).
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OR_CANDLES = 5
STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)

HOLD_MINUTES = [2, 3]
TARGET_GRID = [3.5, 4.0, 5.0, 6.0]

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


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


def build_ce_premium_series(day_df, strike, day_open_spot):
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    intrinsic0 = max(day_open_spot - strike, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE

    p_open = np.maximum(base + (idx_open - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + (idx_close - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (idx_high - day_open_spot) * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + (idx_low - day_open_spot) * ASSUMED_DELTA, 0.0)

    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close},
                         index=day_df.index)


def find_entry_for_day(prem_df):
    """Single confirmed close above the opening 5-min high. Returns only
    the FIRST such bar -- one trade per day, no re-entry. Independent of
    the exit rule, so it's computed ONCE per day and reused across every
    (hold, target) combo below."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade(prem_df, entry_idx, day, target_pts, hold_minutes):
    """Conservative ordering within a bar: TARGET is checked before the
    time deadline on any bar where both would technically be
    satisfiable. No stop-loss."""
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + target_pts
    deadline = idx[entry_idx] + pd.Timedelta(minutes=hold_minutes)

    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
            break
        if t >= deadline:
            exit_i, exit_premium, reason = i, closes[i], f"TIME_{hold_minutes}MIN"
            break
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, closes[i], "EOD"
            break

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {
        "date": day, "entry_time": idx[entry_idx], "exit_time": idx[exit_i],
        "entry_premium": entry_premium, "exit_premium": exit_premium,
        "exit_reason": reason, "points_gained": points, "rupee_pnl": rupee_pnl,
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    rng = pd.date_range("2026-01-06 09:15", periods=12, freq="1min")

    # Target hit cleanly, well before ANY hold deadline (test across the
    # full target grid and both hold durations).
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110, 111, 112]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            t = simulate_trade(prem_df, entry_idx=5, day=dt.date(2026, 1, 6), target_pts=tgt, hold_minutes=hold)
            assert t["exit_reason"] == "TARGET", f"[hold={hold} tgt={tgt}] expected TARGET, got {t['exit_reason']}"
            assert abs(t["points_gained"] - tgt) < 0.01
    print("  simulate_trade TARGET-hit-before-any-deadline check ok (full grid)")

    # Target never reached -> each (hold, any target) combo must force a
    # TIME exit at the first candle whose timestamp is >= entry+hold.
    close2 = [100, 100, 100, 100, 100, 100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2]
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [c + 0.3 for c in close2], "low": [c - 0.3 for c in close2],
        "close": close2,
    }, index=rng)
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            t2 = simulate_trade(prem_df2, entry_idx=5, day=dt.date(2026, 1, 6), target_pts=tgt, hold_minutes=hold)
            assert t2["exit_reason"] == f"TIME_{hold}MIN", \
                f"[hold={hold} tgt={tgt}] expected TIME_{hold}MIN, got {t2['exit_reason']}"
            assert t2["exit_time"] == rng[5] + pd.Timedelta(minutes=hold)
    print("  simulate_trade TIME-exit-when-target-never-hit check ok (full grid)")

    # Target and the deadline both satisfiable on the SAME bar -> TARGET
    # must win, for the smallest target (3.5, easiest to trigger early)
    # and the shortest hold (2min, tightest deadline) -- the tightest
    # possible ambiguous case.
    close3 = [100, 100, 100, 100, 100, 100, 100.1, 108.0, 108.0, 108.0, 108.0, 108.0]
    prem_df3 = pd.DataFrame({
        "open": [100] + close3[:-1], "high": [c + 0.1 for c in close3], "low": [c - 0.3 for c in close3],
        "close": close3,
    }, index=rng)
    t3 = simulate_trade(prem_df3, entry_idx=5, day=dt.date(2026, 1, 6), target_pts=3.5, hold_minutes=2)
    assert t3["exit_reason"] == "TARGET", f"expected TARGET to win on the ambiguous bar, got {t3['exit_reason']}"
    assert t3["exit_time"] == rng[7]  # bar index 5+2=7, exactly the 2-min deadline bar
    print("  simulate_trade same-bar target-vs-deadline ordering check ok (tightest combo)")

    # Single-trade-per-day entry detection.
    close4 = [100] * 5 + [101, 99, 98, 97, 105, 96, 95, 110, 94, 93]
    prem_df4 = pd.DataFrame({
        "open": [100] + close4[:-1], "high": [c + 0.5 for c in close4], "low": [c - 0.5 for c in close4],
        "close": close4,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx, day_high = find_entry_for_day(prem_df4)
    assert day_high == 100.5
    assert entry_idx == 5
    print("  find_entry_for_day single-trade-per-day check ok")

    # No breakout at all -> no entry.
    close5 = [100] * 15
    prem_df5 = pd.DataFrame({
        "open": [100] + close5[:-1], "high": [100.5] * 15, "low": [99.5] * 15, "close": close5,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx5, _ = find_entry_for_day(prem_df5)
    assert entry_idx5 is None
    print("  find_entry_for_day no-breakout-no-trade check ok")

    print("All self-checks passed.\n")


def stats_for(g, target_pts):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_pts=0.0, net_rs=0.0,
                    max_dd=0.0, sqn=float("nan"), final_equity=STARTING_CAPITAL)
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    g = g.sort_values("entry_time")
    equity = STARTING_CAPITAL + g["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    r = g["points_gained"] / target_pts
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_pts=g["points_gained"].sum(),
                net_rs=g["rupee_pnl"].sum(), max_dd=max_dd, sqn=sqn,
                final_equity=equity.iloc[-1])


def main():
    run_self_checks()
    print("=== Fixed-Strike ORB -- Target x Time-Exit GRID sweep -- 2-year backtest ===\n")
    print("Entry: confirmed 1-min close above opening 5-min premium high, one trade/day (same entries used for every combo below).")
    print(f"Grid: hold in {HOLD_MINUTES} minutes  x  target in {TARGET_GRID} points. NO stop-loss in any combo.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    # Find one entry per day (shared across every combo -- apples-to-apples).
    day_entries = []
    no_entry_days = 0
    for d in days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < OR_CANDLES + 1:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_call_strike(day_open_spot)
        prem_df = build_ce_premium_series(day_df, strike, day_open_spot)
        entry_idx, _ = find_entry_for_day(prem_df)
        if entry_idx is None:
            no_entry_days += 1
            continue
        day_entries.append((d, prem_df, entry_idx, strike))

    print(f"Days with a qualifying entry: {len(day_entries)} / {len(days)}  (no breakout on {no_entry_days} days)\n")

    results = {}
    trade_logs = {}
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            label = f"{tgt}pt_{hold}min"
            trades = []
            for d, prem_df, entry_idx, strike in day_entries:
                t = simulate_trade(prem_df, entry_idx, d, target_pts=tgt, hold_minutes=hold)
                t["strike"] = strike
                trades.append(t)
            tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
            trade_logs[label] = tdf
            results[label] = stats_for(tdf, tgt)
            tdf.to_csv(f"nifty_fixed_strike_grid_{label}_2yr_trades.csv", index=False)

    print(f"{'Combo':<14} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'SQN':>6} {'End_Rs':>12}")
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            label = f"{tgt}pt_{hold}min"
            r = results[label]
            print(f"{label:<14} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
                  f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['sqn']:>6.2f} "
                  f"{r.get('final_equity', STARTING_CAPITAL):>12,.2f}")

    print("\nStability check (in-sample/out-of-sample split) per combo:")
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            label = f"{tgt}pt_{hold}min"
            tdf = trade_logs[label]
            dates = pd.to_datetime(tdf["date"])
            mid = dates.min() + (dates.max() - dates.min()) / 2
            fh, sh = tdf[dates <= mid], tdf[dates > mid]
            m1, m2 = stats_for(fh, tgt), stats_for(sh, tgt)
            print(f"  {label:<14} first_half: net_Rs={m1['net_rs']:>10,.2f} PF={m1['pf']:>5.2f} SQN={m1['sqn']:>5.2f}  |  "
                  f"second_half: net_Rs={m2['net_rs']:>10,.2f} PF={m2['pf']:>5.2f} SQN={m2['sqn']:>5.2f}")

    print("\nPer-combo exit-reason breakdown:")
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            label = f"{tgt}pt_{hold}min"
            tdf = trade_logs[label]
            print(f"\n-- {label} --")
            for reason, g in tdf.groupby("exit_reason"):
                print(f"  {reason}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                      f"net_Rs={g['rupee_pnl'].sum():,.2f}  avg_pts={g['points_gained'].mean():.2f}")

    print("\nRed-flag check:")
    for hold in HOLD_MINUTES:
        for tgt in TARGET_GRID:
            label = f"{tgt}pt_{hold}min"
            r = results[label]
            if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
                print(f"  *** {label}: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny. ***")

    print("\nTrade logs saved: nifty_fixed_strike_grid_<target>pt_<hold>min_2yr_trades.csv (one file per combo)")


if __name__ == "__main__":
    main()
