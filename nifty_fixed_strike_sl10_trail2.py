"""
Fixed-Strike Opening-Range Breakout -- -10pt Initial Stop, -2pt Trailing
Standalone, 6-month backtest, plus 30-minute session-wise breakdown.
=================================================================
Third variant of the Fixed-Strike ORB family (nifty_fixed_strike_5pt_target.py
had no stop and wiped out; nifty_fixed_strike_5pt_target_sl3.py used a
fixed -3pt stop / +5pt target). This one replaces the fixed target with
a TRAILING stop, matching the same stop/trail convention used
throughout this project (ORB: 10/4, Velez final: 8/4):
  - Initial stop-loss: -10 points, fixed while flat or underwater.
  - Once the trade moves into profit, switch to a trailing stop of
    -2 points behind the current candle's low, tightening only, never
    loosening. This lets a winner run past what a fixed +5pt target
    would have captured, at the cost of giving back more before
    exiting on a reversal.
  - No separate fixed profit target -- exit is via the (trailing)
    stop or the EOD squareoff only.
Same call-only rule and same same-day re-entry rule (a fresh new-high
close after an exit re-enters) as the other Fixed-Strike variants.
Same fixed nearest-ITM strike picked once at market open.

NEW: a 30-minute session-wise breakdown (matching the "30-Minute
Session Analysis" module from the ORB spec doc used earlier in this
project) -- entries are bucketed into 09:15-09:45, 09:45-10:15, ...,
15:00-15:30 windows, with trade count / win rate / avg P&L / total
P&L reported per bucket, to see whether entries taken early vs. late
in the session behave differently.

PROXY PREMIUM MODE: same convention as the other option scripts in
this project (delta=0.6, floored at 0) -- real Dhan option-premium
data is still blocked (DH-901).
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=182)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OR_CANDLES = 5
STRIKE_STEP = 50
INITIAL_SL_PTS = 10.0
TRAIL_PTS = 2.0
EOD_SQUAREOFF = dt.time(15, 20)

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000

SESSION_START = dt.time(9, 15)
SESSION_END = dt.time(15, 30)


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


def simulate_trade(prem_df, entry_idx, day):
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    lows = prem_df["low"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]

    stop = entry_premium - INITIAL_SL_PTS
    trailing_active = False
    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"

    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        premium_low = lows[i]
        premium_now = closes[i]
        if premium_low <= stop:
            exit_i, exit_premium, reason = i, stop, ("TRAIL_SL" if trailing_active else "INITIAL_SL")
            break
        if premium_now > entry_premium:
            trailing_active = True
        if trailing_active:
            new_stop = premium_low - TRAIL_PTS
            stop = max(stop, new_stop)
        if t.time() >= EOD_SQUAREOFF:
            exit_i, exit_premium, reason = i, premium_now, "EOD"
            break

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {
        "date": day, "entry_time": idx[entry_idx], "exit_time": idx[exit_i],
        "entry_premium": entry_premium, "exit_premium": exit_premium,
        "exit_reason": reason, "points_gained": points, "rupee_pnl": rupee_pnl,
    }


def session_bucket(t):
    """Buckets a datetime.time into a 30-min session window label, e.g.
    '09:15-09:45'. Windows run from SESSION_START to SESSION_END."""
    anchor = dt.datetime.combine(dt.date(2000, 1, 1), SESSION_START)
    tt = dt.datetime.combine(dt.date(2000, 1, 1), t)
    minutes_in = (tt - anchor).total_seconds() / 60.0
    bucket_idx = int(minutes_in // 30)
    b_start = anchor + dt.timedelta(minutes=30 * bucket_idx)
    b_end = b_start + dt.timedelta(minutes=30)
    return f"{b_start.strftime('%H:%M')}-{b_end.strftime('%H:%M')}"


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    rng = pd.date_range("2026-01-06 09:15", periods=6, freq="1min")

    # Trailing-stop hand trace: entry=100, initial stop=90.
    # bar1: close=110 (in profit) -> trailing on, stop=max(90, low(109.5)-2=107.5)=107.5
    # bar2: low=108.0 -> not <=107.5, stop=max(107.5, 108.0-2=106.0)=107.5 (no loosening)
    # bar3: low=114.5 -> not <=107.5, stop=max(107.5, 114.5-2=112.5)=112.5
    # bar4: low=111.5 -> 111.5<=112.5 TRUE -> exit at stop=112.5
    close2 = [100, 110, 108.5, 115, 111.5, 103]
    low2 = [99.5, 109.5, 108.0, 114.5, 111.5, 102.5]
    prem_df2 = pd.DataFrame({
        "open": [100, 100, 110, 108.5, 115, 111.5], "high": [c + 0.5 for c in close2],
        "low": low2, "close": close2,
    }, index=rng)
    t2 = simulate_trade(prem_df2, entry_idx=0, day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {t2['exit_reason']}"
    assert abs(t2["exit_premium"] - 112.5) < 0.01, f"expected exit at 112.5, got {t2['exit_premium']}"
    print("  simulate_trade trailing-stop hand-trace check ok")

    # Initial stop-loss: never goes into profit, drops straight through -10.
    close3 = [100, 95, 89, 85]
    rng3 = pd.date_range("2026-01-06 09:15", periods=4, freq="1min")
    prem_df3 = pd.DataFrame({
        "open": [100, 100, 95, 89], "high": [c + 0.5 for c in close3],
        "low": [c - 0.5 for c in close3], "close": close3,
    }, index=rng3)
    t3 = simulate_trade(prem_df3, entry_idx=0, day=dt.date(2026, 1, 6))
    assert t3["exit_reason"] == "INITIAL_SL"
    assert abs(t3["exit_premium"] - 90.0) < 0.01, f"expected exit at 90.0 (100-10), got {t3['exit_premium']}"
    print("  simulate_trade initial-stop-loss check ok")

    # Session bucketing
    assert session_bucket(dt.time(9, 17)) == "09:15-09:45"
    assert session_bucket(dt.time(9, 45)) == "09:45-10:15"
    assert session_bucket(dt.time(15, 20)) == "15:15-15:45"
    print("  session_bucket check ok")

    print("All self-checks passed.\n")


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
    print("=== Fixed-Strike ORB -- -10pt initial stop / -2pt trailing, call-only, re-entry allowed ===\n")
    print(f"Initial stop: -{INITIAL_SL_PTS}pts  Trailing: -{TRAIL_PTS}pts once in profit  "
          f"No fixed target  Re-entry: allowed same day  Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    trades = []
    for d in days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < OR_CANDLES + 1:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_call_strike(day_open_spot)
        prem_df = build_ce_premium_series(day_df, strike, day_open_spot)

        closes = prem_df["close"].tolist()
        reference_high = prem_df["high"].iloc[:OR_CANDLES].max()
        running_high_seen = reference_high
        i = OR_CANDLES
        n = len(prem_df)
        while i < n:
            if closes[i] > running_high_seen:
                running_high_seen = closes[i]
                t = simulate_trade(prem_df, i, d)
                t["strike"] = strike
                trades.append(t)
                exit_pos = prem_df.index.get_loc(t["exit_time"])
                i = exit_pos + 1
                continue
            i += 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades generated.")
        return

    tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_fixed_strike_sl10_trail2_trades.csv", index=False)

    print(f"Total trades: {len(tdf)}\n")
    stats_block(tdf, "Overall")
    print()
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)

    trades_per_day = tdf.groupby("date").size()
    print(f"\nDays with at least 1 trade: {trades_per_day.shape[0]} / {len(days)}")
    print(f"Days with re-entries (2+ trades same day): {(trades_per_day >= 2).sum()}")
    print(f"Max trades in a single day: {trades_per_day.max()}")

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd = (equity - running_peak).min()
    print(f"\nMax drawdown: Rs {max_dd:,.2f}   {'WIPES OUT' if (equity<=0).any() else 'never wipes out'}")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Return: {100*(equity.iloc[-1]/STARTING_CAPITAL-1):.1f}%")

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print(f"Stability: first half n={mid} Rs{h1:,.2f}  |  second half n={len(tdf)-mid} Rs{h2:,.2f}")

    r = tdf["points_gained"] / INITIAL_SL_PTS
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    print(f"SQN: {sqn:.2f}")

    pnls = tdf["rupee_pnl"].to_numpy()
    rng_mc = np.random.RandomState(42)
    touches_zero = below_half = 0
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = rng_mc.permutation(pnls)
        eq = STARTING_CAPITAL + np.cumsum(shuffled)
        if (eq <= 0).any():
            touches_zero += 1
        if (eq <= STARTING_CAPITAL * 0.5).any():
            below_half += 1
    print(f"Monte Carlo ({MONTE_CARLO_ITERS} reshuffles): "
          f"{100*touches_zero/MONTE_CARLO_ITERS:.2f}% of paths touch <=0, "
          f"{100*below_half/MONTE_CARLO_ITERS:.2f}% fall below 50% of starting capital")

    # 30-minute session-wise breakdown
    tdf["session"] = tdf["entry_time"].dt.time.apply(session_bucket)
    print("\n=== 30-Minute Session-Wise Breakdown ===")
    session_order = sorted(tdf["session"].unique(), key=lambda s: s)
    for s in session_order:
        stats_block(tdf[tdf["session"] == s], s)

    print(f"\nTrade log saved: nifty_fixed_strike_sl10_trail2_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
