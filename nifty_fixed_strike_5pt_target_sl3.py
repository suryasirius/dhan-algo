"""
Fixed-Strike Opening-Range Breakout -- 5-Point Target, -3pt Stop-Loss
Standalone, 6-month backtest.
=================================================================
Same base strategy as nifty_fixed_strike_5pt_target.py (that file is
untouched), with ONE change: a fixed -3pt stop-loss added, replacing
the "no stop-loss, as literally written" version. That version proved
the doc's own flagged risk was real -- 91% win rate but a -46.8pt
average loss on the 80 trades/6mo that never reached target before EOD
wiped the account out (100% of Monte Carlo paths touched zero).

This variant risks 3 points to make 5 (a tighter risk:reward than the
doc's own suggested 10pt fallback stop) -- exit the moment premium
drops 3pts below entry, or hits the +5pt target, or EOD, whichever
comes first. Same call-only rule, same same-day re-entry rule (a
fresh new-high close after an exit re-enters), same fixed nearest-ITM
strike picked once at market open.

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
TARGET_PTS = 5.0
STOP_PTS = 3.0
EOD_SQUAREOFF = dt.time(15, 20)

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 10_000
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000


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
    highs = prem_df["high"].tolist()
    lows = prem_df["low"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + TARGET_PTS
    stop = entry_premium - STOP_PTS

    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        if lows[i] <= stop:
            exit_i, exit_premium, reason = i, stop, "STOP"
            break
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
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

    rng = pd.date_range("2026-01-06 09:15", periods=15, freq="1min")

    # Target-hit: steady rally exits at exactly entry+5.
    close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    t = simulate_trade(prem_df, entry_idx=0, day=dt.date(2026, 1, 6))
    assert t["exit_reason"] == "TARGET"
    assert abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade target-hit check ok")

    # Stop-hit: steady decline exits at exactly entry-3, bounded (not worse).
    close2 = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86]
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [c + 0.5 for c in close2], "low": [c - 0.5 for c in close2],
        "close": close2,
    }, index=rng)
    t2 = simulate_trade(prem_df2, entry_idx=0, day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "STOP", f"expected STOP, got {t2['exit_reason']}"
    assert abs(t2["points_gained"] + STOP_PTS) < 0.01, "loss must be bounded to exactly -3pts"
    print("  simulate_trade stop-loss (bounded at -3) check ok")

    # Flat/going-nowhere: neither level touched, rides to EOD.
    rng3 = pd.date_range("2026-01-06 09:15", periods=3, freq="1min")
    close3 = [100.0, 100.2, 99.9]
    prem_df3 = pd.DataFrame({
        "open": [100, 100, 100.2], "high": [c + 0.3 for c in close3], "low": [c - 0.3 for c in close3],
        "close": close3,
    }, index=rng3)
    t3 = simulate_trade(prem_df3, entry_idx=0, day=dt.date(2026, 1, 6))
    assert t3["exit_reason"] == "EOD"
    print("  simulate_trade EOD (neither level touched) check ok")

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
    print("=== Fixed-Strike ORB -- 5pt Target / -3pt Stop, call-only, re-entry allowed ===\n")
    print(f"Target: +{TARGET_PTS}pts  Stop: -{STOP_PTS}pts  Re-entry: allowed same day  "
          f"Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage\n")

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
    tdf.to_csv("nifty_fixed_strike_5pt_target_sl3_trades.csv", index=False)

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

    r = tdf["points_gained"] / STOP_PTS
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

    print(f"\nTrade log saved: nifty_fixed_strike_5pt_target_sl3_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
