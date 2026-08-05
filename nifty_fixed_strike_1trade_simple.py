"""
Fixed-Strike ORB -- SIMPLE backtest, no extra conditions
=================================================================
Straight implementation of the strategy exactly as described in
"Fixed-Strike Opening-Range Breakout -- 5-Point Target Strategy"
(Nilus, 29 July 2026): one fixed nearest-ITM CE strike per day, mark
the opening 5-minute high/low on that strike's own premium, enter on
the first confirmed 1-min close above the opening high, exit at
+5pts target, one trade per day, no re-entry.

NO extra conditions on top of that: no stop-loss (matches the doc --
flagged as a real risk, same as the doc itself flags it), no time-based
deadline/hold-duration sweep (that was the extra condition in
nifty_fixed_strike_1trade_timesweep.py -- removed here per your
request). The ONLY exit besides the +5pt target is the standard
15:20 EOD squareoff already used everywhere else in this project,
since a real trade can't be held open past market close.

PROXY PREMIUM MODE: same convention as every other option script in
this project (delta=0.6, floored at 0) -- real Dhan option-premium
data is still blocked (DH-901).

2-year window, matching this project's default backtest length.
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
TARGET_PTS = 5.0
EOD_SQUAREOFF = dt.time(15, 20)

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
    """Single confirmed close above the opening-range high (first
    OR_CANDLES candles). Returns only the FIRST such bar -- one trade
    per day, no re-entry."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade(prem_df, entry_idx, day):
    """No stop, no time deadline. Exit ONLY at +5pt target, or EOD
    squareoff if target never hits that day."""
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + TARGET_PTS

    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, len(idx)):
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
            break
        if idx[i].time() >= EOD_SQUAREOFF:
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

    # Target hit cleanly.
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110, 111, 112]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    t = simulate_trade(prem_df, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t["exit_reason"] == "TARGET" and abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade clean TARGET check ok")

    # Target never reached -> must ride to EOD squareoff, no early time exit.
    rng_long = pd.date_range("2026-01-06 09:15", periods=400, freq="1min")  # covers to ~15:55
    close2 = [100] * 400
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [100.5] * 400, "low": [99.5] * 400, "close": close2,
    }, index=rng_long)
    t2 = simulate_trade(prem_df2, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "EOD"
    assert t2["exit_time"].time() >= EOD_SQUAREOFF
    print("  simulate_trade flat-premium rides to EOD (no early time exit) check ok")

    # Single-trade-per-day entry detection.
    close3 = [100] * 5 + [101, 99, 98, 97, 105, 96, 95, 110, 94, 93]
    prem_df3 = pd.DataFrame({
        "open": [100] + close3[:-1], "high": [c + 0.5 for c in close3], "low": [c - 0.5 for c in close3],
        "close": close3,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx, day_high = find_entry_for_day(prem_df3)
    assert day_high == 100.5
    assert entry_idx == 5
    print("  find_entry_for_day single-trade-per-day check ok")

    # No breakout at all -> no entry.
    close4 = [100] * 15
    prem_df4 = pd.DataFrame({
        "open": [100] + close4[:-1], "high": [100.5] * 15, "low": [99.5] * 15, "close": close4,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx4, _ = find_entry_for_day(prem_df4)
    assert entry_idx4 is None
    print("  find_entry_for_day no-breakout-no-trade check ok")

    print("All self-checks passed.\n")


def stats_for(g):
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
    r = g["points_gained"] / TARGET_PTS
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_pts=g["points_gained"].sum(),
                net_rs=g["rupee_pnl"].sum(), max_dd=max_dd, sqn=sqn,
                final_equity=equity.iloc[-1])


def main():
    run_self_checks()
    print("=== Fixed-Strike ORB -- SIMPLE backtest, no extra conditions ===\n")
    print("Entry: confirmed 1-min close above opening 5-min premium high, one trade/day.")
    print("Exit: +5pt target, or EOD squareoff (15:20) if target never hits. NO stop-loss, NO time deadline.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    trades = []
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
        t = simulate_trade(prem_df, entry_idx, d)
        t["strike"] = strike
        trades.append(t)

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    print(f"Days with a qualifying entry: {len(tdf)} / {len(days)}  (no breakout on {no_entry_days} days)\n")

    r = stats_for(tdf)
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_pts={r['net_pts']:.2f}  "
          f"net_Rs={r['net_rs']:,.2f}  maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  "
          f"final_equity_Rs={r['final_equity']:,.2f}\n")

    print("By exit reason:")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(tdf):.1f}%)  win%={100*(g['points_gained']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}  avg_pts={g['points_gained'].mean():.2f}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = stats_for(fh), stats_for(sh)
    print(f"\nStability check (in-sample/out-of-sample split):")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']:.2f} SQN={m1['sqn']:.2f}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']:.2f} SQN={m2['sqn']:.2f}")

    if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny "
              "per this project's standing calibration. ***")

    tdf.to_csv("nifty_fixed_strike_1trade_simple_trades.csv", index=False)
    print(f"\nSaved: nifty_fixed_strike_1trade_simple_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
