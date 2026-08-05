"""
Fixed-Strike ORB -- PUT side (mirror of the CE strategy), 6.5-Point Target
OR 3-Minute Time Exit, whichever first -- 6-month backtest
=================================================================
Mirror of nifty_fixed_strike_3p5pt_3min_backtest_2yr.py, flipped to the
downside: one fixed nearest-ITM PE strike per day, mark the opening
5-minute (09:15-09:20) range, enter on the first confirmed 1-min
premium close that breaks the favorable side of that range, buy the
PUT, one trade per day, no re-entry.

IMPORTANT clarification on direction (read this before trusting the
numbers): you described marking the "high and low" of the studied
strike and entering when it "breaks the low." Taken completely
literally on the PUT's own premium candles, that would mean buying the
PUT exactly when ITS OWN premium is falling to a new low within the
first few minutes -- which, because a PE's premium moves opposite to
the index, is mathematically the SAME MOMENT the index is breaking
ABOVE its own opening high (a bullish move). That would mean buying
puts into index strength -- the opposite of your CE strategy's logic
(buy calls on index strength) and not a sensible mirror.

The sensible, intended mirror -- matching your CE strategy's logic
exactly (buy in the direction of a confirmed breakout) -- is: the
index breaks BELOW its own opening 5-minute low, confirmed by a 1-min
close, and you buy the PUT at the resulting (now higher) premium. This
script implements THAT version. In premium terms this is: mark the PE
premium's own opening-5-min HIGH (which corresponds to the index's
opening-5-min LOW, since a PE's premium peaks when the index dips),
then enter when the premium's close breaks ABOVE that high (which
corresponds to the index closing below its opening low) -- the exact
same code shape as the CE script, just built from the PE premium
formula. If you actually meant the literal PE-premium-breakdown
version instead, say so and I'll build that one separately.

Target: your worked example (₹150 -> ₹156.5) confirms a 6.5-point
target, even though the exit-rule sentence also said "5 point" --
used 6.5 here since the numbers you gave are unambiguous. Time exit:
3 minutes, matching your CE version's exit rule. NO stop-loss (same
convention as every other fixed-strike ORB variant in this project).

PROXY PREMIUM MODE: same convention as every other option script in
this project (delta=0.6, floored at 0) -- real Dhan option-premium
data is still blocked (DH-901).

6-month window, per your request.
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
TARGET_PTS = 6.5
MAX_HOLD = pd.Timedelta(minutes=3)
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


def build_pe_premium_series(day_df, strike, day_open_spot):
    """PE premium moves OPPOSITE to the index: falling index -> rising
    premium. Constructed so that the premium's own HIGH corresponds to
    the index's LOW (best case for a put) and the premium's own LOW
    corresponds to the index's HIGH (worst case for a put) -- same
    convention used for concept-2 PE trades elsewhere in this project."""
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_open = day_df["open"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    intrinsic0 = max(strike - day_open_spot, 0.0)
    base = intrinsic0 + ASSUMED_TIME_VALUE

    p_open = np.maximum(base + (day_open_spot - idx_open) * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + (day_open_spot - idx_low) * ASSUMED_DELTA, 0.0)   # index LOW -> premium HIGH
    p_low = np.maximum(base + (day_open_spot - idx_high) * ASSUMED_DELTA, 0.0)   # index HIGH -> premium LOW
    p_close = np.maximum(base + (day_open_spot - idx_close) * ASSUMED_DELTA, 0.0)

    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close},
                         index=day_df.index)


def find_entry_for_day(prem_df):
    """Single confirmed close above the opening 5-min premium high --
    which, for a PUT built with the formula above, corresponds exactly
    to the INDEX confirming a close below ITS OWN opening 5-min low.
    Returns only the FIRST such bar -- one trade per day, no re-entry."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade(prem_df, entry_idx, day):
    """Conservative ordering within a bar: TARGET is checked before the
    time deadline on any bar where both would technically be
    satisfiable. No stop-loss."""
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + TARGET_PTS
    deadline = idx[entry_idx] + MAX_HOLD

    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
            break
        if t >= deadline:
            exit_i, exit_premium, reason = i, closes[i], "TIME_3MIN"
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

    # build_pe_premium_series direction check: index falling should raise
    # the PE premium; index rising should lower it.
    day_df_check = pd.DataFrame({
        "open": [22000, 22000], "high": [22005, 22000], "low": [21995, 21980],
        "close": [22000, 21980],
    }, index=pd.date_range("2026-01-06 09:15", periods=2, freq="1min"))
    p = build_pe_premium_series(day_df_check, strike=itm_put_strike(22000), day_open_spot=22000)
    assert p["close"].iloc[1] > p["close"].iloc[0], "PE premium should RISE as the index falls"
    print("  build_pe_premium_series direction check ok (index down -> PE premium up)")

    # Target hit cleanly, well before the 3-min deadline.
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110, 111, 112]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    t = simulate_trade(prem_df, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t["exit_reason"] == "TARGET" and abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade clean TARGET (before 3min) check ok")

    # Target never reached within 3 min -> must force TIME_3MIN exit at
    # the first candle whose timestamp is >= entry+3min.
    close2 = [100, 100, 100, 100, 100, 100, 100.5, 101.0, 101.5, 102.0, 102.3, 102.5]
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [c + 0.3 for c in close2], "low": [c - 0.3 for c in close2],
        "close": close2,
    }, index=rng)
    t2 = simulate_trade(prem_df2, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "TIME_3MIN", f"expected TIME_3MIN, got {t2['exit_reason']}"
    assert t2["exit_time"] == rng[5] + pd.Timedelta(minutes=3)
    assert abs(t2["exit_premium"] - close2[8]) < 1e-9  # bar index 5+3=8
    print("  simulate_trade TIME_3MIN-when-target-not-hit check ok")

    # Target and the 3-min deadline both satisfiable on the SAME bar ->
    # TARGET must win.
    close3 = [100, 100, 100, 100, 100, 100, 100.1, 100.1, 110.0, 110.0, 110.0, 110.0]
    prem_df3 = pd.DataFrame({
        "open": [100] + close3[:-1], "high": [c + 0.1 for c in close3], "low": [c - 0.3 for c in close3],
        "close": close3,
    }, index=rng)
    t3 = simulate_trade(prem_df3, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t3["exit_reason"] == "TARGET", f"expected TARGET to win on the ambiguous bar, got {t3['exit_reason']}"
    assert t3["exit_time"] == rng[8]
    print("  simulate_trade same-bar target-vs-deadline ordering check ok")

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
    print("=== Fixed-Strike ORB -- PUT side -- 6.5pt Target OR 3-min Time Exit (whichever first) -- 6-month backtest ===\n")
    print("Entry: confirmed 1-min index close BELOW its opening 5-min low (buys the PUT), one trade/day.")
    print("Exit: +6.5pt target on the premium OR 3 minutes after entry, whichever first. NO stop-loss.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~6 months)...")
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
        strike = itm_put_strike(day_open_spot)
        prem_df = build_pe_premium_series(day_df, strike, day_open_spot)
        entry_idx, _ = find_entry_for_day(prem_df)
        if entry_idx is None:
            no_entry_days += 1
            continue
        t = simulate_trade(prem_df, entry_idx, d)
        t["strike"] = strike
        trades.append(t)

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    print(f"Days with a qualifying entry: {len(tdf)} / {len(days)}  (no breakdown on {no_entry_days} days)\n")

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

    tdf.to_csv("nifty_fixed_strike_put_6p5pt_3min_backtest_6mo_trades.csv", index=False)
    print(f"\nSaved: nifty_fixed_strike_put_6p5pt_3min_backtest_6mo_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
