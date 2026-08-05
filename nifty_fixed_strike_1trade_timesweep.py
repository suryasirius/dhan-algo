"""
Fixed-Strike ORB -- ONE trade/day, +5pt target, NO stop -- max-hold sweep
Standalone, 6-month backtest, comparing 90s / 2min / 3min / 4min / 5min.
=================================================================
Same base strategy as nifty_fixed_strike_1trade_3min.py (untouched):
one fixed nearest-ITM CE strike per day, single trade per day (no
re-entry), entry on the first confirmed close above the Option Day-High
(first 5 candles), exit at +5pts target or the max-hold deadline,
whichever comes first, NO stop-loss.

This variant sweeps the max-hold duration across 90 seconds, 2, 3, 4,
and 5 minutes, using the SAME entry points across all five (the entry
signal doesn't depend on hold time) so the comparison is apples-to-
apples -- only the exit timing changes. Reports a side-by-side summary
table, matching the sensitivity-sweep convention used earlier in this
project (e.g. the Velez stop-loss sweep).

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
EOD_SQUAREOFF = dt.time(15, 20)

HOLD_DURATIONS = [
    ("90sec", pd.Timedelta(seconds=90)),
    ("2min", pd.Timedelta(minutes=2)),
    ("3min", pd.Timedelta(minutes=3)),
    ("4min", pd.Timedelta(minutes=4)),
    ("5min", pd.Timedelta(minutes=5)),
]

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


def find_entry_for_day(prem_df):
    """Single confirmed close above the Strike High marked from the first
    OR_CANDLES candles. Returns only the FIRST such bar index (one trade
    per day, no re-entry)."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade(prem_df, entry_idx, day, hold_delta):
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + TARGET_PTS
    deadline = idx[entry_idx] + hold_delta

    exit_i, exit_premium, reason = len(idx) - 1, closes[-1], "EOD"
    for i in range(entry_idx + 1, len(idx)):
        t = idx[i]
        if highs[i] >= target:
            exit_i, exit_premium, reason = i, target, "TARGET"
            break
        if t >= deadline:
            exit_i, exit_premium, reason = i, closes[i], "TIME"
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

    # Target-hit before ANY of the hold windows elapse.
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110, 111, 112]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    for label, delta in HOLD_DURATIONS:
        t = simulate_trade(prem_df, entry_idx=5, day=dt.date(2026, 1, 6), hold_delta=delta)
        assert t["exit_reason"] == "TARGET", f"[{label}] expected TARGET, got {t['exit_reason']}"
        assert abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade target-hit-before-any-deadline check ok")

    # No target ever reached -> each hold duration should force a TIME exit
    # at the first available 1-min candle whose timestamp is >= its deadline.
    # NOTE: since the underlying data is 1-min bars, sub-minute precision
    # (the 90sec case) is NOT achievable -- it rounds UP to the next full
    # candle boundary, same as it would for any deadline in (1min, 2min].
    # With an entry that lands exactly on a minute boundary, 90sec and 2min
    # deadlines both resolve to the SAME candle (entry+2min) -- this is a
    # real data-granularity limitation, not a bug, and is called out to the
    # user in the results.
    close2 = [100, 100, 100, 100, 100, 100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2]
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [c + 0.5 for c in close2], "low": [c - 0.5 for c in close2],
        "close": close2,
    }, index=rng)
    expected_exit_bar = {}
    for label, delta in HOLD_DURATIONS:
        t2 = simulate_trade(prem_df2, entry_idx=5, day=dt.date(2026, 1, 6), hold_delta=delta)
        assert t2["exit_reason"] == "TIME", f"[{label}] expected TIME, got {t2['exit_reason']}"
        assert t2["exit_time"] >= rng[5] + delta, f"[{label}] exit time must be >= entry+{delta}"
        assert t2["exit_time"] < rng[5] + delta + pd.Timedelta(minutes=1), \
            f"[{label}] exit time should be the FIRST candle at/after the deadline"
        expected_exit_bar[label] = t2["exit_time"]
    # 90sec and 2min must land on the same rounded-up candle (both entry+2min)
    assert expected_exit_bar["90sec"] == expected_exit_bar["2min"] == rng[5] + pd.Timedelta(minutes=2)
    assert expected_exit_bar["3min"] == rng[5] + pd.Timedelta(minutes=3)
    assert expected_exit_bar["4min"] == rng[5] + pd.Timedelta(minutes=4)
    assert expected_exit_bar["5min"] == rng[5] + pd.Timedelta(minutes=5)
    print("  simulate_trade deadline-rounds-up-to-next-candle check ok "
          "(90sec == 2min exit bar, as expected on 1-min data)")

    # Only-one-trade-per-day entry detection still holds regardless of hold duration.
    close3 = [100]*5 + [101, 99, 98, 97, 105, 96, 95, 110, 94, 93]
    prem_df3 = pd.DataFrame({
        "open": [100] + close3[:-1], "high": [c + 0.5 for c in close3], "low": [c - 0.5 for c in close3],
        "close": close3,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx, day_high = find_entry_for_day(prem_df3)
    assert day_high == 100.5
    assert entry_idx == 5
    print("  find_entry_for_day single-trade-per-day check ok")

    print("All self-checks passed.\n")


def stats_for(g):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_pts=0.0, net_rs=0.0,
                    max_dd=0.0, sqn=float("nan"))
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
    print("=== Fixed-Strike ORB -- ONE trade/day, +5pt target, NO stop -- max-hold sweep ===\n")
    print("Comparing max-hold durations: 90sec, 2min, 3min, 4min, 5min (same entries across all)\n")
    print("NOTE: source data is 1-min candles, so a 90sec hold cannot be modeled with sub-minute")
    print("precision -- a TIME-exit deadline of 90sec rounds UP to the next full-minute candle,")
    print("same as it would for any deadline between 1 and 2 minutes. In practice this means the")
    print("90sec column will often be identical or near-identical to the 2min column below.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    # Find one entry per day (shared across all hold durations).
    day_entries = []
    for d in days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < OR_CANDLES + 1:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike = itm_call_strike(day_open_spot)
        prem_df = build_ce_premium_series(day_df, strike, day_open_spot)
        entry_idx, _ = find_entry_for_day(prem_df)
        if entry_idx is None:
            continue
        day_entries.append((d, prem_df, entry_idx, strike))

    print(f"Days with a qualifying entry: {len(day_entries)} / {len(days)}\n")

    results = {}
    trade_logs = {}
    for label, delta in HOLD_DURATIONS:
        trades = []
        for d, prem_df, entry_idx, strike in day_entries:
            t = simulate_trade(prem_df, entry_idx, d, delta)
            t["strike"] = strike
            trades.append(t)
        tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
        trade_logs[label] = tdf
        results[label] = stats_for(tdf)
        tdf.to_csv(f"nifty_fixed_strike_1trade_timesweep_{label}_trades.csv", index=False)

    print(f"{'Duration':<8} {'n':>5} {'win%':>7} {'PF':>7} {'net_pts':>10} {'net_Rs':>12} "
          f"{'max_DD':>12} {'SQN':>6} {'End_Rs':>12}")
    for label, _ in HOLD_DURATIONS:
        r = results[label]
        print(f"{label:<8} {r['n']:>5} {r['win_pct']:>6.1f}% {r['pf']:>7.2f} {r['net_pts']:>10.2f} "
              f"{r['net_rs']:>12,.2f} {r['max_dd']:>12,.2f} {r['sqn']:>6.2f} "
              f"{r.get('final_equity', STARTING_CAPITAL):>12,.2f}")

    print("\nPer-duration exit-reason breakdown:")
    for label, _ in HOLD_DURATIONS:
        tdf = trade_logs[label]
        print(f"\n-- {label} --")
        for reason, g in tdf.groupby("exit_reason"):
            wins = g[g["points_gained"] > 0]
            losses = g[g["points_gained"] <= 0]
            pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
                  if losses["points_gained"].sum() != 0 else float("inf"))
            print(f"  {reason}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                  f"net_Rs={g['rupee_pnl'].sum():,.2f}  avg_pts={g['points_gained'].mean():.2f}")

    print("\nTrade logs saved: nifty_fixed_strike_1trade_timesweep_<duration>_trades.csv "
          "(one file per duration)")


if __name__ == "__main__":
    main()
