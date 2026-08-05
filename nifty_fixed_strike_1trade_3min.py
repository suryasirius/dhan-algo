"""
Fixed-Strike Opening-Range Breakout -- ONE trade/day, 3-min max hold, +5pt target
Standalone, 6-month backtest.
=================================================================
Another variant of the Fixed-Strike ORB family (base doc:
Nifty_Fixed_Strike_5pt_Target_Strategy_Summary.docx). Two changes from
the original "no re-entry limit" versions, per this instruction:
  - ONLY ONE trade per day -- no re-entry at all, even if the premium
    breaks a fresh new high later. Once today's single trade exits
    (for any reason), scanning stops for the rest of the day.
  - Exit within 3 MINUTES of entry, or when the premium is +5 points
    above entry, whichever comes first.
NO stop-loss is applied here, matching this instruction exactly (it
wasn't mentioned) -- this is closer to the original doc's own "no
stop-loss, as literally written" version, except the 3-minute cap now
bounds how long a losing trade can run (a few minutes of adverse
index movement, not a full session), rather than riding all the way
to EOD like nifty_fixed_strike_5pt_target.py did (which is what wiped
that version out). EOD squareoff (15:20) is still kept as an ultimate
fallback for the rare case a trade enters very late in the day, but
in practice the 3-minute cap will almost always trigger first.

Same fixed nearest-ITM CE strike picked once at market open, same
Option Day-High marked from the first 5 one-minute candles of that
strike's own chart, same close-based confirmation.

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
MAX_HOLD_MINUTES = 3
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


def find_entry_for_day(prem_df):
    """Single confirmed close above the Strike High marked from the first
    OR_CANDLES candles. Returns the FIRST such bar index only (no re-entry
    scanning at all) -- once this is found, this day is done regardless
    of what happens afterward."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            return i, day_high
    return None, day_high


def simulate_trade(prem_df, entry_idx, day):
    idx = prem_df.index
    closes = prem_df["close"].tolist()
    highs = prem_df["high"].tolist()
    lot = lot_size_for_date(day)
    entry_premium = closes[entry_idx]
    target = entry_premium + TARGET_PTS
    deadline = idx[entry_idx] + pd.Timedelta(minutes=MAX_HOLD_MINUTES)

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

    # Target-hit within 3 minutes: rally reaches +5 on the 2nd bar after entry.
    rng = pd.date_range("2026-01-06 09:15", periods=10, freq="1min")
    close = [100, 100, 100, 100, 100, 100, 103, 106, 108, 110]
    prem_df = pd.DataFrame({
        "open": [100] + close[:-1], "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close,
    }, index=rng)
    t = simulate_trade(prem_df, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t["exit_reason"] == "TARGET", f"expected TARGET, got {t['exit_reason']}"
    assert abs(t["points_gained"] - TARGET_PTS) < 0.01
    print("  simulate_trade target-hit-within-3min check ok")

    # No target reached within 3 minutes -> forced TIME exit at exactly the
    # 3-minute mark, regardless of price (even if price is flat/small move).
    close2 = [100, 100, 100, 100, 100, 100, 100.5, 100.8, 101.0, 101.2]
    prem_df2 = pd.DataFrame({
        "open": [100] + close2[:-1], "high": [c + 0.5 for c in close2], "low": [c - 0.5 for c in close2],
        "close": close2,
    }, index=rng)
    t2 = simulate_trade(prem_df2, entry_idx=5, day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "TIME", f"expected TIME, got {t2['exit_reason']}"
    assert t2["exit_time"] == rng[5] + pd.Timedelta(minutes=MAX_HOLD_MINUTES)
    print("  simulate_trade 3-minute forced-time-exit check ok")

    # Only-one-trade-per-day: find_entry_for_day must return just the FIRST
    # breakout, even though a later, separate breakout also exists.
    close3 = [100]*5 + [101, 99, 98, 97, 105, 96, 95, 110, 94, 93]
    prem_df3 = pd.DataFrame({
        "open": [100] + close3[:-1], "high": [c + 0.5 for c in close3], "low": [c - 0.5 for c in close3],
        "close": close3,
    }, index=pd.date_range("2026-01-06 09:15", periods=15, freq="1min"))
    entry_idx, day_high = find_entry_for_day(prem_df3)
    assert day_high == 100.5
    assert entry_idx == 5, f"expected the first breakout at bar 5, got {entry_idx}"
    print("  find_entry_for_day single-trade-per-day check ok")

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
    print("=== Fixed-Strike ORB -- ONE trade/day, 3-min max hold, +5pt target, NO stop ===\n")
    print(f"Target: +{TARGET_PTS}pts  Max hold: {MAX_HOLD_MINUTES} min  No stop-loss  "
          f"One trade per day  Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage\n")

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

        entry_idx, _ = find_entry_for_day(prem_df)
        if entry_idx is None:
            continue
        t = simulate_trade(prem_df, entry_idx, d)
        t["strike"] = strike
        trades.append(t)

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades generated.")
        return

    tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_fixed_strike_1trade_3min_trades.csv", index=False)

    print(f"Total trades: {len(tdf)}  (days scanned: {len(days)}, "
          f"days with no breakout at all: {len(days) - len(tdf)})\n")
    stats_block(tdf, "Overall")
    print()
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)

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

    r = tdf["points_gained"] / TARGET_PTS
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

    print(f"\nTrade log saved: nifty_fixed_strike_1trade_3min_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
