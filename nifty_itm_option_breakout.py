"""
Nifty ITM Option Breakout Strategy -- standalone, 2-year backtest
=================================================================
Based on Nifty_ITM_Option_Breakout_Strategy_Summary.docx (Concept by
Nilus). Unlike the index-based ORB strategy already tested in this
project, this one marks its reference range on the OPTION CONTRACT'S
OWN 1-minute premium chart, not the index -- the index is used only
once, at market open, to pick which CE/PE strike counts as "nearest
ITM" for the day.

The doc flagged several genuinely ambiguous points and asked for
confirmation before coding. Since VIX was answered directly ("backtest
without vix concept"), the other three were resolved using the doc's
own suggested defaults (each was the "Recommended" option offered):
  - PE side MIRRORS the CE rule: PE premium breaks above its OWN
    Option Day-High and holds, exactly like the call side (not a
    break-below-Day-Low rule, and not "no PE trade at all").
  - Only the FIRST side to confirm (CE or PE, whichever happens
    first in time) takes the trade for the day, matching the
    same-day convention used in every other strategy in this project.
  - "Holds for 1-2 candles" is treated as: 2 CONSECUTIVE 1-min candles
    must CLOSE above the Day-High (not just trade above intrabar) --
    the strictest, cleanest reading for a repeatable OHLC backtest.
If any of these three assumptions is wrong, only the entry-detection
logic (find_entry_signal) needs to change -- everything else (range
marking, stop/trail, EOD squareoff, P&L) is independent of them.

VIX FILTER: per your instruction, NOT applied. Every trading day is
tested regardless of volatility. This means the results below do NOT
reflect the "VIX < 12" condition the original doc describes -- treat
this as an unfiltered baseline, not the strategy as fully specified.

PROXY PREMIUM MODE: real Dhan historical option-premium data is still
blocked (DH-901 auth error, established earlier in this project), so
each contract's full-day 1-minute premium OHLC is simulated from the
INDEX's own 1-minute OHLC, scaled by an assumed delta -- entry_premium
= intrinsic_at_open + ASSUMED_TIME_VALUE, and premium moves with the
index scaled by ASSUMED_DELTA (0.6) in the option's favorable
direction (CE tracks the index up, PE tracks the index down). This
lets the option's own Day-High/Day-Low be computed just like the doc
describes, but treat absolute Rs figures as indicative, not exact.
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
CONFIRM_CANDLES = 2          # consecutive closes above Day-High required

INITIAL_SL_PTS = 10.0
TRAIL_PTS = 5.0
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


def build_option_premium_series(day_df, side, strike, day_open_spot):
    """Simulates the option's own full-day 1-min OHLC premium series from
    the index's OHLC, scaled by ASSUMED_DELTA. Strike/intrinsic is fixed
    at the day's opening spot (matching the doc's "picked once at open"
    assumption)."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()

    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        base = intrinsic0 + ASSUMED_TIME_VALUE
        p_open = base + (idx_open - day_open_spot) * ASSUMED_DELTA
        p_close = base + (idx_close - day_open_spot) * ASSUMED_DELTA
        p_high = base + (idx_high - day_open_spot) * ASSUMED_DELTA   # index up -> CE premium up
        p_low = base + (idx_low - day_open_spot) * ASSUMED_DELTA
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        base = intrinsic0 + ASSUMED_TIME_VALUE
        p_open = base + (day_open_spot - idx_open) * ASSUMED_DELTA
        p_close = base + (day_open_spot - idx_close) * ASSUMED_DELTA
        p_high = base + (day_open_spot - idx_low) * ASSUMED_DELTA    # index down -> PE premium up
        p_low = base + (day_open_spot - idx_high) * ASSUMED_DELTA

    return pd.DataFrame({
        "open": p_open, "high": p_high, "low": p_low, "close": p_close,
    }, index=day_df.index)


def find_entry_signal(prem_df):
    """Returns the bar index of entry (the 2nd of 2 consecutive candles
    that CLOSE above the Day-High marked from the first OR_CANDLES
    candles of this option's own chart), or None if it never fires."""
    day_high = prem_df["high"].iloc[:OR_CANDLES].max()
    closes = prem_df["close"].tolist()
    streak = 0
    for i in range(OR_CANDLES, len(prem_df)):
        if closes[i] > day_high:
            streak += 1
            if streak >= CONFIRM_CANDLES:
                return i, day_high
        else:
            streak = 0
    return None, day_high


def simulate_trade(prem_df, entry_idx, side, day):
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
        "date": day, "side": side, "entry_time": idx[entry_idx], "exit_time": idx[exit_i],
        "entry_premium": entry_premium, "exit_premium": exit_premium,
        "exit_reason": reason, "points_gained": points, "rupee_pnl": rupee_pnl,
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # Day-high/entry-trigger: 5 flat candles then a rally that closes
    # above the day-high for 2 consecutive candles.
    rng = pd.date_range("2026-01-06 09:15", periods=20, freq="1min")
    close = np.concatenate([np.full(5, 100.0), [100, 101, 106, 107, 108, 110, 111,
                                                  112, 113, 114, 115, 116, 117, 118, 119]])
    prem_df = pd.DataFrame({
        "open": np.roll(close, 1), "high": close + 0.5, "low": close - 0.5, "close": close,
    }, index=rng)
    prem_df.iloc[0, prem_df.columns.get_loc("open")] = close[0]
    entry_idx, day_high = find_entry_signal(prem_df)
    assert day_high == 100.5, f"expected day_high 100.5, got {day_high}"
    assert entry_idx is not None
    assert close[entry_idx] > day_high and close[entry_idx - 1] > day_high, \
        "entry must be the 2nd of 2 consecutive closes above day-high"
    print("  find_entry_signal 2-consecutive-close check ok")

    # A single close above day-high, then a drop back below, must NOT trigger
    close2 = np.concatenate([np.full(5, 100.0), [106, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]])
    prem_df2 = pd.DataFrame({
        "open": np.roll(close2, 1), "high": close2 + 0.5, "low": close2 - 0.5, "close": close2,
    }, index=rng)
    prem_df2.iloc[0, prem_df2.columns.get_loc("open")] = close2[0]
    entry_idx2, _ = find_entry_signal(prem_df2)
    assert entry_idx2 is None, "a single-candle spike followed by a drop must not confirm entry"
    print("  find_entry_signal single-spike-rejected check ok")

    # Trailing stop hand trace: entry=100, initial stop=90. Price rises to
    # 110 (trail activates, stop -> 105), dips to 92 (still above 105? no --
    # wait: once trailing, stop only tightens upward from the trailing calc,
    # so after reaching 110 stop=105; a subsequent low of 108 keeps stop at
    # 103 only if that's tighter -- stop must never loosen).
    rng2 = pd.date_range("2026-01-06 09:15", periods=6, freq="1min")
    close3 = [100, 110, 108, 115, 112, 103]
    low3 = [99.5, 109.5, 107.5, 114.5, 111.5, 102.5]
    prem_df3 = pd.DataFrame({
        "open": [100, 100, 110, 108, 115, 112], "high": [c + 0.5 for c in close3],
        "low": low3, "close": close3,
    }, index=rng2)
    t = simulate_trade(prem_df3, entry_idx=0, side="CE", day=dt.date(2026, 1, 6))
    # bar1 close=110>100 -> trailing_active=True, stop=max(90, 109.5-5=104.5)=104.5
    # bar2 low=107.5 -> not <=104.5, stop=max(104.5, 107.5-5=102.5)=104.5 (tighter one kept, no loosening)
    # bar3 low=114.5 -> not <=104.5, stop=max(104.5, 114.5-5=109.5)=109.5
    # bar4 low=111.5 -> not <=109.5, stop=max(109.5, 111.5-5=106.5)=109.5 (no loosening)
    # bar5 low=102.5 -> 102.5<=109.5 -> TRAIL_SL hit, exit at stop=109.5
    assert t["exit_reason"] == "TRAIL_SL", f"expected TRAIL_SL, got {t['exit_reason']}"
    assert abs(t["exit_premium"] - 109.5) < 0.01, f"expected exit at 109.5, got {t['exit_premium']}"
    print("  simulate_trade trailing-stop hand-trace check ok")

    # Initial (non-trailing) stop-loss: price never goes into profit, drops
    # straight through the fixed -10pt stop.
    rng3 = pd.date_range("2026-01-06 09:15", periods=4, freq="1min")
    close4 = [100, 95, 89, 85]
    prem_df4 = pd.DataFrame({
        "open": [100, 100, 95, 89], "high": [c + 0.5 for c in close4],
        "low": [c - 0.5 for c in close4], "close": close4,
    }, index=rng3)
    t2 = simulate_trade(prem_df4, entry_idx=0, side="CE", day=dt.date(2026, 1, 6))
    assert t2["exit_reason"] == "INITIAL_SL"
    assert abs(t2["exit_premium"] - 90.0) < 0.01, f"expected exit at 90.0 (100-10), got {t2['exit_premium']}"
    print("  simulate_trade initial-stop-loss check ok")

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
    print("=== Nifty ITM Option Breakout (option's-own-range) -- 2yr backtest ===\n")
    print("VIX filter: NOT applied (per instruction). PE mirrors CE rule. Only first "
          "side (CE or PE) to confirm each day is traded. 2 consecutive closes above "
          "the option's own Day-High required. PROXY premium mode -- Rs 40/trade brokerage.\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    trades = []
    for d in days:
        day_df = df1m[df1m.index.date == d]
        if len(day_df) < OR_CANDLES + CONFIRM_CANDLES + 1:
            continue
        day_open_spot = day_df["open"].iloc[0]
        ce_strike = itm_call_strike(day_open_spot)
        pe_strike = itm_put_strike(day_open_spot)

        ce_prem = build_option_premium_series(day_df, "CE", ce_strike, day_open_spot)
        pe_prem = build_option_premium_series(day_df, "PE", pe_strike, day_open_spot)

        ce_entry_idx, _ = find_entry_signal(ce_prem)
        pe_entry_idx, _ = find_entry_signal(pe_prem)

        # Only the first side to confirm (by time) is traded.
        candidates = []
        if ce_entry_idx is not None:
            candidates.append((ce_entry_idx, "CE", ce_prem))
        if pe_entry_idx is not None:
            candidates.append((pe_entry_idx, "PE", pe_prem))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        entry_idx, side, prem_df = candidates[0]

        trades.append(simulate_trade(prem_df, entry_idx, side, d))

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades generated.")
        return

    tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_itm_option_breakout_trades.csv", index=False)

    print(f"Total trades: {len(tdf)}\n")
    stats_block(tdf, "Overall")
    stats_block(tdf[tdf.side == "CE"], "CE only")
    stats_block(tdf[tdf.side == "PE"], "PE only")
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

    print(f"\nTrade log saved: nifty_itm_option_breakout_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
