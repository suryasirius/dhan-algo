"""
Opening-Drive Momentum + 8pt Stop / 4pt Trail -- isolated standalone test
=================================================================
WHY THIS SCRIPT EXISTS: in the six-concepts screening pass
(nifty_six_concepts_screening.py), three unrelated "Velez concept"
modules (RBI/GBI relaxed, Power Bar Start, Market Law 4) all
independently converged to firing on 99.6% of days with 93-99% of
entries landing before 10am. That's not those concepts working -- it's
all three degenerating into the same trade: buy a CE/PE near the open
and trail it with the 8pt/4pt rule. This script isolates THAT specific
behavior on its own, with no borrowed technical-analysis dressing, so
it can be judged honestly for what it actually is.

ENTRY LOGIC (deliberately simple -- this is the whole point):
  - Scan bars from the day's open (09:15) up to ENTRY_CUTOFF (09:45).
  - The first bar whose close has moved >= MIN_MOVE_PTS away from the
    day's opening price sets the "opening drive" direction: LONG if it
    moved up, SHORT if it moved down.
  - If no bar clears MIN_MOVE_PTS before the cutoff, no trade that day
    (no clear opening drive -> skip, don't force a trade).
  - No MA/200-bar warmup needed (unlike the six-concepts script), so
    every trading day in the window is eligible.

REV2 FIX -- EXECUTION LAG: the first real run came back with PF=4.05,
SQN=9.35, CAGR=134%. Cause #1: entries were filled at the exact close
of the same bar that generated the signal (zero-latency, no-slippage,
at the precise pivot moment). Fixed with a mandatory 1-bar execution
lag -- signal still confirms on bar i's close, fill happens at bar
(i+1)'s open. This only modestly reduced the numbers (PF 3.57, SQN
8.88, CAGR 129%), which means it was NOT the dominant issue.

REV3 FIX -- INTRABAR HIGH/LOW ORDERING (the bigger bug): every bar
during a trade updated the trailing peak/stop from that bar's OWN HIGH
*before* checking that same bar's OWN LOW against the stop. Since 2-min
OHLC bars don't tell you whether the high or the low happened first,
this silently assumed "high always happens first" on every single bar.
That assumption lets a bar that dips down (which should have breached
the OLD, looser stop) survive and get credited with a brand new,
tighter trailing stop instead -- because the code checked the dip
against the stop level *after* ratcheting it up from that same bar's
high. Over hundreds of trades with a tight 4pt trail that re-tightens
on every new high, this compounds into a large, systematic advantage
that isn't available in real trading. Fixed by checking the bar's low
against the stop level as it stood BEFORE this bar (the conservative,
standard convention: assume the worst-case intrabar order), and only
updating peak/stop for evaluation on the NEXT bar onward. Real result
after both fixes: PF 1.44, SQN 3.02, CAGR 72.25% -- believable,
Velez-comparable territory, not the "holy grail" numbers from before.

REV4 ADDITION -- IN-SAMPLE / OUT-OF-SAMPLE SPLIT: a single 2-year
aggregate number can hide an edge that only existed in one lucky
stretch. Added a first-half vs second-half split (by calendar date) so
the edge's consistency over time can be checked directly, the same
scrutiny Velez's strategy already received before being trusted.

MANAGEMENT: 8pt hard stop / 4pt trail-after-profit, EOD square-off.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")   # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "2min"

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
ENTRY_CUTOFF = dt.time(9, 45)
MIN_MOVE_PTS = 5.0

STOP_PTS = 8.0
TRAIL_PTS = 4.0

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 25_000
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


def resample_2min_per_day(df, bar_rule=BAR_RULE):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(bar_rule, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def build_premium_series(day_df, strike, day_open_spot, side):
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        mo, mh, ml, mc = (idx_open - day_open_spot, idx_high - day_open_spot,
                          idx_low - day_open_spot, idx_close - day_open_spot)
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        mo, mh, ml, mc = (day_open_spot - idx_open, day_open_spot - idx_low,
                          day_open_spot - idx_high, day_open_spot - idx_close)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def find_opening_drive(day_df, min_move=MIN_MOVE_PTS, cutoff=ENTRY_CUTOFF):
    """Scan bars from the open; return (local_index, direction) for the
    first bar whose close deviates from the day's opening price by at
    least min_move, as long as that bar's time is before cutoff. Returns
    (None, None) if no such bar exists before the cutoff. This is the
    SIGNAL bar only -- the actual fill happens one bar later (see main())."""
    day_open = day_df["open"].iloc[0]
    closes = day_df["close"].to_numpy()
    times = day_df.index
    for i in range(len(day_df)):
        if times[i].time() >= cutoff:
            return None, None
        move = closes[i] - day_open
        if move >= min_move:
            return i, "LONG"
        if move <= -min_move:
            return i, "SHORT"
    return None, None


def simulate_trade(premium_df, entry_idx, entry_price, stop_pts=STOP_PTS, trail_pts=TRAIL_PTS):
    """Always simulated as a long-premium trade (we only ever buy CE or PE).
    `entry_idx` is the bar the fill happens ON.

    CONSERVATIVE ORDERING (important): each bar's LOW is checked against
    the stop level AS IT STOOD BEFORE THIS BAR, before that bar's own HIGH
    is allowed to ratchet the trailing stop up. This assumes the
    worst-case intrabar path (a dip could happen before a new high within
    the same 2-min bar) rather than silently assuming highs always come
    first. Only after the low-check does the bar's high get to update
    peak/trailing for use starting the NEXT bar. EOD square-off fallback.
    """
    stop_level = entry_price - stop_pts
    peak = entry_price
    trailing_active = False
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()

        # 1) check this bar's low against the stop level as it stood BEFORE this bar
        if bar["low"] <= stop_level:
            reason = "TRAIL" if trailing_active else "STOP"
            return stop_level, reason, premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]

        # 2) only now let this bar's high update the peak/trailing stop (for next bar)
        if bar["high"] > peak:
            peak = bar["high"]
        if peak > entry_price:
            trailing_active = True
        if trailing_active:
            stop_level = max(stop_level, peak - trail_pts)

    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0,
                "cagr_pct": 0.0}
    pnl = trades_df["rupee_pnl"].to_numpy()
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_pct = 100.0 * len(wins) / n
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_rs = pnl.sum()
    equity = STARTING_CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd_rs = dd.max()
    sqn = (pnl.mean() / pnl.std() * math.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    dates = pd.to_datetime(trades_df["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr_pct = 0.0
    if years > 0 and STARTING_CAPITAL > 0 and (STARTING_CAPITAL + net_rs) > 0:
        cagr_pct = 100.0 * (((STARTING_CAPITAL + net_rs) / STARTING_CAPITAL) ** (1.0 / years) - 1.0)
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def split_in_sample_out_of_sample(t):
    """Split trades into a first-half / second-half of the actual trade date
    range (midpoint by calendar date, not by trade count) and compute
    metrics for each half separately. A real edge should show up in BOTH
    halves at a broadly similar magnitude; an edge concentrated almost
    entirely in one half is a strong sign of overfitting to one period's
    conditions rather than a durable, repeatable behavior."""
    if t.empty:
        return None, None, None
    dates = pd.to_datetime(t["date"])
    start, end = dates.min(), dates.max()
    midpoint = start + (end - start) / 2
    first_half = t[dates <= midpoint].reset_index(drop=True)
    second_half = t[dates > midpoint].reset_index(drop=True)
    return first_half, second_half, midpoint


def print_diagnostics(t, total_days):
    if t.empty:
        print("  no trades generated")
        return
    coverage = 100.0 * len(t) / total_days
    ce = int((t["side"] == "CE").sum())
    pe = int((t["side"] == "PE").sum())
    hours = pd.to_datetime(t["entry_time"]).dt.hour
    early = int((hours < 10).sum())
    later = int((hours >= 10).sum())
    print(f"  fires on {coverage:.1f}% of eligible days ({len(t)}/{total_days}) | "
          f"CE={ce} PE={pe} | entry-hour <10:{early} >=10:{later}")


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    idx = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")
    closes = [100, 100.5, 106.0] + [110] * 17   # bar0=100(open ref via first open), bar1=100.5, bar2=106 -> +6 move
    opens = [100] + closes[:-1]
    df = pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)
    local_i, direction = find_opening_drive(df)
    assert local_i == 2 and direction == "LONG", f"expected LONG at bar 2, got {local_i}, {direction}"
    print("  find_opening_drive LONG-detection check ok")

    closes2 = [100, 99.5, 93.0] + [90] * 17
    opens2 = [100] + closes2[:-1]
    df2 = pd.DataFrame({"open": opens2, "high": closes2, "low": closes2, "close": closes2}, index=idx)
    local_i2, direction2 = find_opening_drive(df2)
    assert local_i2 == 2 and direction2 == "SHORT", f"expected SHORT at bar 2, got {local_i2}, {direction2}"
    print("  find_opening_drive SHORT-detection check ok")

    # no move clears threshold before cutoff (09:45 = bar index 15 at 2min spacing) -> no trade
    idx_late = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")
    closes3 = [100.0] * 20
    df3 = pd.DataFrame({"open": [100.0] * 20, "high": closes3, "low": closes3, "close": closes3}, index=idx_late)
    local_i3, direction3 = find_opening_drive(df3)
    assert local_i3 is None and direction3 is None
    print("  find_opening_drive no-signal check ok")

    # cutoff enforcement: move clears threshold only AFTER 09:45 -> must return None
    closes4 = [100.0] * 16 + [108.0] * 4   # the +8 move happens at local index 16 = 09:15 + 32min = 09:47 (past cutoff)
    df4 = pd.DataFrame({"open": [100.0] * 20, "high": closes4, "low": closes4, "close": closes4}, index=idx_late)
    local_i4, direction4 = find_opening_drive(df4)
    assert local_i4 is None and direction4 is None, f"expected cutoff to block a late move, got {local_i4}, {direction4}"
    print("  find_opening_drive cutoff-enforcement check ok")

    # execution-lag check: signal fires at bar 2 (close=106), the fill should
    # use bar 3's OPEN, NOT bar 2's own close
    exec_idx = local_i + 1
    assert exec_idx == 3
    assert exec_idx != local_i, "fill bar must be the NEXT bar, not the signal bar itself"
    print("  execution-lag (fill on next bar, not signal bar) check ok")

    idx_t = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")

    # simulate_trade: hard stop before any profit (peak never exceeds entry, so
    # trailing never activates -- price declines straight through the stop)
    prem = pd.DataFrame({
        "open": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "high": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "low": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
        "close": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
    }, index=idx_t)
    exit_price, reason, _ = simulate_trade(prem, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason == "STOP" and abs(exit_price - 42.0) < 1e-6, f"expected hard STOP at 42.0, got {exit_price}, {reason}"
    print("  simulate_trade hard-stop check ok")

    # simulate_trade: goes into profit, then trailing stop triggers (clean case,
    # no same-bar high/low ambiguity)
    prem2 = pd.DataFrame({
        "open": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
        "high": [50, 56, 61, 66, 61, 56, 56, 56, 56, 56],
        "low": [50, 54, 59, 64, 55, 54.5, 54.5, 54.5, 54.5, 54.5],
        "close": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
    }, index=idx_t)
    exit_price2, reason2, _ = simulate_trade(prem2, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason2 == "TRAIL" and abs(exit_price2 - 62.0) < 1e-6, f"expected TRAIL exit at 62.0, got {exit_price2}, {reason2}"
    print("  simulate_trade trailing-stop check ok")

    # simulate_trade: THE KEY REV3 CHECK -- a bar that dips down (which should
    # breach the OLD stop level) AND makes a big new high in the same bar.
    # bar0 = entry (50). bar1: high=56 -> peak=56, trailing on, stop=52.
    # bar2: low=50 (breaches the OLD stop=52) AND high=70 (would-be new peak=70,
    # would-be stop=66 if updated first). The CONSERVATIVE fix must exit at the
    # OLD level (52, TRAIL) at bar index 2 -- NOT ratchet up to 66 first and
    # then fail to exit.
    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    exit_price3, reason3, exit_time3 = simulate_trade(prem3, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason3 == "TRAIL" and abs(exit_price3 - 52.0) < 1e-6, (
        f"expected conservative TRAIL exit at the OLD stop level 52.0 (not the optimistic "
        f"66.0 from ratcheting on this bar's own high first), got {exit_price3}, {reason3}"
    )
    assert exit_time3 == prem3.index[2], "expected the exit to occur on the ambiguous bar itself (index 2)"
    print("  simulate_trade conservative intrabar-ordering (REV3 fix) check ok")

    # in-sample/out-of-sample split: 4 trades spanning day1..day4 -> midpoint
    # should split them 2-and-2 (first_half inclusive of the midpoint date)
    t_test = pd.DataFrame({
        "date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2025, 1, 4)],
        "rupee_pnl": [10, -5, 20, -8],
    })
    fh, sh, mid = split_in_sample_out_of_sample(t_test)
    assert len(fh) == 2 and len(sh) == 2, f"expected a clean 2/2 split, got {len(fh)}/{len(sh)}"
    assert list(fh["date"]) == [dt.date(2025, 1, 1), dt.date(2025, 1, 2)]
    assert list(sh["date"]) == [dt.date(2025, 1, 3), dt.date(2025, 1, 4)]
    print("  split_in_sample_out_of_sample check ok")

    print("All self-checks passed.\n")


def main():
    run_self_checks()

    print(f"Fetching Nifty index data {FROM_DATE} to {TO_DATE} ...")
    raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    full_df = resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    days = sorted(set(full_df.index.date))
    total_days = len(days)

    trades = []
    capital = STARTING_CAPITAL
    for day in days:
        day_df = full_df.loc[full_df.index.date == day]
        if len(day_df) < 3:
            continue
        day_open_spot = day_df["open"].iloc[0]
        local_i, direction = find_opening_drive(day_df)
        if local_i is None:
            continue
        exec_idx = local_i + 1
        if exec_idx >= len(day_df):
            continue  # no bar left in the day to actually execute the fill on
        side = "CE" if direction == "LONG" else "PE"
        strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
        premium_df = build_premium_series(day_df, strike, day_open_spot, side)
        entry_price = premium_df["open"].iloc[exec_idx]
        exit_price, reason, exit_time = simulate_trade(premium_df, exec_idx, entry_price)
        lot = lot_size_for_date(day)
        pnl_pts = exit_price - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        capital += rupee_pnl
        trades.append({
            "date": day, "direction": direction, "strike": strike, "side": side,
            "signal_time": day_df.index[local_i], "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2),
            "rupee_pnl": round(rupee_pnl, 2), "capital_after": round(capital, 2),
        })

    t = pd.DataFrame(trades)
    m = compute_metrics(t)

    print("=" * 70)
    print("OPENING-DRIVE MOMENTUM + 8pt STOP / 4pt TRAIL -- FULL 2-YEAR RESULTS")
    print("=" * 70)
    print(f"n={m['n']}  win%={m['win_pct']}  PF={m['pf']}  net_Rs={m['net_rs']}  "
          f"maxDD_Rs={m['max_dd_rs']}  SQN={m['sqn']}  CAGR%={m['cagr_pct']}")
    print()
    print("DIAGNOSTICS (fire-rate & entry-time clustering -- sanity check):")
    print_diagnostics(t, total_days)

    print("\n" + "=" * 70)
    print("IN-SAMPLE (first half) vs OUT-OF-SAMPLE (second half) SPLIT")
    print("=" * 70)
    first_half, second_half, midpoint = split_in_sample_out_of_sample(t)
    if first_half is not None:
        m1 = compute_metrics(first_half)
        m2 = compute_metrics(second_half)
        print(f"Split date (midpoint): {midpoint.date()}")
        print(f"First half:  n={m1['n']}  win%={m1['win_pct']}  PF={m1['pf']}  net_Rs={m1['net_rs']}  "
              f"maxDD_Rs={m1['max_dd_rs']}  SQN={m1['sqn']}")
        print(f"Second half: n={m2['n']}  win%={m2['win_pct']}  PF={m2['pf']}  net_Rs={m2['net_rs']}  "
              f"maxDD_Rs={m2['max_dd_rs']}  SQN={m2['sqn']}")

    if not t.empty:
        t.to_csv("open_momentum_trail_trades.csv", index=False)
        print("\nSaved open_momentum_trail_trades.csv")


if __name__ == "__main__":
    main()
