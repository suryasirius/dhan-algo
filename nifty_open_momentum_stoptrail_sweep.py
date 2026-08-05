"""
Opening-Drive Momentum -- Stop x Trail Sweep (EXTENDED GRID)
=================================================================
Builds on nifty_open_momentum_trail.py, which survived a genuine bug
hunt (look-ahead entry fixed, intrabar high/low ordering fixed) and an
in-sample/out-of-sample split (PF 1.39 vs 1.48, SQN 1.83 vs 2.41 --
consistent, not a one-period fluke). The entry logic (opening-drive
detection, 1-bar execution lag, conservative intrabar stop/trail
ordering) is unchanged and reused verbatim here. This script only
varies the exit parameters -- STOP_PTS x TRAIL_PTS.

GRID HISTORY: the first pass used STOP_VALUES=[3,5,8,10] x
TRAIL_VALUES=[2,4,6]. The best combo (sl10_trail2, SQN 6.08) sat at the
EDGE of both dimensions -- widest stop, tightest trail -- with
performance monotonically improving toward that corner across the
whole grid. That's a red flag per the project's own rigor checklist:
either the grid was too narrow, or the result is fitting this specific
2-year path's tail characteristics rather than a real, generalizable
edge. This revision extends the grid wider (STOP up to 20) and tighter
(TRAIL down to 1) to see whether performance keeps climbing toward the
new extreme (bad sign -- still hasn't found a real edge) or levels off
at an interior combo (good sign -- an actual sweet spot exists).
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

STOP_VALUES = [3, 5, 8, 10, 12, 15, 20]
TRAIL_VALUES = [1, 2, 4, 6]

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
    least min_move, as long as that bar's time is before cutoff."""
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


def simulate_trade(premium_df, entry_idx, entry_price, stop_pts, trail_pts):
    """Unchanged from nifty_open_momentum_trail.py (REV3): conservative
    intrabar ordering -- each bar's low is checked against the stop level
    as it stood BEFORE this bar, before that bar's own high is allowed to
    ratchet the trailing stop up."""
    stop_level = entry_price - stop_pts
    peak = entry_price
    trailing_active = False
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= stop_level:
            reason = "TRAIL" if trailing_active else "STOP"
            return stop_level, reason, premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
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
    if t.empty:
        return None, None, None
    dates = pd.to_datetime(t["date"])
    start, end = dates.min(), dates.max()
    midpoint = start + (end - start) / 2
    first_half = t[dates <= midpoint].reset_index(drop=True)
    second_half = t[dates > midpoint].reset_index(drop=True)
    return first_half, second_half, midpoint


def run_for_combo(full_df, day_groups, stop_pts, trail_pts):
    trades = []
    capital = STARTING_CAPITAL
    for day, day_df in day_groups:
        if len(day_df) < 3:
            continue
        day_open_spot = day_df["open"].iloc[0]
        local_i, direction = find_opening_drive(day_df)
        if local_i is None:
            continue
        exec_idx = local_i + 1
        if exec_idx >= len(day_df):
            continue
        side = "CE" if direction == "LONG" else "PE"
        strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
        premium_df = build_premium_series(day_df, strike, day_open_spot, side)
        entry_price = premium_df["open"].iloc[exec_idx]
        exit_price, reason, exit_time = simulate_trade(premium_df, exec_idx, entry_price, stop_pts, trail_pts)
        lot = lot_size_for_date(day)
        pnl_pts = exit_price - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        capital += rupee_pnl
        trades.append({
            "date": day, "direction": direction, "side": side,
            "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2),
            "rupee_pnl": round(rupee_pnl, 2), "capital_after": round(capital, 2),
        })
    return pd.DataFrame(trades)


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    idx = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")
    closes = [100, 100.5, 106.0] + [110] * 17
    opens = [100] + closes[:-1]
    df = pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)
    local_i, direction = find_opening_drive(df)
    assert local_i == 2 and direction == "LONG"
    print("  find_opening_drive check ok")

    idx_t = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")
    prem = pd.DataFrame({
        "open": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "high": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "low": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
        "close": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
    }, index=idx_t)
    exit_price, reason, _ = simulate_trade(prem, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason == "STOP" and abs(exit_price - 42.0) < 1e-6
    print("  simulate_trade hard-stop check ok")

    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    exit_price3, reason3, _ = simulate_trade(prem3, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason3 == "TRAIL" and abs(exit_price3 - 52.0) < 1e-6, "conservative intrabar ordering must be preserved"
    print("  simulate_trade conservative-ordering check ok")

    exit_tight, reason_tight, _ = simulate_trade(prem, 0, entry_price=50, stop_pts=3, trail_pts=2)
    assert abs(exit_tight - 47.0) < 1e-6 and reason_tight == "STOP", (
        f"expected a tighter 3pt stop to exit earlier at 47.0, got {exit_tight}, {reason_tight}"
    )
    print("  stop/trail parametrization check ok (tighter stop exits earlier, as expected)")

    # NEW: extended-grid check -- trail_pts=1 (tightest in the new grid) locks
    # in profit almost immediately once the position is in the money at all
    prem4 = pd.DataFrame({
        "open": [50, 53, 53, 53],
        "high": [50, 54, 54, 54],
        "low": [50, 52, 51, 51],
        "close": [50, 53, 53, 53],
    }, index=idx_t[:4])
    exit_price4, reason4, exit_time4 = simulate_trade(prem4, 0, entry_price=50, stop_pts=20, trail_pts=1)
    # bar1: low=52 checked vs old stop(30, from 50-20) -> no exit; high=54>peak(50)->peak=54,
    # trailing on, stop=max(30,54-1=53)=53. bar2: low=51 <=53 -> exit at 53, TRAIL.
    assert reason4 == "TRAIL" and abs(exit_price4 - 53.0) < 1e-6, (
        f"expected a tight trail=1 to lock in a quick profit at 53.0, got {exit_price4}, {reason4}"
    )
    print("  extended-grid trail=1 check ok")

    # run_for_combo end-to-end smoke on 2 synthetic days
    d1 = dt.date(2025, 2, 3)
    d2 = dt.date(2025, 2, 4)
    times1 = pd.date_range(dt.datetime.combine(d1, dt.time(9, 15)), periods=30, freq="2min")
    times2 = pd.date_range(dt.datetime.combine(d2, dt.time(9, 15)), periods=30, freq="2min")
    closes_d = [22000, 22001, 22008] + [22015] * 27
    opens_d = [22000] + closes_d[:-1]
    day1_df = pd.DataFrame({"open": opens_d, "high": [c + 1 for c in closes_d],
                             "low": [c - 1 for c in closes_d], "close": closes_d}, index=times1)
    day2_df = day1_df.copy()
    day2_df.index = times2
    fake_groups = [(d1, day1_df), (d2, day2_df)]
    t = run_for_combo(None, fake_groups, stop_pts=8, trail_pts=4)
    assert len(t) == 2, f"expected 2 trades from 2 synthetic days, got {len(t)}"
    print("  run_for_combo end-to-end smoke check ok")

    print("All self-checks passed.\n")


def main():
    run_self_checks()

    print(f"Fetching Nifty index data {FROM_DATE} to {TO_DATE} ...")
    raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    full_df = resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    day_groups = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]

    results = {}
    for stop_pts in STOP_VALUES:
        for trail_pts in TRAIL_VALUES:
            if trail_pts >= stop_pts:
                continue  # trail wider than the hard stop doesn't make sense
            key = f"sl{stop_pts}_trail{trail_pts}"
            t = run_for_combo(full_df, day_groups, stop_pts, trail_pts)
            results[key] = (t, compute_metrics(t))

    print("=" * 78)
    print("STOP x TRAIL SWEEP -- EXTENDED GRID, FULL 2-YEAR RESULTS")
    print("=" * 78)
    print(f"{'Combo':<14}{'n':>6}{'win%':>8}{'PF':>8}{'net_Rs':>12}{'maxDD_Rs':>12}{'SQN':>7}{'CAGR%':>9}")
    for key, (t, m) in results.items():
        pf_str = "inf" if m["pf"] == float("inf") else str(m["pf"])
        print(f"{key:<14}{m['n']:>6}{m['win_pct']:>8}{pf_str:>8}{m['net_rs']:>12}{m['max_dd_rs']:>12}"
              f"{m['sqn']:>7}{m['cagr_pct']:>9}")

    best_key = max(results, key=lambda k: results[k][1]["sqn"])
    best_t, best_m = results[best_key]
    print(f"\nBest combo by SQN: {best_key}  (n={best_m['n']}, PF={best_m['pf']}, SQN={best_m['sqn']}, "
          f"net_Rs={best_m['net_rs']}, CAGR%={best_m['cagr_pct']})")

    # grid-edge check: is the best combo sitting at the boundary of the tested range?
    stop_used = int(best_key.split("_")[0].replace("sl", ""))
    trail_used = int(best_key.split("_")[1].replace("trail", ""))
    at_stop_edge = stop_used == max(STOP_VALUES)
    at_trail_edge = trail_used == min(TRAIL_VALUES)
    print(f"Grid-edge check: stop={stop_used} is {'AT' if at_stop_edge else 'within'} the widest tested value "
          f"({max(STOP_VALUES)}); trail={trail_used} is {'AT' if at_trail_edge else 'within'} the tightest "
          f"tested value ({min(TRAIL_VALUES)}).")
    if at_stop_edge or at_trail_edge:
        print("WARNING: best combo is still at a grid boundary -- treat with continued suspicion "
              "(see nifty_backtest_playbook.md, checklist item 8).")
    else:
        print("Best combo is an INTERIOR point of the grid -- a real sweet spot, not a boundary artifact.")

    print("\n" + "=" * 78)
    print(f"IN-SAMPLE vs OUT-OF-SAMPLE SPLIT FOR BEST COMBO ({best_key})")
    print("=" * 78)
    first_half, second_half, midpoint = split_in_sample_out_of_sample(best_t)
    if first_half is not None:
        m1 = compute_metrics(first_half)
        m2 = compute_metrics(second_half)
        print(f"Split date (midpoint): {midpoint.date()}")
        print(f"First half:  n={m1['n']}  win%={m1['win_pct']}  PF={m1['pf']}  net_Rs={m1['net_rs']}  "
              f"maxDD_Rs={m1['max_dd_rs']}  SQN={m1['sqn']}")
        print(f"Second half: n={m2['n']}  win%={m2['win_pct']}  PF={m2['pf']}  net_Rs={m2['net_rs']}  "
              f"maxDD_Rs={m2['max_dd_rs']}  SQN={m2['sqn']}")

    for key, (t, _) in results.items():
        if not t.empty:
            fname = f"open_momentum_sweep_{key}_trades.csv"
            t.to_csv(fname, index=False)
    print("\nSaved per-combo trade CSVs (open_momentum_sweep_*_trades.csv)")


if __name__ == "__main__":
    main()
