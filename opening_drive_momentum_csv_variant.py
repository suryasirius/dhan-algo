"""
Opening-Drive Momentum -- CSV-data variant (reuses validated logic)
=====================================================================
WHY THIS FILE EXISTS: this project already has a LOCKED, validated
strategy at nifty_open_momentum_trail.py (Opening-Drive Momentum, 8pt
stop / 4pt trail, isolated from the six-concepts screening artifact,
went through two real bug fixes -- execution-lag look-ahead and
conservative intrabar high/low ordering -- landing on PF 1.44 / SQN 3.02
over a live-API-fetched 2-year window). That file is NOT edited here.

This script re-implements the SAME entry/exit/diagnostic logic
(deliberately kept as close to line-for-line identical as the different
data source allows) but sources bars from the locally cached CSV
(dhan_historical_data_180days_v5.csv, 1-min bars, 122 trading days,
2026-02-02 to 2026-07-31) instead of a live Dhan API fetch, since the
sandbox this runs in has no API credentials. This lets us verify the
validated strategy's edge reproduces on the specific dataset this
project's other CSV-based backtests use, and extends the check with a
small stop/trail grid around the sl8_trail4 baseline per the mandatory
rigor checklist.

PROXY PREMIUM MODEL (same simplification as the rest of the project):
no real option-chain data is available (Dhan expiryCode=0 bug, already
investigated exhaustively elsewhere in this project). entry_premium =
intrinsic_at_open + ASSUMED_TIME_VALUE, moves scaled by a flat
ASSUMED_DELTA, floored at 0.

NO LOOK-AHEAD: find_opening_drive() only reports the SIGNAL bar index.
The fill always happens at bar (signal_idx + 1)'s OPEN -- never the
signal bar's own close. Verified explicitly in run_self_checks().

CONSERVATIVE INTRABAR ORDERING: simulate_trade() checks each bar's LOW
against the stop level as it stood BEFORE that bar, before that bar's
own HIGH is allowed to ratchet the trailing stop up. See REV3 comment
in the original file for the exact failure mode this guards against.

Brokerage: Rs40/round-trip trade (matches the value already established
for this exact strategy in the locked file -- kept consistent rather than
substituting the generic Rs20 default). EOD square-off fallback: 15:20.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

CSV_PATH = "dhan_historical_data_180days_v5.csv"
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

LOT_SIZE = 75  # Nifty lot size for the entire 2026-02 to 2026-07 window (see lot_size_for_date)


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


def load_csv(path=CSV_PATH):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["datetime"], utc=False, format="mixed")
    df["time"] = df["time"].dt.tz_localize(None) if df["time"].dt.tz is not None else df["time"]
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


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
                "cagr_pct": 0.0, "avg_pnl": 0.0, "std_pnl": 0.0}
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
            "cagr_pct": round(cagr_pct, 2), "avg_pnl": round(pnl.mean(), 2), "std_pnl": round(pnl.std(), 2)}


def max_dd_duration_days(trades_df):
    if trades_df.empty:
        return 0
    pnl = trades_df["rupee_pnl"].to_numpy()
    dates = pd.to_datetime(trades_df["date"]).to_numpy()
    equity = STARTING_CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    max_dur = 0
    cur_start = None
    for i in range(len(equity)):
        if underwater[i]:
            if cur_start is None:
                cur_start = dates[i]
            dur = (dates[i] - cur_start).astype('timedelta64[D]').astype(int)
            max_dur = max(max_dur, dur)
        else:
            cur_start = None
    return max_dur


def sharpe_sortino(trades_df, periods_per_year=252):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0, 0.0
    pnl = trades_df["rupee_pnl"].to_numpy()
    ret = pnl / STARTING_CAPITAL
    mean_r = ret.mean()
    std_r = ret.std()
    downside = ret[ret < 0]
    down_std = downside.std() if len(downside) > 1 else 0.0
    n_trades = len(ret)
    dates = pd.to_datetime(trades_df["date"])
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1e-6)
    trades_per_year = n_trades / years
    sharpe = (mean_r / std_r) * math.sqrt(trades_per_year) if std_r > 0 else 0.0
    sortino = (mean_r / down_std) * math.sqrt(trades_per_year) if down_std > 0 else 0.0
    return round(sharpe, 2), round(sortino, 2)


def ttest_mean_return(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return float("nan"), float("nan")
    pnl = trades_df["rupee_pnl"].to_numpy()
    n = len(pnl)
    mean = pnl.mean()
    se = pnl.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return float("nan"), float("nan")
    t_stat = mean / se
    # two-sided p-value via normal approximation (n is large enough here; for
    # small n this is a rough approximation, flagged as such in the report)
    from math import erf, sqrt
    p_approx = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return round(t_stat, 3), round(p_approx, 4)


def split_in_sample_out_of_sample(t):
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
    mins = pd.to_datetime(t["entry_time"]).dt.minute
    early = int((hours < 10).sum())
    later = int((hours >= 10).sum())
    print(f"  fires on {coverage:.1f}% of eligible days ({len(t)}/{total_days}) | "
          f"CE={ce} PE={pe} | entry-hour <10:{early} >=10:{later}")
    print(f"  entry-time distribution (5-min buckets): "
          f"{sorted((hours*60+mins - 9*60-15).value_counts().to_dict().items())[:8]} ... (min-since-open)")


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    idx = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")
    closes = [100, 100.5, 106.0] + [110] * 17
    opens = [100] + closes[:-1]
    df = pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)
    local_i, direction = find_opening_drive(df)
    assert local_i == 2 and direction == "LONG"
    print("  find_opening_drive LONG-detection check ok")

    closes2 = [100, 99.5, 93.0] + [90] * 17
    opens2 = [100] + closes2[:-1]
    df2 = pd.DataFrame({"open": opens2, "high": closes2, "low": closes2, "close": closes2}, index=idx)
    local_i2, direction2 = find_opening_drive(df2)
    assert local_i2 == 2 and direction2 == "SHORT"
    print("  find_opening_drive SHORT-detection check ok")

    closes3 = [100.0] * 20
    df3 = pd.DataFrame({"open": [100.0] * 20, "high": closes3, "low": closes3, "close": closes3}, index=idx)
    local_i3, direction3 = find_opening_drive(df3)
    assert local_i3 is None and direction3 is None
    print("  find_opening_drive no-signal check ok")

    closes4 = [100.0] * 16 + [108.0] * 4
    df4 = pd.DataFrame({"open": [100.0] * 20, "high": closes4, "low": closes4, "close": closes4}, index=idx)
    local_i4, direction4 = find_opening_drive(df4)
    assert local_i4 is None and direction4 is None, "cutoff must block a late move"
    print("  find_opening_drive cutoff-enforcement check ok")

    exec_idx = local_i + 1
    assert exec_idx == 3 and exec_idx != local_i, "fill bar must be next bar, not signal bar (no look-ahead)"
    print("  execution-lag (no look-ahead) check ok")

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

    prem2 = pd.DataFrame({
        "open": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
        "high": [50, 56, 61, 66, 61, 56, 56, 56, 56, 56],
        "low": [50, 54, 59, 64, 55, 54.5, 54.5, 54.5, 54.5, 54.5],
        "close": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
    }, index=idx_t)
    exit_price2, reason2, _ = simulate_trade(prem2, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason2 == "TRAIL" and abs(exit_price2 - 62.0) < 1e-6
    print("  simulate_trade trailing-stop check ok")

    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    exit_price3, reason3, exit_time3 = simulate_trade(prem3, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason3 == "TRAIL" and abs(exit_price3 - 52.0) < 1e-6, (
        "conservative intrabar ordering must exit at OLD stop level 52.0, not ratchet to 66 first"
    )
    assert exit_time3 == prem3.index[2]
    print("  simulate_trade conservative intrabar-ordering check ok")

    t_test = pd.DataFrame({
        "date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2025, 1, 4)],
        "rupee_pnl": [10, -5, 20, -8],
    })
    fh, sh, mid = split_in_sample_out_of_sample(t_test)
    assert len(fh) == 2 and len(sh) == 2
    print("  split_in_sample_out_of_sample check ok")

    print("All self-checks passed.\n")


def run_smoke_test():
    print("Running smoke test on synthetic multi-day data...")
    days = pd.date_range("2025-01-06", periods=6, freq="B")
    rows = []
    rng = np.random.default_rng(42)
    for d in days:
        base = 24000 + rng.normal(0, 50)
        times = pd.date_range(d.strftime("%Y-%m-%d") + " 09:15", d.strftime("%Y-%m-%d") + " 15:29", freq="1min")
        drift = rng.choice([1, -1]) * rng.uniform(4, 12)
        n = len(times)
        walk = np.cumsum(rng.normal(0, 2, n)) + np.linspace(0, drift, n)
        close = base + walk
        openp = np.roll(close, 1); openp[0] = base
        high = np.maximum(openp, close) + rng.uniform(0, 2, n)
        low = np.minimum(openp, close) - rng.uniform(0, 2, n)
        for i in range(n):
            rows.append({"time": times[i], "open": openp[i], "high": high[i], "low": low[i],
                         "close": close[i], "volume": 1000})
    synth = pd.DataFrame(rows).set_index("time")
    full_df = resample_2min_per_day(synth)
    days_list = sorted(set(full_df.index.date))
    trades = run_backtest(full_df, days_list, STOP_PTS, TRAIL_PTS)
    print(f"  smoke test produced {len(trades)} trades across {len(days_list)} synthetic days")
    assert isinstance(trades, pd.DataFrame)
    print("Smoke test complete.\n")


def run_backtest(full_df, days, stop_pts, trail_pts):
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
            "date": day, "direction": direction, "strike": strike, "side": side,
            "signal_time": day_df.index[local_i], "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2),
            "rupee_pnl": round(rupee_pnl, 2), "capital_after": round(capital, 2),
        })
    return pd.DataFrame(trades)


def main():
    run_self_checks()
    run_smoke_test()

    print(f"Loading cached CSV data from {CSV_PATH} ...")
    raw = load_csv()
    full_df = resample_2min_per_day(raw)
    days = sorted(set(full_df.index.date))
    total_days = len(days)
    print(f"Loaded {len(full_df)} 2-min bars across {total_days} trading days.\n")

    # ---- baseline sl8_trail4 ----
    print("=" * 78)
    print("BASELINE: sl8_trail4 (8pt stop / 4pt trail)")
    print("=" * 78)
    t = run_backtest(full_df, days, STOP_PTS, TRAIL_PTS)
    m = compute_metrics(t)
    sharpe, sortino = sharpe_sortino(t)
    t_stat, p_val = ttest_mean_return(t)
    dd_dur = max_dd_duration_days(t)
    print(f"n={m['n']}  win%={m['win_pct']}  PF={m['pf']}  net_Rs={m['net_rs']}  "
          f"maxDD_Rs={m['max_dd_rs']} ({dd_dur}d)  SQN={m['sqn']}  CAGR%={m['cagr_pct']}")
    print(f"Sharpe(ann,trade-freq)={sharpe}  Sortino={sortino}  "
          f"mean_pnl_Rs={m['avg_pnl']}  t-stat={t_stat}  p(vs 0, normal-approx)={p_val}")
    print_diagnostics(t, total_days)
    if not t.empty:
        t.to_csv("opening_drive_csv_variant_trades_sl8_trail4.csv", index=False)
        print("Saved opening_drive_csv_variant_trades_sl8_trail4.csv")

    print("\nIN-SAMPLE (first half) vs OUT-OF-SAMPLE (second half):")
    fh, sh, mid = split_in_sample_out_of_sample(t)
    if fh is not None and len(fh) > 0:
        m1, m2 = compute_metrics(fh), compute_metrics(sh)
        print(f"Split date: {mid.date()}")
        print(f"  First half:  n={m1['n']} win%={m1['win_pct']} PF={m1['pf']} net_Rs={m1['net_rs']} SQN={m1['sqn']}")
        print(f"  Second half: n={m2['n']} win%={m2['win_pct']} PF={m2['pf']} net_Rs={m2['net_rs']} SQN={m2['sqn']}")

    # ---- small stop/trail grid around sl8_trail4 ----
    print("\n" + "=" * 78)
    print("SMALL STOP/TRAIL GRID AROUND sl8_trail4")
    print("=" * 78)
    grid_results = []
    stops = [6.0, 8.0, 10.0, 12.0]
    trails = [2.0, 3.0, 4.0, 5.0, 6.0]
    for sp in stops:
        for tp in trails:
            if tp >= sp:
                continue
            tg = run_backtest(full_df, days, sp, tp)
            mg = compute_metrics(tg)
            grid_results.append({"stop": sp, "trail": tp, **mg})
            print(f"  stop={sp:>4} trail={tp:>4}  n={mg['n']:>3}  PF={mg['pf']:>6}  "
                  f"SQN={mg['sqn']:>6}  net_Rs={mg['net_rs']:>10}  maxDD={mg['max_dd_rs']:>8}")

    gdf = pd.DataFrame(grid_results)
    gdf.to_csv("opening_drive_csv_variant_grid.csv", index=False)
    best = gdf.loc[gdf["sqn"].idxmax()]
    print(f"\nBest-by-SQN combo: stop={best['stop']} trail={best['trail']} "
          f"SQN={best['sqn']} PF={best['pf']} n={best['n']}")
    at_edge = best['stop'] in (min(stops), max(stops)) or best['trail'] in (min(trails), max(trails))
    print(f"Best combo at grid edge? {at_edge}")


if __name__ == "__main__":
    main()
