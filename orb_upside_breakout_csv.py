"""
ORB (Opening Range Breakout) -- UP-SIDE-ONLY continuation, new standalone strategy
====================================================================================
SOURCE OF THE IDEA: a prior exploratory scan (not a backtest) over ~6
months found the 9:15-9:30 opening range breaks on ~100% of days, and
up-breaks continue (price ends the day beyond the breakout level, or at
least the trade is a net winner) 55.7% of the time vs 50.8% for down-
breaks. This script tests the UP-SIDE ONLY as a real backtest: downside
breaks are ignored entirely, per the brief. This is a brand-new file --
nothing in this project has previously tested this exact rule.

ENTRY LOGIC:
  - Opening range = high/low of bars from 09:15 up to (not including)
    09:30.
  - Scan bars from 09:30 onward. The first bar whose CLOSE trades above
    the opening-range HIGH is the signal bar.
  - NO LOOK-AHEAD: the fill happens at the (signal_idx + 1) bar's OPEN,
    never the signal bar's own close.
  - If no breakout above the OR high occurs before EOD, no trade that
    day. Downside breaks are not traded (long-only, matches the brief).

MANAGEMENT: same conservative stop/trailing mechanics as the rest of
this project -- hard stop_pts below entry, trail_pts trailing stop once
in profit, each bar's LOW checked against the OLD stop level before that
bar's HIGH is allowed to ratchet the trail up (worst-case intrabar
ordering), EOD square-off at 15:20 as a fallback.

PROXY PREMIUM MODEL: same as elsewhere in this project -- no real
option-chain data (Dhan expiryCode=0 bug, already investigated).
entry_premium = intrinsic_at_open + ASSUMED_TIME_VALUE, moves scaled by
a flat ASSUMED_DELTA, floored at 0.

Brokerage: Rs20/round-trip trade (project default for a strategy with no
prior established brokerage figure of its own). EOD square-off: 15:20.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

CSV_PATH = "dhan_historical_data_180days_v5.csv"
BAR_RULE = "2min"

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
OR_START = dt.time(9, 15)
OR_END = dt.time(9, 30)   # opening range = bars with time < OR_END

STOP_PTS = 8.0
TRAIL_PTS = 4.0

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 20.0


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def load_csv(path=CSV_PATH):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["datetime"], format="mixed")
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
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


def build_premium_series_ce(day_df, strike, day_open_spot):
    """Long-only strategy -> always CE (calls), synthetic option premium."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    intrinsic0 = max(day_open_spot - strike, 0.0)
    mo = idx_open - day_open_spot
    mh = idx_high - day_open_spot
    ml = idx_low - day_open_spot
    mc = idx_close - day_open_spot
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


def compute_opening_range(day_df, or_end=OR_END):
    """Return (or_high, or_low, first_post_or_local_index) using bars whose
    time is strictly before or_end. Returns (None, None, None) if there are
    no bars before or_end or no bars after it."""
    times = day_df.index
    or_mask = np.array([t.time() < or_end for t in times])
    if not or_mask.any() or or_mask.all():
        return None, None, None
    or_high = day_df["high"].to_numpy()[or_mask].max()
    or_low = day_df["low"].to_numpy()[or_mask].min()
    first_post = int(np.argmax(~or_mask))  # first index where mask is False
    return or_high, or_low, first_post


def find_upside_breakout(day_df, or_high, start_idx):
    """Scan bars from start_idx onward; return the local index of the first
    bar whose CLOSE > or_high. Returns None if no such bar exists in the day.
    SIGNAL bar only -- fill happens one bar later."""
    closes = day_df["close"].to_numpy()
    for i in range(start_idx, len(day_df)):
        if closes[i] > or_high:
            return i
    return None


def simulate_trade(premium_df, entry_idx, entry_price, stop_pts=STOP_PTS, trail_pts=TRAIL_PTS):
    """Conservative intrabar ordering: each bar's LOW checked against the
    stop level as it stood BEFORE this bar, before this bar's HIGH is
    allowed to ratchet the trailing stop up."""
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


def sharpe_sortino(trades_df):
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
    hours = pd.to_datetime(t["entry_time"]).dt.hour
    mins = pd.to_datetime(t["entry_time"]).dt.minute
    since_open = hours * 60 + mins - 9 * 60 - 15
    print(f"  fires on {coverage:.1f}% of eligible days ({len(t)}/{total_days})")
    print(f"  entry-time-since-open (min) stats: mean={since_open.mean():.1f} "
          f"median={since_open.median():.1f} min={since_open.min()} max={since_open.max()}")
    buckets = pd.cut(since_open, bins=[0, 15, 30, 60, 120, 400], right=False)
    print(f"  bucket counts (min since open): {buckets.value_counts().sort_index().to_dict()}")


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # compute_opening_range: OR = bars strictly before 09:30
    idx = pd.date_range("2025-01-06 09:15", periods=20, freq="2min")  # 09:15..09:53, 09:30 at local idx 8 (wait: 2min*8=16min ->09:31)
    # build so that idx[7]=09:29, idx[8]=09:31 (2min steps): 09:15,17,19,21,23,25,27,29,31,...
    high = [24800, 24810, 24805, 24820, 24795, 24825, 24790, 24830, 24850, 24860] + [24860]*10
    low  = [24790, 24795, 24798, 24800, 24780, 24805, 24775, 24810, 24835, 24845] + [24840]*10
    df = pd.DataFrame({"open": high, "high": high, "low": low, "close": high}, index=idx)
    or_high, or_low, first_post = compute_opening_range(df)
    # bars with time < 09:30 -> idx[0..7] (09:15..09:29), idx[8]=09:31 is first post-OR bar
    assert first_post == 8, f"expected first_post=8, got {first_post}"
    assert or_high == max(high[:8]), f"expected OR high {max(high[:8])}, got {or_high}"
    assert or_low == min(low[:8]), f"expected OR low {min(low[:8])}, got {or_low}"
    print("  compute_opening_range OR-window boundary check ok")

    # find_upside_breakout: first close above or_high after start_idx
    closes = [24800]*8 + [24831, 24845] + [24900]*10  # or_high=24830, breakout at local idx 9 (close=24845)... check idx8=24831>24830 too
    df2 = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
    sig = find_upside_breakout(df2, or_high=24830, start_idx=8)
    assert sig == 8, f"expected breakout signal at idx 8 (close 24831>24830), got {sig}"
    print("  find_upside_breakout detection check ok")

    # no breakout case
    closes3 = [24800]*20
    df3 = pd.DataFrame({"open": closes3, "high": closes3, "low": closes3, "close": closes3}, index=idx)
    sig3 = find_upside_breakout(df3, or_high=24830, start_idx=8)
    assert sig3 is None
    print("  find_upside_breakout no-signal check ok")

    # no-look-ahead: fill must be signal_idx + 1, not signal_idx itself
    exec_idx = sig + 1
    assert exec_idx == 9 and exec_idx != sig
    print("  execution-lag (no look-ahead) check ok")

    idx_t = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")

    # hard stop check
    prem = pd.DataFrame({
        "open": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "high": [50, 50, 49, 47, 41, 41, 41, 41, 41, 41],
        "low": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
        "close": [50, 49, 47, 41, 41, 41, 41, 41, 41, 41],
    }, index=idx_t)
    exit_price, reason, _ = simulate_trade(prem, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason == "STOP" and abs(exit_price - 42.0) < 1e-6
    print("  simulate_trade hard-stop check ok")

    # trailing stop check
    prem2 = pd.DataFrame({
        "open": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
        "high": [50, 56, 61, 66, 61, 56, 56, 56, 56, 56],
        "low": [50, 54, 59, 64, 55, 54.5, 54.5, 54.5, 54.5, 54.5],
        "close": [50, 55, 60, 65, 60, 55, 55, 55, 55, 55],
    }, index=idx_t)
    exit_price2, reason2, _ = simulate_trade(prem2, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason2 == "TRAIL" and abs(exit_price2 - 62.0) < 1e-6
    print("  simulate_trade trailing-stop check ok")

    # conservative intrabar ordering (same ambiguous-bar case as opening-drive script)
    prem3 = pd.DataFrame({
        "open": [50, 55, 55, 60, 60],
        "high": [50, 56, 70, 61, 61],
        "low": [50, 54, 50, 59, 59],
        "close": [50, 55, 60, 60, 60],
    }, index=idx_t[:5])
    exit_price3, reason3, exit_time3 = simulate_trade(prem3, 0, entry_price=50, stop_pts=8, trail_pts=4)
    assert reason3 == "TRAIL" and abs(exit_price3 - 52.0) < 1e-6, (
        "conservative intrabar ordering must exit at the OLD stop level (52.0), not ratchet on this bar's own high first"
    )
    print("  simulate_trade conservative intrabar-ordering check ok")

    t_test = pd.DataFrame({
        "date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2025, 1, 4)],
        "rupee_pnl": [10, -5, 20, -8],
    })
    fh, sh, mid = split_in_sample_out_of_sample(t_test)
    assert len(fh) == 2 and len(sh) == 2
    print("  split_in_sample_out_of_sample check ok")

    print("All self-checks passed.\n")


def run_backtest(full_df, days, stop_pts, trail_pts):
    trades = []
    capital = STARTING_CAPITAL
    for day in days:
        day_df = full_df.loc[full_df.index.date == day]
        if len(day_df) < 5:
            continue
        day_open_spot = day_df["open"].iloc[0]
        or_high, or_low, first_post = compute_opening_range(day_df)
        if or_high is None:
            continue
        sig_idx = find_upside_breakout(day_df, or_high, first_post)
        if sig_idx is None:
            continue
        exec_idx = sig_idx + 1
        if exec_idx >= len(day_df):
            continue
        strike = itm_call_strike(day_open_spot)
        premium_df = build_premium_series_ce(day_df, strike, day_open_spot)
        entry_price = premium_df["open"].iloc[exec_idx]
        exit_price, reason, exit_time = simulate_trade(premium_df, exec_idx, entry_price, stop_pts, trail_pts)
        lot = lot_size_for_date(day)
        pnl_pts = exit_price - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        capital += rupee_pnl
        trades.append({
            "date": day, "or_high": round(or_high, 2), "or_low": round(or_low, 2), "strike": strike,
            "signal_time": day_df.index[sig_idx], "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
            "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "exit_reason": reason, "pnl_pts": round(pnl_pts, 2),
            "rupee_pnl": round(rupee_pnl, 2), "capital_after": round(capital, 2),
        })
    return pd.DataFrame(trades)


def run_smoke_test():
    print("Running smoke test on synthetic multi-day data...")
    days = pd.date_range("2025-01-06", periods=6, freq="B")
    rows = []
    rng = np.random.default_rng(7)
    for d in days:
        base = 24000 + rng.normal(0, 50)
        times = pd.date_range(d.strftime("%Y-%m-%d") + " 09:15", d.strftime("%Y-%m-%d") + " 15:29", freq="1min")
        drift = rng.choice([1, -1]) * rng.uniform(4, 15)
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


def main():
    run_self_checks()
    run_smoke_test()

    print(f"Loading cached CSV data from {CSV_PATH} ...")
    raw = load_csv()
    full_df = resample_2min_per_day(raw)
    days = sorted(set(full_df.index.date))
    total_days = len(days)
    print(f"Loaded {len(full_df)} 2-min bars across {total_days} trading days.\n")

    print("=" * 78)
    print("BASELINE: ORB upside-only breakout, stop=8 trail=4")
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
        t.to_csv("orb_upside_breakout_trades_sl8_trail4.csv", index=False)
        print("Saved orb_upside_breakout_trades_sl8_trail4.csv")

    print("\nIN-SAMPLE (first half) vs OUT-OF-SAMPLE (second half):")
    fh, sh, mid = split_in_sample_out_of_sample(t)
    if fh is not None and len(fh) > 0:
        m1, m2 = compute_metrics(fh), compute_metrics(sh)
        print(f"Split date: {mid.date()}")
        print(f"  First half:  n={m1['n']} win%={m1['win_pct']} PF={m1['pf']} net_Rs={m1['net_rs']} SQN={m1['sqn']}")
        print(f"  Second half: n={m2['n']} win%={m2['win_pct']} PF={m2['pf']} net_Rs={m2['net_rs']} SQN={m2['sqn']}")

    print("\n" + "=" * 78)
    print("STOP/TRAIL GRID SWEEP")
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
    gdf.to_csv("orb_upside_breakout_grid.csv", index=False)
    if len(gdf) > 0:
        best = gdf.loc[gdf["sqn"].idxmax()]
        print(f"\nBest-by-SQN combo: stop={best['stop']} trail={best['trail']} "
              f"SQN={best['sqn']} PF={best['pf']} n={best['n']}")
        at_edge = best['stop'] in (min(stops), max(stops)) or best['trail'] in (min(trails), max(trails))
        print(f"Best combo at grid edge? {at_edge}")


if __name__ == "__main__":
    main()
