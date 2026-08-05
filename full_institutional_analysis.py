"""
Full institutional-grade analysis for our two validated "top" strategies,
reading their REAL trade logs directly (no re-fetch needed):
  - velez_8pt_proxy_premium_trades.csv    (Velez 8pt, proxy premium + exec-lag fix)
  - open_momentum_sweep_sl8_trail4_trades.csv (Opening-Drive Momentum, sl8_trail4)

This closes the gap flagged when asked "did you check with ALL tests and
analysis" -- prior rounds covered self-checks, smoke tests, look-ahead
fix, intrabar ordering, entry-frequency diagnostics, in-sample/out-of-
sample split, and the grid-edge check, but NOT: Sharpe, Sortino, max
drawdown duration, VaR/CVaR, a significance test on mean return, or a
Monte Carlo trade-order-reshuffle capital-stress test (the ORIGINAL
locked velez_final_8pt.py already had this Monte Carlo check -- the new
proxy-premium retest never got it, and Opening-Drive Momentum never had
it either). This adds all of it, built without scipy (same convention as
the earlier full-metrics-sweep script in this project).
"""

import math
import datetime as dt
import random
import pandas as pd
import numpy as np

STARTING_CAPITAL = 25_000
MONTE_CARLO_ITERS = 5000


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_daily_pnl(trades_df):
    """Aggregate trade-level rupee P&L into one total per calendar day
    (a strategy can have multiple trades per day, e.g. Velez 8pt)."""
    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    daily = df.groupby("date")["rupee_pnl"].sum().sort_index()
    return daily


def full_time_capital_series(daily_pnl):
    """Reindex a sparse daily P&L series (only days with a trade) onto
    every business day between the first and last trade date, filling
    zero P&L on days with no trade. This answers a different, important
    question than the active-day-only series: 'how does this look for
    the WHOLE account, including all the days I wasn't in a trade?'
    Approximates the real NSE trading calendar with business days
    (Mon-Fri) -- doesn't account for market holidays, a minor
    simplification worth knowing but not a big source of error."""
    idx = pd.to_datetime(pd.Series(daily_pnl.index))
    full_range = pd.bdate_range(idx.iloc[0], idx.iloc[-1])
    reindexed = daily_pnl.copy()
    reindexed.index = pd.to_datetime(reindexed.index)
    return reindexed.reindex(full_range, fill_value=0.0)


def full_metrics(daily_pnl, starting_capital=STARTING_CAPITAL):
    equity = starting_capital + daily_pnl.cumsum()
    prior_equity = pd.concat([pd.Series([starting_capital]), equity.iloc[:-1]])
    prior_equity.index = equity.index
    daily_returns = (daily_pnl.to_numpy() / prior_equity.to_numpy())

    n_days = len(daily_pnl)
    # IMPORTANT: years-elapsed must come from the actual CALENDAR date span
    # (last trade date - first trade date), NOT from n_days/252. A strategy
    # that only trades on a subset of days (e.g. Velez 8pt trades ~176 of
    # ~497 days in a 2-year window) would otherwise have its timeline
    # compressed to ~0.7 "years" instead of the real ~2, wildly inflating
    # CAGR. Confirmed as a real bug during this analysis -- Velez showed
    # CAGR=186% before this fix vs the correct ~44% once real elapsed time
    # is used.
    date_index = pd.to_datetime(pd.Series(daily_pnl.index))
    years = (date_index.iloc[-1] - date_index.iloc[0]).days / 365.25
    cagr_pct = 100.0 * ((equity.iloc[-1] / starting_capital) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    mean_r = daily_returns.mean()
    std_r = daily_returns.std(ddof=1) if len(daily_returns) > 1 else 0.0
    sharpe_active = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino_active = (mean_r / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    # Full-time-capital version: reindex onto every business day in the real
    # span (zero-filling non-trade days) before computing Sharpe/Sortino.
    # For a SPARSE strategy (trades on a minority of days), the active-day-
    # only Sharpe above overstates risk-adjusted return relative to how it'd
    # look holding the whole account through the quiet days too -- this is
    # the more conservative, standard-convention number.
    ft_series = full_time_capital_series(daily_pnl)
    ft_equity = starting_capital + ft_series.cumsum()
    ft_prior_equity = pd.concat([pd.Series([starting_capital]), ft_equity.iloc[:-1]])
    ft_prior_equity.index = ft_equity.index
    ft_returns = (ft_series.to_numpy() / ft_prior_equity.to_numpy())
    ft_mean = ft_returns.mean()
    ft_std = ft_returns.std(ddof=1) if len(ft_returns) > 1 else 0.0
    sharpe_fulltime = (ft_mean / ft_std * math.sqrt(252)) if ft_std > 0 else 0.0
    ft_downside = ft_returns[ft_returns < 0]
    ft_downside_std = ft_downside.std(ddof=1) if len(ft_downside) > 1 else 0.0
    sortino_fulltime = (ft_mean / ft_downside_std * math.sqrt(252)) if ft_downside_std > 0 else 0.0

    peak = np.maximum.accumulate(equity.to_numpy())
    dd_pct = (equity.to_numpy() - peak) / peak * 100.0
    max_dd_pct = dd_pct.min()

    # drawdown duration: days strictly below the running peak, up to but
    # not including the day a new peak is set (established convention)
    below_peak = equity.to_numpy() < peak
    max_duration = 0
    cur = 0
    for b in below_peak:
        if b:
            cur += 1
            max_duration = max(max_duration, cur)
        else:
            cur = 0

    var95_pct = np.percentile(daily_returns, 5) * 100.0
    cvar_threshold = np.percentile(daily_returns, 5)
    tail = daily_returns[daily_returns <= cvar_threshold]
    cvar95_pct = (tail.mean() if len(tail) > 0 else cvar_threshold) * 100.0

    t_stat = (mean_r / (std_r / math.sqrt(n_days))) if std_r > 0 and n_days > 1 else 0.0
    p_value = 2 * (1 - norm_cdf(abs(t_stat)))

    return {
        "n_days": n_days, "n_calendar_bdays": len(ft_series), "cagr_pct": round(cagr_pct, 2),
        "sharpe_active_days": round(sharpe_active, 2), "sortino_active_days": round(sortino_active, 2),
        "sharpe_fulltime": round(sharpe_fulltime, 2), "sortino_fulltime": round(sortino_fulltime, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dd_duration_days": max_duration, "var95_pct": round(var95_pct, 2),
        "cvar95_pct": round(cvar95_pct, 2), "t_stat": round(t_stat, 2), "p_value": round(p_value, 4),
        "final_equity": round(equity.iloc[-1], 2),
    }


def monte_carlo_reshuffle(trades_df, starting_capital=STARTING_CAPITAL, iters=MONTE_CARLO_ITERS, seed=42):
    """Reshuffle TRADE-level (not daily) P&L order N times -- same
    convention as the original velez_final_8pt.py's Monte Carlo check --
    to see how sensitive the outcome is to the specific sequence of wins
    and losses that happened to occur."""
    random.seed(seed)
    pnl_list = trades_df["rupee_pnl"].tolist()
    final_equities, max_dds_pct, min_equities = [], [], []
    for _ in range(iters):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = starting_capital + np.cumsum(shuffled)
        peak = np.maximum.accumulate(curve)
        dd_pct = (curve - peak) / peak * 100.0
        final_equities.append(curve[-1])
        max_dds_pct.append(dd_pct.min())
        min_equities.append(curve.min())
    fe = np.array(final_equities)
    mdd = np.array(max_dds_pct)
    me = np.array(min_equities)
    return {
        "final_eq_p5": round(np.percentile(fe, 5), 2), "final_eq_p50": round(np.percentile(fe, 50), 2),
        "final_eq_p95": round(np.percentile(fe, 95), 2),
        "max_dd_pct_p5": round(np.percentile(mdd, 5), 2), "max_dd_pct_p50": round(np.percentile(mdd, 50), 2),
        "max_dd_pct_p95": round(np.percentile(mdd, 95), 2),
        "pct_paths_wipeout": round(100.0 * (me <= 0).mean(), 3),
        "pct_paths_below_50pct_capital": round(100.0 * (fe < starting_capital * 0.5).mean(), 2),
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # --- norm_cdf sanity ---
    assert abs(norm_cdf(0) - 0.5) < 1e-9
    assert norm_cdf(3) > 0.998
    print("  norm_cdf check ok")

    # --- full_metrics on a hand-traceable series ---
    # 5 days: +1000, -500, +1000, -500, +1000 on a 10,000 base
    dates = pd.date_range("2025-01-06", periods=5, freq="D").date
    pnl = pd.Series([1000, -500, 1000, -500, 1000], index=dates)
    m = full_metrics(pnl, starting_capital=10_000)
    # equity: 11000, 10500, 11500, 11000, 12000 -- final should be 12000
    assert abs(m["final_equity"] - 12000) < 1e-6, f"expected final equity 12000, got {m['final_equity']}"
    # peak sequence: 11000,11000,11500,11500,12000 -> dd% at day2 = (10500-11000)/11000*100 = -4.545...
    assert m["max_dd_pct"] < 0, "expected a real (negative) drawdown at day 2"
    assert m["sharpe_active_days"] > 0, "expected a positive Sharpe for a net-winning, mostly-up series"
    print(f"  full_metrics hand-traced check ok (final_equity={m['final_equity']}, max_dd_pct={m['max_dd_pct']})")

    # --- CAGR must use the real calendar date span, not the count of
    # active trading days -- a sparse strategy (few trades spread across
    # a wide date range) must NOT get its timeline compressed ---
    sparse_dates = [dt.date(2024, 1, 1), dt.date(2024, 7, 1), dt.date(2026, 1, 1)]  # only 3 trades, but spans ~2 years
    sparse_pnl = pd.Series([5000, 5000, 5000], index=sparse_dates)  # 15,000 total on 10,000 capital -> 25,000 final
    m_sparse = full_metrics(sparse_pnl, starting_capital=10_000)
    real_years = (sparse_dates[-1] - sparse_dates[0]).days / 365.25
    expected_cagr = 100.0 * ((25_000 / 10_000) ** (1.0 / real_years) - 1.0)
    assert abs(m_sparse["cagr_pct"] - expected_cagr) < 0.5, (
        f"CAGR must use the real ~2yr calendar span (expected ~{expected_cagr:.1f}%), "
        f"not n_days=3 as a stand-in for years (which would wildly overstate CAGR), got {m_sparse['cagr_pct']}%"
    )
    print(f"  CAGR uses real calendar span, not active-day count, check ok (cagr={m_sparse['cagr_pct']}%)")

    # --- full-time-capital Sharpe must differ from active-day Sharpe for a
    # SPARSE strategy (trades on a small fraction of the real business days
    # in its span) -- diluting with zero-return days is the whole point ---
    assert m_sparse["n_calendar_bdays"] > m_sparse["n_days"] * 5, (
        "expected many more business days in the real span than the 3 active trade days"
    )
    assert m_sparse["sharpe_fulltime"] != m_sparse["sharpe_active_days"], (
        "full-time-capital Sharpe must differ from active-day-only Sharpe for a sparse strategy"
    )
    print(f"  full-time-capital vs active-day Sharpe distinction check ok "
          f"(active={m_sparse['sharpe_active_days']}, fulltime={m_sparse['sharpe_fulltime']}, "
          f"active_days={m_sparse['n_days']}, calendar_bdays={m_sparse['n_calendar_bdays']})")

    # --- drawdown duration hand-trace: down for exactly 2 days before a new peak ---
    dates2 = pd.date_range("2025-02-03", periods=6, freq="D").date
    pnl2 = pd.Series([1000, -800, -100, 2000, 500, -100], index=dates2)
    # equity: 10000+1000=11000(peak), 10200, 10100, 12100(new peak), 12600(new peak), 12500
    m2 = full_metrics(pnl2, starting_capital=10_000)
    assert m2["max_dd_duration_days"] == 2, f"expected a 2-day drawdown duration, got {m2['max_dd_duration_days']}"
    print(f"  drawdown-duration hand-trace check ok (duration={m2['max_dd_duration_days']} days)")

    # --- Monte Carlo: a strategy that never loses should show 0% wipeout risk ---
    trades_all_win = pd.DataFrame({"rupee_pnl": [500, 300, 400, 600, 200] * 20})
    mc = monte_carlo_reshuffle(trades_all_win, starting_capital=10_000, iters=200, seed=1)
    assert mc["pct_paths_wipeout"] == 0.0, "an all-winning trade set should never wipe out under any reshuffle"
    print("  monte_carlo_reshuffle all-winners-never-wipeout check ok")

    # --- Monte Carlo: a strategy that guarantees wipeout (one huge loss exceeding capital) should show 100% ---
    trades_guaranteed_wipeout = pd.DataFrame({"rupee_pnl": [-20_000, 100, 100, 100]})
    mc2 = monte_carlo_reshuffle(trades_guaranteed_wipeout, starting_capital=10_000, iters=200, seed=1)
    assert mc2["pct_paths_wipeout"] == 100.0, "a loss bigger than capital must wipe out regardless of order"
    print("  monte_carlo_reshuffle guaranteed-wipeout check ok")

    print("All self-checks passed.\n")


def analyze_strategy(name, csv_path, starting_capital=STARTING_CAPITAL):
    print("=" * 78)
    print(f"{name}")
    print("=" * 78)
    trades = pd.read_csv(csv_path)
    daily_pnl = build_daily_pnl(trades)
    m = full_metrics(daily_pnl, starting_capital)
    print(f"Trading days: {m['n_days']} active / {m['n_calendar_bdays']} calendar business days in span   "
          f"Final equity: Rs {m['final_equity']:,.2f}   CAGR: {m['cagr_pct']}%")
    print(f"Sharpe (active trading days only): {m['sharpe_active_days']}   "
          f"Sortino (active only): {m['sortino_active_days']}")
    print(f"Sharpe (full-time capital, zero-filled quiet days): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}")
    print(f"Max drawdown: {m['max_dd_pct']}%   Max drawdown duration: {m['max_dd_duration_days']} trading days")
    print(f"VaR(95%, daily): {m['var95_pct']}%   CVaR(95%, daily): {m['cvar95_pct']}%")
    print(f"t-stat on mean daily return: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")

    print(f"\n--- Monte Carlo trade-order reshuffle ({MONTE_CARLO_ITERS} iterations) ---")
    mc = monte_carlo_reshuffle(trades, starting_capital)
    print(f"Final equity -> 5th pct: Rs {mc['final_eq_p5']:,.2f}   50th: Rs {mc['final_eq_p50']:,.2f}   "
          f"95th: Rs {mc['final_eq_p95']:,.2f}")
    print(f"Max drawdown% -> 5th pct: {mc['max_dd_pct_p5']}%   50th: {mc['max_dd_pct_p50']}%   "
          f"95th: {mc['max_dd_pct_p95']}%")
    print(f"Probability of ever wiping out (equity <= 0) across reshuffled trade orders: "
          f"{mc['pct_paths_wipeout']}%")
    print(f"Probability of falling below 50% of starting capital: {mc['pct_paths_below_50pct_capital']}%")
    print()
    return m, mc


def main():
    run_self_checks()
    results = {}
    results["Velez 8pt (proxy premium)"] = analyze_strategy(
        "VELEZ 8pt -- proxy premium + execution-lag fix", "velez_8pt_proxy_premium_trades.csv")
    results["Opening-Drive Momentum (sl8_trail4)"] = analyze_strategy(
        "OPENING-DRIVE MOMENTUM -- sl8_trail4", "open_momentum_sweep_sl8_trail4_trades.csv")

    print("=" * 78)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 78)
    print(f"{'Metric':<28}{'Velez 8pt':>22}{'Opening-Drive Mom.':>22}")
    m1, mc1 = results["Velez 8pt (proxy premium)"]
    m2, mc2 = results["Opening-Drive Momentum (sl8_trail4)"]
    rows = [
        ("CAGR %", m1["cagr_pct"], m2["cagr_pct"]),
        ("Sharpe (active days)", m1["sharpe_active_days"], m2["sharpe_active_days"]),
        ("Sharpe (full-time capital)", m1["sharpe_fulltime"], m2["sharpe_fulltime"]),
        ("Sortino (active days)", m1["sortino_active_days"], m2["sortino_active_days"]),
        ("Sortino (full-time capital)", m1["sortino_fulltime"], m2["sortino_fulltime"]),
        ("Max DD %", m1["max_dd_pct"], m2["max_dd_pct"]),
        ("Max DD duration (days)", m1["max_dd_duration_days"], m2["max_dd_duration_days"]),
        ("VaR 95% (daily)", m1["var95_pct"], m2["var95_pct"]),
        ("CVaR 95% (daily)", m1["cvar95_pct"], m2["cvar95_pct"]),
        ("t-stat / p-value", f"{m1['t_stat']}/{m1['p_value']}", f"{m2['t_stat']}/{m2['p_value']}"),
        ("MC wipeout probability %", mc1["pct_paths_wipeout"], mc2["pct_paths_wipeout"]),
        ("MC final eq. 5th pct (Rs)", mc1["final_eq_p5"], mc2["final_eq_p5"]),
    ]
    for label, v1, v2 in rows:
        print(f"{label:<28}{str(v1):>22}{str(v2):>22}")


if __name__ == "__main__":
    main()
