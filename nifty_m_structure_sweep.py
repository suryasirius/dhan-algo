"""
M-STRUCTURE -- diagnostic + optimization sweep
=================================================================
Two jobs, in order, because the second isn't trustworthy until the first
is answered:

JOB 1 -- DIAGNOSE THE ZERO-TRADES-FOR-15-MONTHS ANOMALY. The first full
2-year run of nifty_m_structure_backtest.py produced 1058 trades, but ALL
of them fell between 2025-11-03 and 2026-07-29 (the OOS split's own H1
start date gave this away) -- zero trades fired across the first ~15
months of the requested 2-year window (2024-07-30 to ~2025-11-03), even
though structure/breakout patterns should occur at some rate in any
multi-month window. Per this project's standing rigor rule, an entry-
frequency pattern this extreme (0% for over a year, then a sudden
regime) needs a bug hunt before any headline number is trusted -- the
leading hypothesis is that the Nifty INDEX feed's volume field is zero or
unpopulated before some cutover date, which would make the strategy's
"volume > 20-bar average" filter permanently False (0 > 0) for that whole
period, silently blocking every signal. This script fetches the same data
and prints month-by-month volume coverage BEFORE running anything else,
so we know for certain whether that's the cause (and if so, what the
REAL usable sample size actually is) before trusting any sweep result.

JOB 2 -- OPTIMIZATION SWEEP, once the real usable window is known:
  - Stop/trail grid sweep (signals are detected ONCE and reused across
    every combo -- only the exit simulation changes, since entry logic
    doesn't depend on stop/trail at all).
  - A time-capped-exit variant (force exit at N bars if stop/trail hasn't
    fired yet) tested against the no-cap baseline.
  - Day-of-week and 30-minute-session breakdowns on the current default
    combo (5pt stop / 4pt trail, no cap), to see whether the edge is
    broad-based or concentrated in specific days/times.
  - Grid-edge check per the project's standing rule: if the best combo
    sits at the edge of the tested range, that's a tail-risk-harvesting
    red flag, not a real edge, until stress-tested further.

Run:
    cd "C:\\dhan algo"
    python nifty_m_structure_sweep.py --smoke-test
    python nifty_m_structure_sweep.py
"""

import sys
import math
import datetime as dt
import pandas as pd
import numpy as np

import nifty_paper_trade_top3_live as top3
import nifty_m_structure_backtest as ms
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000

STOP_GRID = [3.0, 5.0, 8.0, 10.0]
TRAIL_GRID = [2.0, 4.0, 6.0, 8.0]
TIME_CAP_GRID_BARS = [15, 30, 60, None]  # None = no cap (current default behaviour)
DEFAULT_STOP, DEFAULT_TRAIL = 5.0, 4.0   # your originally specified values, used for day/session breakdown

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def classify_session(ts):
    t = ts.time()
    for label in SESSIONS:
        start_s, end_s = label.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        if dt.time(sh, sm) <= t < dt.time(eh, em):
            return label
    return "Outside usual session"


# =============================================================================
# JOB 1 -- volume coverage diagnostic
# =============================================================================
def print_volume_coverage(df1m):
    print("\n################ VOLUME COVERAGE DIAGNOSTIC (JOB 1) ################")
    df = df1m.copy()
    df["month"] = df.index.to_period("M")
    monthly = df.groupby("month")["volume"].apply(lambda v: (v > 0).mean() * 100)
    first_good_month = None
    for month, pct_nonzero in monthly.items():
        flag = ""
        if pct_nonzero < 5 and first_good_month is None:
            flag = "  <-- effectively NO real volume this month"
        elif pct_nonzero >= 50 and first_good_month is None:
            first_good_month = month
            flag = "  <-- volume coverage turns on here"
        print(f"  {month}   {pct_nonzero:6.1f}% of bars have volume>0{flag}")
    if first_good_month is not None:
        print(f"\n  CONCLUSION: real volume data effectively starts around {first_good_month}. "
              f"Before that, the 'volume > 20-bar average' filter is comparing against an average "
              f"that's also ~0, which can make it permanently unsatisfiable -- this is the most likely "
              f"explanation for the zero-trades-for-15-months pattern seen in the first full run.")
    else:
        print("\n  CONCLUSION: volume coverage looks consistently populated across the whole window -- "
              "the zero-trades pattern is NOT a volume-data availability issue. Some other cause "
              "(structure/rejection-hold filters too strict in that regime, or a genuine low-volatility "
              "period with no clean M/W structures) needs to be investigated instead.")
    print("#" * 70)


# =============================================================================
# JOB 2 -- precompute once, reuse across every stop/trail combo
# =============================================================================
def precompute_day_contexts(df1m):
    days_covered = sorted(set(df1m.index.date))
    contexts = []
    for day in days_covered:
        day_df = df1m[df1m.index.date == day]
        if len(day_df) < ms.STRUCT_LOOKBACK + ms.REJECT_HOLD_BARS + ms.HOLD_CONFIRM_BARS + 3:
            continue
        sigs = ms.detect_m_structure_signals(day_df)
        if not sigs:
            continue
        day_open_spot = day_df["open"].iloc[0]
        strike_ce = top3.itm_call_strike(day_open_spot)
        strike_pe = top3.itm_put_strike(day_open_spot)
        premium_ce = top3.build_premium_series(day_df, strike_ce, day_open_spot, side="CE")
        premium_pe = top3.build_premium_series(day_df, strike_pe, day_open_spot, side="PE")
        lot = top3.lot_size_for_date(day)
        contexts.append({
            "day": day, "sigs": sigs, "premium_ce": premium_ce, "premium_pe": premium_pe,
            "strike_ce": strike_ce, "strike_pe": strike_pe, "lot": lot, "n": len(day_df),
        })
    return contexts


def simulate_combo(contexts, stop_pts, trail_pts, max_bars=None):
    trades = []
    for ctx in contexts:
        open_until_idx = -1
        for sig in ctx["sigs"]:
            j0 = sig["bar"]
            if j0 <= open_until_idx:
                continue
            exec_idx = j0 + ms.HOLD_CONFIRM_BARS + 1
            if exec_idx >= ctx["n"]:
                continue
            premium_df = ctx["premium_ce"] if sig["side"] == "CE" else ctx["premium_pe"]
            strike = ctx["strike_ce"] if sig["side"] == "CE" else ctx["strike_pe"]
            entry_price = premium_df["open"].iloc[exec_idx]
            pos = {"entry_price": entry_price, "stop_level": entry_price - stop_pts, "peak": entry_price,
                   "trailing_active": False, "last_checked_idx": exec_idx - 1}
            if max_bars is not None:
                cap_idx = min(exec_idx + max_bars, ctx["n"] - 1)
                pdf_slice = premium_df.iloc[:cap_idx + 1]
            else:
                pdf_slice = premium_df
            exit_info = top3.step_long_premium_position(pos, pdf_slice, trail_pts)
            if exit_info is None:
                reason = "TIME_EXIT" if max_bars is not None else "EOD_FALLBACK"
                exit_info = {"exit_price": pdf_slice["close"].iloc[-1], "reason": reason,
                             "exit_time": pdf_slice.index[-1], "exit_idx": len(pdf_slice) - 1}
            pnl_pts = exit_info["exit_price"] - entry_price
            rupee_pnl = pnl_pts * ctx["lot"] - BROKERAGE_PER_TRADE
            trades.append({
                "side": sig["side"], "strike": strike, "event": sig["event"],
                "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
                "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
                "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": ctx["day"],
            })
            open_until_idx = exit_info["exit_idx"]
    return pd.DataFrame(trades)


def compute_pf_sqn(trades_df):
    pnl = trades_df["rupee_pnl"]
    n = len(pnl)
    if n == 0:
        return 0.0, 0.0, 0
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    std = pnl.std(ddof=1) if n > 1 else 0.0
    sqn = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    return pf, sqn, n


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    ms.run_self_checks()

    # --- equivalence check: simulate_combo(stop=5, trail=4, no cap) must match
    # ms.run_m_structure_standalone_for_day exactly on the same synthetic day
    # used in nifty_m_structure_backtest's own self-checks (real breakout case) ---
    idx2 = pd.date_range("2025-01-06 09:15", periods=ms.STRUCT_LOOKBACK + 10, freq="1min")
    base_h = [100.0] * ms.STRUCT_LOOKBACK
    base_l = [95.0] * ms.STRUCT_LOOKBACK
    base_v = [1000.0] * ms.STRUCT_LOOKBACK
    h1 = base_h + [105, 106, 107, 106, 106, 106, 106, 106, 106, 106]
    l1 = base_l + [99, 101, 102, 101, 101, 101, 101, 101, 101, 101]
    v1 = base_v + [3000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    df1 = pd.DataFrame({"open": h1, "high": h1, "low": l1, "close": h1, "volume": v1}, index=idx2)

    direct_trades = ms.run_m_structure_standalone_for_day(df1)
    ctx = precompute_day_contexts(df1.assign(_dummy=0).drop(columns="_dummy"))
    # precompute_day_contexts expects a multi-day frame; feed it as a single "day"
    ctx = [c for c in [{
        "day": df1.index[0].date(), "sigs": ms.detect_m_structure_signals(df1),
        "premium_ce": top3.build_premium_series(df1, top3.itm_call_strike(df1["open"].iloc[0]), df1["open"].iloc[0], "CE"),
        "premium_pe": top3.build_premium_series(df1, top3.itm_put_strike(df1["open"].iloc[0]), df1["open"].iloc[0], "PE"),
        "strike_ce": top3.itm_call_strike(df1["open"].iloc[0]), "strike_pe": top3.itm_put_strike(df1["open"].iloc[0]),
        "lot": top3.lot_size_for_date(df1.index[0].date()), "n": len(df1),
    }] if c["sigs"]]
    swept_trades = simulate_combo(ctx, ms.STRUCT_STOP_PTS, ms.STRUCT_TRAIL_PTS, max_bars=None)
    assert len(direct_trades) == len(swept_trades) == 1, (
        f"expected exactly 1 trade from both paths, got direct={len(direct_trades)} swept={len(swept_trades)}"
    )
    assert abs(direct_trades[0]["rupee_pnl"] - swept_trades.iloc[0]["rupee_pnl"]) < 1e-6, (
        f"sweep engine diverged from the original verified backtest: "
        f"{direct_trades[0]['rupee_pnl']} vs {swept_trades.iloc[0]['rupee_pnl']}"
    )
    print("  simulate_combo(default stop/trail) matches the original verified backtest exactly ok")

    # --- time-cap forces an earlier exit than the uncapped run on the same trade ---
    capped = simulate_combo(ctx, ms.STRUCT_STOP_PTS, ms.STRUCT_TRAIL_PTS, max_bars=1)
    assert len(capped) == 1 and capped.iloc[0]["reason"] == "TIME_EXIT", (
        f"expected a 1-bar cap to force a TIME_EXIT, got {capped.iloc[0].to_dict() if len(capped) else 'no trade'}"
    )
    print("  time-capped exit forces an earlier TIME_EXIT than the uncapped run ok")

    print("All self-checks passed.\n")


def print_combo_row(label, trades_df):
    pf, sqn, n = compute_pf_sqn(trades_df)
    net = trades_df["rupee_pnl"].sum() if n else 0.0
    win = 100 * (trades_df["rupee_pnl"] > 0).mean() if n else 0.0
    print(f"  {label:<22} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")
    return pf, sqn, n, net


def run_full_pipeline(df1m):
    print_volume_coverage(df1m)

    print("\nPrecomputing signals + premium series once (reused across every combo)...")
    contexts = precompute_day_contexts(df1m)
    total_sigs = sum(len(c["sigs"]) for c in contexts)
    print(f"{total_sigs} candidate signals across {len(contexts)} days with at least one signal.")
    if total_sigs == 0:
        print("No signals at all -- nothing to sweep.")
        return

    print(f"\n################ STOP/TRAIL GRID SWEEP ################")
    results = []
    for stop_pts in STOP_GRID:
        for trail_pts in TRAIL_GRID:
            trades_df = simulate_combo(contexts, stop_pts, trail_pts, max_bars=None)
            pf, sqn, n, net = print_combo_row(f"stop={stop_pts} trail={trail_pts}", trades_df)
            results.append({"stop": stop_pts, "trail": trail_pts, "pf": pf, "sqn": sqn, "n": n, "net": net})

    res_df = pd.DataFrame(results)
    best = res_df.sort_values("sqn", ascending=False).iloc[0]
    print(f"\nBest by SQN: stop={best['stop']} trail={best['trail']}  SQN={best['sqn']:.2f}  "
          f"PF={best['pf']:.2f}  net=Rs{best['net']:,.2f}")
    at_edge = (best["stop"] in (STOP_GRID[0], STOP_GRID[-1])) or (best["trail"] in (TRAIL_GRID[0], TRAIL_GRID[-1]))
    if at_edge:
        print("  *** GRID-EDGE WARNING per project rigor rule: the best combo sits at the edge of the "
              "tested range. This can mean either the grid is too narrow, or -- more likely for very "
              "tight-trail/wide-stop corners -- this is a tail-risk-harvesting artifact (locks in tiny "
              "wins on favorable ticks while the real loss scenario hasn't occurred enough in this "
              "sample yet). Treat this combo as needing dedicated stress-testing, NOT as a finding, "
              "before trading it. The originally-specified 5/4 combo is the one with an actual basis "
              "(your own stated rule), not whatever the grid says is 'best'.")
    if best["pf"] > 2.5 or best["sqn"] > 5:
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as needing a bug hunt before trusting, per "
              "project rigor rule. (Calibration: real clean results here look like PF ~1.4-1.7, SQN ~2.5-3.) ***")

    print(f"\n################ TIME-CAPPED EXIT TEST (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}) ################")
    for cap in TIME_CAP_GRID_BARS:
        label = f"cap={cap} bars" if cap is not None else "no cap (baseline)"
        trades_df = simulate_combo(contexts, DEFAULT_STOP, DEFAULT_TRAIL, max_bars=cap)
        print_combo_row(label, trades_df)

    print(f"\n################ DAY-OF-WEEK BREAKDOWN (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}, no cap) ################")
    default_trades = simulate_combo(contexts, DEFAULT_STOP, DEFAULT_TRAIL, max_bars=None)
    default_trades["weekday"] = pd.to_datetime(default_trades["entry_time"]).dt.day_name()
    for wd in WEEKDAY_ORDER:
        g = default_trades[default_trades["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

    print(f"\n################ SESSION-OF-DAY BREAKDOWN (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}, no cap) ################")
    default_trades["session"] = default_trades["entry_time"].apply(classify_session)
    for sess in SESSIONS:
        g = default_trades[default_trades["session"] == sess]
        if len(g):
            print_combo_row(sess, g)

    default_trades.to_csv("m_structure_sweep_default_combo_trades.csv", index=False)
    print(f"\nDefault-combo (stop={DEFAULT_STOP}, trail={DEFAULT_TRAIL}) trades saved to "
          f"m_structure_sweep_default_combo_trades.csv")

    if len(default_trades):
        daily_pnl = build_daily_pnl(default_trades.rename(columns={"date": "date"}))
        m = full_metrics(daily_pnl, STARTING_CAPITAL)
        print(f"\n################ DEFAULT COMBO ({DEFAULT_STOP}/{DEFAULT_TRAIL}) -- FULL INSTITUTIONAL METRICS ################")
        print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
              f"Sortino (full-time): {m['sortino_fulltime']}   Max DD: {m['max_dd_pct']}%   "
              f"p-value: {m['p_value']}")


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = ms.fetch_index_chunked_with_volume(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). This failure is specifically about "
              "reaching the Dhan API from wherever this was just run -- re-run on the machine with a "
              "working Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m)
    print("\nSmoke test complete -- no crashes, all self-checks passed, full sweep pipeline ran "
          "end-to-end on real data.")


def main():
    run_self_checks()
    print("=== M-STRUCTURE DIAGNOSTIC + OPTIMIZATION SWEEP -- 2-YEAR REAL DATA ===")
    print("Fetching Nifty index 1-min data (2 years, with volume)...")
    df1m = ms.fetch_index_chunked_with_volume(ms.FROM_DATE, ms.TO_DATE)
    if df1m.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")
    run_full_pipeline(df1m)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main()
