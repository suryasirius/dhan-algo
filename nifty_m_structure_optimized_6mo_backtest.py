"""
M-STRUCTURE, OPTIMIZED VERSION -- 6-MONTH REAL BACKTEST
=================================================================
The tuned strategy from nifty_m_structure_paper_trade_live.py (PE-only,
Tuesday excluded, stop=8/trail=4 -- see that file's docstring for the
full reasoning from the sweep), run as a proper backtest over the last
~6 months instead of a live paper loop.

WHY 6 MONTHS, NOT 2 YEARS: nifty_m_structure_sweep.py's volume-coverage
diagnostic confirmed real Nifty index volume only exists from ~2025-11
onward -- before that, the volume-confirmation filter is permanently
unsatisfiable (0 > 0), so a longer request window wouldn't add any real
trades anyway. A 6-month window ending today sits entirely inside the
confirmed-good-volume era, so this is a clean, honest read -- not a
truncated one.

All signal-detection and position-management logic is reused verbatim
from nifty_m_structure_backtest.py (`ms`) and nifty_paper_trade_top3_live.py
(`top3`) -- nothing is reimplemented here, only the day-loop is filtered
to PE-only + non-Tuesday and the stop/trail is fixed at the tuned 8/4.

Run:
    cd "C:\\dhan algo"
    python nifty_m_structure_optimized_6mo_backtest.py --smoke-test
    python nifty_m_structure_optimized_6mo_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_m_structure_backtest as ms
from oi_trend_strategy_backtest import fetch_futures_oi_chunked, find_current_nifty_future
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== tuned config ===============================
BACKTEST_DAYS = 183  # ~6 months
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")

STOP_PTS, TRAIL_PTS = 8.0, 4.0
EXCLUDED_WEEKDAYS = {1}  # Tuesday
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000

TRADES_CSV = "m_structure_optimized_6mo_trades.csv"


# =============================================================================
# Day-level simulation -- PE-only, tuned stop/trail, non-Tuesday
# =============================================================================
def run_optimized_standalone_for_day(day_df):
    trades = []
    today = day_df.index[0].date()
    if today.weekday() in EXCLUDED_WEEKDAYS:
        return trades
    sigs = [s for s in ms.detect_m_structure_signals(day_df) if s["side"] == "PE"]
    if not sigs:
        return trades
    n = len(day_df)
    day_open_spot = day_df["open"].iloc[0]
    strike = top3.itm_put_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="PE")
    lot = top3.lot_size_for_date(today)
    open_until_idx = -1
    for sig in sigs:
        j0 = sig["bar"]
        if j0 <= open_until_idx:
            continue
        exec_idx = j0 + ms.HOLD_CONFIRM_BARS + 1
        if exec_idx >= n:
            continue
        entry_price = premium_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": entry_price - STOP_PTS, "peak": entry_price,
               "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": n - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        trades.append({
            "strategy": "MStructureOptimized", "side": "PE", "strike": strike, "event": sig["event"],
            "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
            "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
            "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": today,
        })
        open_until_idx = exit_info["exit_idx"]
    return trades


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    ms.run_self_checks()

    # --- Tuesday must be skipped entirely, even with a real signal present ---
    idx2 = pd.date_range("2025-01-07 09:15", periods=ms.STRUCT_LOOKBACK + 10, freq="1min")  # 2025-01-07 = Tuesday
    assert dt.date(2025, 1, 7).weekday() == 1, "test setup sanity check failed"
    base_h = [100.0] * ms.STRUCT_LOOKBACK
    base_l_low = [95.0] * (ms.STRUCT_LOOKBACK - ms.REJECT_HOLD_BARS) + [90.0] * ms.REJECT_HOLD_BARS  # clean breakdown setup
    breakdown_tail_h = [88, 85, 84, 85, 85, 85, 85, 85, 85, 85]  # bars 31,32 (hold-confirm) stay below sup=90
    breakdown_tail_l = [82, 80, 79, 80, 80, 80, 80, 80, 80, 80]
    v = [1000.0] * ms.STRUCT_LOOKBACK + [3000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    df_tue = pd.DataFrame({
        "open": base_h + breakdown_tail_h, "high": base_h + breakdown_tail_h,
        "low": base_l_low + breakdown_tail_l, "close": base_h + breakdown_tail_h, "volume": v,
    }, index=idx2)
    trades_tue = run_optimized_standalone_for_day(df_tue)
    assert trades_tue == [], f"Tuesday must produce zero trades regardless of signals, got {trades_tue}"
    print("  Tuesday exclusion holds even with a valid signal present ok")

    # --- same day shifted to a Wednesday must produce a trade (proves the block is weekday-specific) ---
    idx3 = pd.date_range("2025-01-08 09:15", periods=ms.STRUCT_LOOKBACK + 10, freq="1min")  # Wednesday
    assert dt.date(2025, 1, 8).weekday() == 2
    df_wed = df_tue.set_axis(idx3)
    trades_wed = run_optimized_standalone_for_day(df_wed)
    assert len(trades_wed) == 1 and trades_wed[0]["side"] == "PE", (
        f"expected exactly 1 PE trade on the non-excluded weekday, got {trades_wed}"
    )
    print("  identical setup on a non-excluded weekday correctly produces a PE trade ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics
# =============================================================================
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


def run_full_pipeline(df1m, label="6-MONTH OPTIMIZED BACKTEST", fetch_oi=True):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))
    excluded_days = [d for d in days_covered if d.weekday() in EXCLUDED_WEEKDAYS]
    print(f"{len(excluded_days)} of {len(days_covered)} calendar trading days are Tuesdays (sat out by design).")

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        if len(day_df) >= ms.STRUCT_LOOKBACK + ms.REJECT_HOLD_BARS + ms.HOLD_CONFIRM_BARS + 3:
            all_trades.extend(run_optimized_standalone_for_day(day_df))

    if not all_trades:
        print("No trades generated -- either the window doesn't have enough real-volume days, or the "
              "filter stack is too strict for this particular 6-month period.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)

    traded_days = trades_df["date"].nunique()
    non_tuesday_days = len(days_covered) - len(excluded_days)
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {non_tuesday_days} eligible (non-Tuesday) days "
          f"({100*traded_days/non_tuesday_days:.1f}%)")
    if traded_days / non_tuesday_days > 0.95:
        print("  *** WARNING: firing on >95% of eligible days -- worth double-checking this hasn't "
              "degenerated into something looser than the specific structure it claims to test. ***")

    if fetch_oi:
        print("\nFetching real NIFTY futures OI for soft-tagging...")
        try:
            fut_id, fut_symbol, fut_expiry = find_current_nifty_future()
            oi_df = fetch_futures_oi_chunked(fut_id, FROM_DATE, TO_DATE)
        except Exception as e:
            print(f"  Could not fetch OI data ({e}) -- proceeding with all trades tagged NO_DATA.")
            oi_df = pd.DataFrame(columns=["oi"])
        regime_df = ms.build_oi_regime_series(oi_df, df1m) if not oi_df.empty else pd.DataFrame(columns=["oi_regime"])
        trades_df = ms.tag_trades_with_oi(trades_df, regime_df)
        n_with_oi = (trades_df["oi_regime"] != "NO_DATA").sum()
        print(f"  {n_with_oi} of {len(trades_df)} trades fall within the real-OI window "
              f"({100*n_with_oi/len(trades_df):.1f}%).")
        if n_with_oi > 0:
            agree = trades_df[trades_df["oi_agrees"] == True]
            print(f"  Of those: {len(agree)} OI-agreed (net Rs{agree['rupee_pnl'].sum():,.2f}, "
                  f"win%={100*(agree['rupee_pnl']>0).mean():.1f})")
    else:
        trades_df["oi_regime"] = "SKIPPED"

    trades_df.to_csv(TRADES_CSV, index=False)
    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ M-STRUCTURE OPTIMIZED (PE-only, no-Tue, 8/4) -- HEADLINE ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. (Calibration: real clean results look like PF ~1.4-1.7, SQN ~2.5-3.) ***")

    # in-sample / out-of-sample split
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    dates = sorted(trades_df["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = trades_df[trades_df["date"] < mid]
        h2 = trades_df[trades_df["date"] >= mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split: H1 ({dates[0].date()} to {mid.date()}) n={n1} PF={pf1:.2f} SQN={sqn1:.2f} "
              f"net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 ({mid.date()} to {dates[-1].date()}) n={n2} PF={pf2:.2f} SQN={sqn2:.2f} "
              f"net=Rs{h2['rupee_pnl'].sum():,.2f}")

    # capital adequacy
    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING: worst drawdown approaches/exceeds starting capital. ***")
    if (equity_curve <= 0).any():
        print("  WARNING: equity touches Rs0 or below at some point in this trade sequence.")
    else:
        print("  Equity never touches Rs0 or below across this trade sequence.")

    # full institutional metrics
    daily_pnl = build_daily_pnl(trades_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(trades_df, STARTING_CAPITAL)
    print(f"\n################ FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (active days): {m['sharpe_active_days']}   "
          f"Sharpe (full-time capital): {m['sharpe_fulltime']}")
    print(f"Sortino (active days): {m['sortino_active_days']}   Sortino (full-time capital): {m['sortino_fulltime']}")
    print(f"Max DD: {m['max_dd_pct']}%   Max DD duration: {m['max_dd_duration_days']} trading days")
    print(f"VaR95 (daily): {m['var95_pct']}%   CVaR95 (daily): {m['cvar95_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    return {"trades": trades_df, "metrics": m, "monte_carlo": mc}


# =============================================================================
# Smoke test / main
# =============================================================================
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
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)", fetch_oi=False)
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data.")


def main():
    run_self_checks()
    print(f"=== M-STRUCTURE OPTIMIZED -- 6-MONTH REAL BACKTEST ===")
    print(f"Window: {FROM_DATE} to {TO_DATE} (entirely inside the confirmed real-volume era)")
    print("Fetching Nifty index 1-min data (with volume)...")
    df1m = ms.fetch_index_chunked_with_volume(FROM_DATE, TO_DATE)
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
