"""
FIXED-STRIKE OPENING-RANGE BREAKOUT -- 5-POINT TARGET, 3-MINUTE TIME CAP
=================================================================
Variant of nifty_orb_fixed_strike_5pt_target_backtest.py (which used a
10-point stop). This version replaces the stop with your stated rule:
exit at the 5-point target OR after 3 minutes in the trade, whichever
comes first -- NO separate stop-loss. Everything else (ATM strike fixed
at open, opening-range high from the first 5 one-minute candles,
confirmed-CLOSE-only entry trigger, CE-only, one trade per day, next-bar-
open fill) is reused verbatim from that script via import -- not
reimplemented.

IMPORTANT, CARRIED FORWARD FROM THE CONCEPT DOC: there is genuinely no
point-based stop in this version. A losing trade is only bounded by
whatever the premium happens to be 3 minutes after entry -- if the
underlying moves sharply against the position in those 3 minutes, this
strategy has no mechanism to cut the loss short. The report below prints
the worst single-trade loss explicitly so this risk is visible in the
real numbers, not just the docstring.

Run:
    cd "C:\\dhan algo"
    python nifty_orb_fixed_strike_5pt_3mincap_backtest.py --smoke-test
    python nifty_orb_fixed_strike_5pt_3mincap_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_orb_fixed_strike_5pt_target_backtest as orb
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== config ===============================
FROM_DATE = orb.FROM_DATE
TO_DATE = orb.TO_DATE
TARGET_PTS = 5.0
TIME_CAP_SECONDS = 180  # 3 minutes
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000

TRADES_CSV = "nifty_orb_fixed_strike_5pt_3mincap_trades.csv"


# =============================================================================
# Exit stepper: target OR time-cap, whichever first -- NO stop-loss.
# If both would be satisfied on the same bar, the time-cap is checked
# first (the more conservative assumption: we can't tell within a bar
# whether the target or the 3-minute mark came first, so this doesn't
# give the trade the benefit of the doubt).
# =============================================================================
def step_target_or_timecap(pos, premium_df):
    n = len(premium_df)
    entry_time = pos["entry_time"]
    for j in range(pos["last_checked_idx"] + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j]
        elapsed = (t - entry_time).total_seconds()
        if elapsed >= TIME_CAP_SECONDS:
            return {"exit_price": bar["close"], "reason": "TIME_CAP_3MIN", "exit_time": t, "exit_idx": j}
        if bar["high"] >= pos["target_level"]:
            return {"exit_price": pos["target_level"], "reason": "TARGET", "exit_time": t, "exit_idx": j}
        if t.time() >= orb.EOD_SQUAREOFF:
            return {"exit_price": bar["close"], "reason": "EOD", "exit_time": t, "exit_idx": j}
        pos["last_checked_idx"] = j
    return None


def run_standalone_for_day(day_df):
    n = len(day_df)
    if n < orb.OPENING_RANGE_BARS + 2:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = orb.atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    strike_high = premium_df["high"].iloc[:orb.OPENING_RANGE_BARS].max()
    sig_bar = orb.detect_orb_signal_bar(premium_df, strike_high)
    if sig_bar is None:
        return []
    exec_idx = sig_bar + 1
    if exec_idx >= n:
        return []

    entry_price = premium_df["open"].iloc[exec_idx]
    pos = {"entry_time": premium_df.index[exec_idx], "target_level": entry_price + TARGET_PTS,
           "last_checked_idx": exec_idx - 1}
    exit_info = step_target_or_timecap(pos, premium_df)
    if exit_info is None:
        exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                     "exit_time": premium_df.index[-1], "exit_idx": n - 1}
    pnl_pts = exit_info["exit_price"] - entry_price
    rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
    return [{
        "strategy": "ORB_FixedStrike_5pt_3minCap", "side": "CE", "strike": strike, "strike_high": strike_high,
        "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    }]


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    orb.run_self_checks()  # re-verify the shared strike/opening-range/premium primitives

    idx = pd.date_range("2025-01-06 09:20", periods=5, freq="1min")

    # --- target hit well within 3 minutes must fire TARGET, not the time cap ---
    prem1 = pd.DataFrame({
        "open": [150, 150, 158, 158, 158], "high": [150, 151, 158, 158, 158],
        "low": [150, 149, 156, 156, 156], "close": [150, 150.5, 157, 157, 157],
    }, index=idx)
    pos1 = {"entry_time": idx[0], "target_level": 155, "last_checked_idx": -1}
    exit1 = step_target_or_timecap(pos1, prem1)
    assert exit1["reason"] == "TARGET" and exit1["exit_price"] == 155, exit1
    print("  target hit within 3 minutes fires TARGET (not the time cap) ok")

    # --- neither hit -> exits at the 3-minute mark (elapsed=180s = bar index 3, since bars are 60s apart) ---
    prem2 = pd.DataFrame({
        "open": [150, 151, 152, 153, 154], "high": [150, 152, 153, 154, 155],
        "low": [150, 150, 151, 152, 153], "close": [150, 151.5, 152.5, 153.5, 154.5],
    }, index=idx)
    pos2 = {"entry_time": idx[0], "target_level": 155, "last_checked_idx": -1}
    exit2 = step_target_or_timecap(pos2, prem2)
    assert exit2["reason"] == "TIME_CAP_3MIN" and exit2["exit_idx"] == 3, (
        f"expected the 3-minute mark (bar index 3, elapsed=180s) to force an exit, got {exit2}"
    )
    print("  neither target nor cap-time reached early -> exits exactly at the 3-minute mark ok")

    # --- no stop: a sharp adverse move does NOT force an early exit before the time cap ---
    prem3 = pd.DataFrame({
        "open": [150, 120, 120, 120, 120], "high": [150, 121, 121, 121, 121],
        "low": [150, 100, 100, 100, 100],   # a huge adverse move on bar 1 -- must NOT force an exit (no stop exists)
        "close": [150, 110, 110, 110, 110],
    }, index=idx)
    pos3 = {"entry_time": idx[0], "target_level": 155, "last_checked_idx": -1}
    exit3 = step_target_or_timecap(pos3, prem3)
    assert exit3["reason"] == "TIME_CAP_3MIN", (
        f"a sharp adverse move must NOT trigger an early exit -- there is no stop in this version, "
        f"got {exit3}"
    )
    print("  confirmed: a sharp adverse move does not force an early exit (no stop-loss exists here) ok")

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


def run_full_pipeline(df1m, label="FULL 2-YEAR BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(run_standalone_for_day(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades generated at all.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    print(f"  Exit reason breakdown: {trades_df['reason'].value_counts().to_dict()}")

    worst_single_loss = trades_df["rupee_pnl"].min()
    print(f"\n  *** NO STOP-LOSS IN THIS VERSION *** -- worst single trade: Rs{worst_single_loss:,.2f}. "
          f"Compare this to the ~Rs{-STOP_PTS_REFERENCE_RS():.0f} a 10-point stop would have capped it at, "
          f"to see what the missing stop actually cost on its worst day.")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ ORB FIXED-STRIKE 5PT / 3MIN-CAP -- HEADLINE RESULT ################")
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


def STOP_PTS_REFERENCE_RS():
    # informational only -- what a 10pt stop (this project's standard) would have cost per trade,
    # using the most common recent lot size, just to give the "no stop" number some scale.
    return 10.0 * 65


# =============================================================================
# Smoke test / main
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = orb.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). This failure is specifically about "
              "reaching the Dhan API from wherever this was just run -- re-run on the machine with a "
              "working Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)")
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data.")


def main():
    run_self_checks()
    print(f"=== ORB FIXED-STRIKE 5PT / 3-MIN CAP -- 2-YEAR REAL BACKTEST ===")
    print("Fetching Nifty index 1-min data (2 years)...")
    df1m = orb.fetch_index_chunked(FROM_DATE, TO_DATE)
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
