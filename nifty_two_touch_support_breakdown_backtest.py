"""
TWO-TOUCH SUPPORT BREAKDOWN (3rd-time breach entry) -- PE MIRROR of
nifty_two_touch_resistance_breakout_backtest.py -- 2-YEAR BACKTEST
=================================================================
Per your request to optimize Two-Touch Resistance by adding the PE-side
mirror ("support breakdown -> buy PE") instead of tuning the existing
CE-only version further.

DIRECTION-OF-MIRRORING REASONING (documented explicitly, same convention
used for every PUT-side mirror in this project so far -- e.g. the
fixed-strike ORB PUT variants):

  A PE's premium is a DECREASING function of the index: PE premium goes
  UP when the index goes DOWN. So "the index breaking down below a
  two-touch support level" is, in premium-space, EXACTLY the same shape
  as "the PE's own premium breaking UP above a two-touch resistance
  level" -- because build_premium_series(side="PE") already encodes that
  inversion for you (its own high comes from the index's own low, and
  vice versa).

  That means find_two_touch_resistance_and_entry() -- day-high breach,
  two swing-high pivots within tolerance, third breach above that level
  -- does NOT need to be rewritten or inverted at all. It is completely
  generic: it only ever looks at a premium_df's own high/low/close
  columns and has no idea whether that premium came from a CE or a PE.
  The ONLY change needed is which side you build the premium series
  with. This file reuses find_two_touch_resistance_and_entry() VERBATIM,
  imported from the original (never edited, per project convention).

RULES (identical to the CE version, mirrored side only):
  1. OPENING RANGE: first 30 one-min candles (09:15-09:44), but now on
     the ATM PE's own premium chart.
  2. FIRST BREACH: first bar (from 09:45) whose PE-premium CLOSE exceeds
     that PE's own opening-range high -- this corresponds to the index
     falling meaningfully below its own opening-range low.
  3. TWO-TOUCH RESISTANCE (in PE-premium space): two swing-high pivots on
     the PE's own premium chart within TOUCH_TOLERANCE_PTS of each other.
  4. THIRD BREACH = ENTRY SIGNAL, same look-ahead-safe fill-at-next-open
     rule as everywhere else in this project.
  5. STOP: 8 points below the resistance level (PE-premium points, not
     entry-8). TRAIL: 4-point trail, same conservative-ordering stepper
     (top3.step_long_premium_position) used everywhere else.
  6. PE-ONLY. One trade per day. ATM strike fixed at market open.

Backtest window: 2 years, Rs25,000 starting capital, same DEFAULT_STOP/
DEFAULT_TRAIL=8/4 as the CE side's original spec (not re-swept here --
see the note on the existing project's grid-edge/tail-risk lesson if you
want a resweep later; this file's purpose is testing the PE mirror
itself, not re-optimizing stop/trail on top of it).

Does NOT touch nifty_two_touch_resistance_breakout_backtest.py or
nifty_paper_trade_top3_live.py (both locked/already validated). Imports
both and reuses find_two_touch_resistance_and_entry, atm_strike,
classify_session, SESSIONS, WEEKDAY_ORDER, fetch_index_chunked verbatim.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_support_breakdown_backtest.py --smoke-test
    python nifty_two_touch_support_breakdown_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== config ===============================
BACKTEST_DAYS = 730  # 2 years, matching the CE side's 2yr sweep window for apples-to-apples comparison
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STOP_BELOW_RESISTANCE_PTS = base.STOP_BELOW_RESISTANCE_PTS  # 8.0, same spec as CE side
TRAIL_PTS = base.TRAIL_PTS  # 4.0
BROKERAGE_PER_TRADE = base.BROKERAGE_PER_TRADE
STARTING_CAPITAL = base.STARTING_CAPITAL

TRADES_CSV = "two_touch_support_breakdown_trades.csv"


# =============================================================================
# Day runner -- identical structure to base.run_standalone_for_day, only the
# premium side changes (CE -> PE) and the strategy label/side change to match.
# =============================================================================
def run_standalone_for_day_pe(day_df):
    n = len(day_df)
    if n < base.OPENING_RANGE_BARS + 10:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = base.atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="PE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    sig_bar, resistance, touch1_idx, touch2_idx = base.find_two_touch_resistance_and_entry(premium_df)
    if sig_bar is None:
        return []
    exec_idx = sig_bar + 1
    if exec_idx >= n:
        return []

    entry_price = premium_df["open"].iloc[exec_idx]
    pos = {"entry_price": entry_price, "stop_level": resistance - STOP_BELOW_RESISTANCE_PTS,
           "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
    exit_info = top3.step_long_premium_position(pos, premium_df, TRAIL_PTS)
    if exit_info is None:
        exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                     "exit_time": premium_df.index[-1], "exit_idx": n - 1}
    pnl_pts = exit_info["exit_price"] - entry_price
    rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
    return [{
        "strategy": "TwoTouchSupportBreakdown", "side": "PE", "strike": strike,
        "resistance": resistance, "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    }]


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    base.run_self_checks()  # re-verify the shared primitives + the CE-side hand-traced example

    # --- Check 1: the reused find_two_touch_resistance_and_entry is genuinely
    # side-agnostic -- feed it the EXACT same hand-traced premium shape as the
    # CE self-check and confirm it fires identically (proves the function
    # itself needs zero changes to be reused for PE). ---
    idx = pd.date_range("2025-01-06 09:15", periods=50, freq="1min")
    closes = [100.0] * 30
    closes += [105, 108, 112, 110, 108]
    closes += [105, 104, 106, 111, 109, 107]
    closes += [109, 113]
    closes += [114, 116, 118, 120, 122, 124, 126, 128]
    closes = closes[:50]
    df_premium_shape = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    sig_bar, resistance, t1, t2 = base.find_two_touch_resistance_and_entry(df_premium_shape)
    assert sig_bar == 42 and abs(resistance - 112.5) < 1e-6
    print("  find_two_touch_resistance_and_entry is side-agnostic (same shape -> same signal, reused unchanged) ok")

    # --- Check 2: a REAL support breakdown on the INDEX (falling below its
    # own opening range, forming two touches, then a third breach lower)
    # must, once run through build_premium_series(side='PE'), produce the
    # mirrored day-high/two-touch/third-breach shape in PE-PREMIUM space and
    # generate a completed PE trade. Built by mirroring the same hand-traced
    # closes pattern around a 25000 index level (index falls exactly where
    # the premium shape above rose), so PE premium (which rises as the index
    # falls) reproduces that same rising shape. ---
    day_open_spot = 25000.0
    mirrored_index_closes = [day_open_spot - (c - 100.0) for c in closes]
    mirrored_index_opens = [day_open_spot] + mirrored_index_closes[:-1]
    day_df = pd.DataFrame({
        "open": mirrored_index_opens,
        "high": [c + 0.5 for c in mirrored_index_closes],
        "low": [c - 0.5 for c in mirrored_index_closes],
        "close": mirrored_index_closes,
    }, index=idx)
    assert abs(day_df["open"].iloc[0] - 25000.0) < 1e-9

    strike = base.atm_strike(day_open_spot)
    premium_df_direct = top3.build_premium_series(day_df, strike, day_open_spot, side="PE")
    sig_bar_direct, resistance_direct, _, _ = base.find_two_touch_resistance_and_entry(premium_df_direct)
    assert sig_bar_direct is not None, "expected a support-breakdown signal on the mirrored falling-index day"

    trades = run_standalone_for_day_pe(day_df)
    assert len(trades) == 1, f"expected exactly 1 PE trade, got {len(trades)}"
    tr = trades[0]
    assert tr["side"] == "PE" and tr["strategy"] == "TwoTouchSupportBreakdown"
    exec_idx_direct = sig_bar_direct + 1
    assert abs(tr["entry_price"] - premium_df_direct["open"].iloc[exec_idx_direct]) < 1e-9
    assert abs(tr["resistance"] - resistance_direct) < 1e-9
    print(f"  falling-index (support-breakdown) day -> PE premium mirrors the pattern -> "
          f"1 completed PE trade, entry/resistance match direct computation exactly ok")

    # --- Check 3 (negative control): the SAME closes pattern used as-is for
    # the INDEX (i.e. an index that keeps RISING, the CE-side shape, not
    # mirrored) must NOT produce a PE support-breakdown signal -- PE premium
    # falls as a rising index rises, so no upward breakout shape can form in
    # PE-premium space. This rules out "the function fires regardless of
    # actual direction" as a bug. ---
    day_df_rising = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    trades_none = run_standalone_for_day_pe(day_df_rising)
    assert trades_none == [], f"expected no PE signal on a rising-index day, got {trades_none}"
    print("  negative control: a rising-index day (CE-shaped, not mirrored) correctly produces NO PE signal ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics (identical helpers to the CE-side sweep file)
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


def print_combo_row(label, g):
    pf, sqn, n = compute_pf_sqn(g)
    net = g["rupee_pnl"].sum() if n else 0.0
    win = 100 * (g["rupee_pnl"] > 0).mean() if n else 0.0
    print(f"  {label:<14} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")


def run_full_pipeline(df1m, label="2-YEAR BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(run_standalone_for_day_pe(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades generated at all -- the support-breakdown pattern may be rare with these "
              "tolerance/pivot settings.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    if traded_days / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days. ***")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ TWO-TOUCH SUPPORT BREAKDOWN (PE) -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. ***")

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

    print(f"\n################ 30-MINUTE SESSION-OF-DAY BREAKDOWN ################")
    trades_df["session"] = trades_df["entry_time"].apply(base.classify_session)
    for sess in base.SESSIONS:
        g = trades_df[trades_df["session"] == sess]
        if len(g):
            print_combo_row(sess, g)

    print(f"\n################ DAY-OF-WEEK BREAKDOWN ################")
    trades_df["weekday"] = pd.to_datetime(trades_df["entry_time"]).dt.day_name()
    for wd in base.WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING: worst drawdown approaches/exceeds starting capital. ***")

    daily_pnl = build_daily_pnl(trades_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(trades_df, STARTING_CAPITAL)
    print(f"\n################ FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time capital): {m['sharpe_fulltime']}   "
          f"Sortino (full-time capital): {m['sortino_fulltime']}")
    print(f"Max DD: {m['max_dd_pct']}%   Max DD duration: {m['max_dd_duration_days']} trading days")
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
        df1m = base.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). Re-run on the machine with a working "
              "Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)")
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data.")


def main():
    run_self_checks()
    print(f"=== TWO-TOUCH SUPPORT BREAKDOWN (PE) -- 2-YEAR REAL BACKTEST ===")
    print(f"Window: {FROM_DATE} to {TO_DATE}")
    print("Fetching Nifty index 1-min data...")
    df1m = base.fetch_index_chunked(FROM_DATE, TO_DATE)
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
