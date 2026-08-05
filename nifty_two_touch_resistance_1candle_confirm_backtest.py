"""
TWO-TOUCH RESISTANCE BREAKOUT, 1-CANDLE FULL-CONFIRMATION -- 6-MONTH BACKTEST
=================================================================
Variant of nifty_two_touch_resistance_breakout_backtest.py (and a lighter
version of nifty_two_touch_resistance_3candle_confirm_backtest.py) with
your latest condition: instead of entering on the first close above the
two-touch resistance, this now requires just 1 candle that fully stands
above the resistance level -- i.e. that candle's LOW (not just its
close) must also be above resistance, ruling out a bar that only pokes
above on a wick/close while still trading below resistance intrabar. If
a candidate bar closes above resistance but its low doesn't clear it too
(a "fake" breakout bar), this keeps scanning forward for the next bar
that genuinely stands fully above -- it doesn't give up after one failed
candidate.

Everything else is unchanged from the original concept and your stated
rules: 30-min opening range (09:15-09:44) marks Day High on the ATM CALL's
own premium chart; the two-touch ("U shape") resistance is found using
this project's existing pivot_high() swing-detection function (2 bars
each side, touches within 3 points of each other); stop-loss = 8 points
BELOW THE RESISTANCE LEVEL (not below entry); 4-point trail once in
profit; CE-only; one trade per day; ATM strike fixed at market open;
Rs25,000 starting capital; 6-month backtest window.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_1candle_confirm_backtest.py --smoke-test
    python nifty_two_touch_resistance_1candle_confirm_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== config ===============================
BACKTEST_DAYS = 183  # ~6 months, per your instruction
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")

OPENING_RANGE_BARS = base.OPENING_RANGE_BARS
PIVOT_LEFT, PIVOT_RIGHT = base.PIVOT_LEFT, base.PIVOT_RIGHT
TOUCH_TOLERANCE_PTS = base.TOUCH_TOLERANCE_PTS
CONFIRM_BARS = 1  # NEW: just 1 candle must fully stand above the resistance
STOP_BELOW_RESISTANCE_PTS = base.STOP_BELOW_RESISTANCE_PTS  # 8
TRAIL_PTS = base.TRAIL_PTS  # 4
BROKERAGE_PER_TRADE = base.BROKERAGE_PER_TRADE
STARTING_CAPITAL = base.STARTING_CAPITAL

TRADES_CSV = "two_touch_resistance_1candle_confirm_trades.csv"


# =============================================================================
# Pattern detection: day-high breach -> two-touch resistance -> NEW 1-candle
# full-confirmation entry (replaces the plain "first close above resistance").
# =============================================================================
def find_two_touch_resistance(premium_df):
    """Same day-high + two-touch pivot logic as the base script. Returns
    (resistance, touch1_idx, touch2_idx) or (None, None, None)."""
    n = len(premium_df)
    if n < OPENING_RANGE_BARS + 2 * PIVOT_RIGHT + 5:
        return None, None, None
    highs = premium_df["high"].to_numpy()
    closes = premium_df["close"].to_numpy()
    day_high = highs[:OPENING_RANGE_BARS].max()

    breach1 = None
    for i in range(OPENING_RANGE_BARS, n):
        if closes[i] > day_high:
            breach1 = i
            break
    if breach1 is None:
        return None, None, None

    pivot_series = dlb.pivot_high(premium_df["high"], PIVOT_LEFT, PIVOT_RIGHT)
    pivot_idxs = [i for i in range(breach1, n)
                  if not pd.isna(pivot_series.iloc[i]) and pivot_series.iloc[i] > day_high]
    if len(pivot_idxs) < 2:
        return None, None, None

    touch1_idx = touch2_idx = None
    for a in range(len(pivot_idxs)):
        for b in range(a + 1, len(pivot_idxs)):
            ia, ib = pivot_idxs[a], pivot_idxs[b]
            va, vb = pivot_series.iloc[ia], pivot_series.iloc[ib]
            if abs(va - vb) <= TOUCH_TOLERANCE_PTS:
                touch1_idx, touch2_idx = ia, ib
                break
        if touch1_idx is not None:
            break
    if touch1_idx is None:
        return None, None, None

    resistance = max(pivot_series.iloc[touch1_idx], pivot_series.iloc[touch2_idx])
    return resistance, touch1_idx, touch2_idx


def find_entry_with_confirm(premium_df, resistance, search_start):
    """NEW rule: scan forward from search_start for the first bar whose
    CLOSE is above resistance AND whose LOW is also above resistance (the
    whole candle -- not just a wick or the close -- stands above the
    level). If a candidate bar closes above resistance but its low
    doesn't clear it (a fake/wick breakout), this keeps scanning for the
    next candidate rather than giving up."""
    closes = premium_df["close"].to_numpy()
    lows = premium_df["low"].to_numpy()
    n = len(premium_df)
    for i in range(search_start, n - CONFIRM_BARS + 1):
        if closes[i] > resistance and all(lows[i + k] > resistance for k in range(CONFIRM_BARS)):
            return i + CONFIRM_BARS  # exec_idx: fill at this bar's open
    return None


def find_signal(premium_df):
    resistance, touch1_idx, touch2_idx = find_two_touch_resistance(premium_df)
    if resistance is None:
        return None, None
    search_start = touch2_idx + PIVOT_RIGHT + 1  # look-ahead safe, same convention as the base script
    exec_idx = find_entry_with_confirm(premium_df, resistance, search_start)
    if exec_idx is None:
        return None, None
    return exec_idx, resistance


def run_standalone_for_day(day_df):
    n = len(day_df)
    if n < OPENING_RANGE_BARS + 10:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = base.atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    exec_idx, resistance = find_signal(premium_df)
    if exec_idx is None or exec_idx >= n:
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
        "strategy": "TwoTouchResistance_1CandleConfirm", "side": "CE", "strike": strike,
        "resistance": resistance, "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    }]


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    base.run_self_checks()  # re-verify the shared day-high/two-touch/premium/position primitives

    idx = pd.date_range("2025-01-06 09:15", periods=50, freq="1min")
    closes = [100.0] * 30
    closes += [105, 108, 112, 110, 108]      # bars 30-34: breach + peak1 (112) at bar32
    closes += [105, 104, 106, 111, 109, 107]  # bars 35-40: dip + peak2 (111) at bar38
    # resistance = max(112,111) = 112 (highs, so 112.5). search_start = 38+2+1 = 41.
    closes += [109, 114, 115]                 # bars 41-43
    closes += [116] * 12                      # bars 44-49
    closes = closes[:50]
    df = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)

    resistance, t1, t2 = find_two_touch_resistance(df)
    assert abs(resistance - 112.5) < 1e-6, f"expected resistance ~112.5, got {resistance}"

    # --- clean case: bar42 (close=114, low=113.5) fully stands above 112.5 on its own -- fires immediately ---
    exec_idx, _ = find_signal(df)
    assert exec_idx == 42 + CONFIRM_BARS, f"expected entry at bar {42+CONFIRM_BARS} (bar42 qualifies on its own), got {exec_idx}"
    print("  a single candle fully standing above resistance fires the entry immediately (1 bar later) ok")

    # --- fake/wick breakout: bar42 CLOSES above resistance but its LOW doesn't clear it (a wick dip) --
    # must be rejected, and the search must find the next bar that genuinely stands fully above ---
    lows_wick = [c - 0.5 for c in closes]
    lows_wick[42] = 110.0  # bar42 closes at 114 (>112.5) but its low dips to 110 (<112.5) -- a fake breakout
    df_wick = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": lows_wick, "close": closes,
    }, index=idx)
    resistance_w, _, _ = find_two_touch_resistance(df_wick)
    exec_idx_w, _ = find_signal(df_wick)
    assert exec_idx_w is not None and exec_idx_w != 42 + CONFIRM_BARS, (
        f"bar42's wick-below-resistance low should have rejected it as a candidate, got {exec_idx_w}"
    )
    assert exec_idx_w == 43 + CONFIRM_BARS, (
        f"expected bar43 (close=115, low=114.5, genuinely above resistance) to fire the entry at "
        f"bar {43+CONFIRM_BARS}, got {exec_idx_w}"
    )
    print("  a fake/wick breakout (closes above resistance but low dips back below) is correctly "
          "rejected, and the next genuine candidate fires instead ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics (same structure as the base/3-candle scripts)
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


def run_full_pipeline(df1m, label="6-MONTH BACKTEST"):
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
    print(f"\n-- Comparison to the other variants on the same 6-month window --")
    print(f"  Plain (no confirm):    n=61  PF=1.82  SQN=1.28  net=Rs4,806.85  p=0.1853 (not significant)")
    print(f"  3-candle confirm:      (see that script's own run for its numbers)")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ 1-CANDLE-CONFIRM -- HEADLINE RESULT ################")
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
    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    for wd in base.WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING: worst drawdown approaches/exceeds starting capital. ***")
    if (equity_curve <= 0).any():
        print("  WARNING: equity touches Rs0 or below at some point in this trade sequence.")
    else:
        print("  Equity never touches Rs0 or below across this trade sequence.")

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
        df1m = base.fetch_index_chunked(from_date, to_date)
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
    print(f"=== TWO-TOUCH RESISTANCE, 1-CANDLE CONFIRM -- 6-MONTH REAL BACKTEST ===")
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
