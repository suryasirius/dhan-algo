"""
TWO-TOUCH RESISTANCE BREAKOUT, 3-CANDLE FULL-CONFIRMATION -- 6-MONTH BACKTEST
=================================================================
Variant of nifty_two_touch_resistance_breakout_backtest.py with your new
condition added: instead of entering on the first close above the
two-touch resistance, this now requires 3 CONSECUTIVE 1-minute candles
that fully stand above the resistance level -- i.e. each of the 3
candles' LOW (not just its close) must be above resistance, so no wick
dips back through the level during those 3 candles. If a candidate run
breaks partway through (a low dips back to/below resistance), this keeps
scanning forward for the next bar where a fresh 3-candle fully-above run
begins -- it does not just give up after the first failed attempt.

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
    python nifty_two_touch_resistance_3candle_confirm_backtest.py --smoke-test
    python nifty_two_touch_resistance_3candle_confirm_backtest.py
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
CONFIRM_BARS = 3  # NEW: 3 consecutive candles must fully stand above the resistance
STOP_BELOW_RESISTANCE_PTS = base.STOP_BELOW_RESISTANCE_PTS  # 8
TRAIL_PTS = base.TRAIL_PTS  # 4
BROKERAGE_PER_TRADE = base.BROKERAGE_PER_TRADE
STARTING_CAPITAL = base.STARTING_CAPITAL

TRADES_CSV = "two_touch_resistance_3candle_confirm_trades.csv"


# =============================================================================
# Pattern detection: day-high breach -> two-touch resistance -> NEW 3-candle
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


def find_entry_with_3candle_confirm(premium_df, resistance, search_start):
    """NEW rule: scan forward from search_start. Whenever a bar's CLOSE is
    above resistance, check whether that bar AND the next (CONFIRM_BARS-1)
    bars all have their LOW fully above resistance too (no wick dipping
    back through the level). If that 3-candle run holds, entry fires at
    the open of the bar right after the run completes. If the run breaks
    (a low dips back to/below resistance before completing), this keeps
    scanning forward for the next candidate -- it doesn't give up after
    one failed attempt."""
    closes = premium_df["close"].to_numpy()
    lows = premium_df["low"].to_numpy()
    n = len(premium_df)
    i = search_start
    while i < n:
        if closes[i] > resistance and i + CONFIRM_BARS <= n:
            if all(lows[i + k] > resistance for k in range(CONFIRM_BARS)):
                return i + CONFIRM_BARS  # exec_idx: fill at this bar's open
        i += 1
    return None


def find_signal(premium_df):
    resistance, touch1_idx, touch2_idx = find_two_touch_resistance(premium_df)
    if resistance is None:
        return None, None
    search_start = touch2_idx + PIVOT_RIGHT + 1  # look-ahead safe, same convention as the base script
    exec_idx = find_entry_with_3candle_confirm(premium_df, resistance, search_start)
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
        "strategy": "TwoTouchResistance_3CandleConfirm", "side": "CE", "strike": strike,
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

    # --- reuse the base script's exact hand-traced two-touch setup, but this
    # time verify the NEW 3-candle confirm logic on top of it ---
    idx = pd.date_range("2025-01-06 09:15", periods=60, freq="1min")
    closes = [100.0] * 30
    closes += [105, 108, 112, 110, 108]                 # bars 30-34: breach + peak1 (112) at bar32
    closes += [105, 104, 106, 111, 109, 107]             # bars 35-40: dip + peak2 (111) at bar38
    # resistance = max(112,111) = 112. search_start = 38+2+1 = 41.
    # bar41 close=109 (below 112, not a candidate). bar42 close=113 (candidate!) but its own low
    # will be set to close-0.5=112.5 > 112 -- fully above. Need bars 42,43,44 all low>112.
    closes += [109, 114, 115, 116]                       # bars 41-44: candidate run starts at 42, holds through 44
    closes += [116] * 15                                 # bars 45-59: plenty of room for entry + exit testing
    closes = closes[:60]
    df = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)

    resistance, t1, t2 = find_two_touch_resistance(df)
    assert abs(resistance - 112.5) < 1e-6, f"expected resistance ~112.5 (same as the base script's test), got {resistance}"

    exec_idx, res2 = find_signal(df)
    # bar42 close=114>112.5 -- candidate. lows[42]=113.5, lows[43]=114.5, lows[44]=115.5 -- all clearly
    # above the 112.5 resistance. Run holds -> entry fires 3 bars later, at bar 45.
    assert exec_idx == 42 + CONFIRM_BARS, f"expected entry at bar {42+CONFIRM_BARS} (candidate bar 42 + 3-bar confirm), got {exec_idx}"
    print("  3-candle full-confirmation entry fires exactly 3 bars after a clean qualifying run ok")

    # --- a run that breaks partway (a low dips back through resistance) must be skipped,
    # and the search must find the NEXT qualifying run instead ---
    closes_break = [100.0] * 30
    closes_break += [105, 108, 112, 110, 108]
    closes_break += [105, 104, 106, 111, 109, 107]
    # candidate at bar42 (close=114, low=113.5>112.5 -- run STARTS), but bar43's close drops back to 108
    # (below resistance), so its low (107.5) is also well below resistance -- the 3-bar run starting at
    # 42 (bars 42,43,44) fails at bar43. The next candidate that actually holds is bar44.
    closes_break += [109, 114, 108, 115, 114, 118]  # bars 41-46
    closes_break += [119, 120]                       # bars 47-48
    closes_break += [121] * 11                       # bars 49-59
    closes_break = closes_break[:60]
    df_break = pd.DataFrame({
        "open": [100.0] + closes_break[:-1], "high": [c + 0.5 for c in closes_break],
        "low": [c - 0.5 for c in closes_break], "close": closes_break,
    }, index=idx)
    resistance_b, _, _ = find_two_touch_resistance(df_break)
    exec_idx_b, _ = find_signal(df_break)
    assert exec_idx_b is not None and exec_idx_b != 42 + CONFIRM_BARS, (
        f"the bar-42 run should have broken (bar43's low fell back through resistance); expected a "
        f"LATER entry bar, got {exec_idx_b}"
    )
    assert exec_idx_b == 44 + CONFIRM_BARS, (
        f"expected the next qualifying run (starting at bar 44, since bar42's run broke at bar43) to "
        f"fire the entry at bar {44+CONFIRM_BARS}, got {exec_idx_b}"
    )
    print("  a broken run (close dips back below resistance) is correctly rejected, and the next "
          "qualifying run is found instead ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics (same structure as the base script)
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
        print("No trades generated at all -- the 3-candle full-confirmation requirement may be too "
              "strict on top of the two-touch pattern for this window.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    print(f"\n-- Comparison to the plain (no 3-candle confirm) version on the same 6-month window --")
    print(f"  Plain version had: n=61  PF=1.82  SQN=1.28  net=Rs4,806.85  p=0.1853 (not significant)")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ 3-CANDLE-CONFIRM -- HEADLINE RESULT ################")
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
    print(f"=== TWO-TOUCH RESISTANCE, 3-CANDLE CONFIRM -- 6-MONTH REAL BACKTEST ===")
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
