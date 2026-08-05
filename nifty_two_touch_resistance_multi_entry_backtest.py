"""
TWO-TOUCH RESISTANCE BREAKOUT -- MULTI-ENTRY (RE-ARMING) VARIANT -- 2-YEAR BACKTEST
=================================================================
Per your annotated chart: day-high breaks, price consolidates into a
two-touch resistance, then breaks that resistance -- and this can happen
SEVERAL TIMES in one trending day (your picture showed 3 separate
red-circled setups as price stair-stepped higher), not just once.

The original nifty_two_touch_resistance_breakout_backtest.py takes only
the FIRST such pattern each day, using the opening range (first 30 bars)
as a fixed "day high" reference, then stops scanning for the rest of the
day. This file changes that to RE-ARM after each trade exits:

  1. First pattern: reference high = opening-range high (bars 0-29),
     exactly as before.
  2. Once that trade EXITS (stop, trail, or EOD), reset reference high
     to the HIGHEST point the option premium has reached so far today
     (not the opening range anymore -- the market has moved), and
     immediately resume scanning from the bar right after the exit for
     the NEXT breach -> two-touch -> third-breach sequence.
  3. Repeat until no further pattern completes or the day ends (capped
     at MAX_TRADES_PER_DAY=6 as a defensive bound, not expected to bind
     in practice since each re-arm only starts after a real exit).

Everything else is UNCHANGED from the original spec: ATM CE premium
chart, TOUCH_TOLERANCE_PTS=3.0, PIVOT_LEFT=PIVOT_RIGHT=2, stop = 8pts
BELOW THE RESISTANCE level (not entry-8), 4pt trail, look-ahead-safe
fill at the next bar's open.

Does NOT touch nifty_two_touch_resistance_breakout_backtest.py or
nifty_paper_trade_top3_live.py (both locked/already validated). Reuses
atm_strike / classify_session / SESSIONS / WEEKDAY_ORDER /
fetch_index_chunked / TOUCH_TOLERANCE_PTS / PIVOT_LEFT / PIVOT_RIGHT /
STOP_BELOW_RESISTANCE_PTS / TRAIL_PTS / OPENING_RANGE_BARS from the
original file verbatim (imported as `base`), and step_long_premium_
position / build_premium_series / lot_size_for_date from top3. The
per-signal pattern-matching math (breach / pivot-tolerance / third-
breach) is the SAME logic as base.find_two_touch_resistance_and_entry,
just parametrized by a reference-high and a search-start bar instead of
hardcoded to the opening range, so it can be called repeatedly within a
day with an updated reference each time.

Also reports the stop-loss usage breakdown you asked for explicitly:
what fraction of ALL trades exit via the 8pt initial stop vs the 4pt
trail vs EOD, both overall and per re-entry number (1st/2nd/3rd trade of
the day), since a multi-entry design can behave differently on later
entries within the same day than on the first.

Backtest window: 2 years, Rs25,000 starting capital.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_multi_entry_backtest.py --smoke-test
    python nifty_two_touch_resistance_multi_entry_backtest.py
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
BACKTEST_DAYS = 730  # 2 years, per your request
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")

MAX_TRADES_PER_DAY = 6  # defensive bound; each re-arm only starts after a real exit, so shouldn't bind
BROKERAGE_PER_TRADE = base.BROKERAGE_PER_TRADE
STARTING_CAPITAL = base.STARTING_CAPITAL

TRADES_CSV = "two_touch_resistance_multi_entry_trades.csv"


# =============================================================================
# Single-pattern search, parametrized by reference_high/search_from instead
# of the hardcoded opening-range day-high -- same breach/pivot-tolerance/
# third-breach logic as base.find_two_touch_resistance_and_entry.
# =============================================================================
def _find_next_signal(premium_df, pivot_series, reference_high, search_from):
    n = len(premium_df)
    closes = premium_df["close"].to_numpy()

    breach1 = None
    for i in range(search_from, n):
        if closes[i] > reference_high:
            breach1 = i
            break
    if breach1 is None:
        return None

    pivot_idxs = [i for i in range(breach1, n)
                  if not pd.isna(pivot_series.iloc[i]) and pivot_series.iloc[i] > reference_high]
    if len(pivot_idxs) < 2:
        return None

    touch1_idx = touch2_idx = None
    for a in range(len(pivot_idxs)):
        for b in range(a + 1, len(pivot_idxs)):
            ia, ib = pivot_idxs[a], pivot_idxs[b]
            va, vb = pivot_series.iloc[ia], pivot_series.iloc[ib]
            if abs(va - vb) <= base.TOUCH_TOLERANCE_PTS:
                touch1_idx, touch2_idx = ia, ib
                break
        if touch1_idx is not None:
            break
    if touch1_idx is None:
        return None

    resistance = max(pivot_series.iloc[touch1_idx], pivot_series.iloc[touch2_idx])
    search_start = touch2_idx + base.PIVOT_RIGHT + 1
    for i in range(search_start, n):
        if closes[i] > resistance:
            return i, resistance, touch1_idx, touch2_idx
    return None


def run_standalone_for_day_multi(day_df, max_trades_per_day=MAX_TRADES_PER_DAY):
    n = len(day_df)
    if n < base.OPENING_RANGE_BARS + 10:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = base.atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    pivot_series = dlb.pivot_high(premium_df["high"], base.PIVOT_LEFT, base.PIVOT_RIGHT)
    highs = premium_df["high"].to_numpy()

    reference_high = highs[:base.OPENING_RANGE_BARS].max()
    search_from = base.OPENING_RANGE_BARS
    trades = []

    while len(trades) < max_trades_per_day:
        result = _find_next_signal(premium_df, pivot_series, reference_high, search_from)
        if result is None:
            break
        sig_bar, resistance, touch1_idx, touch2_idx = result
        exec_idx = sig_bar + 1
        if exec_idx >= n:
            break

        entry_price = premium_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": resistance - base.STOP_BELOW_RESISTANCE_PTS,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, base.TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": n - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        trades.append({
            "strategy": "TwoTouchResistanceMultiEntry", "side": "CE", "strike": strike,
            "entry_num_today": len(trades) + 1, "reference_high": round(reference_high, 2),
            "resistance": round(resistance, 2), "entry_time": premium_df.index[exec_idx],
            "exit_time": exit_info["exit_time"], "entry_price": entry_price,
            "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
            "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
        })

        exit_idx = exit_info.get("exit_idx", n - 1)
        if exit_idx >= n - 1:
            break  # day is effectively over
        reference_high = max(reference_high, highs[:exit_idx + 1].max())
        search_from = exit_idx + 1

    return trades


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    base.run_self_checks()  # re-verify shared primitives + the original single-entry pattern logic
    print("  base.run_self_checks() re-verified ok")

    # --- Hand-built two-pattern day: pattern 1 (identical shape to base's own
    # hand-traced example, base level 100), a sharp pullback to force a real
    # TRAIL_SL exit, then pattern 2 built on TOP of pattern 1's peak (base
    # level 120 = pattern 1's high), verifying re-arm picks up the SECOND
    # pattern using an updated (higher) reference high, strictly after the
    # first trade's exit. Exact bar indices below were verified by running
    # this exact synthetic day through the algorithm in a sandbox first (not
    # hand-guessed) -- see the delivery notes for the raw run. ---
    def _pattern_block(base_level):
        vals = [base_level] * 30
        vals += [base_level + 5, base_level + 8, base_level + 12, base_level + 10, base_level + 8]
        vals += [base_level + 5, base_level + 4, base_level + 6, base_level + 11, base_level + 9, base_level + 7]
        vals += [base_level + 9, base_level + 13]
        vals += [base_level + 14, base_level + 16, base_level + 18, base_level + 20]
        return vals

    block1 = _pattern_block(100.0)
    pullback = [116.0, 112.0, 111.0]
    block2_base = 120.0  # block1's peak
    consolidation = [119.0, 118.0, 120.0, 119.0]
    block2 = consolidation + _pattern_block(block2_base)[30:]  # skip the redundant flat-30 prefix
    closes = block1 + pullback + block2
    idx = pd.date_range("2025-01-06 09:15", periods=len(closes), freq="1min")
    day_df = pd.DataFrame({
        "open": [25000.0 - (100.0 - closes[0])] + [None] * (len(closes) - 1),  # placeholder, fixed below
        "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    # Build this as a raw premium-shaped series fed straight in (same convention
    # base.run_self_checks() uses -- the day_df IS the premium chart here, ATM
    # strike/side don't matter for this structural check).
    day_df["open"] = [closes[0]] + closes[:-1]

    pivot_series = dlb.pivot_high(day_df["high"], base.PIVOT_LEFT, base.PIVOT_RIGHT)
    highs = day_df["high"].to_numpy()
    reference_high = highs[:base.OPENING_RANGE_BARS].max()
    search_from = base.OPENING_RANGE_BARS
    trades = []
    while len(trades) < MAX_TRADES_PER_DAY:
        result = _find_next_signal(day_df, pivot_series, reference_high, search_from)
        if result is None:
            break
        sig_bar, resistance, t1, t2 = result
        exec_idx = sig_bar + 1
        if exec_idx >= len(day_df):
            break
        entry_price = day_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": resistance - base.STOP_BELOW_RESISTANCE_PTS,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, day_df, base.TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": day_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": day_df.index[-1], "exit_idx": len(day_df) - 1}
        trades.append(dict(sig_bar=sig_bar, resistance=resistance, reference_high=reference_high,
                            exit_idx=exit_info["exit_idx"], reason=exit_info["reason"],
                            exit_time=exit_info["exit_time"], entry_time=day_df.index[exec_idx]))
        exit_idx = exit_info.get("exit_idx", len(day_df) - 1)
        if exit_idx >= len(day_df) - 1:
            break
        reference_high = max(reference_high, highs[:exit_idx + 1].max())
        search_from = exit_idx + 1

    assert len(trades) == 2, f"expected exactly 2 re-armed trades on this synthetic two-pattern day, got {len(trades)}"
    assert trades[0]["sig_bar"] == 42 and abs(trades[0]["resistance"] - 112.5) < 1e-6
    assert trades[0]["reason"] == "TRAIL_SL", f"expected trade 1 to exit via TRAIL_SL, got {trades[0]['reason']}"
    assert trades[1]["reference_high"] > trades[0]["reference_high"], (
        "expected the 2nd pattern's reference_high to be higher than the 1st (re-armed off the day's "
        "highest point reached so far, not reset back to the opening range)")
    assert trades[1]["resistance"] > trades[0]["resistance"], "expected the 2nd pattern's resistance to sit higher too"
    assert trades[1]["entry_time"] > trades[0]["exit_time"], (
        "expected the 2nd trade to start strictly after the 1st trade's own exit (no overlap)")
    print(f"  multi-entry re-arm: 2 trades fire on the synthetic two-pattern day, trade 2 uses an updated "
          f"(higher) reference high ({trades[1]['reference_high']:.1f} > {trades[0]['reference_high']:.1f}) "
          f"and starts strictly after trade 1's exit ok")

    # --- single-pattern day (no second breakout available) must still
    # produce exactly the SAME single trade as the original base script,
    # proving the multi-entry loop doesn't change day-1 behavior. ---
    idx0 = pd.date_range("2025-01-06 09:15", periods=50, freq="1min")
    closes0 = [100.0] * 30 + [105, 108, 112, 110, 108] + [105, 104, 106, 111, 109, 107] + [109, 113] + \
              [114, 116, 118, 120, 122, 124, 126, 128]
    closes0 = closes0[:50]
    df0 = pd.DataFrame({"open": [100.0] + closes0[:-1], "high": [c + 0.5 for c in closes0],
                         "low": [c - 0.5 for c in closes0], "close": closes0}, index=idx0)
    sig_bar0, resistance0, *_ = base.find_two_touch_resistance_and_entry(df0)
    pivot_series0 = dlb.pivot_high(df0["high"], base.PIVOT_LEFT, base.PIVOT_RIGHT)
    highs0 = df0["high"].to_numpy()
    result0 = _find_next_signal(df0, pivot_series0, highs0[:base.OPENING_RANGE_BARS].max(), base.OPENING_RANGE_BARS)
    assert result0 is not None and result0[0] == sig_bar0 and abs(result0[1] - resistance0) < 1e-6, (
        "multi-entry's single-pattern search must match the original single-entry function exactly "
        "on a day where only one pattern exists")
    print("  single-pattern day: multi-entry's first signal matches base.find_two_touch_resistance_and_entry exactly ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics
# =============================================================================
def compute_pf_sqn(trades_df):
    if trades_df is None or trades_df.empty:
        return 0.0, 0.0, 0
    pnl = trades_df["rupee_pnl"]
    n = len(pnl)
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
    print(f"  {label:<20} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")


def run_full_pipeline(df1m, label="2-YEAR MULTI-ENTRY BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(run_standalone_for_day_multi(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades generated at all.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    trades_per_day = trades_df.groupby("date").size()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    print(f"  Trades per traded day: mean={trades_per_day.mean():.2f}  max={trades_per_day.max()}  "
          f"distribution: {trades_per_day.value_counts().sort_index().to_dict()}")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ MULTI-ENTRY TWO-TOUCH RESISTANCE -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. ***")

    # --- stop-loss probability breakdown, overall AND by entry-number-of-day
    # (this is the "probability of stop loss" you specifically asked for) ---
    print(f"\n################ STOP-LOSS USAGE BREAKDOWN ################")
    print("-- Overall, by exit reason --")
    for reason, g in trades_df.groupby("reason"):
        pct = 100 * len(g) / len(trades_df)
        print(f"  {reason:<14} n={len(g):<5} ({pct:5.1f}% of all trades)  net=Rs{g['rupee_pnl'].sum():>10,.2f}")
    initial_sl = trades_df[trades_df["reason"] == "INITIAL_SL"]
    print(f"\n  Probability any given trade hits the initial 8pt stop: "
          f"{100*len(initial_sl)/len(trades_df):.1f}% ({len(initial_sl)}/{len(trades_df)} trades)")

    print("\n-- By entry number within the day (1st, 2nd, 3rd... re-arm) --")
    for entry_num, g in trades_df.groupby("entry_num_today"):
        sl_pct = 100 * (g["reason"] == "INITIAL_SL").mean()
        print(f"  entry #{entry_num}: n={len(g):<5} stop-loss rate={sl_pct:5.1f}%  "
              f"win%={100*(g['rupee_pnl']>0).mean():5.1f}  net=Rs{g['rupee_pnl'].sum():>10,.2f}")

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
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (full-time): {m['sharpe_fulltime']}   "
          f"Sortino (full-time): {m['sortino_fulltime']}")
    print(f"Max DD: {m['max_dd_pct']}%   Max DD duration: {m['max_dd_duration_days']} trading days")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    print(f"\n################ COMPARISON vs single-entry (previously reported 2yr default combo) ################")
    print("  Single-entry (previous):  n=228  PF=1.19  SQN=0.83  net_Rs=4,967.45  CAGR=9.49%  p=0.36 (not significant)")
    print(f"  Multi-entry (this run):   n={n}  PF={pf:.2f}  SQN={sqn:.2f}  net_Rs={total_pnl:,.2f}  "
          f"CAGR={m['cagr_pct']}%  p={m['p_value']} ({'significant' if m['p_value'] < 0.05 else 'not significant'})")

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
    print("\nSmoke test complete -- no crashes, all self-checks passed, full multi-entry pipeline ran "
          "end-to-end on real data.")


def main():
    run_self_checks()
    print(f"=== TWO-TOUCH RESISTANCE -- MULTI-ENTRY (RE-ARMING) -- 2-YEAR REAL BACKTEST ===")
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
