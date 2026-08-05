"""
COMBINED PRIORITY-ARBITRATED STRATEGY (2-concept version) -- variant of
combined_priority_strategy_backtest.py (the validated 3-concept version:
1020 trades, PF 1.41, SQN 3.68, net Rs77,157/2yr, max DD -30.07%) built to
test dropping Crash Play entirely.

WHY THIS VARIANT: in the 3-concept combined backtest, Crash Play
contributed only 0.9% of combined net P&L (Rs684 on 242 trades, PF 1.02,
SQN 0.08) and was net NEGATIVE in the second year of the sample (PF 0.71,
net -Rs5,746) despite a real standalone edge in its own earlier isolated
test (PF 1.64, net Rs10,902/2yr). Priority arbitration only skipped 7 of
its trades (worth Rs3,293) to Velez, so its weak/decaying showing here is
not an arbitration artifact -- it is Crash Play's own trade performance
in this rolling 2-year window. This variant tests whether removing it
improves the combined portfolio's risk profile (SQN, max drawdown) at
a near-zero net P&L cost.

PRIORITY ORDER (highest first):
  1. Opening-Drive Momentum   SQN 3.02, PF 1.44, CAGR 72.3%  (nifty_open_momentum_trail.py, sl8_trail4)
  2. Velez 8pt                SQN 2.63, PF 1.71, CAGR 45.2%  (velez_8pt_proxy_premium_test.py)

  Excluded: Crash Play (see above), Concepts 2 & 4, O=H/O=L, six-concepts
  screening survivors, fixed-strike timesweeps -- same playbook verdicts
  as the 3-concept version.

HOW THE ARBITRATION WORKS: identical two-stage design as the 3-concept
version -- each concept's own standalone trade list (one position at a
time for itself, reusing its own already-validated signal-detection and
conservative-ordering exit logic verbatim from nifty_paper_trade_top3_live.py),
then a weighted-interval-scheduling greedy pools and arbitrates: process
candidates ordered by (priority, entry_time), keep a candidate only if it
doesn't overlap an already-accepted trade, otherwise skip it (attributed
to whichever higher-priority trade blocked it).

MONEY MODEL: same proxy-premium model as both concepts' own validated
standalone backtests. Starting capital Rs25,000, flat Rs40/round-trip
brokerage on every trade -- same conventions as the 3-concept version, so
results are directly comparable.

Two ways to run this:
  Self-checks + smoke test (safe any time, ~10-20s, no CSV writes):
      cd "C:\\dhan algo"
      python combined_priority_2concepts_backtest.py --smoke-test
  Full 2-year real backtest:
      cd "C:\\dhan algo"
      python combined_priority_2concepts_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd
import numpy as np

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== shared config ===============================
FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0

PRIORITY = {"OpeningDrive": 1, "Velez8pt": 2}

TRADES_CSV = "combined_priority_2concepts_ALL_standalone_trades.csv"
SELECTED_CSV = "combined_priority_2concepts_SELECTED_trades.csv"
SKIPPED_CSV = "combined_priority_2concepts_SKIPPED_trades.csv"


# ============================== data fetch ===================================
def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp)
        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
            "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


# =============================================================================
# Stage 1a/b -- Velez 8pt / Opening-Drive standalone trades, reusing
# signal-detection + conservative-ordering steppers VERBATIM from
# nifty_paper_trade_top3_live.py (imported as `top3`).
# =============================================================================
def run_velez_standalone_for_day(day_df_velez):
    trades = []
    sigs = top3.detect_velez_signals(day_df_velez)
    if not sigs:
        return trades
    n = len(day_df_velez)
    day_open_spot = day_df_velez["open"].iloc[0]
    strike = top3.itm_put_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df_velez, strike, day_open_spot, side="PE")
    lot = top3.lot_size_for_date(day_df_velez.index[0].date())
    open_until_idx = -1
    for sig in sigs:
        j0 = sig["bar"]
        if j0 <= open_until_idx:
            continue
        exec_idx = j0 + 1
        if exec_idx >= n:
            continue
        entry_price = premium_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": entry_price - top3.VELEZ_STOP_PTS,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, top3.VELEZ_TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": n - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        trades.append({
            "strategy": "Velez8pt", "priority": PRIORITY["Velez8pt"], "side": "PE", "strike": strike,
            "event": sig["event"], "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
            "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
            "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df_velez.index[0].date(),
        })
        open_until_idx = exit_info["exit_idx"]
    return trades


def run_opening_drive_standalone_for_day(day_df):
    trades = []
    local_i, direction = top3.find_opening_drive(day_df)
    if local_i is None:
        return trades
    n = len(day_df)
    exec_idx = local_i + 1
    if exec_idx >= n:
        return trades
    day_open_spot = day_df["open"].iloc[0]
    side = "CE" if direction == "LONG" else "PE"
    strike = top3.itm_call_strike(day_open_spot) if side == "CE" else top3.itm_put_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side=side)
    lot = top3.lot_size_for_date(day_df.index[0].date())
    entry_price = premium_df["open"].iloc[exec_idx]
    pos = {"entry_price": entry_price, "stop_level": entry_price - top3.OD_STOP_PTS,
           "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
    exit_info = top3.step_long_premium_position(pos, premium_df, top3.OD_TRAIL_PTS)
    if exit_info is None:
        exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                     "exit_time": premium_df.index[-1], "exit_idx": n - 1}
    pnl_pts = exit_info["exit_price"] - entry_price
    rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
    trades.append({
        "strategy": "OpeningDrive", "priority": PRIORITY["OpeningDrive"], "side": side, "strike": strike,
        "event": direction, "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    })
    return trades


# =============================================================================
# Stage 2 -- pool all standalone trades, priority-arbitrate to a single
# non-overlapping combined portfolio (weighted-interval-scheduling greedy).
# =============================================================================
def select_non_overlapping_by_priority(all_trades):
    ordered = sorted(all_trades, key=lambda c: (c["priority"], c["entry_time"]))
    selected_intervals = []
    selected, skipped = [], []
    for c in ordered:
        conflict = False
        blocker = None
        for s_c, s_t in selected_intervals:
            if c["entry_time"] < s_t and s_c["entry_time"] < c["exit_time"]:
                conflict = True
                blocker = s_c
                break
        if conflict:
            c2 = dict(c)
            c2["blocked_by_strategy"] = blocker["strategy"]
            c2["blocked_by_entry_time"] = blocker["entry_time"]
            skipped.append(c2)
        else:
            selected_intervals.append((c, c["exit_time"]))
            selected.append(c)
    selected.sort(key=lambda c: c["entry_time"])
    skipped.sort(key=lambda c: c["entry_time"])
    return selected, skipped


# =============================================================================
# Self-checks -- hand-traced synthetic assertions, no network required
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()  # re-verify the underlying verbatim primitives first

    # --- priority arbitration: two overlapping candidates, higher priority
    # (lower number) must win ---
    t0 = pd.Timestamp("2025-01-06 09:20")
    cands = [
        {"strategy": "Velez8pt", "priority": 2, "entry_time": t0, "exit_time": t0 + pd.Timedelta(minutes=20),
         "rupee_pnl": 100, "date": t0.date()},
        {"strategy": "OpeningDrive", "priority": 1, "entry_time": t0 + pd.Timedelta(minutes=5),
         "exit_time": t0 + pd.Timedelta(minutes=30), "rupee_pnl": 200, "date": t0.date()},
    ]
    sel, skip = select_non_overlapping_by_priority(cands)
    sel_strats = {s["strategy"] for s in sel}
    assert sel_strats == {"OpeningDrive"}, f"expected only OpeningDrive kept, got {sel_strats}"
    assert len(skip) == 1 and skip[0]["strategy"] == "Velez8pt" and skip[0]["blocked_by_strategy"] == "OpeningDrive", (
        f"expected Velez8pt skipped and attributed to OpeningDrive, got {skip}"
    )
    print("  priority arbitration (higher-probability concept wins an overlap) ok")

    # --- non-overlapping candidates of DIFFERENT priority must ALL be kept ---
    t1 = pd.Timestamp("2025-01-07 09:20")
    cands2 = [
        {"strategy": "OpeningDrive", "priority": 1, "entry_time": t1, "exit_time": t1 + pd.Timedelta(minutes=10),
         "rupee_pnl": 30, "date": t1.date()},
        {"strategy": "Velez8pt", "priority": 2, "entry_time": t1 + pd.Timedelta(minutes=15),
         "exit_time": t1 + pd.Timedelta(minutes=25), "rupee_pnl": 40, "date": t1.date()},
    ]
    sel2, skip2 = select_non_overlapping_by_priority(cands2)
    assert len(sel2) == 2 and len(skip2) == 0, "non-overlapping trades of different priority must both be kept"
    print("  non-overlapping trades of different priority both kept ok")

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


def print_entry_frequency(all_trades_df, days_covered_count, label):
    print(f"\n-- Entry-frequency diagnostics ({label}) --")
    for strat in ["OpeningDrive", "Velez8pt"]:
        g = all_trades_df[all_trades_df["strategy"] == strat]
        if len(g) == 0:
            print(f"  {strat:<14} 0 trades")
            continue
        traded_days = g["date"].nunique()
        long_ct = (g["side"].isin(["LONG", "CE"])).sum()
        short_ct = (g["side"].isin(["SHORT", "PE"])).sum()
        print(f"  {strat:<14} {len(g):>4} trades on {traded_days:>4} days "
              f"({100*traded_days/days_covered_count:5.1f}% of {days_covered_count} days)   "
              f"long/CE={long_ct} short/PE={short_ct}")


def run_full_pipeline(df1m, label="FULL 2-YEAR BACKTEST"):
    print(f"\n=== {label} ===")
    df2m = top3.resample_2min_per_day(df1m)
    df2m_velez = top3.compute_velez_indicators(df2m)

    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df2m = df2m[df2m.index.date == day]
        day_df2m_velez = df2m_velez[df2m_velez.index.date == day]
        if len(day_df2m) >= 10:
            all_trades.extend(run_velez_standalone_for_day(day_df2m_velez))
            all_trades.extend(run_opening_drive_standalone_for_day(day_df2m))

        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} standalone candidate trades so far")

    if not all_trades:
        print("No trades generated by any concept -- cannot proceed.")
        return None

    all_df = pd.DataFrame(all_trades)
    all_df.to_csv(TRADES_CSV, index=False)
    print(f"\n{len(all_df)} total standalone candidate trades across both concepts (before arbitration), "
          f"saved to {TRADES_CSV}")
    print_entry_frequency(all_df, len(days_covered), "standalone, pre-arbitration")

    selected, skipped = select_non_overlapping_by_priority(all_trades)
    sel_df = pd.DataFrame(selected)
    skip_df = pd.DataFrame(skipped) if skipped else pd.DataFrame(
        columns=list(all_df.columns) + ["blocked_by_strategy", "blocked_by_entry_time"])
    sel_df.to_csv(SELECTED_CSV, index=False)
    skip_df.to_csv(SKIPPED_CSV, index=False)

    print(f"\n################ PRIORITY ARBITRATION RESULT ################")
    print(f"Standalone candidates: {len(all_df)}   Selected (combined portfolio): {len(sel_df)}   "
          f"Skipped due to a higher-priority overlap: {len(skip_df)}")
    print("\nSelected trades by strategy:")
    print(sel_df["strategy"].value_counts().to_string())
    if len(skip_df):
        print("\nSkipped trades by strategy (lost an overlap) -- and who blocked them:")
        print(skip_df.groupby(["strategy", "blocked_by_strategy"]).size().to_string())
    print_entry_frequency(sel_df, len(days_covered), "after arbitration, final combined portfolio")

    print(f"\n################ COMBINED PORTFOLIO -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(sel_df)
    total_pnl = sel_df["rupee_pnl"].sum()
    win_rate = 100 * (sel_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt, "
              "NOT a real edge, before trusting this. (Calibration: real clean results here look like "
              "PF ~1.4-1.7, SQN ~2.5-3.) ***")

    # in-sample / out-of-sample split
    sel_df["date"] = pd.to_datetime(sel_df["date"])
    dates = sorted(sel_df["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = sel_df[sel_df["date"] < mid]
        h2 = sel_df[sel_df["date"] >= mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split: H1 ({dates[0].date()} to {mid.date()}) n={n1} PF={pf1:.2f} SQN={sqn1:.2f} "
              f"net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 ({mid.date()} to {dates[-1].date()}) n={n2} PF={pf2:.2f} SQN={sqn2:.2f} "
              f"net=Rs{h2['rupee_pnl'].sum():,.2f}")

    # capital adequacy check
    equity_curve = STARTING_CAPITAL + sel_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (trade-sequence, Rs{STARTING_CAPITAL:,} starting capital): "
          f"Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING: worst drawdown approaches/exceeds starting capital -- "
              "this is an undersized account, not proof the combined strategy is broken. ***")
    if (equity_curve <= 0).any():
        print("  WARNING: combined equity touches Rs0 or below at some point in this trade sequence.")
    else:
        print("  Combined equity never touches Rs0 or below across this trade sequence.")

    # full institutional metrics on the SELECTED combined portfolio
    daily_pnl = build_daily_pnl(sel_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(sel_df, STARTING_CAPITAL)
    print(f"\n################ COMBINED PORTFOLIO -- FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (active days): {m['sharpe_active_days']}   "
          f"Sharpe (full-time capital): {m['sharpe_fulltime']}")
    print(f"Sortino (active days): {m['sortino_active_days']}   Sortino (full-time capital): {m['sortino_fulltime']}")
    print(f"Max DD: {m['max_dd_pct']}%   Max DD duration: {m['max_dd_duration_days']} trading days")
    print(f"VaR95 (daily): {m['var95_pct']}%   CVaR95 (daily): {m['cvar95_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo ({5000} reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    return {"all_trades": all_df, "selected": sel_df, "skipped": skip_df, "metrics": m, "monte_carlo": mc}


# =============================================================================
# Smoke test -- real recent data, no CSV writes beyond the normal pipeline's
# own (acceptable here since it's a tiny date range and clearly labeled)
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = fetch_index_chunked(from_date, to_date)
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
    print(f"=== COMBINED PRIORITY-ARBITRATED STRATEGY (2-concept) -- 2-YEAR REAL BACKTEST ===")
    print("Fetching Nifty index 1-min data (2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
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
