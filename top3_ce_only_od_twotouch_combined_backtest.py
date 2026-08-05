"""
TOP-3 COMBINED -- Velez 8pt (PE) + Opening-Drive CE-only + Two-Touch
Resistance Multi-Entry Optimized -- 2-YEAR BACKTEST
=================================================================
Adds the newly-optimized Two-Touch variant (multi-entry re-arm, capped to
entries #1-2/day, Wednesday + 13:15-13:45 excluded -- see
nifty_two_touch_resistance_multi_entry_optimized_backtest.py) to your
confirmed top-2 combo (top2_ce_only_opening_drive_backtest.py: Velez 8pt
+ Opening-Drive CE-only). Same independent-state, overlap-allowed, one-
shared-account convention as every combined test in this project.

Does NOT touch any of the three underlying files (all locked/already
validated). Pure reuse via import:
  - top2_ce_only_opening_drive_backtest (Velez + OD-CE-only day driver,
    the find_opening_drive_ce_only patch, log-capture hooks, stats_for)
  - nifty_two_touch_resistance_multi_entry_backtest (day-runner)
  - nifty_two_touch_resistance_multi_entry_optimized_backtest (apply_filters)

Two-Touch runs on the raw 1-min index data (its own opening-range/pivot
logic needs 1-min bars); Velez/OD run on 2-min bars, exactly as before.
Both are computed from the SAME single 1-min data pull.

Rs25,000 shared starting capital, 2-year window.

Run:
    cd "C:\\dhan algo"
    python top3_ce_only_od_twotouch_combined_backtest.py
"""

import datetime as dt
import time
import pandas as pd

import nifty_paper_trade_top3_live as top3
import top2_ce_only_opening_drive_backtest as top2
import nifty_two_touch_resistance_multi_entry_backtest as multi
import nifty_two_touch_resistance_multi_entry_optimized_backtest as opt

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
STARTING_CAPITAL = 25_000
MAX_REPEATS_PER_DAY = 12

TRADES_CSV = "top3_ce_only_od_twotouch_combined_trades.csv"


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top2.run_self_checks()
    print("  top2.run_self_checks() re-verified ok (Velez + Opening-Drive CE-only)")
    opt.run_self_checks()
    print("  opt.run_self_checks() re-verified ok (Two-Touch multi-entry, optimized filter)")
    print("All self-checks passed.\n")


def stats_for(g):
    return top2.stats_for(g)


def main():
    run_self_checks()
    print("=== TOP-3 COMBINED: Velez 8pt + Opening-Drive CE-only + Two-Touch Optimized -- 2yr backtest ===\n")
    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    raw = top2.fetch_index_chunked(FROM_DATE, TO_DATE)
    full_2min = top3.resample_2min_per_day(raw)
    full_velez = top3.compute_velez_indicators(full_2min)
    days = sorted(set(full_2min.index.date))
    print(f"Got {len(full_2min)} 2-min bars across {len(days)} day(s): {days[0]} to {days[-1]}\n")

    top2._captured_trades, top2._pending_entries = [], {}
    top3.log_entry = top2._capture_log_entry
    top3.log_exit = top2._capture_log_exit
    top3.find_opening_drive = top2.find_opening_drive_ce_only

    tt_raw_trades = []
    t_start = time.time()
    for day in days:
        day_df_1min = raw[raw.index.date == day].copy()
        day_df_2min = full_2min[full_2min.index.date == day].copy()
        day_df_velez = full_velez[full_velez.index.date == day].copy()
        if day_df_2min.empty:
            continue

        states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}
        for name, fn, df_to_use in (("velez", top3.process_velez, day_df_velez),
                                     ("od", top3.process_opening_drive, day_df_2min)):
            st = states[name]
            for _ in range(MAX_REPEATS_PER_DAY):
                before = (st["position"] is not None, st["trades_today"])
                fn(df_to_use, st, verbose=False)
                after = (st["position"] is not None, st["trades_today"])
                if before == after:
                    break

        tt_raw_trades.extend(multi.run_standalone_for_day_multi(day_df_1min))

    top3.log_entry, top3.log_exit = top2._orig_log_entry, top2._orig_log_exit
    top3.find_opening_drive = top2._orig_find_opening_drive
    print(f"Scan complete in {time.time() - t_start:.1f}s.\n")

    velez_od = [{
        "strategy": t["strategy"], "side": t["side"], "strike": t["strike"],
        "entry_time": t["entry_time"], "exit_time": t["exit_time"],
        "entry_price": t["entry_price"], "exit_price": t["exit_price"],
        "exit_reason": t["exit_reason"], "pnl_pts": round(t["pnl_pts"], 2),
        "rupee_pnl": round(t["rupee_pnl"], 2), "date": t["date"],
    } for t in top2._captured_trades]

    tt_raw_df = pd.DataFrame(tt_raw_trades)
    tt_filtered = opt.apply_filters(tt_raw_df) if not tt_raw_df.empty else tt_raw_df
    if not tt_filtered.empty:
        tt_filtered = tt_filtered.rename(columns={"reason": "exit_reason"})
        tt_filtered["strategy"] = "TwoTouchOptimized"

    velez_od_df = pd.DataFrame(velez_od)
    combined = pd.concat([velez_od_df, tt_filtered], ignore_index=True, sort=False)
    combined = combined.sort_values("entry_time").reset_index(drop=True)

    if combined.empty:
        print("No trades generated across any strategy.")
        return

    combined.to_csv(TRADES_CSV, index=False)

    r = stats_for(combined)
    print("--- COMBINED (Velez 8pt + Opening-Drive CE-only + Two-Touch Optimized, one shared Rs25,000 account) ---")
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  "
          f"maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  final_equity_Rs={r['final_equity']:,.2f}\n")

    print("--- By strategy (isolated stats) ---")
    for strat, g in combined.groupby("strategy"):
        rg = stats_for(g)
        print(f"  [{strat}] n={rg['n']}  win%={rg['win_pct']:.1f}  PF={rg['pf']:.2f}  "
              f"net_Rs={rg['net_rs']:,.2f}  SQN={rg['sqn']:.2f}")

    print("\n--- By exit reason (combined) ---")
    for reason, g in combined.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(combined):.1f}%)  win%={100*(g['rupee_pnl']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")

    events = []
    for _, row in combined.iterrows():
        events.append((row["entry_time"], 1))
        events.append((row["exit_time"], -1))
    events.sort()
    concurrent, max_concurrent = 0, 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    print(f"\n--- Overlap diagnostic ---\nMax simultaneous open positions across all 3 strategies: {max_concurrent}")

    dates = pd.to_datetime(combined["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = combined[dates <= mid], combined[dates > mid]
    m1, m2 = stats_for(fh), stats_for(sh)
    print(f"\n--- Stability check (in-sample/out-of-sample split) ---")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']:.2f} SQN={m1['sqn']:.2f}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']:.2f} SQN={m2['sqn']:.2f}")

    if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny "
              "per this project's standing calibration. ***")

    print(f"\nSaved: {TRADES_CSV} ({len(combined)} rows)")
    print("\n*** REMINDER: Two-Touch Optimized is several cuts deep on its own sample already -- treat its "
          "contribution here as a hypothesis, not a validated edge, same as its own standalone report says. ***")


if __name__ == "__main__":
    main()
