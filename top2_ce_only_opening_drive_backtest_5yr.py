"""
Top 2 CE-only Combined (Velez 8pt + Opening-Drive CE-only) -- 5-YEAR BACKTEST
=================================================================
Same strategy as top2_ce_only_opening_drive_backtest.py, extended from
2 years to 5 years per your request. Nothing about the strategy logic
changes -- only FROM_DATE. Reuses that file's self-checks, fetch,
CE-only patch, and log-capture hooks verbatim (does not touch it).

NOTE: Dhan's historical data availability may not actually extend a
full 5 years back. If fewer trading days come back than requested,
that's a genuine data-availability limit, not a bug here -- the output
tells you exactly how many days were actually returned.

Adds a yearly breakdown on top of the usual first-half/second-half
split, since a 5-year window is long enough to make a year-by-year view
more informative than just two halves.

Run:
    cd "C:\\dhan algo"
    python top2_ce_only_opening_drive_backtest_5yr.py
"""

import datetime as dt
import time
import pandas as pd

import nifty_paper_trade_top3_live as top3
import top2_ce_only_opening_drive_backtest as top2

FROM_DATE = (dt.date.today() - dt.timedelta(days=1825)).strftime("%Y-%m-%d")  # 5 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
MAX_REPEATS_PER_DAY = top2.MAX_REPEATS_PER_DAY
STARTING_CAPITAL = top2.STARTING_CAPITAL
TRADES_CSV = "top2_ce_only_opening_drive_5yr_trades.csv"


def main():
    top2.run_self_checks()
    print("=== Top 2 CE-only Combined -- 5-YEAR BACKTEST ===\n")
    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~5 years)...")
    raw = top2.fetch_index_chunked(FROM_DATE, TO_DATE)
    full_2min = top3.resample_2min_per_day(raw)
    full_velez = top3.compute_velez_indicators(full_2min)
    days = sorted(set(full_2min.index.date))
    print(f"Got {len(full_2min)} 2-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")
    if len(days) < 1000:
        print(f"NOTE: requested 5 years (~1250 trading days) but only {len(days)} trading days were "
              f"returned -- this reflects actual Dhan data availability, not a script limitation.\n")

    top2._captured_trades, top2._pending_entries = [], {}
    top3.log_entry = top2._capture_log_entry
    top3.log_exit = top2._capture_log_exit
    top3.find_opening_drive = top2.find_opening_drive_ce_only

    t_start = time.time()
    for day in days:
        day_df = full_2min[full_2min.index.date == day].copy()
        day_df_velez = full_velez[full_velez.index.date == day].copy()
        if day_df.empty:
            continue
        states = {"velez": top3.new_strategy_state(), "od": top3.new_strategy_state()}
        for name, fn, df_to_use in (("velez", top3.process_velez, day_df_velez),
                                     ("od", top3.process_opening_drive, day_df)):
            st = states[name]
            for _ in range(MAX_REPEATS_PER_DAY):
                before = (st["position"] is not None, st["trades_today"])
                fn(df_to_use, st, verbose=False)
                after = (st["position"] is not None, st["trades_today"])
                if before == after:
                    break

    top3.log_entry, top3.log_exit = top2._orig_log_entry, top2._orig_log_exit
    top3.find_opening_drive = top2._orig_find_opening_drive
    print(f"Scan complete in {time.time() - t_start:.1f}s.\n")

    combined = []
    for t in top2._captured_trades:
        combined.append({
            "strategy": t["strategy"], "side": t["side"], "strike": t["strike"],
            "entry_time": t["entry_time"], "exit_time": t["exit_time"],
            "entry_price": t["entry_price"], "exit_price": t["exit_price"],
            "exit_reason": t["exit_reason"], "pnl_pts": round(t["pnl_pts"], 2),
            "rupee_pnl": round(t["rupee_pnl"], 2), "date": t["date"],
        })

    if not combined:
        print("No trades generated across either strategy.")
        return

    tdf = pd.DataFrame(combined).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv(TRADES_CSV, index=False)

    r = top2.stats_for(tdf)
    print("--- COMBINED (Velez 8pt + Opening-Drive CE-only, one shared Rs25,000 account, 5yr) ---")
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  "
          f"maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  final_equity_Rs={r['final_equity']:,.2f}\n")

    print("--- By strategy (isolated stats) ---")
    for strat, g in tdf.groupby("strategy"):
        rg = top2.stats_for(g)
        print(f"  [{strat}] n={rg['n']}  win%={rg['win_pct']:.1f}  PF={rg['pf']:.2f}  "
              f"net_Rs={rg['net_rs']:,.2f}  SQN={rg['sqn']:.2f}")

    print("\n--- By exit reason (combined) ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(tdf):.1f}%)  win%={100*(g['rupee_pnl']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")

    dates = pd.to_datetime(tdf["date"])
    tdf["year_bucket"] = dates.dt.year
    print("\n--- Yearly breakdown (stability check across the longer window) ---")
    for yr, g in tdf.groupby("year_bucket"):
        rg = top2.stats_for(g)
        print(f"  {yr}: n={rg['n']}  net_Rs={rg['net_rs']:,.2f}  PF={rg['pf']:.2f}  SQN={rg['sqn']:.2f}")

    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = top2.stats_for(fh), top2.stats_for(sh)
    print(f"\n--- Stability check (in-sample/out-of-sample split) ---")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']:.2f} SQN={m1['sqn']:.2f}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']:.2f} SQN={m2['sqn']:.2f}")

    if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny "
              "per this project's standing calibration. ***")

    print(f"\nSaved: {TRADES_CSV} ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
