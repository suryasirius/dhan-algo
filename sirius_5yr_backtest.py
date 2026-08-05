"""
Module: sirius_5yr_backtest.py
Description: 5-year backtest of "Sirius" -- your alias for the combined
             open_momentum + six_concepts strategy (originally built in
             best_two_of_claude_5_8_26.py, live-traded via
             step10_live_paper_best2.py).

WHY A SEPARATE SCRIPT: best_two_of_claude_5_8_26.py fetches
om.FROM_DATE/om.TO_DATE, which is a rolling "today minus 2 years" window.
This script instead requests the maximum real history Dhan's intraday API
actually provides -- 5 years (confirmed from Dhan's own documentation: 1/5/
15/25/60-min candles are available for up to 5 years, fetched in <=90-day
chunks, which om.fetch_index_chunked already handles). Everything else is
reused verbatim from the existing, already-tested scripts -- no logic is
re-transcribed, so there's no risk of a copy-paste bug changing either
strategy's behaviour or the combine/priority rule.

WHAT IT RUNS (identical logic to best_two_of_claude_5_8_26.py):
  1. Opening-Drive Momentum, stop=20pts / trail=1pts
  2. Six Concepts -- RBI/GBI (relaxed rule-3 variant)
  Combined onto ONE shared Rs 25,000 starting capital, open_momentum wins
  any same-day time-overlap conflict.

HONEST CAVEAT WORTH KEEPING IN MIND: the earlier 2-year backtest already
carries real caveats (open_momentum's sl20/trail1 sits at the edge of its
tested grid; six_concepts RBI/GBI relaxed was flagged by its own source
script as a possible entry-frequency artifact). Five years of data doesn't
resolve either caveat -- it just shows whether the same combo holds up over
a longer, more varied stretch of market conditions (different volatility
regimes, different lot-size eras, etc.). Read the result as more data, not
as a fix for those open questions.

DATA-ACCESS NOTE: needs live Dhan connectivity (5 years of chunked intraday
fetches -- this will take a while, expect several minutes). Run on the
machine where dhan_live_bot.py normally works:

    cd "C:\\dhan algo"
    python sirius_5yr_backtest.py

Standalone -- does not modify best_two_of_claude_5_8_26.py, step10, or
either source strategy script.
"""

import datetime as dt
import pandas as pd

import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc
import best_two_of_claude_5_8_26 as bt

# Dhan's own documented cap for intraday candle history is 5 years. Using
# 1825 days (5 x 365, no leap-year padding) to stay safely inside that
# window rather than requesting the exact boundary and risking a rejection.
FROM_DATE_5YR = (dt.date.today() - dt.timedelta(days=1825)).strftime("%Y-%m-%d")
TO_DATE_5YR = dt.date.today().strftime("%Y-%m-%d")


def main():
    print("Running Sirius self-checks (synthetic data, reused from the source scripts)...")
    bt.run_self_checks()

    print("=" * 78)
    print(f"SIRIUS 5-YEAR BACKTEST -- open_momentum(sl{bt.OM_STOP_PTS}_trail{bt.OM_TRAIL_PTS}) "
          f"+ six_concepts(RBI_GBI_relaxed), shared Rs {bt.STARTING_CAPITAL:,} capital")
    print(f"Requesting {FROM_DATE_5YR} to {TO_DATE_5YR} (~5 years, Dhan's documented intraday history limit).")
    print("Requires live Dhan connectivity. This fetches ~5x the data of the 2-year backtest,")
    print("expect this to take noticeably longer (several minutes).")
    print("=" * 78)

    raw = om.fetch_index_chunked(FROM_DATE_5YR, TO_DATE_5YR)
    full_df = om.resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)
    om_metrics = om.compute_metrics(om_trades)
    print(f"open_momentum(sl{bt.OM_STOP_PTS}_trail{bt.OM_TRAIL_PTS}) alone: {om_metrics}")

    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        day_groups_sc.setdefault(d, []).append(pos)
    sc_trades = sc.run_module(full_df_sc, day_groups_sc, lambda d, i: sc.rbi_gbi_signal(d, i, relaxed=True))
    sc_metrics = sc.compute_metrics(sc_trades)
    print(f"six_concepts(RBI_GBI_relaxed) alone: {sc_metrics}\n")

    combined, dropped = bt.merge_with_priority(om_trades, sc_trades, priority=bt.PRIORITY_STRATEGY)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    combined["equity"] = bt.STARTING_CAPITAL + combined["rupee_pnl"].cumsum()

    m = bt.compute_combined_metrics(combined, bt.STARTING_CAPITAL)
    print("=" * 78)
    print(f"SIRIUS COMBINED RESULT -- 5 years, shared Rs {bt.STARTING_CAPITAL:,} capital, "
          f"{bt.PRIORITY_STRATEGY} priority on same-day overlaps")
    print("=" * 78)
    print(f"  Trades kept: {m['n']}  (open_momentum={len(om_trades)}, "
          f"six_concepts kept={m['n'] - len(om_trades)}, six_concepts dropped={len(dropped)})")
    print(f"  Win rate: {m['win_pct']}%")
    print(f"  Net profit: Rs {m['net_rs']:,}")
    print(f"  Max drawdown: Rs {m['max_dd_rs']:,} ({100*m['max_dd_rs']/bt.STARTING_CAPITAL:.1f}% of capital)")
    print(f"  SQN: {m['sqn']}")
    print(f"  Final equity: Rs {bt.STARTING_CAPITAL + m['net_rs']:,.0f}")

    # --- year-by-year breakdown, so a strong overall number can't hide one bad year ---
    combined["year"] = pd.to_datetime(combined["entry_time"]).dt.year
    print("\n  Year-by-year breakdown:")
    for yr, g in combined.groupby("year"):
        yr_metrics = bt.compute_combined_metrics(g, bt.STARTING_CAPITAL)
        print(f"    {yr}: n={yr_metrics['n']:3d}  win={yr_metrics['win_pct']}%  "
              f"net=Rs{yr_metrics['net_rs']:,.0f}  maxDD=Rs{yr_metrics['max_dd_rs']:,.0f}  SQN={yr_metrics['sqn']}")

    if not dropped.empty:
        print(f"\n  {len(dropped)} same-day overlap(s) resolved in {bt.PRIORITY_STRATEGY}'s favour.")

    combined.to_csv("sirius_5yr_trades.csv", index=False)
    dropped.to_csv("sirius_5yr_dropped_overlaps.csv", index=False)
    print("\nSaved: sirius_5yr_trades.csv / sirius_5yr_dropped_overlaps.csv")


if __name__ == "__main__":
    main()
