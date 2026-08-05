"""
Best Two of Claude -- 5/8/26 (5 August 2026)
=================================================================
Combines the two strategies picked after the full ranking/review pass on
this project's 306 backtested trade logs:

  1. Opening-Drive Momentum, stop=20pts / trail=1pts
     (source: nifty_open_momentum_stoptrail_sweep.py, combo "sl20_trail1")
     Rank #1 by SQN (9.82) across the whole project.
     CAUTION (from that script's own docstring): this combo sits at the
     EDGE of the tested grid (widest stop tested, tightest trail tested).
     The script explicitly flags this as a possible curve-fitting red
     flag, not a confirmed interior sweet spot. Treat accordingly.

  2. Six Concepts -- RBI/GBI (relaxed rule-3 variant)
     (source: nifty_six_concepts_screening.py, module "RBI_GBI_relaxed")
     Rank #6 by SQN (6.79).
     CAUTION (from that script's own docstring, written by the run that
     produced this exact trade log): "three modules (RBI/GBI relaxed,
     Power Bar Start, Market Law 4) came back showing SQN > 6.6 and
     near-identical net P&L/drawdown -- both are red flags [...] smells
     like an entry-frequency artifact, not a shared edge." That warning
     applies directly to this strategy. It was picked anyway because it
     ranked well and is not from a huge sweep (only 3 six_concepts
     variants existed, vs. 11 for open_momentum), but this is a real,
     documented reason for caution, not a clean pick.

WHAT THIS SCRIPT DOES
  - Reuses the exact, unmodified signal/simulation functions from both
    source scripts (imported as modules below) -- no logic is
    re-transcribed, so there is no risk of a copy-paste bug changing
    either strategy's behaviour.
  - Runs both over the same 2-year Nifty history.
  - Merges the two trade logs onto ONE shared starting capital of
    Rs 25,000 (not Rs 25,000 each / Rs 50,000 total).
  - Applies the capital-sharing rule agreed with the user: when both
    strategies want an open position at the same time on the same day,
    open_momentum gets priority and the conflicting six_concepts trade
    for that day is dropped (six_concepts only ever takes one trade/day,
    same for open_momentum, so this is at most a 1-vs-1 same-day check,
    not a complex multi-trade scheduler).
  - Recomputes net profit / drawdown / SQN / win rate on the COMBINED,
    priority-resolved trade list against the single Rs 25,000 base.

DATA-ACCESS NOTE: same as every other real-backtest script in this
project -- this sandbox cannot reach api.dhan.co. Self-checks below run
entirely on synthetic data and prove the merge/priority logic is
correct without network access. The REAL 2-year result can only be
produced by running this file on the machine where dhan_live_bot.py /
step1_config_and_connector.py normally work:

    cd "C:\\dhan algo"
    python best_two_of_claude_5_8_26.py

Standalone -- does not modify either source script, step1..step9, or
step8_dashboard.py.
"""

import datetime as dt
import pandas as pd
import numpy as np

import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc

STARTING_CAPITAL = 25_000
OM_STOP_PTS = 20
OM_TRAIL_PTS = 1
PRIORITY_STRATEGY = "open_momentum"   # wins on same-day time-overlap conflicts


# --------------------------------------------------------------------
# Merge + capital-sharing priority rule
# --------------------------------------------------------------------

def merge_with_priority(om_trades: pd.DataFrame, sc_trades: pd.DataFrame,
                         priority: str = PRIORITY_STRATEGY):
    """Combines the two trade logs onto one shared capital pool. Each
    source strategy only ever takes at most ONE trade per calendar day
    (verified: both run_for_combo and run_module `break`/return after the
    first signal found each day), so a conflict can only ever be a single
    open_momentum trade vs. a single six_concepts trade on the SAME date.
    If their [entry_time, exit_time] windows overlap, the priority
    strategy's trade is kept and the other day's trade is dropped
    (there wasn't enough shared capital/margin to hold both at once).
    If they don't overlap in time (one already exited before the other
    entered), both are kept -- sequential same-day trades don't need
    extra capital beyond the shared Rs 25,000 base since lot size is
    fixed by NSE rules, not equity-dependent in this project's model."""
    om_t = om_trades.copy()
    om_t["strategy"] = "open_momentum"
    sc_t = sc_trades.copy()
    sc_t["strategy"] = "six_concepts"

    for df in (om_t, sc_t):
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["date"] = pd.to_datetime(df["date"]).dt.date

    om_by_date = {row["date"]: row for _, row in om_t.iterrows()}
    sc_by_date = {row["date"]: row for _, row in sc_t.iterrows()}

    kept_rows = []
    dropped_log = []

    for d, row in om_by_date.items():
        kept_rows.append(row)

    for d, srow in sc_by_date.items():
        orow = om_by_date.get(d)
        if orow is None:
            kept_rows.append(srow)
            continue
        overlap = (srow["entry_time"] < orow["exit_time"]) and (orow["entry_time"] < srow["exit_time"])
        if overlap:
            winner, loser = (orow, srow) if priority == "open_momentum" else (srow, orow)
            dropped_log.append({
                "date": d,
                "kept_strategy": winner["strategy"], "kept_entry": winner["entry_time"],
                "dropped_strategy": loser["strategy"], "dropped_entry": loser["entry_time"],
            })
            # orow already added above; nothing further to add for this date
        else:
            kept_rows.append(srow)

    combined = pd.DataFrame(kept_rows).sort_values("entry_time").reset_index(drop=True)
    dropped_df = pd.DataFrame(dropped_log)
    return combined, dropped_df


def compute_combined_metrics(combined: pd.DataFrame, starting_capital: float = STARTING_CAPITAL):
    if combined.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0}
    pnl = combined["rupee_pnl"].to_numpy()
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_pct = 100.0 * len(wins) / n
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_rs = pnl.sum()
    equity = starting_capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd_rs = dd.max()
    sqn = (pnl.mean() / pnl.std() * np.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2)}


# --------------------------------------------------------------------
# Self-checks (synthetic data, no network access needed)
# --------------------------------------------------------------------

def run_self_checks():
    print("Running best_two_of_claude_5_8_26 self-checks (synthetic data)...")

    # Reuse both source scripts' own self-checks first -- if either
    # strategy's underlying logic is broken, this whole file is moot.
    om.run_self_checks()
    sc.run_self_checks()
    sc.run_module_self_checks()

    # --- merge_with_priority: overlapping same-day trades -> priority wins ---
    om_trades = pd.DataFrame([{
        "date": "2025-01-06", "direction": "LONG", "side": "CE",
        "entry_time": "2025-01-06 09:20:00", "exit_time": "2025-01-06 09:40:00",
        "entry_price": 100.0, "exit_price": 105.0, "exit_reason": "TRAIL",
        "pnl_pts": 5.0, "rupee_pnl": 335.0, "capital_after": 25335.0,
    }])
    sc_trades_overlap = pd.DataFrame([{
        "date": "2025-01-06", "direction": "SHORT", "side": "PE", "strike": 24000,
        "entry_time": "2025-01-06 09:30:00", "exit_time": "2025-01-06 09:50:00",
        "entry_price": 50.0, "exit_price": 45.0, "exit_reason": "STOP",
        "pnl_pts": -5.0, "rupee_pnl": -415.0, "capital_after": 24585.0,
    }])
    combined, dropped = merge_with_priority(om_trades, sc_trades_overlap)
    assert len(combined) == 1 and combined.iloc[0]["strategy"] == "open_momentum", (
        "overlapping same-day trade: open_momentum must win, six_concepts must be dropped"
    )
    assert len(dropped) == 1 and dropped.iloc[0]["dropped_strategy"] == "six_concepts"
    print("  merge_with_priority overlap-resolution check ok (open_momentum wins)")

    # --- merge_with_priority: NON-overlapping same-day trades -> both kept ---
    sc_trades_no_overlap = sc_trades_overlap.copy()
    sc_trades_no_overlap["entry_time"] = "2025-01-06 11:00:00"
    sc_trades_no_overlap["exit_time"] = "2025-01-06 11:20:00"
    combined2, dropped2 = merge_with_priority(om_trades, sc_trades_no_overlap)
    assert len(combined2) == 2 and len(dropped2) == 0, (
        "non-overlapping same-day trades should both be kept"
    )
    print("  merge_with_priority non-overlap check ok (both kept)")

    # --- merge_with_priority: different days -> both kept regardless of time ---
    sc_trades_diff_day = sc_trades_overlap.copy()
    sc_trades_diff_day["date"] = "2025-01-07"
    sc_trades_diff_day["entry_time"] = "2025-01-07 09:30:00"
    sc_trades_diff_day["exit_time"] = "2025-01-07 09:50:00"
    combined3, dropped3 = merge_with_priority(om_trades, sc_trades_diff_day)
    assert len(combined3) == 2 and len(dropped3) == 0
    print("  merge_with_priority different-day check ok (both kept)")

    # --- compute_combined_metrics: basic sanity on a known small sample ---
    sample = pd.DataFrame({"rupee_pnl": [500.0, -200.0, 300.0, -100.0]})
    m = compute_combined_metrics(sample, starting_capital=25000)
    assert m["n"] == 4 and abs(m["net_rs"] - 500.0) < 1e-6
    assert m["win_pct"] == 50.0
    print("  compute_combined_metrics sanity check ok")

    print("All best_two_of_claude_5_8_26 self-checks passed.\n")


# --------------------------------------------------------------------
# Real backtest (needs live Dhan connectivity -- run this file locally)
# --------------------------------------------------------------------

def main():
    run_self_checks()

    print("=" * 78)
    print(f"REAL COMBINED BACKTEST -- open_momentum(sl{OM_STOP_PTS}_trail{OM_TRAIL_PTS}) "
          f"+ six_concepts(RBI_GBI_relaxed), shared Rs {STARTING_CAPITAL:,} capital")
    print("Requires live Dhan connectivity. If this errors with a connection/proxy")
    print("issue, run this file on the machine where dhan_live_bot.py normally works.")
    print("=" * 78)

    print(f"Fetching Nifty index data {om.FROM_DATE} to {om.TO_DATE} ...")
    raw = om.fetch_index_chunked(om.FROM_DATE, om.TO_DATE)
    full_df = om.resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    # --- open_momentum: our exact picked combo (sl20_trail1) ---
    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, OM_STOP_PTS, OM_TRAIL_PTS)
    om_metrics = om.compute_metrics(om_trades)
    print(f"open_momentum(sl{OM_STOP_PTS}_trail{OM_TRAIL_PTS}) alone: {om_metrics}")

    # --- six_concepts RBI/GBI relaxed: needs the indicator columns first ---
    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        day_groups_sc.setdefault(d, []).append(pos)

    highs_list = full_df_sc["high"].tolist()
    lows_list = full_df_sc["low"].tolist()

    def zone_tag_fn(df, abs_p, direction):
        zone = sc.fifty_pct_zone_check(highs_list, lows_list, abs_p)
        return {"retracement_zone": zone}

    sc_trades = sc.run_module(
        full_df_sc, day_groups_sc,
        lambda df, i: sc.rbi_gbi_signal(df, i, relaxed=True),
        tag_fn=zone_tag_fn,
    )
    sc_metrics = sc.compute_metrics(sc_trades)
    print(f"six_concepts(RBI_GBI_relaxed) alone: {sc_metrics}\n")

    # --- merge onto one shared Rs 25,000 pool, open_momentum priority on overlap ---
    combined, dropped = merge_with_priority(om_trades, sc_trades, priority=PRIORITY_STRATEGY)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    combined["equity"] = STARTING_CAPITAL + combined["rupee_pnl"].cumsum()

    m = compute_combined_metrics(combined, STARTING_CAPITAL)
    print("=" * 78)
    print(f"COMBINED RESULT (shared Rs {STARTING_CAPITAL:,} capital, "
          f"{PRIORITY_STRATEGY} priority on same-day overlaps)")
    print("=" * 78)
    print(f"  Trades kept: {m['n']}  (open_momentum={len(om_trades)}, "
          f"six_concepts kept={m['n'] - len(om_trades)}, six_concepts dropped={len(dropped)})")
    print(f"  Win rate: {m['win_pct']}%")
    print(f"  Net profit: Rs {m['net_rs']:,}")
    print(f"  Max drawdown: Rs {m['max_dd_rs']:,} ({100*m['max_dd_rs']/STARTING_CAPITAL:.1f}% of capital)")
    print(f"  SQN: {m['sqn']}")
    print(f"  Final equity: Rs {STARTING_CAPITAL + m['net_rs']:,.0f}")

    if not dropped.empty:
        print(f"\n  {len(dropped)} same-day overlap(s) resolved in {PRIORITY_STRATEGY}'s favour:")
        for _, r in dropped.iterrows():
            print(f"    {r['date']}: kept {r['kept_strategy']} @ {r['kept_entry']}, "
                  f"dropped {r['dropped_strategy']} @ {r['dropped_entry']}")

    combined.to_csv("best_two_of_claude_5_8_26_trades.csv", index=False)
    dropped.to_csv("best_two_of_claude_5_8_26_dropped_overlaps.csv", index=False)
    print("\nSaved: best_two_of_claude_5_8_26_trades.csv / _dropped_overlaps.csv")


if __name__ == "__main__":
    main()
