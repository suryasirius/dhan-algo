"""
Module: dna_filter_out_of_sample_test.py
Description: Out-of-sample validation for the "Market DNA trend filter" found
             while reviewing best_two_of_claude_5_8_26.py's strategies.

BACKGROUND
  On the real 6-month window already cached in this project
  (dhan_historical_data_180days_v5.csv, 2026-02-02 to 2026-07-31), adding a
  filter of "only take a trade when step9_live_engine.py's Market DNA trend
  label (spot vs VWAP) agrees with the trade's own direction" improved the
  combined open_momentum + six_concepts portfolio:
      Baseline : 239 trades, 66.5% win, net Rs 60,097, max DD Rs 4,853, SQN 5.61
      Filtered : 169 trades, 71.6% win, net Rs 66,220, max DD Rs 1,891, SQN 7.75
  A first-half/second-half split of that SAME 6 months held up directionally
  (better win rate, much smaller drawdown, higher SQN both times) but was
  mixed on raw profit (won by a lot in the first half, lost slightly in the
  second half).

  IMPORTANT CAVEAT (from that same review, repeated here so it isn't lost):
  the filter's benefit is concentrated almost entirely in open_momentum. For
  open_momentum there's a real tautology risk -- its own entry signal already
  fires off "which way did price move in the first 30 minutes", and "trend =
  spot above/below VWAP" measures something very similar. The filter may be
  partly re-confirming the same price move the strategy already detected,
  not adding independent information. six_concepts showed no such benefit
  (if anything, mildly the opposite).

WHAT THIS SCRIPT DOES
  Both halves of the 6-month check above are still the SAME 6 months (Feb-Jul
  2026) -- same overall market conditions. This script re-runs the identical
  test on a genuinely separate period the filter was never looked at:
  2024-08-05 to 2026-02-01 (~18 months, chosen to butt right up against the
  already-tested window with zero overlap). If the win-rate/drawdown pattern
  holds here too, that's real out-of-sample evidence, not curve-fitting to
  one window.

  Market DNA is reproduced using ONLY the price-based half of step9's real
  formula (trend from spot vs VWAP, momentum from % change) -- there is no
  stored historical option-chain OI data anywhere in this project, so the
  OI-based half (volume_profile / smart_money) is forced to the same neutral
  50 fallback step9_live_engine.py itself uses when OI is unavailable. This
  is not an invented shortcut; it's the project's own documented fallback
  behavior, reused here for consistency.

DATA-ACCESS NOTE: same as every other real-backtest script in this project --
this needs live Dhan connectivity to fetch ~18 months of 2-min candles. Run
it on the machine where dhan_live_bot.py normally works:

    cd "C:\\dhan algo"
    python dna_filter_out_of_sample_test.py

Standalone -- does not modify step9_live_engine.py, step10_live_paper_best2.py,
best_two_of_claude_5_8_26.py, or either source strategy script.
"""

import pandas as pd
import numpy as np

import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc
import best_two_of_claude_5_8_26 as bt

# Right after the already-tested window ends (2026-02-01), going back ~18
# months. Deliberately fixed dates (not "today - N days") so this always
# tests the same out-of-sample period regardless of when it's rerun.
OOS_FROM_DATE = "2024-08-05"
OOS_TO_DATE = "2026-02-01"


# --------------------------------------------------------------------
# Market DNA (price-only reproduction of step9_live_engine.py's formula)
# --------------------------------------------------------------------

def compute_dna_series(full_df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar trend label + overall_score, computed day-by-day with only
    data available up to that bar (no lookahead) -- exactly mirrors what
    step9_live_engine.py's score_market_dna would have produced live, with
    the OI-dependent half (volume_score, smart_money_score) forced to the
    neutral 50 fallback since no historical OI dataset exists to replay."""
    rows = []
    for day in sorted(set(full_df.index.date)):
        d = full_df.loc[full_df.index.date == day].copy()
        typical = (d["high"] + d["low"] + d["close"]) / 3.0
        cum_vol = d["volume"].cumsum().replace(0, 1)
        vwap = (typical * d["volume"]).cumsum() / cum_vol
        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - d["close"].shift()).abs(),
            (d["low"] - d["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean()
        atr_safe = atr.where(atr != 0, 1.0)
        day_open = d["open"].iloc[0]
        pct_change = (d["close"] - day_open) / day_open * 100.0

        trend_score = 50 + ((d["close"] - vwap) / atr_safe * 15).clip(-40, 40)
        trend = np.where(d["close"] > vwap, "Bullish", np.where(d["close"] < vwap, "Bearish", "Neutral"))
        momentum_score = 50 + (pct_change * 8).clip(-40, 40)
        overall = ((trend_score + momentum_score + 50 + 50) / 4.0).clip(0, 100)

        out = pd.DataFrame({"trend": trend, "overall_score": overall.round(1)}, index=d.index)
        rows.append(out)
    return pd.concat(rows)


def direction_matches_trend(row) -> bool:
    """Filter A: does the DNA trend label (spot vs VWAP) agree with the
    direction the strategy actually traded?"""
    return row["trend"] == "Bullish" if row["direction"] == "LONG" else row["trend"] == "Bearish"


def run_period(full_df: pd.DataFrame, label: str):
    """Runs both strategies + the DNA filter comparison over one dataset,
    reusing best_two_of_claude_5_8_26's exact merge/metrics functions so the
    comparison is apples-to-apples with the original 6-month result."""
    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)

    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        day_groups_sc.setdefault(d, []).append(pos)
    sc_trades = sc.run_module(full_df_sc, day_groups_sc, lambda df, i: sc.rbi_gbi_signal(df, i, relaxed=True))

    dna = compute_dna_series(full_df)
    om_t = om_trades.copy()
    om_t["entry_time"] = pd.to_datetime(om_t["entry_time"])
    om_t = om_t.join(dna, on="entry_time")
    sc_t = sc_trades.copy()
    sc_t["entry_time"] = pd.to_datetime(sc_t["entry_time"])
    sc_t = sc_t.join(dna, on="entry_time")

    combined_base, _ = bt.merge_with_priority(om_t, sc_t)
    m_base = bt.compute_combined_metrics(combined_base, bt.STARTING_CAPITAL)

    om_f = om_t[om_t.apply(direction_matches_trend, axis=1)].copy()
    sc_f = sc_t[sc_t.apply(direction_matches_trend, axis=1)].copy()
    combined_f, dropped_f = bt.merge_with_priority(om_f, sc_f)
    m_f = bt.compute_combined_metrics(combined_f, bt.STARTING_CAPITAL)

    print(f"\n=== {label} ({len(set(full_df.index.date))} trading days) ===")
    print(f"  Baseline (no filter) : n={m_base['n']:3d}  win={m_base['win_pct']}%  "
          f"net=Rs{m_base['net_rs']:,.0f}  maxDD=Rs{m_base['max_dd_rs']:,.0f}  SQN={m_base['sqn']}")
    print(f"  Filter A (DNA trend)  : n={m_f['n']:3d}  win={m_f['win_pct']}%  "
          f"net=Rs{m_f['net_rs']:,.0f}  maxDD=Rs{m_f['max_dd_rs']:,.0f}  SQN={m_f['sqn']}")
    print(f"  Kept: om={len(om_f)}/{len(om_t)}, sc={len(sc_f)}/{len(sc_t)}")

    # Per-strategy trend-agree vs disagree breakdown (the detail that showed
    # the effect was concentrated in open_momentum on the original 6mo test)
    for name, t in [("open_momentum", om_t), ("six_concepts", sc_t)]:
        t = t.copy()
        t["trend_agrees"] = t.apply(direction_matches_trend, axis=1)
        g = t.groupby("trend_agrees")["rupee_pnl"].agg(
            n="count", win_pct=lambda s: round(100 * (s > 0).mean(), 1), net=lambda s: round(s.sum(), 2)
        )
        print(f"  {name} by trend-agreement:\n{g.to_string()}")

    return m_base, m_f, combined_base, combined_f


# --------------------------------------------------------------------
# Self-checks (synthetic data, no network access needed)
# --------------------------------------------------------------------

def run_self_checks():
    print("Running dna_filter_out_of_sample_test self-checks (synthetic data)...")
    om.run_self_checks()
    sc.run_self_checks()
    sc.run_module_self_checks()

    # compute_dna_series: rising-price day should read Bullish once price
    # clears VWAP, with overall_score climbing above the neutral 50 baseline.
    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="2min")
    synthetic = pd.DataFrame({
        "open":  [100, 100.5, 101, 102, 103, 104],
        "high":  [100.5, 101.2, 101.8, 102.5, 103.5, 104.5],
        "low":   [99.8, 100.2, 100.8, 101.5, 102.5, 103.5],
        "close": [100.5, 101, 101.8, 102.4, 103.4, 104.2],
        "volume": [1000] * 6,
    }, index=idx)
    dna = compute_dna_series(synthetic)
    assert dna["trend"].iloc[-1] == "Bullish", "rising day should end Bullish (close above cumulative VWAP)"
    assert dna["overall_score"].iloc[-1] > 50, "sustained rise should push overall_score above neutral 50"
    assert dna["overall_score"].between(0, 100).all(), "overall_score must stay within its 0-100 bound"
    print("  compute_dna_series check ok (rising synthetic day reads Bullish, score > 50)")

    # direction_matches_trend: sanity on both directions
    assert direction_matches_trend({"direction": "LONG", "trend": "Bullish"}) is True
    assert direction_matches_trend({"direction": "LONG", "trend": "Bearish"}) is False
    assert direction_matches_trend({"direction": "SHORT", "trend": "Bearish"}) is True
    assert direction_matches_trend({"direction": "SHORT", "trend": "Bullish"}) is False
    print("  direction_matches_trend check ok")

    print("All dna_filter_out_of_sample_test self-checks passed.\n")


# --------------------------------------------------------------------
# Real out-of-sample backtest (needs live Dhan connectivity)
# --------------------------------------------------------------------

def main():
    run_self_checks()

    print("=" * 78)
    print("OUT-OF-SAMPLE CHECK -- Market DNA trend filter (Filter A)")
    print(f"Period NOT used in the original 6-month test: {OOS_FROM_DATE} to {OOS_TO_DATE}")
    print("Requires live Dhan connectivity. If this errors with a connection/proxy")
    print("issue, run this file on the machine where dhan_live_bot.py normally works.")
    print("=" * 78)

    print(f"Fetching Nifty index data {OOS_FROM_DATE} to {OOS_TO_DATE} ...")
    raw = om.fetch_index_chunked(OOS_FROM_DATE, OOS_TO_DATE)
    full_df = om.resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.")

    m_base, m_f, combined_base, combined_f = run_period(full_df, "OUT-OF-SAMPLE PERIOD (2024-08-05 to 2026-02-01)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    win_improved = m_f["win_pct"] > m_base["win_pct"]
    dd_improved = m_f["max_dd_rs"] < m_base["max_dd_rs"]
    sqn_improved = m_f["sqn"] > m_base["sqn"]
    net_improved = m_f["net_rs"] > m_base["net_rs"]
    print(f"  Win rate improved with filter : {win_improved}  ({m_base['win_pct']}% -> {m_f['win_pct']}%)")
    print(f"  Max drawdown improved         : {dd_improved}  (Rs{m_base['max_dd_rs']:,.0f} -> Rs{m_f['max_dd_rs']:,.0f})")
    print(f"  SQN improved                  : {sqn_improved}  ({m_base['sqn']} -> {m_f['sqn']})")
    print(f"  Net profit improved           : {net_improved}  (Rs{m_base['net_rs']:,.0f} -> Rs{m_f['net_rs']:,.0f})")
    if win_improved and dd_improved and sqn_improved:
        print("\n  Pattern HELD on genuinely unseen data (win rate + drawdown + SQN all "
              "improved, same as the original 6-month test). This is real evidence the "
              "filter is worth adding to step10_live_paper_best2.py, regardless of what "
              "happened to raw net profit this time.")
    else:
        print("\n  Pattern did NOT fully hold on unseen data -- at least one of win rate / "
              "drawdown / SQN did not improve here the way it did on the original 6-month "
              "window. Treat the original result as likely specific to that window, not a "
              "general edge. Do not wire this into step10_live_paper_best2.py based on the "
              "6-month result alone.")

    combined_base.to_csv("dna_filter_oos_baseline_trades.csv", index=False)
    combined_f.to_csv("dna_filter_oos_filtered_trades.csv", index=False)
    print("\nSaved: dna_filter_oos_baseline_trades.csv / dna_filter_oos_filtered_trades.csv")


if __name__ == "__main__":
    main()
