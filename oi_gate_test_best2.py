"""
Module: oi_gate_test_best2.py
Description: Tests whether gating open_momentum + six_concepts (the two
             strategies picked in best_two_of_claude_5_8_26.py) on real NIFTY
             futures Open Interest improves the combined portfolio -- the
             same question we just answered for Concepts 2&4
             (oi_gated_concepts_backtest.py), now asked of OUR strategies.

WHY THIS IS A SEPARATE SCRIPT: Concepts 2&4's OI-gated test showed gating
made results WORSE in 4 of 6 exit-management variants (see chat). That's a
different strategy with a different entry logic, so it doesn't tell us
anything about open_momentum/six_concepts specifically -- entry-signal
quality interacts with an OI gate differently depending on what the base
signal already captures. This has to be tested on our actual two strategies,
not assumed from the Concepts 2&4 result.

METHOD (reuses real, already-tested code, no re-transcription):
  - OI regime classification is REUSED VERBATIM from oi_gated_concepts_backtest.py
    (classify_oi_regime: LONG_BUILDUP / SHORT_BUILDUP / SHORT_COVERING /
    LONG_UNWINDING / NEUTRAL, from price-change vs OI-change over a rolling
    15-minute lookback) -- same futures-OI fetch, same classification logic,
    imported as a module so there is zero risk of a copy-paste bug changing
    either script's behaviour.
  - open_momentum / six_concepts trade generation is REUSED VERBATIM from
    their own source scripts, exactly like best_two_of_claude_5_8_26.py does.
  - Gate rule: a trade is only kept if the OI regime at entry time confirms
    the SAME direction -- LONG_BUILDUP for a LONG trade, SHORT_BUILDUP for a
    SHORT trade. (Identical gate definition used for Concepts 2&4, so the
    comparison method itself is apples-to-apples across both tests.)

HARD DATA LIMITATION (same one already found and confirmed via
oi_gated_concepts_backtest.py -- not new): NIFTY futures contracts roll
monthly and expired ones drop out of Dhan's instrument list entirely, so
real historical futures OI is only available back to whenever the CURRENT
contract started trading -- roughly the last ~90-120 calendar days, not the
full 2-year window best_two_of_claude_5_8_26.py itself was tested on. This
script uses whatever real window Dhan actually returns (printed explicitly)
for BOTH the gated and ungated side, so the comparison is fair even though
it's a shorter window than the original 2-year backtest.

DATA-ACCESS NOTE: needs live Dhan connectivity (futures OI + index price).
Run on the machine where dhan_live_bot.py normally works:

    cd "C:\\dhan algo"
    python oi_gate_test_best2.py

Standalone -- does not modify best_two_of_claude_5_8_26.py,
step10_live_paper_best2.py, or either source strategy script.
"""

import datetime as dt
import pandas as pd

import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc
import best_two_of_claude_5_8_26 as bt
import oi_gated_concepts_backtest as og   # reused verbatim: OI fetch + regime classification

WARMUP_DAYS = 10   # extra calendar days fetched before the real OI window, for six_concepts' SMA200


# --------------------------------------------------------------------
# Self-checks (synthetic data, no network access needed)
# --------------------------------------------------------------------

def run_self_checks():
    print("Running oi_gate_test_best2 self-checks (synthetic data)...")
    om.run_self_checks()
    sc.run_self_checks()
    sc.run_module_self_checks()
    og.run_self_checks()   # covers classify_oi_regime's 5 cases

    # Gate-rule sanity: LONG only survives on LONG_BUILDUP, SHORT only on SHORT_BUILDUP
    assert og.classify_oi_regime(10, 5000) == "LONG_BUILDUP"
    assert og.classify_oi_regime(-10, 5000) == "SHORT_BUILDUP"

    def trade_passes_gate(direction, regime):
        return (direction == "LONG" and regime == "LONG_BUILDUP") or \
               (direction == "SHORT" and regime == "SHORT_BUILDUP")

    assert trade_passes_gate("LONG", "LONG_BUILDUP") is True
    assert trade_passes_gate("LONG", "SHORT_BUILDUP") is False
    assert trade_passes_gate("SHORT", "SHORT_BUILDUP") is True
    assert trade_passes_gate("SHORT", "NEUTRAL") is False
    print("  trade_passes_gate check ok")

    # attach_oi_regime: backward-asof join must pick the OI reading in effect
    # AT OR BEFORE each trade's entry, never a later (lookahead) one.
    oi_series = pd.Series(
        ["NEUTRAL", "LONG_BUILDUP", "LONG_BUILDUP", "SHORT_BUILDUP"],
        index=pd.to_datetime(["2025-01-06 09:15", "2025-01-06 09:20", "2025-01-06 09:25", "2025-01-06 09:35"]),
    )
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(["2025-01-06 09:18", "2025-01-06 09:30", "2025-01-06 09:10"]),
        "direction": ["LONG", "SHORT", "LONG"],
        "rupee_pnl": [100.0, -50.0, 20.0],
    })
    out = attach_oi_regime(trades, oi_series)
    got = dict(zip(out["entry_time"].dt.strftime("%H:%M"), out["oi_regime"]))
    assert got["09:18"] == "NEUTRAL", "09:18 entry should see the 09:15 reading (last one at/before it)"
    assert got["09:30"] == "LONG_BUILDUP", "09:30 entry should see the 09:25 reading, not the later 09:35 one"
    assert pd.isna(got["09:10"]), "09:10 entry is before any OI reading exists -- must be NaN, not a lookahead guess"
    print("  attach_oi_regime check ok (no-lookahead backward join verified)")

    print("All oi_gate_test_best2 self-checks passed.\n")


def trade_passes_gate(direction, regime):
    return (direction == "LONG" and regime == "LONG_BUILDUP") or \
           (direction == "SHORT" and regime == "SHORT_BUILDUP")


def attach_oi_regime(trades_df: pd.DataFrame, oi_regime_series: pd.Series) -> pd.DataFrame:
    """Joins the OI regime label onto each trade at its entry_time, using the
    nearest OI reading AT OR BEFORE that entry (merge_asof, backward) -- so
    this can never look ahead at OI data that wasn't known yet when the trade
    was actually entered. oi_regime_series must be a Series indexed by time
    (same convention as om.fetch_index_chunked's "time"-named index)."""
    t = trades_df.copy()
    if t.empty:
        t["oi_regime"] = pd.Series(dtype=object)
        return t
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t = t.sort_values("entry_time")
    oi_lookup = oi_regime_series.rename("oi_regime").sort_index().reset_index()
    oi_lookup = oi_lookup.rename(columns={oi_lookup.columns[0]: "entry_time"})
    return pd.merge_asof(t, oi_lookup, on="entry_time", direction="backward")


# --------------------------------------------------------------------
# Real test (needs live Dhan connectivity)
# --------------------------------------------------------------------

def main():
    run_self_checks()

    print("=" * 78)
    print("OI-GATE TEST -- open_momentum + six_concepts (best_two_of_claude_5_8_26)")
    print("Requires live Dhan connectivity. If this errors with a connection/proxy")
    print("issue, run this file on the machine where dhan_live_bot.py normally works.")
    print("=" * 78)

    print("Finding the current NIFTY futures contract...")
    fut_security_id, fut_symbol, fut_expiry = og.find_current_nifty_future()
    print(f"Using {fut_symbol} (security_id={fut_security_id}, expiry={fut_expiry})\n")

    print("Fetching NIFTY futures OI history (this sets the real usable window)...")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date_oi = (dt.date.today() - dt.timedelta(days=og.OI_REQUEST_DAYS_BACK)).strftime("%Y-%m-%d")
    oi_df = og.fetch_futures_oi_chunked(fut_security_id, from_date_oi, to_date)
    if oi_df.empty:
        print("No OI data returned at all -- stopping here.")
        return
    oi_start, oi_end = oi_df.index.min(), oi_df.index.max()
    print(f"\nReal OI data available from {oi_start} to {oi_end} "
          f"({(oi_end - oi_start).days} calendar days). This is the actual test window "
          f"for BOTH the gated and ungated side below -- NOT the full 2-year window "
          f"best_two_of_claude_5_8_26.py itself was originally tested on.\n")

    print("Fetching Nifty index 1-min price data (+ warmup for six_concepts' SMA200)...")
    warmup_start = (oi_start.date() - dt.timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    raw_1min = om.fetch_index_chunked(warmup_start, oi_end.strftime("%Y-%m-%d"))

    print("Joining futures OI onto the 1-min price series and classifying OI regime...")
    df = raw_1min.join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()
    oi_change = df["oi"].diff(og.OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(og.OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [og.classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]
    oi_regime_1min = df["oi_regime"]

    full_df = om.resample_2min_per_day(raw_1min[["open", "high", "low", "close", "volume"]])
    days_covered = sorted(d for d in set(full_df.index.date) if d >= oi_start.date())
    print(f"Test window: {len(days_covered)} trading day(s) with real OI coverage: "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    # --- open_momentum trades, restricted to the real-OI test window ---
    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in days_covered]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)
    print(f"open_momentum: {len(om_trades)} trades in the test window.")

    # --- six_concepts trades (indicators computed on the FULL warmup+test range,
    #     but only test-window days are eligible to produce a trade) ---
    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        if d in set(days_covered):
            day_groups_sc.setdefault(d, []).append(pos)
    sc_trades = sc.run_module(full_df_sc, day_groups_sc, lambda d, i: sc.rbi_gbi_signal(d, i, relaxed=True))
    print(f"six_concepts: {len(sc_trades)} trades in the test window.\n")

    # --- attach OI regime at entry time (asof, no lookahead: nearest OI reading AT or BEFORE entry) ---
    om_t = attach_oi_regime(om_trades, oi_regime_1min)
    sc_t = attach_oi_regime(sc_trades, oi_regime_1min)

    # --- baseline vs OI-gated, using best_two's own priority-merge + metrics ---
    combined_base, _ = bt.merge_with_priority(om_t, sc_t)
    m_base = bt.compute_combined_metrics(combined_base, bt.STARTING_CAPITAL)

    om_g = om_t[om_t.apply(lambda r: trade_passes_gate(r["direction"], r.get("oi_regime")), axis=1)].copy()
    sc_g = sc_t[sc_t.apply(lambda r: trade_passes_gate(r["direction"], r.get("oi_regime")), axis=1)].copy()
    combined_gated, dropped_gated = bt.merge_with_priority(om_g, sc_g)
    m_gated = bt.compute_combined_metrics(combined_gated, bt.STARTING_CAPITAL)

    print("=" * 78)
    print("RESULT -- combined portfolio, real OI window")
    print("=" * 78)
    print(f"  Baseline (no OI gate) : n={m_base['n']:3d}  win={m_base['win_pct']}%  "
          f"net=Rs{m_base['net_rs']:,.0f}  maxDD=Rs{m_base['max_dd_rs']:,.0f}  SQN={m_base['sqn']}")
    print(f"  OI-Gated               : n={m_gated['n']:3d}  win={m_gated['win_pct']}%  "
          f"net=Rs{m_gated['net_rs']:,.0f}  maxDD=Rs{m_gated['max_dd_rs']:,.0f}  SQN={m_gated['sqn']}")
    print(f"  Kept: om={len(om_g)}/{len(om_t)}, sc={len(sc_g)}/{len(sc_t)}\n")

    for name, t in [("open_momentum", om_t), ("six_concepts", sc_t)]:
        if t.empty:
            continue
        t = t.copy()
        t["gate_pass"] = t.apply(lambda r: trade_passes_gate(r["direction"], r.get("oi_regime")), axis=1)
        g = t.groupby("gate_pass")["rupee_pnl"].agg(
            n="count", win_pct=lambda s: round(100 * (s > 0).mean(), 1), net=lambda s: round(s.sum(), 2)
        )
        print(f"  {name} by OI-gate pass/fail:\n{g.to_string()}\n")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    win_ok = m_gated["win_pct"] > m_base["win_pct"]
    dd_ok = m_gated["max_dd_rs"] < m_base["max_dd_rs"]
    sqn_ok = m_gated["sqn"] > m_base["sqn"]
    net_ok = m_gated["net_rs"] > m_base["net_rs"]
    print(f"  Win rate improved   : {win_ok}  ({m_base['win_pct']}% -> {m_gated['win_pct']}%)")
    print(f"  Max drawdown improved: {dd_ok}  (Rs{m_base['max_dd_rs']:,.0f} -> Rs{m_gated['max_dd_rs']:,.0f})")
    print(f"  SQN improved        : {sqn_ok}  ({m_base['sqn']} -> {m_gated['sqn']})")
    print(f"  Net profit improved : {net_ok}  (Rs{m_base['net_rs']:,.0f} -> Rs{m_gated['net_rs']:,.0f})")
    if sum([win_ok, dd_ok, sqn_ok, net_ok]) >= 3:
        print("\n  OI gate looks genuinely helpful here (majority of metrics improved). Still only "
              "one real window (~90-120 days) -- treat as promising, not proven, same standard "
              "applied to the DNA filter test.")
    else:
        print("\n  OI gate does NOT clearly help open_momentum/six_concepts either -- same conclusion "
              "as Concepts 2&4. Do not add this to step10_live_paper_best2.py based on this result.")

    combined_base.to_csv("oi_gate_test_best2_baseline_trades.csv", index=False)
    combined_gated.to_csv("oi_gate_test_best2_gated_trades.csv", index=False)
    print("\nSaved: oi_gate_test_best2_baseline_trades.csv / _gated_trades.csv")


if __name__ == "__main__":
    main()
