"""
Module: sirius_synthetic_candle_test.py
Description: Tests Sirius (open_momentum + six_concepts combined) on
             SYNTHETIC 2-month sequences built from REAL candle patterns --
             not fabricated random data, and not the exact real historical
             order either. This is a bootstrap robustness check: does Sirius
             still show a real edge when the same real days it was
             backtested on get reshuffled/recombined into sequences that
             never actually happened, or was the original result partly a
             coincidence of that one specific 6-month ordering?

METHOD (day-block bootstrap, not a random walk):
  - Pulls real 2-min candle SHAPES from the cached 6-month dataset
    (dhan_historical_data_180days_v5.csv, 122 real trading days).
  - Builds a synthetic ~42-trading-day (2 month) sequence by randomly
    drawing real days WITH replacement (the same real day can appear more
    than once, or not at all, in a given synthetic run).
  - Each day's OHLC shape is 100% real, untouched -- only WHICH real day
    appears WHERE in the sequence is randomized. This preserves genuine
    intraday patterns (the opening-drive shapes, RBI/GBI patterns, etc.)
    instead of replacing them with statistical noise, which would just
    test "does this work on a random walk" (answer: obviously not, these
    are pattern-based strategies, that wouldn't tell us anything useful).
  - Each synthetic day is price-shifted so its open exactly matches the
    previous synthetic day's close (zero overnight gap) -- avoids
    introducing FAKE overnight gap spikes purely as an artifact of
    stitching together days that were never actually adjacent. Real
    calendar date LABELS are kept in their real sequential order so
    date-dependent logic (NSE lot-size schedule) still resolves correctly.
  - Runs MANY independent random draws (not just one), because a single
    bootstrap draw can be lucky or unlucky by chance -- reports the spread
    of outcomes, not a single number.

WHAT THIS DOES AND DOES NOT TELL YOU:
  - DOES tell you whether Sirius's edge depends on the exact specific
    6-month sequence it was tested on, or holds up across many different
    recombinations of the same real day-patterns.
  - Does NOT tell you anything about true future performance -- these are
    still the same 122 real days, just reordered. A strategy that's
    curve-fit to a handful of specific days could still look fine here if
    those days just get resampled a lot; this is a coincidence check, not
    a substitute for real out-of-sample testing on genuinely new data
    (like dna_filter_out_of_sample_test.py did on a separate 18-month
    window).

Run (uses cached real data, no live Dhan connection needed):
    cd "C:\\dhan algo"
    python sirius_synthetic_candle_test.py
"""

import numpy as np
import pandas as pd

import nifty_open_momentum_stoptrail_sweep as om
import nifty_six_concepts_screening as sc
import best_two_of_claude_5_8_26 as bt

CSV = "dhan_historical_data_180days_v5.csv"
SYNTHETIC_DAYS = 42     # ~2 trading months
N_TRIALS = 30            # independent random draws, not just one
RANDOM_SEED = 20260805   # fixed for reproducibility of this specific run


# ======================================================================
# Synthetic sequence builder (pure, testable without live data)
# ======================================================================

def build_synthetic_sequence(full_df: pd.DataFrame, real_days: list, label_dates: list,
                              rng: np.random.Generator) -> pd.DataFrame:
    """Draws len(label_dates) real days WITH replacement from real_days,
    stitches their REAL candle shapes together under the label_dates'
    calendar sequence, rebasing each day so its open matches the previous
    synthetic day's close (no fake overnight gaps introduced by stitching)."""
    pieces = []
    prev_close = None
    for label_date in label_dates:
        source_day = real_days[rng.integers(0, len(real_days))]
        day_df = full_df.loc[full_df.index.date == source_day].copy()
        offset = 0.0 if prev_close is None else (prev_close - float(day_df["open"].iloc[0]))
        for col in ["open", "high", "low", "close"]:
            day_df[col] = day_df[col] + offset
        time_of_day = day_df.index - pd.Timestamp(day_df.index[0].date())
        day_df.index = pd.Timestamp(label_date) + time_of_day
        pieces.append(day_df)
        prev_close = float(day_df["close"].iloc[-1])
    return pd.concat(pieces).sort_index()


def run_sirius_on(full_df: pd.DataFrame) -> dict:
    """Runs the real, unmodified Sirius pipeline (same functions as every
    other real backtest in this project) on whatever full_df it's given."""
    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)

    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        day_groups_sc.setdefault(d, []).append(pos)
    sc_trades = sc.run_module(full_df_sc, day_groups_sc, lambda d, i: sc.rbi_gbi_signal(d, i, relaxed=True))

    combined, _ = bt.merge_with_priority(om_trades, sc_trades)
    return bt.compute_combined_metrics(combined, bt.STARTING_CAPITAL)


# ======================================================================
# Self-checks (synthetic-of-synthetic, no network needed)
# ======================================================================

def run_self_checks():
    print("Running sirius_synthetic_candle_test self-checks (synthetic data)...")

    idx1 = pd.date_range("2025-01-06 09:15", periods=5, freq="2min")
    day1 = pd.DataFrame({"open": [100, 101, 102, 101, 100], "high": [101]*5, "low": [99]*5,
                          "close": [100, 101, 102, 101, 100], "volume": [10]*5}, index=idx1)
    idx2 = pd.date_range("2025-02-10 09:15", periods=5, freq="2min")
    day2 = pd.DataFrame({"open": [500, 501, 500, 499, 498], "high": [502]*5, "low": [497]*5,
                          "close": [500, 501, 500, 499, 498], "volume": [10]*5}, index=idx2)
    full_df = pd.concat([day1, day2])

    rng = np.random.default_rng(0)
    label_dates = [pd.Timestamp("2030-01-01").date(), pd.Timestamp("2030-01-02").date(),
                   pd.Timestamp("2030-01-03").date()]
    synth = build_synthetic_sequence(full_df, [idx1[0].date(), idx2[0].date()], label_dates, rng)

    days_out = sorted(set(synth.index.date))
    assert days_out == label_dates, "output must use the real calendar label dates, in order"
    print("  build_synthetic_sequence: uses the correct label dates, in order")

    # no-gap check: every day after the first must open exactly at the previous day's close
    for i in range(1, len(label_dates)):
        prev_day = synth.loc[synth.index.date == label_dates[i - 1]]
        this_day = synth.loc[synth.index.date == label_dates[i]]
        prev_close = float(prev_day["close"].iloc[-1])
        this_open = float(this_day["open"].iloc[0])
        assert abs(prev_close - this_open) < 1e-9, \
            f"day {label_dates[i]} must open exactly at the prior synthetic day's close (no fake gap)"
    print("  build_synthetic_sequence: no artificial overnight gaps between stitched days")

    # each day's SHAPE (relative moves) must be untouched -- only its absolute level shifts.
    # The first synthetic day is drawn randomly from the two source days, so check against
    # whichever one it actually matches rather than assuming which was picked.
    a_day = synth.loc[synth.index.date == label_dates[0]]
    shape = (a_day["close"] - a_day["open"].iloc[0]).round(6).tolist()
    shape_day1 = (day1["close"] - day1["open"].iloc[0]).round(6).tolist()
    shape_day2 = (day2["close"] - day2["open"].iloc[0]).round(6).tolist()
    assert shape in (shape_day1, shape_day2), \
        "intraday shape (relative moves) must exactly match one of the real source days, only rebased in level"
    print("  build_synthetic_sequence: real intraday shape preserved exactly (only price level shifts)")

    print("All sirius_synthetic_candle_test self-checks passed.\n")


# ======================================================================
# Real test: many synthetic 2-month draws from real candle patterns
# ======================================================================

def main():
    run_self_checks()

    print("=" * 78)
    print(f"SIRIUS ON SYNTHETIC CANDLES -- {N_TRIALS} random {SYNTHETIC_DAYS}-day sequences, "
          f"built from real day-patterns (bootstrap, not random-walk noise)")
    print("=" * 78)

    raw = pd.read_csv(CSV, parse_dates=["datetime"])
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.tz_localize(None)
    raw = raw.set_index("datetime").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    full_df = om.resample_2min_per_day(raw)
    real_days = sorted(set(full_df.index.date))
    label_dates = real_days[:SYNTHETIC_DAYS]
    print(f"Real source pool: {len(real_days)} real trading days. "
          f"Each synthetic run: {SYNTHETIC_DAYS} days drawn with replacement from that pool.\n")

    real_metrics_full = run_sirius_on(full_df)
    print(f"For reference -- Sirius on the REAL, unshuffled 6-month sequence: "
          f"n={real_metrics_full['n']}  win={real_metrics_full['win_pct']}%  "
          f"net=Rs{real_metrics_full['net_rs']:,.0f}  SQN={real_metrics_full['sqn']}\n")

    rng = np.random.default_rng(RANDOM_SEED)
    results = []
    for trial in range(N_TRIALS):
        synth = build_synthetic_sequence(full_df, real_days, label_dates, rng)
        m = run_sirius_on(synth)
        results.append(m)
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: n={m['n']:3d}  win={m['win_pct']:5.1f}%  "
              f"net=Rs{m['net_rs']:>9,.0f}  maxDD=Rs{m['max_dd_rs']:>7,.0f}  SQN={m['sqn']:5.2f}")

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print(f"SUMMARY ACROSS {N_TRIALS} SYNTHETIC 2-MONTH SEQUENCES")
    print("=" * 78)
    for col, label in [("win_pct", "Win rate %"), ("net_rs", "Net profit Rs"),
                        ("max_dd_rs", "Max drawdown Rs"), ("sqn", "SQN")]:
        print(f"  {label:<18} mean={res_df[col].mean():>10,.2f}  median={res_df[col].median():>10,.2f}  "
              f"min={res_df[col].min():>10,.2f}  max={res_df[col].max():>10,.2f}")

    losing_trials = (res_df["net_rs"] < 0).sum()
    print(f"\n  Trials that lost money: {losing_trials}/{N_TRIALS}")
    print(f"  Trials with SQN < 2 (weak/no real edge by conventional SQN reading): "
          f"{(res_df['sqn'] < 2).sum()}/{N_TRIALS}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if losing_trials == 0 and res_df["sqn"].median() > 3:
        print("  Sirius held up across essentially every random recombination of real candle "
              "patterns -- the edge does not appear to depend on the one specific historical "
              "order it was originally tested on. Reassuring, though remember this is still the "
              "same 122 real days reshuffled, not genuinely new data.")
    elif losing_trials > N_TRIALS * 0.2:
        print("  A meaningful fraction of reshuffled sequences lost money -- the original 6-month "
              "result may depend more on the SPECIFIC ORDER/TIMING of real events than on a "
              "durable pattern-level edge. Worth treating the original result with more caution.")
    else:
        print("  Mixed but mostly positive -- edge appears real but with real variance depending "
              "on which days/order show up. Normal for a live strategy; not a red flag by itself.")

    res_df.to_csv("sirius_synthetic_candle_test_results.csv", index=False)
    print("\nSaved: sirius_synthetic_candle_test_results.csv")


if __name__ == "__main__":
    main()
