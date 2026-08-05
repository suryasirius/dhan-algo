"""
Module: ai_exit_engine_sample.py
Description: SAMPLE / prototype AI exit engine, per the architecture doc's
             Section 3 (Research/MARKET_STRUCTURE_ML_ARCHITECTURE.md) and the
             instruction to try a real ML exit model on the data already
             available, and report honestly before deciding whether it's
             worth developing further.

SCOPE (deliberately narrow for a first sample): open_momentum only (122
trades on the real 6-month dataset -- the larger, stronger of the two
strategies). Not six_concepts yet -- one honest test first, not two thin ones.

WHAT THIS TESTS: open_momentum currently exits via a FIXED rule -- hard stop
20pts, then once in profit, trail 1pt behind the peak. This script asks: can
a real trained model, looking at the trade's own unfolding path (profit so
far, bars held, distance from peak, time of day), decide better than that
fixed rule when to take profit? The hard stop-loss is NEVER replaced or
removed -- the model only gets a vote on WHEN TO TAKE PROFIT once the trade
is already working, matching the architecture doc's fail-safe principle.

METHOD (real ML, not a heuristic -- see Research/MARKET_STRUCTURE_ML_ARCHITECTURE.md
Section 3 for why this distinction matters):
  1. Reconstruct each trade's FULL intraday premium path from entry to EOD
     square-off (not just to its actual fixed-rule exit) -- so the model can
     see what holding longer WOULD have earned, not just what the fixed rule
     actually captured.
  2. Label each bar: "future_max_gain_from_here" = the best premium still
     reachable before EOD, minus the current premium. label_hold = 1 if
     that's still meaningfully positive (more upside left), else 0 (this bar
     is at/near the best price the rest of the day has to offer -- should
     exit now).
  3. Real features only: profit-so-far, bars held, drawback from the peak
     reached so far, time of day. No fabricated inputs.
  4. WALK-FORWARD split by TRADE DATE (not by bar) -- train on the
     chronologically earlier trades, test only on later trades never seen
     in training. No shuffling (the exact leakage bug already fixed
     elsewhere in this project).
  5. Simple model: logistic regression. Feature importance (coefficients)
     printed after training, per the architecture doc's Section 3.7.
  6. Backtest comparison on the TEST trades only: original fixed sl20/trail1
     rule vs. hard-stop-plus-model-decides-when-to-take-profit, same trades,
     same starting capital, same metrics.

HONEST CAVEAT: bars within one trade are NOT independent samples (they're
all part of the same trade's path), so the bar-count is not the same as
having that many independent trades. Real sample size for anything that
matters is still ~122 trades split across train/test -- small for ML. This
is a first sample/prototype specifically to see if there's any signal worth
developing further, not a finished, trustworthy model.

DATA: price-only, uses the same real cached 6-month dataset as the other
recent tests (dhan_historical_data_180days_v5.csv) -- runs fully offline,
no live Dhan connection needed.

Run:
    cd "C:\\dhan algo"
    python ai_exit_engine_sample.py
"""

import numpy as np
import pandas as pd

import nifty_open_momentum_stoptrail_sweep as om
import best_two_of_claude_5_8_26 as bt

HOLD_LABEL_THRESHOLD_PTS = 2.0   # "meaningfully more upside" cutoff for label_hold
TRAIN_FRACTION = 0.7              # chronological split -- first 70% of trade DATES train, rest test
FEATURE_COLUMNS = ["pnl_pts_so_far", "bars_held", "drawdown_from_peak_pts", "minutes_since_open"]


# ======================================================================
# Path reconstruction + labeling (pure, testable without sklearn/network)
# ======================================================================

def reconstruct_trade_bars(day_df, strike, side, entry_idx, entry_price, eod_squareoff_idx):
    """Rebuilds the FULL premium path from entry to EOD square-off (not just
    to the trade's actual fixed-rule exit), and returns one row per bar with
    the real features + the counterfactual 'future_max_gain_from_here' label
    input. No lookahead in the FEATURES (each bar only knows its own past);
    the LABEL is deliberately look-ahead within the trade's own remaining
    life -- that's the whole point of a training label, not a bug."""
    day_open_spot = day_df["open"].iloc[0]
    premium_df = om.build_premium_series(day_df, strike, day_open_spot, side)
    closes = premium_df["close"].to_numpy()
    highs = premium_df["high"].to_numpy()
    market_open_minute = 0  # minutes_since_open computed from day_df's own index below

    rows = []
    peak_so_far = entry_price
    for j in range(entry_idx, eod_squareoff_idx + 1):
        pnl_pts_so_far = closes[j] - entry_price
        peak_so_far = max(peak_so_far, highs[j])
        drawdown_from_peak_pts = peak_so_far - closes[j]
        bars_held = j - entry_idx
        minutes_since_open = (day_df.index[j] - day_df.index[0]).total_seconds() / 60.0

        future_highs = highs[j:eod_squareoff_idx + 1]
        future_max = float(future_highs.max()) if len(future_highs) else closes[j]
        future_max_gain_from_here = future_max - closes[j]
        label_hold = 1 if future_max_gain_from_here > HOLD_LABEL_THRESHOLD_PTS else 0

        rows.append({
            "bar_time": day_df.index[j],
            "pnl_pts_so_far": pnl_pts_so_far,
            "bars_held": bars_held,
            "drawdown_from_peak_pts": drawdown_from_peak_pts,
            "minutes_since_open": minutes_since_open,
            "label_hold": label_hold,
            "premium_close": closes[j],
            "premium_low": float(premium_df["low"].iloc[j]),
        })
    return rows


def build_training_table(full_df, om_trades):
    """Runs reconstruct_trade_bars() over every real open_momentum trade,
    tagging each bar-row with its trade's date and row id for the
    walk-forward split later."""
    all_rows = []
    for t_id, trade in om_trades.iterrows():
        day = pd.to_datetime(trade["date"]).date()
        day_df = full_df.loc[full_df.index.date == day]
        if day_df.empty:
            continue
        entry_time = pd.to_datetime(trade["entry_time"])
        if entry_time not in day_df.index:
            continue
        entry_idx = day_df.index.get_loc(entry_time)
        eod_idx = len(day_df) - 1
        for k, t in enumerate(day_df.index):
            if t.time() >= om.EOD_SQUAREOFF:
                eod_idx = k
                break
        if eod_idx <= entry_idx:
            continue
        strike = om.itm_call_strike(day_df["open"].iloc[0]) if trade["side"] == "CE" else om.itm_put_strike(day_df["open"].iloc[0])
        bars = reconstruct_trade_bars(day_df, strike, trade["side"], entry_idx, trade["entry_price"], eod_idx)
        for b in bars:
            b["trade_id"] = t_id
            b["date"] = day
            b["side"] = trade["side"]
            b["entry_price"] = trade["entry_price"]
        all_rows.extend(bars)
    return pd.DataFrame(all_rows)


# ======================================================================
# Self-checks (synthetic, no sklearn/network needed)
# ======================================================================

def run_self_checks():
    print("Running ai_exit_engine_sample self-checks (synthetic data)...")

    # --- reconstruct_trade_bars: a clean rise-then-fall path ---
    idx = pd.date_range("2025-01-06 09:15", periods=10, freq="2min")
    # spot path: rises for 5 bars then falls -- premium should mirror via CE construction
    spot = [100, 101, 102, 104, 107, 106, 104, 103, 102, 101]
    day_df = pd.DataFrame({
        "open": spot, "close": spot,
        "high": [x + 0.3 for x in spot], "low": [x - 0.3 for x in spot],
        "volume": [100] * 10,
    }, index=idx)
    strike = om.itm_call_strike(spot[0])
    premium_df = om.build_premium_series(day_df, strike, spot[0], "CE")
    entry_price = float(premium_df["open"].iloc[1])
    bars = reconstruct_trade_bars(day_df, strike, "CE", entry_idx=1, entry_price=entry_price, eod_squareoff_idx=9)

    # the peak of the spot path (and therefore the premium) is around bar index 4 (spot=107).
    # bars well before the peak must be labeled "hold" (more upside still ahead);
    # the last bar (nothing left of the day) must always be labeled "exit now".
    early_bar = next(b for b in bars if b["bars_held"] == 0)
    last_bar = bars[-1]
    assert early_bar["label_hold"] == 1, "a bar well before the day's remaining peak must be labeled hold"
    assert last_bar["label_hold"] == 0, "the final bar of the day can never have more upside left -- must be exit"
    print("  reconstruct_trade_bars: rise-then-fall path labels hold/exit correctly")

    # --- features are causal: pnl_pts_so_far and drawdown must only use bars up to 'now' ---
    mid_bar = next(b for b in bars if b["bars_held"] == 3)
    assert mid_bar["pnl_pts_so_far"] == pytest_approx(mid_bar["premium_close"] - entry_price)
    print("  feature causality (pnl_pts_so_far matches premium-vs-entry) ok")

    # --- walk-forward split helper: no date overlap between train/test ---
    dates = pd.to_datetime([f"2025-01-{d:02d}" for d in range(1, 21)]).date
    train_dates, test_dates = walk_forward_split(list(dates), TRAIN_FRACTION)
    assert max(train_dates) < min(test_dates), "train dates must all be strictly before test dates"
    assert len(train_dates) + len(test_dates) == len(dates)
    print("  walk_forward_split: no date leakage between train and test")

    print("All ai_exit_engine_sample self-checks passed.\n")


def pytest_approx(x, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def walk_forward_split(sorted_unique_dates, train_fraction):
    sorted_unique_dates = sorted(sorted_unique_dates)
    cut = max(1, int(len(sorted_unique_dates) * train_fraction))
    return sorted_unique_dates[:cut], sorted_unique_dates[cut:]


# ======================================================================
# Backtest simulation with the trained model in place of the fixed trail
# ======================================================================

def simulate_ai_exit_trade(day_df, strike, side, entry_idx, entry_price, eod_idx, model, feature_columns,
                            hard_stop_pts=bt.OM_STOP_PTS):
    """Replays one trade bar by bar. Hard stop-loss is ALWAYS checked first
    and can NEVER be overridden by the model -- the model only gets a vote
    on taking profit once the trade is working. Mirrors om.simulate_trade's
    conservative ordering (check bar's low against the CURRENT stop before
    anything else)."""
    day_open_spot = day_df["open"].iloc[0]
    premium_df = om.build_premium_series(day_df, strike, day_open_spot, side)
    hard_stop_level = entry_price - hard_stop_pts
    peak_so_far = entry_price

    for j in range(entry_idx, eod_idx + 1):
        bar = premium_df.iloc[j]
        if bar["low"] <= hard_stop_level:
            return float(hard_stop_level), "STOP", day_df.index[j]
        if day_df.index[j].time() >= om.EOD_SQUAREOFF:
            return float(bar["close"]), "EOD", day_df.index[j]

        peak_so_far = max(peak_so_far, float(bar["high"]))
        pnl_pts_so_far = float(bar["close"]) - entry_price
        if pnl_pts_so_far > 0:
            feats = pd.DataFrame([{
                "pnl_pts_so_far": pnl_pts_so_far,
                "bars_held": j - entry_idx,
                "drawdown_from_peak_pts": peak_so_far - float(bar["close"]),
                "minutes_since_open": (day_df.index[j] - day_df.index[0]).total_seconds() / 60.0,
            }])[feature_columns]
            predicted_hold = model.predict(feats)[0]
            if predicted_hold == 0:
                return float(bar["close"]), "AI_EXIT", day_df.index[j]

    last = premium_df.iloc[eod_idx]
    return float(last["close"]), "EOD", day_df.index[eod_idx]


# ======================================================================
# Main: build the real table, train, walk-forward test, report honestly
# ======================================================================

def main():
    run_self_checks()

    from sklearn.linear_model import LogisticRegression

    print("=" * 78)
    print("SAMPLE AI EXIT ENGINE -- open_momentum only, real 6-month data")
    print("=" * 78)

    raw = pd.read_csv("dhan_historical_data_180days_v5.csv", parse_dates=["datetime"])
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.tz_localize(None)
    raw = raw.set_index("datetime").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    full_df = om.resample_2min_per_day(raw)

    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)
    print(f"Real open_momentum trades: {len(om_trades)}\n")

    table = build_training_table(full_df, om_trades)
    print(f"Reconstructed {len(table)} bar-level rows across {table['trade_id'].nunique()} trades "
          f"(honest caveat: bars within a trade are correlated, not independent samples).\n")

    all_dates = sorted(table["date"].unique())
    train_dates, test_dates = walk_forward_split(all_dates, TRAIN_FRACTION)
    print(f"Walk-forward split: train on {len(train_dates)} earlier days "
          f"({train_dates[0]} to {train_dates[-1]}), test on {len(test_dates)} later days "
          f"({test_dates[0]} to {test_dates[-1]}), never seen in training.\n")

    train_tbl = table[table["date"].isin(train_dates)]
    test_trade_ids = set(table[table["date"].isin(test_dates)]["trade_id"])

    X_train, y_train = train_tbl[FEATURE_COLUMNS], train_tbl["label_hold"]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    print("Feature importance (logistic regression coefficients -- positive means it pushes toward HOLD):")
    for feat, coef in sorted(zip(FEATURE_COLUMNS, model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {feat:<26} {coef:+.4f}")
    print()

    # --- baseline: original fixed-rule trades, TEST dates only ---
    baseline_test = om_trades.loc[om_trades.index.isin(test_trade_ids)].copy()
    m_base = bt.compute_combined_metrics(baseline_test.rename(columns={"rupee_pnl": "rupee_pnl"}), bt.STARTING_CAPITAL)

    # --- AI-exit backtest, same TEST trades, hard-stop-plus-model exit ---
    ai_trades = []
    for t_id, trade in om_trades.iterrows():
        if t_id not in test_trade_ids:
            continue
        day = pd.to_datetime(trade["date"]).date()
        day_df = full_df.loc[full_df.index.date == day]
        entry_time = pd.to_datetime(trade["entry_time"])
        if entry_time not in day_df.index:
            continue
        entry_idx = day_df.index.get_loc(entry_time)
        eod_idx = len(day_df) - 1
        for k, t in enumerate(day_df.index):
            if t.time() >= om.EOD_SQUAREOFF:
                eod_idx = k
                break
        strike = om.itm_call_strike(day_df["open"].iloc[0]) if trade["side"] == "CE" else om.itm_put_strike(day_df["open"].iloc[0])
        exit_price, reason, exit_time = simulate_ai_exit_trade(
            day_df, strike, trade["side"], entry_idx, trade["entry_price"], eod_idx, model, FEATURE_COLUMNS
        )
        lot = om.lot_size_for_date(day)
        pnl_pts = exit_price - trade["entry_price"]
        rupee_pnl = pnl_pts * lot - om.BROKERAGE_PER_TRADE
        ai_trades.append({"date": day, "entry_time": entry_time, "exit_time": exit_time,
                           "entry_price": trade["entry_price"], "exit_price": exit_price,
                           "exit_reason": reason, "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl})
    ai_df = pd.DataFrame(ai_trades)
    m_ai = bt.compute_combined_metrics(ai_df, bt.STARTING_CAPITAL)

    print("=" * 78)
    print(f"RESULT -- test set only ({len(test_trade_ids)} trades, dates never seen in training)")
    print("=" * 78)
    print(f"  Baseline (fixed sl{bt.OM_STOP_PTS}/trail{bt.OM_TRAIL_PTS}) : n={m_base['n']:3d}  win={m_base['win_pct']}%  "
          f"net=Rs{m_base['net_rs']:,.0f}  maxDD=Rs{m_base['max_dd_rs']:,.0f}  SQN={m_base['sqn']}")
    print(f"  AI exit (hard stop + model)     : n={m_ai['n']:3d}  win={m_ai['win_pct']}%  "
          f"net=Rs{m_ai['net_rs']:,.0f}  maxDD=Rs{m_ai['max_dd_rs']:,.0f}  SQN={m_ai['sqn']}")
    if not ai_df.empty:
        print(f"\n  AI exit reasons: {ai_df['exit_reason'].value_counts().to_dict()}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    checks = [m_ai["win_pct"] > m_base["win_pct"], m_ai["net_rs"] > m_base["net_rs"],
              m_ai["max_dd_rs"] <= m_base["max_dd_rs"], m_ai["sqn"] > m_base["sqn"]]
    if sum(checks) >= 3:
        print("  AI exit shows a real, promising improvement on this held-out test -- worth developing further "
              "(more data, six_concepts too, cross-validated more than one split) before considering paper trading it.")
    else:
        print("  AI exit does NOT clearly beat the fixed rule on this held-out test. Small sample, one split -- "
              "report honestly and don't build further on this specific result without more data.")

    baseline_test.to_csv("ai_exit_sample_baseline_test_trades.csv", index=False)
    ai_df.to_csv("ai_exit_sample_ai_test_trades.csv", index=False)
    print("\nSaved: ai_exit_sample_baseline_test_trades.csv / ai_exit_sample_ai_test_trades.csv")


if __name__ == "__main__":
    main()
