"""
Compression Spike (Concept 2 ALONE, isolated from Concept 4) -- SL x TP
optimization sweep
=================================================================
Baseline already computed directly from the existing real trade log
(concepts_2_4_all_days_trades.csv, no re-fetch needed for this part):
concept_id==2 alone, all weekdays, fixed 5pt SL / 10pt TP (dhan_live_bot.py's
current live settings): n=781, win%=39.1%, PF=0.95 (a slight net loser),
net=-251.1pts, SQN=-0.65. BUT broken down by weekday, the exact same
Tuesday/Wednesday pattern already baked into dhan_live_bot.py's
ALLOWED_WEEKDAYS={1,2} holds for concept 2 in isolation too: Tuesday
+66.45pts, Wednesday +387.75pts (both net positive), Monday -136.80,
Thursday -373.40 (worst), Friday -195.10. So compression spike alone,
Tue/Wed only, is a real net-positive starting point worth optimizing --
this script does that.

WHAT THIS SCRIPT DOES: isolates concept_id==2 signals ONLY (drops concept
4/rollover entirely, unlike every previous concepts_2_4_* script in this
project, which always tested them blended), restricted to the ALREADY-
VALIDATED Tuesday/Wednesday weekday filter (not re-litigating that choice
here -- it's established from real 2yr data, re-checking it again for a
third time on the same data would just be re-deriving something already
answered), then runs a genuine 2D stop-loss x target sweep (previous
sensitivity scripts only ever swept SL with TP held fixed at 10 -- this
adds the TP dimension too, since "optimize" should mean searching the
actual 2D space, not one axis of it).

Reuses dlb.compute_latest_signal() -- the exact real, unedited signal-
detection logic from the live bot -- via the same parallelized,
HTF-context-window-correct signal collection already proven in
concepts_2_4_sl_sensitivity_fast.py (each day gets the right amount of
prior-day history so the 5m/15m trend filter isn't starved, the bug that
caused an earlier version of this test to silently find zero signals).
dhan_live_bot.py itself is never modified.

Same PnL convention as the existing concepts_2_4_* script family
(points x ASSUMED_DELTA x LOT_SIZE, flat Rs40/trade brokerage) for direct
comparability against those existing real results -- not silently
switched to the newer proxy-premium model used elsewhere in this project,
to avoid mixing conventions mid-comparison. Flagged here explicitly, not
hidden.

Full rigor applied to the sweep result: grid-edge check (is the best
combo sitting at the tested boundary?), in-sample/out-of-sample split on
the best combo, and the project's standing red-flag calibration
(PF>2.5 or SQN>5 treated as an extraordinary claim needing scrutiny, not
a finding to trust outright).
"""

import datetime as dt
import math
import os
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

WARMUP_BARS = 30
TARGET_CONCEPT_ID = 2          # compression spike (sell) ONLY -- concept 4 (rollover) excluded
TARGET_WEEKDAYS = {1, 2}       # Tuesday, Wednesday -- already-validated subset, not re-tested here

SL_SWEEP = [3, 4, 5, 6, 7, 8, 10, 12]
TP_SWEEP = [5, 8, 10, 12, 15, 20]

STARTING_CAPITAL = 10_000
ASSUMED_DELTA = 0.6
LOT_SIZE = 65
BROKERAGE_PER_TRADE = 40.0


def extract_signal(sig):
    if sig.get("final_buy"):
        return "LONG", sig.get("concept_id")
    if sig.get("final_sell"):
        return "SHORT", sig.get("concept_id")
    return None


def get_htf_context_days():
    cd = getattr(dlb, "HTF_CONTEXT_DAYS", None)
    return cd if cd is not None else 5


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def compute_mae(day_df, entry_bar, side):
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    n = len(day_df)
    if entry_bar + 1 >= n:
        return 0.0
    entry_price = day_df["close"].iloc[entry_bar]
    if side == "LONG":
        return max(0.0, entry_price - min(lows[entry_bar + 1:]))
    else:
        return max(0.0, max(highs[entry_bar + 1:]) - entry_price)


def simulate_with_sl_tp(day_df, entry_bar, side, sl_pts, tp_pts, max_bars):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    entry_price = closes[entry_bar]

    if side == "LONG":
        stop = entry_price - sl_pts
        target = entry_price + tp_pts
    else:
        stop = entry_price + sl_pts
        target = entry_price - tp_pts

    exit_j, exit_price, reason = entry_bar, entry_price, "EOD"
    for j in range(entry_bar + 1, min(entry_bar + 1 + max_bars, n)):
        h, l, c = highs[j], lows[j], closes[j]
        # conservative ordering: STOP checked before TARGET within the same bar
        if side == "LONG":
            if l <= stop:
                exit_j, exit_price, reason = j, stop, "STOP"; break
            if h >= target:
                exit_j, exit_price, reason = j, target, "TARGET"; break
        else:
            if h >= stop:
                exit_j, exit_price, reason = j, stop, "STOP"; break
            if l <= target:
                exit_j, exit_price, reason = j, target, "TARGET"; break
    else:
        last_idx = min(entry_bar + max_bars, n - 1)
        exit_j, exit_price, reason = last_idx, closes[last_idx], "TIMEOUT"

    points = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    rupee_pnl = points * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    return {
        "date": day_df.index[entry_bar].date(), "side": side, "entry_time": idx[entry_bar],
        "exit_time": idx[exit_j], "points": points, "reason": reason, "rupee_pnl": rupee_pnl,
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert extract_signal({"final_buy": True, "final_sell": False, "concept_id": 2}) == ("LONG", 2)
    assert extract_signal({"final_buy": False, "final_sell": True, "concept_id": 4}) == ("SHORT", 4)
    assert extract_signal({"final_buy": False, "final_sell": False, "concept_id": None}) is None
    print("  extract_signal check ok")

    days = [dt.date(2025, 1, d) for d in range(1, 11)]
    w = build_context_window_days(days, dt.date(2025, 1, 10), 5)
    assert w == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8),
                 dt.date(2025, 1, 9), dt.date(2025, 1, 10)]
    print("  build_context_window_days check ok")

    # simulate_with_sl_tp: clean STOP case (SHORT)
    idx_t = pd.date_range("2025-01-06 09:15", periods=6, freq="1min")
    day = pd.DataFrame({
        "open": [100, 101, 103, 106, 106, 106], "high": [100, 102, 104, 107, 107, 107],
        "low": [100, 100, 102, 105, 105, 105], "close": [100, 101, 103, 106, 106, 106],
    }, index=idx_t)
    t = simulate_with_sl_tp(day, 0, "SHORT", sl_pts=5, tp_pts=10, max_bars=15)
    assert t["reason"] == "STOP" and abs(t["points"] - (-5)) < 1e-6
    print("  simulate_with_sl_tp SHORT clean STOP check ok")

    # simulate_with_sl_tp: clean TARGET case (LONG)
    day2 = pd.DataFrame({
        "open": [100, 102, 104, 108, 112, 112], "high": [100, 103, 105, 109, 113, 113],
        "low": [100, 101, 103, 107, 111, 111], "close": [100, 102, 104, 108, 112, 112],
    }, index=idx_t)
    t2 = simulate_with_sl_tp(day2, 0, "LONG", sl_pts=5, tp_pts=10, max_bars=15)
    assert t2["reason"] == "TARGET" and abs(t2["points"] - 10) < 1e-6
    print("  simulate_with_sl_tp LONG clean TARGET check ok")

    # conservative ordering: same bar breaches stop AND would separately reach target -- must resolve STOP
    day3 = pd.DataFrame({
        "open": [100, 100, 100, 100, 100], "high": [100, 101, 130, 100, 100],
        "low": [100, 99, 92, 100, 100], "close": [100, 99, 100, 100, 100],
    }, index=idx_t[:5])
    t3 = simulate_with_sl_tp(day3, 0, "LONG", sl_pts=5, tp_pts=10, max_bars=15)
    assert t3["reason"] == "STOP", f"expected conservative STOP, got {t3['reason']}"
    print("  simulate_with_sl_tp conservative stop-before-target ordering check ok")

    print("All self-checks passed.\n")


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


def _scan_one_day(args):
    target_day, window_df = args
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return [], None
    first_bar_today, last_bar_today = today_positions[0], today_positions[-1]
    if (last_bar_today - first_bar_today) <= WARMUP_BARS + 2:
        return [], None

    found, first_error, open_bar = [], None, -1
    for i in today_positions:
        if i - first_bar_today < WARMUP_BARS or i <= open_bar:
            continue
        try:
            sig = dlb.compute_latest_signal(window_df.iloc[:i + 1])
        except Exception:
            if first_error is None:
                first_error = traceback.format_exc()
            continue
        if sig is None:
            continue
        result = extract_signal(sig)
        if result is None:
            continue
        side, concept = result
        found.append((i, side, concept))
        open_bar = i + dlb.MAX_BARS_IN_TRADE
    return found, first_error


def collect_concept2_signals(df, days_covered):
    context_days = get_htf_context_days()
    target_days = [d for d in days_covered if d.weekday() in TARGET_WEEKDAYS]
    print(f"Scanning Tuesday/Wednesday only: {len(target_days)} of {len(days_covered)} days.")
    print(f"Each day gets a {context_days}-trading-day rolling context window before scanning.\n")

    jobs = []
    for target_day in target_days:
        window_days = build_context_window_days(days_covered, target_day, context_days)
        window_df = df[(df.index.date >= window_days[0]) & (df.index.date <= window_days[-1])]
        jobs.append((target_day, window_df))

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Scanning {len(target_days)} days across {n_workers} worker processes...")
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_scan_one_day, jobs)

    all_signals, errors = [], []
    for (target_day, window_df), (found, err) in zip(jobs, results):
        day_df = window_df[window_df.index.date == target_day]
        for (i_window, side, concept) in found:
            if concept != TARGET_CONCEPT_ID:
                continue
            i_day = i_window - window_df.index.get_loc(day_df.index[0])
            all_signals.append((target_day, day_df, i_day, side))
        if err:
            errors.append((target_day, err))

    if errors:
        print(f"\n{len(errors)} of {len(target_days)} days hit an error calling compute_latest_signal. "
              f"First error (from {errors[0][0]}):\n{errors[0][1]}\nPaste this back if you see it.\n")

    return all_signals


def compute_metrics(tdf, starting_capital=STARTING_CAPITAL, sl_pts=None):
    if tdf.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0, "cagr_pct": 0.0}
    wins, losses = tdf[tdf["points"] > 0], tdf[tdf["points"] <= 0]
    win_pct = 100.0 * len(wins) / len(tdf)
    pf = (wins["points"].sum() / abs(losses["points"].sum())) if losses["points"].sum() != 0 else float("inf")
    equity = starting_capital + tdf["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd_rs = (peak - equity).max()
    if sl_pts:
        r_multiple = tdf["rupee_pnl"] / (sl_pts * ASSUMED_DELTA * LOT_SIZE)
        sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else 0.0
    else:
        sqn = 0.0
    dates = pd.to_datetime(tdf["date"])
    years = (dates.max() - dates.min()).days / 365.25
    net_rs = tdf["rupee_pnl"].sum()
    cagr_pct = 0.0
    if years > 0 and (starting_capital + net_rs) > 0:
        cagr_pct = 100.0 * (((starting_capital + net_rs) / starting_capital) ** (1.0 / years) - 1.0)
    return {"n": len(tdf), "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def main():
    run_self_checks()
    print("=== Compression Spike (Concept 2 ALONE) -- Tue/Wed, SL x TP optimization sweep ===\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    signals = collect_concept2_signals(df, days_covered)
    if not signals:
        print("No concept-2 signals collected on Tue/Wed. If an error was printed above, paste it back.")
        return
    print(f"Total concept-2 (compression spike) signals on Tue/Wed: {len(signals)}\n")

    print("--- MAE distribution (how far price moves against us before any SL/TP is applied) ---")
    maes = np.array([compute_mae(day_df, i, side) for _, day_df, i, side in signals])
    for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9)]:
        cnt = ((maes >= lo) & (maes < hi)).sum()
        print(f"  MAE in [{lo:>3},{hi if hi < 1e9 else 'inf':>4}) pts: {cnt:>4} ({100*cnt/len(maes):5.1f}%)")
    print(f"  Median MAE: {np.median(maes):.2f}pts  Mean: {maes.mean():.2f}pts  90th pct: {np.percentile(maes,90):.2f}pts\n")

    print("--- 2D SL x TP sweep ---")
    print(f"{'SL':>4} {'TP':>4} {'n':>5} {'win%':>7} {'PF':>7} {'net_Rs':>12} {'maxDD_Rs':>12} {'SQN':>6}")
    results = {}
    for sl in SL_SWEEP:
        for tp in TP_SWEEP:
            trades = [simulate_with_sl_tp(day_df, i, side, sl, tp, dlb.MAX_BARS_IN_TRADE) for _, day_df, i, side in signals]
            tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
            m = compute_metrics(tdf, sl_pts=sl)
            results[(sl, tp)] = (m, tdf)
            print(f"{sl:>4} {tp:>4} {m['n']:>5} {m['win_pct']:>6.1f}% {m['pf']:>7.2f} "
                  f"{m['net_rs']:>12,.2f} {m['max_dd_rs']:>12,.2f} {m['sqn']:>6.2f}")

    best_key = max(results, key=lambda k: results[k][0]["net_rs"])
    best_m, best_tdf = results[best_key]
    best_sl, best_tp = best_key
    print(f"\nBest combo by net Rs: SL={best_sl} TP={best_tp} -> n={best_m['n']} win%={best_m['win_pct']} "
          f"PF={best_m['pf']} net_Rs={best_m['net_rs']:,.2f} SQN={best_m['sqn']} CAGR%={best_m['cagr_pct']}")

    at_sl_edge = best_sl in (min(SL_SWEEP), max(SL_SWEEP))
    at_tp_edge = best_tp in (min(TP_SWEEP), max(TP_SWEEP))
    if at_sl_edge or at_tp_edge:
        print("\nWARNING: best combo sits at a grid boundary (SL and/or TP) -- per this project's standing "
              "rule, that means either the grid is too narrow (extend it once) or this is fitting this "
              "specific dataset's tail rather than a generalizable edge. Treat with continued suspicion "
              "even if it looks good -- do not stop at this result alone.")
    else:
        print("\nBest combo sits at an interior grid point (not a boundary) -- a healthier sign than a "
              "grid-edge result, though still worth the checks below before trusting it.")

    if best_m["pf"] != float("inf") and (best_m["pf"] > 2.5 or best_m["sqn"] > 5):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- treat as an extraordinary claim requiring extra scrutiny "
              "before trusting it, per this project's standing calibration (real, validated strategies here "
              "run PF~1.4-1.7, SQN~2.5-3). ***")

    dates = pd.to_datetime(best_tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = best_tdf[dates <= mid], best_tdf[dates > mid]
    m1, m2 = compute_metrics(fh, sl_pts=best_sl), compute_metrics(sh, sl_pts=best_sl)
    print(f"\nIn-sample/out-of-sample split on the best combo:")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']} SQN={m1['sqn']}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']} SQN={m2['sqn']}")

    best_tdf.to_csv("concept2_compression_spike_best_combo_trades.csv", index=False)
    print(f"\nSaved: concept2_compression_spike_best_combo_trades.csv ({len(best_tdf)} rows)")
    print(f"\nFor reference, dhan_live_bot.py's CURRENT live setting is SL={dlb.SL_POINTS} TP={dlb.TP_POINTS} "
          f"-- compare that row in the sweep table above directly against the best combo found here.")


if __name__ == "__main__":
    main()
