"""
Concepts 2 & 4 -- Initial Stop-Loss Sensitivity Check -- FAST VERSION
=================================================================
Same real dlb.compute_latest_signal() calls as before, weekday-filtered
+ parallelized for speed.

*** CONTEXT-WINDOW FIX (this version) ***
Earlier versions sliced each day in complete isolation (today's bars
only, from bar 0). Your live bot's config includes HTF_CONTEXT_DAYS=5
-- compute_latest_signal likely needs several PRIOR trading days of
history to compute its 5m/15m trend context, and a single isolated day
starting from bar 0 never has that. That's the most likely reason the
last run found zero signals with no errors: the function was probably
returning a valid "no signal" dict every time because it silently
couldn't compute a trend from such a short window, not because nothing
ever happened over 2 years.

Fixed by building a rolling window of HTF_CONTEXT_DAYS trading days
(read from dlb, default 5) ending on each target day, and scanning only
the target day's bars within that larger window -- so
compute_latest_signal always sees the same amount of history it would
have in live trading.

Standalone -- does NOT modify dhan_live_bot.py or any other file.
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

WARMUP_BARS = 30  # bars into the TARGET day before scanning starts
SL_SWEEP = [3, 4, 5, 6, 7, 8, 10, 12]

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


def get_allowed_weekdays():
    wd = getattr(dlb, "ALLOWED_WEEKDAYS", None)
    if wd is None:
        print("  NOTE: dlb.ALLOWED_WEEKDAYS not found -- falling back to {1,2} (Tue/Wed).")
        return {1, 2}
    return set(wd)


def get_htf_context_days():
    cd = getattr(dlb, "HTF_CONTEXT_DAYS", None)
    if cd is None:
        print("  NOTE: dlb.HTF_CONTEXT_DAYS not found -- falling back to 5 (documented default this project).")
        return 5
    return cd


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert extract_signal({"final_buy": True, "final_sell": False, "concept_id": 2}) == ("LONG", 2)
    assert extract_signal({"final_buy": False, "final_sell": True, "concept_id": 4}) == ("SHORT", 4)
    assert extract_signal({"final_buy": False, "final_sell": False, "concept_id": None}) is None
    allowed = get_allowed_weekdays()
    assert isinstance(allowed, set) and len(allowed) > 0
    ctx = get_htf_context_days()
    assert isinstance(ctx, int) and ctx > 0
    print(f"  allowed weekdays: {sorted(allowed)} (0=Mon..4=Fri)   HTF_CONTEXT_DAYS: {ctx}")

    # window-building check
    days = [dt.date(2025, 1, d) for d in range(1, 11)]
    w = build_context_window_days(days, dt.date(2025, 1, 10), 5)
    assert w == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8),
                 dt.date(2025, 1, 9), dt.date(2025, 1, 10)]
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
            "volume": data["volume"], "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    """Returns the list of trading days (context_days of them, ending at
    and including target_day) that should be included in the window."""
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def _scan_one_day(args):
    """Worker function -- runs in a separate process. `args` is
    (target_day, window_df) where window_df spans HTF_CONTEXT_DAYS
    trading days ending on target_day. Only scans bars belonging to
    target_day, but compute_latest_signal sees the full window as
    history. Returns (signals_list, first_error_string_or_None)."""
    target_day, window_df = args
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return [], None
    n = len(window_df)
    first_bar_today = today_positions[0]
    last_bar_today = today_positions[-1]
    if (last_bar_today - first_bar_today) <= WARMUP_BARS + 2:
        return [], None

    found = []
    first_error = None
    open_bar = -1
    for i in today_positions:
        if i - first_bar_today < WARMUP_BARS:
            continue
        if i <= open_bar:
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


def collect_signals_parallel(df, days_covered):
    allowed_weekdays = get_allowed_weekdays()
    context_days = get_htf_context_days()
    all_trading_days_sorted = days_covered  # already sorted
    target_days = [d for d in days_covered if d.weekday() in allowed_weekdays]
    print(f"Filtering to allowed weekdays only: {len(target_days)} of {len(days_covered)} days to scan.")
    print(f"Each day gets a {context_days}-trading-day rolling context window before scanning.")

    jobs = []
    for target_day in target_days:
        window_days = build_context_window_days(all_trading_days_sorted, target_day, context_days)
        window_df = df[(df.index.date >= window_days[0]) & (df.index.date <= window_days[-1])]
        jobs.append((target_day, window_df))

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Scanning {len(target_days)} days across {n_workers} worker processes...")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_scan_one_day, jobs)

    all_signals = []
    errors = []
    for (target_day, window_df), (found, err) in zip(jobs, results):
        day_df = window_df[window_df.index.date == target_day]
        for (i_window, side, concept) in found:
            # translate window-relative index to day-relative index for downstream sim
            i_day = i_window - window_df.index.get_loc(day_df.index[0])
            all_signals.append((target_day, day_df, i_day, side, concept))
        if err:
            errors.append((target_day, err))

    if errors:
        print(f"\n{len(errors)} of {len(target_days)} days hit an error calling compute_latest_signal.")
        print(f"First error (from {errors[0][0]}):\n{errors[0][1]}")
        print("Paste this back to me.\n")

    return all_signals


def compute_mae(day_df, entry_bar, side):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    n = len(day_df)
    if entry_bar + 1 >= n:
        return 0.0
    entry_price = closes[entry_bar]
    if side == "LONG":
        worst_low = min(lows[entry_bar + 1:])
        return max(0.0, entry_price - worst_low)
    else:
        worst_high = max(highs[entry_bar + 1:])
        return max(0.0, worst_high - entry_price)


def simulate_with_sl(day_df, entry_bar, side, sl_pts, tp_pts, max_bars):
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


def summarize(tdf, sl_label):
    if tdf.empty:
        print(f"SL={sl_label}pts: no trades")
        return
    wins = tdf[tdf["points"] > 0]
    losses = tdf[tdf["points"] <= 0]
    win_rate = len(wins) / len(tdf)
    pf = (wins["points"].sum() / abs(losses["points"].sum())
          if losses["points"].sum() != 0 else float("inf"))
    equity = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    running_peak = equity.cummax()
    max_dd = (equity - running_peak).min()
    wipes_out = (equity <= 0).any()
    r_multiple = tdf["rupee_pnl"] / (sl_label * ASSUMED_DELTA * LOT_SIZE)
    sqn = (r_multiple.mean() / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"SL={sl_label:>3}pts | n={len(tdf):>4} | win%={100*win_rate:5.1f} | PF={pf:5.2f} | "
          f"net_Rs={tdf['rupee_pnl'].sum():>12,.2f} | end_Rs={equity.iloc[-1]:>12,.2f} | "
          f"max_dd_Rs={max_dd:>12,.2f} | SQN={sqn:5.2f} | {'WIPES OUT' if wipes_out else 'ok'}")


def main():
    run_self_checks()
    print("=== Concepts 2 & 4 -- Initial Stop-Loss Sensitivity Check (FAST) ===\n")
    print(f"Using dlb.TP_POINTS={dlb.TP_POINTS}, dlb.MAX_BARS_IN_TRADE={dlb.MAX_BARS_IN_TRADE} "
          f"(read directly from your live bot).\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    signals = collect_signals_parallel(df, days_covered)
    if not signals:
        print("Still no signals collected even with proper context windows. If an error was printed "
              "above, paste it back. If not, something else needs checking -- let me know and I'll dig further.")
        return
    print(f"Total signals found: {len(signals)}\n")

    concept_counts = {}
    for _, _, _, _, concept in signals:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1
    print(f"By concept_id: {concept_counts}\n")

    print("--- Raw MAE distribution (how far price moves against us before stop/target/timeout, no SL applied) ---")
    maes = []
    for day, day_df, i, side, concept in signals:
        maes.append(compute_mae(day_df, i, side))
    maes = np.array(maes)
    buckets = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9)]
    for lo, hi in buckets:
        cnt = ((maes >= lo) & (maes < hi)).sum()
        print(f"  MAE in [{lo:>3},{hi if hi < 1e9 else 'inf':>4}) pts: {cnt:>4} signals ({100*cnt/len(maes):5.1f}%)")
    print(f"  Median MAE: {np.median(maes):.2f} pts   Mean MAE: {maes.mean():.2f} pts   "
          f"90th pct: {np.percentile(maes,90):.2f} pts\n")

    print("--- Stop-size sweep (same signals, dlb.TP_POINTS target, dlb.MAX_BARS_IN_TRADE timeout) ---")
    for sl in SL_SWEEP:
        trades = []
        for day, day_df, i, side, concept in signals:
            t = simulate_with_sl(day_df, i, side, sl, dlb.TP_POINTS, dlb.MAX_BARS_IN_TRADE)
            trades.append(t)
        tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
        summarize(tdf, sl)

    print(f"\n(Starting capital assumed: Rs {STARTING_CAPITAL:,}, delta={ASSUMED_DELTA}, "
          f"lot={LOT_SIZE} -- same convention as the original Concepts 2&4 test this project.)")


if __name__ == "__main__":
    main()
