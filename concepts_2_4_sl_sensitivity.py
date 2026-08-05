"""
Concepts 2 & 4 -- Initial Stop-Loss Sensitivity Check
=================================================================
Same idea as velez_sl_sensitivity.py, applied to Concepts 2 & 4.

Uses the REAL dlb.compute_latest_signal() from your live bot (via the
same expanding-window pattern used earlier this project) -- not a
reconstruction of the detection logic. Confirmed the actual return
shape by running against your live bot:

    {'time': Timestamp, 'close': float, 'bull_votes': int, 'bear_votes': int,
     'final_buy': bool, 'final_sell': bool, 'concept_id': int or None,
     'trend_5m': str or None, 'trend_15m': str or None}

It always returns this dict (never None) -- a signal only counts when
final_buy or final_sell is True. concept_id is None when no signal.
The weekday restriction (Tue/Wed) and disabled-concepts filtering
already built into compute_latest_signal apply automatically here.

Two things reported, same format as the Velez sweep:

1. RAW MAE DISTRIBUTION -- for every signal, before any stop is
   applied, how far did price move against it before hitting target,
   timing out, or end of day?

2. STOP-SIZE SWEEP -- the same real signals re-simulated with several
   different initial stop sizes (3/4/5/6/7/8/10/12 points), holding
   dlb.TP_POINTS and dlb.MAX_BARS_IN_TRADE fixed at whatever they
   currently are in your live bot (read directly from dlb, not
   hardcoded).

Standalone -- does NOT modify dhan_live_bot.py or any other file.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

WARMUP_BARS = 30  # bars into each day before we start calling compute_latest_signal

SL_SWEEP = [3, 4, 5, 6, 7, 8, 10, 12]

STARTING_CAPITAL = 10_000
ASSUMED_DELTA = 0.6
LOT_SIZE = 65
BROKERAGE_PER_TRADE = 40.0


def extract_signal(sig):
    """Returns (side, concept) if this bar has a real signal, else None."""
    if sig.get("final_buy"):
        return "LONG", sig.get("concept_id")
    if sig.get("final_sell"):
        return "SHORT", sig.get("concept_id")
    return None


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert extract_signal({"final_buy": True, "final_sell": False, "concept_id": 2}) == ("LONG", 2)
    assert extract_signal({"final_buy": False, "final_sell": True, "concept_id": 4}) == ("SHORT", 4)
    assert extract_signal({"final_buy": False, "final_sell": False, "concept_id": None}) is None
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


def collect_signals(df, days_covered):
    all_signals = []
    error_shown = False
    for day in days_covered:
        day_df = df[df.index.date == day]
        n = len(day_df)
        if n <= WARMUP_BARS + 2:
            continue
        open_trade_bar = -1
        for i in range(WARMUP_BARS, n):
            if i <= open_trade_bar:
                continue
            try:
                sig = dlb.compute_latest_signal(day_df.iloc[:i + 1])
            except Exception as e:
                if not error_shown:
                    print(f"compute_latest_signal raised an error at bar {i} on {day}: {e}")
                    error_shown = True
                continue
            if sig is None:
                continue
            result = extract_signal(sig)
            if result is None:
                continue
            side, concept = result
            all_signals.append((day, day_df, i, side, concept))
            open_trade_bar = i + dlb.MAX_BARS_IN_TRADE
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
    print("=== Concepts 2 & 4 -- Initial Stop-Loss Sensitivity Check ===\n")
    print(f"Using dlb.TP_POINTS={dlb.TP_POINTS}, dlb.MAX_BARS_IN_TRADE={dlb.MAX_BARS_IN_TRADE} "
          f"(read directly from your live bot).\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    print("Scanning for signals via dlb.compute_latest_signal (this may take a while)...")
    signals = collect_signals(df, days_covered)
    if not signals:
        print("No signals collected -- either none fired, or an error was printed above. Stopping.")
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
