"""
Concepts 2 & 4 -- Scan-Frequency Validation Test
=================================================================
Tests whether checking dlb.compute_latest_signal() every 5 minutes
instead of every 1 minute gives the SAME signals. If it does, that's a
safe 5x speedup for the full 2-year backtest (fewer calls, same
results). If it doesn't, we keep the slower every-1-minute version --
correctness comes first.

Runs on a SMALL window (last ~60 days) specifically because this is a
validation check, not the final backtest -- fast to run, and any
mismatch will show up clearly on a small sample.

Does NOT modify dhan_live_bot.py or any other file.
"""

import datetime as dt
import traceback
import pandas as pd

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=75)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

WARMUP_BARS = 30


def get_allowed_weekdays():
    wd = getattr(dlb, "ALLOWED_WEEKDAYS", None)
    if wd is None:
        print("  NOTE: dlb.ALLOWED_WEEKDAYS not found -- falling back to {1,2}.")
        return {1, 2}
    return set(wd)


def get_htf_context_days():
    cd = getattr(dlb, "HTF_CONTEXT_DAYS", None)
    if cd is None:
        print("  NOTE: dlb.HTF_CONTEXT_DAYS not found -- falling back to 5.")
        return 5
    return cd


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def c_extract_signal(sig):
    if sig.get("final_buy"):
        return "LONG"
    if sig.get("final_sell"):
        return "SHORT"
    return None


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert c_extract_signal({"final_buy": True, "final_sell": False}) == "LONG"
    assert c_extract_signal({"final_buy": False, "final_sell": False}) is None
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


def scan_day(target_day, window_df, every_n_minutes):
    """Scans one day's bars within window_df. If every_n_minutes > 1,
    only checks bars whose minute-of-day position (since the start of
    today's session) is a multiple of every_n_minutes."""
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return []
    first_bar_today = today_positions[0]
    found = []
    open_bar = -1
    for i in today_positions:
        offset = i - first_bar_today
        if offset < WARMUP_BARS:
            continue
        if i <= open_bar:
            continue
        if every_n_minutes > 1 and (offset % every_n_minutes != 0):
            continue
        try:
            sig = dlb.compute_latest_signal(window_df.iloc[:i + 1])
        except Exception as e:
            print(f"  Error at {window_df.index[i]}: {e}")
            continue
        if sig is None:
            continue
        side = c_extract_signal(sig)
        if side is None:
            continue
        found.append({"time": window_df.index[i], "side": side, "concept": sig.get("concept_id")})
        open_bar = i + dlb.MAX_BARS_IN_TRADE
    return found


def main():
    run_self_checks()
    print("=== Concepts 2 & 4 -- Scan-Frequency Validation (every 1min vs every 5min) ===\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~60 days for a fast check)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    allowed_weekdays = get_allowed_weekdays()
    context_days = get_htf_context_days()
    target_days = [d for d in days_covered if d.weekday() in allowed_weekdays]
    print(f"Scanning {len(target_days)} target day(s) with {context_days}-day context windows.\n")

    every_1min_signals = []
    every_5min_signals = []
    for target_day in target_days:
        window_days = build_context_window_days(days_covered, target_day, context_days)
        window_df = df[(df.index.date >= window_days[0]) & (df.index.date <= window_days[-1])]
        s1 = scan_day(target_day, window_df, every_n_minutes=1)
        s5 = scan_day(target_day, window_df, every_n_minutes=5)
        every_1min_signals.extend(s1)
        every_5min_signals.extend(s5)

    print(f"Every-1-minute scan: {len(every_1min_signals)} signals found.")
    print(f"Every-5-minute scan: {len(every_5min_signals)} signals found.\n")

    if len(every_1min_signals) == len(every_5min_signals):
        mismatches = 0
        for a, b in zip(every_1min_signals, every_5min_signals):
            if a["time"] != b["time"] or a["side"] != b["side"] or a["concept"] != b["concept"]:
                mismatches += 1
        if mismatches == 0:
            print("RESULT: Identical signal count AND identical signals (time/side/concept all match).")
            print("Every-5-minute scanning is SAFE to use -- same results, ~5x fewer calls.")
        else:
            print(f"RESULT: Same count but {mismatches} signal(s) differ in details -- NOT safe to swap in "
                  "blindly. Keep the every-1-minute version.")
    else:
        print(f"RESULT: Different signal counts ({len(every_1min_signals)} vs {len(every_5min_signals)}) "
              "-- every-5-minute scanning MISSES or ADDS signals. NOT safe to use -- keep the "
              "every-1-minute version. The exact 1-minute timing matters for this signal.")

    print("\n--- Every-1-minute signals ---")
    for s in every_1min_signals:
        print(f"  {s['time']}  {s['side']}  concept={s['concept']}")
    print("\n--- Every-5-minute signals ---")
    for s in every_5min_signals:
        print(f"  {s['time']}  {s['side']}  concept={s['concept']}")


if __name__ == "__main__":
    main()
