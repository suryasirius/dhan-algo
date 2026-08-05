"""
Concepts 2 & 4 -- Which window theory is right?
=================================================================
Zero signals across 200 days over 2 years contradicts the known
baseline: your original 6-month test found 86 signals, so 2 years
should show roughly 300+. That means the "needs 5 days of prior
context" theory (applied in the *_fast.py scripts) is probably WRONG --
it may have broken something rather than fixed it.

This script tests BOTH theories side by side on a small, fast window
(last ~60 days) and reports which one actually finds a reasonable
number of signals:

  THEORY A (single-day only): day_df = just today's bars, exactly like
  every other backtest script in this project (Gap Fade, ORB, Velez).
  This is what the ORIGINAL 6-month test almost certainly did.

  THEORY B (5-day context window): today's bars plus the prior 4
  trading days included in the slice passed to compute_latest_signal.
  This is what I guessed based on the documented HTF_CONTEXT_DAYS=5
  config -- but that guess may be wrong about HOW that context gets
  used, or compute_latest_signal may not handle multi-day input the
  way I assumed.

Whichever theory produces a believable signal count on this small
window is the one we should use for the full 2-year run.

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


def scan_single_day_only(df, target_day):
    """THEORY A: today's bars only, exactly like every other script."""
    day_df = df[df.index.date == target_day]
    n = len(day_df)
    if n <= WARMUP_BARS + 2:
        return [], None
    closes = day_df.index
    found = []
    first_error = None
    open_bar = -1
    for i in range(WARMUP_BARS, n):
        if i <= open_bar:
            continue
        try:
            sig = dlb.compute_latest_signal(day_df.iloc[:i + 1])
        except Exception:
            if first_error is None:
                first_error = traceback.format_exc()
            continue
        if sig is None:
            continue
        side = c_extract_signal(sig)
        if side is None:
            continue
        found.append({"time": day_df.index[i], "side": side, "concept": sig.get("concept_id")})
        open_bar = i + dlb.MAX_BARS_IN_TRADE
    return found, first_error


def scan_with_context_window(df, days_covered, target_day, context_days):
    """THEORY B: prior context_days trading days included in the slice."""
    window_days = build_context_window_days(days_covered, target_day, context_days)
    window_df = df[(df.index.date >= window_days[0]) & (df.index.date <= window_days[-1])]
    today_positions = [i for i, d in enumerate(window_df.index.date) if d == target_day]
    if not today_positions:
        return [], None
    first_bar_today = today_positions[0]
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
        side = c_extract_signal(sig)
        if side is None:
            continue
        found.append({"time": window_df.index[i], "side": side, "concept": sig.get("concept_id")})
        open_bar = i + dlb.MAX_BARS_IN_TRADE
    return found, first_error


def main():
    run_self_checks()
    print("=== Concepts 2 & 4 -- Window Theory Test (Theory A: single-day vs Theory B: 5-day context) ===\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~60 days for a fast check)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    allowed_weekdays = get_allowed_weekdays()
    context_days = get_htf_context_days()
    target_days = [d for d in days_covered if d.weekday() in allowed_weekdays]
    print(f"Scanning {len(target_days)} target day(s).\n")

    theory_a_signals = []
    theory_a_errors = []
    theory_b_signals = []
    theory_b_errors = []

    for target_day in target_days:
        sa, ea = scan_single_day_only(df, target_day)
        theory_a_signals.extend(sa)
        if ea:
            theory_a_errors.append((target_day, ea))

        sb, eb = scan_with_context_window(df, days_covered, target_day, context_days)
        theory_b_signals.extend(sb)
        if eb:
            theory_b_errors.append((target_day, eb))

    print(f"THEORY A (single-day only): {len(theory_a_signals)} signals found, "
          f"{len(theory_a_errors)} days hit errors.")
    if theory_a_errors:
        print(f"  First error (from {theory_a_errors[0][0]}):\n{theory_a_errors[0][1]}")
    print()

    print(f"THEORY B (5-day context window): {len(theory_b_signals)} signals found, "
          f"{len(theory_b_errors)} days hit errors.")
    if theory_b_errors:
        print(f"  First error (from {theory_b_errors[0][0]}):\n{theory_b_errors[0][1]}")
    print()

    print("--- Theory A signals ---")
    for s in theory_a_signals:
        print(f"  {s['time']}  {s['side']}  concept={s['concept']}")
    print("\n--- Theory B signals ---")
    for s in theory_b_signals:
        print(f"  {s['time']}  {s['side']}  concept={s['concept']}")

    print("\n--- Verdict ---")
    print("Your original 6-month test found 86 signals -- roughly 1 signal every ~7 trading days on "
          "Tue/Wed. Over this ~60-day window with ~17 Tue/Wed days, a believable count is somewhere "
          "around 2-4 signals. Whichever theory lands in a believable range (and especially whichever "
          "one is non-zero) is the one to use for the full 2-year run.")


if __name__ == "__main__":
    main()
