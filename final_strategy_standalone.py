"""
Final Combined Strategy -- STANDALONE (does NOT touch dhan_live_bot.py)
=================================================================
Pools two already-separately-tested pieces into one combined trade log
and one shared equity curve:

  1. Concepts 2 & 4  -- read from your existing trades CSV
     (concepts_2_4_all_days_trades.csv, or pass a different path as
     CONCEPTS_CSV below). I'm reusing that file directly rather than
     regenerating the detection logic, because my scratchpad reset and
     I no longer have the exact original script's source in front of me
     -- reusing your already-produced, already-reviewed output is safer
     than reconstructing the signal logic from memory and risking a
     subtle mismatch. If you want this regenerated fresh instead (e.g.
     to force the locked Tue/Wed-only live-bot rule), send me that
     script's contents or a fresh CSV from it and I'll swap it in.

  2. Gap Fade LONG-only -- run fresh here (same proven logic as
     gap_fade_long_only.py, which you already ran: 9 trades, 66.7% win,
     +108.05 pts, never wipes out).

Both are independent signal sources on the same underlying -- this
script does NOT change either one's rules, it just pools the resulting
trades chronologically into a single equity curve and reports combined
stability, same convention as the rest of this project.
"""

import datetime as dt
import os
import pandas as pd

import dhan_live_bot as dlb

CONCEPTS_CSV = "concepts_2_4_all_days_trades.csv"   # change path if needed

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

GAP_ATR_MIN_MULT = 0.3
GAP_ATR_MAX_MULT = 2.0
DAILY_ATR_PERIOD = 14
REJECTION_WINDOW_MIN = 15
TARGET_GAP_PCT = 0.15
TRAIL_PTS = 4
STOP_BUFFER_PTS = 1.0
PINBAR_WICK_RATIO = 2.0

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------------------------------------------------------------
# Gap Fade LONG-only detection (identical to gap_fade_long_only.py)
# ---------------------------------------------------------------------
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


def build_daily_bars(df):
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def compute_daily_atr(daily_df, period=DAILY_ATR_PERIOD):
    h, l, c = daily_df["high"].tolist(), daily_df["low"].tolist(), daily_df["close"].tolist()
    trs = [h[0] - l[0]]
    for i in range(1, len(h)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    atr = []
    for i in range(len(trs)):
        w = trs[max(0, i-period+1):i+1]
        atr.append(sum(w)/len(w))
    return atr


def classify_daily_trend(daily_df, i):
    if i < 2:
        return None
    prev = daily_df.iloc[i-1]
    prev2 = daily_df.iloc[i-2]
    if prev["close"] > prev["open"] and prev["high"] > prev2["high"]:
        return "UP"
    if prev["close"] < prev["open"] and prev["low"] < prev2["low"]:
        return "DOWN"
    return "NEUTRAL"


def is_bullish_engulfing(o, c, j):
    if j < 1:
        return False
    return (c[j] > o[j] and c[j-1] < o[j-1] and min(o[j], c[j]) <= min(o[j-1], c[j-1])
            and max(o[j], c[j]) >= max(o[j-1], c[j-1]))


def is_bullish_pinbar(o, h, l, c, j, ratio=PINBAR_WICK_RATIO):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and lw >= ratio*body and lw > uw


def detect_long_signal_for_day(day_df, prev_trend, gap, gap_ok, trend_4h_at_open):
    if prev_trend != "UP" or not gap_ok or gap >= 0:
        return None
    if trend_4h_at_open != "UP":
        return None
    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = min(len(day_df), REJECTION_WINDOW_MIN)
    for j in range(1, n):
        if is_bullish_engulfing(o, c, j) or is_bullish_pinbar(o, h, l, c, j):
            return {"bar": j, "side": "LONG", "rej_low": l[j], "rej_high": h[j]}
    return None


def simulate_gap_trade(day_df, signal, gap_size):
    if signal is None:
        return None
    closes, highs, lows = day_df["close"].tolist(), day_df["high"].tolist(), day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = signal["bar"]
    entry = closes[j0]
    target_pts = abs(gap_size) * TARGET_GAP_PCT
    stop = signal["rej_low"] - STOP_BUFFER_PTS
    target = entry + target_pts

    trailing_active = False
    stop_level = None
    for j in range(j0 + 1, n):
        c, h, l = closes[j], highs[j], lows[j]
        if trailing_active:
            if c <= stop_level:
                return _trade(idx, j0, j, entry, c, "TRAIL")
            stop_level = max(stop_level, l - TRAIL_PTS)
        else:
            if c <= stop:
                return _trade(idx, j0, j, entry, c, "STOP")
            elif c >= target:
                trailing_active = True
                stop_level = l - TRAIL_PTS
    return _trade(idx, j0, n-1, entry, closes[-1], "EOD")


def _trade(idx, j0, j, entry, exitp, reason):
    pts = exitp - entry
    return {"entry_time": idx[j0], "exit_time": idx[j], "side": "LONG", "entry": entry,
            "exit": exitp, "points": pts, "reason": reason, "strategy": "GapFadeLong"}


def run_gap_fade_long(df):
    daily = build_daily_bars(df)
    daily_atr = compute_daily_atr(daily)
    daily_dates = daily.index.date.tolist()

    htf4h = dlb.ema_trend_series(dlb.resample_ohlc(df, "4h")["close"], dlb.TREND_FAST_EMA, dlb.TREND_SLOW_EMA)
    combined_idx = df.index.union(htf4h.index)
    aligned = htf4h.reindex(combined_idx).sort_index().ffill()
    df["trend_4h"] = aligned.reindex(df.index)

    trades = []
    for day in sorted(set(df.index.date)):
        if day not in daily_dates:
            continue
        i = daily_dates.index(day)
        prev_trend = classify_daily_trend(daily, i)
        if prev_trend != "UP" or i == 0:
            continue
        prev_close = daily["close"].iloc[i-1]
        today_open = daily["open"].iloc[i]
        gap = today_open - prev_close
        atr_i = daily_atr[i] if i < len(daily_atr) else None
        if not atr_i or atr_i <= 0:
            continue
        gap_ok = GAP_ATR_MIN_MULT * atr_i <= abs(gap) <= GAP_ATR_MAX_MULT * atr_i

        day_df = df[df.index.date == day]
        if len(day_df) < REJECTION_WINDOW_MIN + 2:
            continue
        trend_4h_at_open = day_df["trend_4h"].iloc[0]

        sig = detect_long_signal_for_day(day_df, prev_trend, gap, gap_ok, trend_4h_at_open)
        if sig is not None:
            t = simulate_gap_trade(day_df, sig, gap)
            if t:
                trades.append(t)
    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


def load_concepts_2_4(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- combined report will only include Gap Fade LONG trades.")
        print("Point CONCEPTS_CSV at your Concepts 2&4 trades file and re-run for the real combined test.\n")
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "points", "reason", "strategy"])
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["strategy"] = "Concepts2n4"
    return df


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    daily = pd.DataFrame({
        "open":  [100, 101, 105, 103],
        "high":  [102, 108, 106, 107],
        "low":   [99,  100, 102, 101],
        "close": [101, 107, 104, 106],
    })
    assert classify_daily_trend(daily, 2) == "UP"
    o = [100, 95]; c = [95, 108]
    assert is_bullish_engulfing(o, c, 1) is True
    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Final Combined Strategy: Concepts 2&4 + Gap Fade LONG-only ===\n")

    print("Loading Concepts 2&4 trades ...")
    c24 = load_concepts_2_4(CONCEPTS_CSV)
    print(f"  {len(c24)} Concepts 2&4 trades loaded.\n")

    print("Fetching index data and running Gap Fade LONG-only fresh ...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    gap_trades = run_gap_fade_long(df)
    gap_df = pd.DataFrame(gap_trades)
    print(f"  {len(gap_df)} Gap Fade LONG trades generated.\n")

    combined = pd.concat([
        c24[["entry_time", "exit_time", "side", "points", "reason", "strategy"]],
        gap_df[["entry_time", "exit_time", "side", "points", "reason", "strategy"]] if len(gap_df) else gap_df,
    ], ignore_index=True)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    combined.to_csv("final_combined_trades.csv", index=False)

    if combined.empty:
        print("No trades in combined set.")
        return

    print("--- Combined totals ---")
    wins = combined[combined["points"] > 0]
    print(f"Total: {len(combined)}  Win%: {100*len(wins)/len(combined):.1f}  Net pts: {combined['points'].sum():.2f}")

    print("\n--- By strategy ---")
    for strat, g in combined.groupby("strategy"):
        w = g[g["points"] > 0]
        print(f"{strat}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*len(w)/len(g):.1f}%")

    combined["weekday"] = combined["entry_time"].dt.day_name()
    print("\n--- By weekday (combined) ---")
    for wd in WEEKDAY_ORDER:
        g = combined[combined["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    combined["session"] = combined["entry_time"].apply(classify_30min_session)
    print("\n--- By 30-min session (combined) ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = combined[combined["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    rupee_pnl = combined["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    print(f"\n--- Combined money simulation ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {capital_curve.iloc[-1]:,.2f}   Max DD Rs {max_dd:,.2f}")
    print("WARNING: wipes out at some point." if (capital_curve <= 0).any() else "Never wipes out.")

    mid = len(combined)//2
    h1 = rupee_pnl.iloc[:mid].sum(); h2 = rupee_pnl.iloc[mid:].sum()
    print(f"\n--- Combined stability check ---")
    print(f"First half: {mid} trades, net Rs {h1:,.2f}")
    print(f"Second half: {len(combined)-mid} trades, net Rs {h2:,.2f}")
    print("\nNOTE: Concepts 2&4 trades are reused from your existing CSV, not regenerated here.")
    print("If that CSV used different weekday rules than the current locked live-bot setting,")
    print("this combined result reflects THAT ruleset, not necessarily today's dhan_live_bot.py config.")


if __name__ == "__main__":
    main()
