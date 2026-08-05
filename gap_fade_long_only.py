"""
Overnight Gap Fade -- LONG ONLY, narrowed per your decision
=================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.

This is Variant A (against-trend fade) from gap_fade_strategy_backtest.py,
narrowed to ONLY its LONG case, which showed a consistent edge across two
independent samples (1yr: 8 trades/75% win/+128pts, 2yr: 9 trades/66.7%
win/+108pts) while the SHORT case was a consistent loser both times
(1yr: -35pts, 2yr: -75pts). Everything else (gap-size ATR filter,
rejection-candle trigger, 4H trend confirmation, 15%-of-gap target then
4-point trail, structure-based initial stop) is UNCHANGED from the
already-tested Variant A logic -- only the SHORT case is removed.

The LONG case fires when: yesterday was a confirmed UP day (closed above
its own open AND made a higher high than the day before), today gaps DOWN
against that trend, the gap is a "medium/standard" size (0.3x-2.0x the
14-day daily ATR, not an outlier), the 4-hour trend reads UP at the open,
and a bullish rejection candle (engulfing or pinbar) appears within the
first 15 minutes of trading.

2 years of data, same as the last full test, so results are directly
comparable to the combined (LONG+SHORT) Variant A numbers already seen.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

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


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    daily = pd.DataFrame({
        "open":  [100, 101, 105, 103],
        "high":  [102, 108, 106, 107],
        "low":   [99,  100, 102, 101],
        "close": [101, 107, 104, 106],
    })
    assert classify_daily_trend(daily, 2) == "UP"
    o = [100, 95]; c = [95, 108]  # bar1 bullish engulfing bar0's bearish body
    assert is_bullish_engulfing(o, c, 1) is True
    print("All self-checks passed.\n")


def detect_long_signal_for_day(day_df, prev_trend, gap, gap_ok, trend_4h_at_open):
    """Only the LONG case: prev day UP, gap DOWN, 4H trend UP, bullish rejection."""
    if prev_trend != "UP" or not gap_ok or gap >= 0:
        return None  # need a genuine gap DOWN against an UP prior day
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
            "exit": exitp, "points": pts, "reason": reason}


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


def main():
    run_self_checks()
    print("=== Overnight Gap Fade -- LONG ONLY -- 2 years ===\n")

    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned.")
        return

    daily = build_daily_bars(df)
    daily_atr = compute_daily_atr(daily)
    daily_dates = daily.index.date.tolist()

    htf4h = dlb.ema_trend_series(dlb.resample_ohlc(df, "4h")["close"], dlb.TREND_FAST_EMA, dlb.TREND_SLOW_EMA)
    combined_idx = df.index.union(htf4h.index)
    aligned = htf4h.reindex(combined_idx).sort_index().ffill()
    df["trend_4h"] = aligned.reindex(df.index)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    trades = []
    for day in days_covered:
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

    if not trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("gap_fade_long_only_trades.csv", index=False)

    wins = tdf[tdf["points"] > 0]
    print(f"Total: {len(tdf)}  Win%: {100*len(wins)/len(tdf):.1f}  Net pts: {tdf['points'].sum():.2f}")
    print(f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")

    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    print("\n--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    print("\n--- By 30-min session ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    rupee_pnl = tdf["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    print(f"\n--- Money simulation ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {capital_curve.iloc[-1]:,.2f}   Max DD Rs {max_dd:,.2f}")
    print("WARNING: wipes out at some point." if (capital_curve <= 0).any() else "Never wipes out.")

    mid = len(tdf)//2
    h1 = rupee_pnl.iloc[:mid].sum(); h2 = rupee_pnl.iloc[mid:].sum()
    print(f"\n--- Stability check ---")
    print(f"First half: {mid} trades, net Rs {h1:,.2f}")
    print(f"Second half: {len(tdf)-mid} trades, net Rs {h2:,.2f}")
    if len(tdf) < 20:
        print("NOTE: sample is still thin -- treat this as a promising direction, not a validated result.")


if __name__ == "__main__":
    main()
