"""
Overnight Gap Strategy -- Nifty 1-min, TWO variants tested separately
=========================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.
No OI involved, no other concept mixed in.

2-YEAR VERSION -- same exact logic as the 1-year run (which showed Variant A
promising but thin at 13 trades, Variant B clearly rejected), just extended
to gather more sample size before deciding on any narrowing.

Your idea, translated into rules:
  - PREVIOUS DAY TREND: a day counts as "UP" if it closed above its own
    open AND its high was higher than the day before that (a genuine
    higher-high, not just a green candle). "DOWN" is the mirror (closed
    below its own open AND a lower low than the day before). Anything
    else = NEUTRAL (no signal either variant).
  - GAP: today's open minus yesterday's close. Direction is just whether
    that's positive (gap up) or negative (gap down).
  - GAP SIZE FILTER ("medium/standard... not extremely big"): the gap must
    be between 0.3x and 2.0x the 14-day ATR of the DAILY candles -- big
    enough to be a real gap, not so big it's an outlier/news-gap that
    behaves differently. Gaps outside this range are skipped entirely.
  - TWO VARIANTS (tested separately, not combined):
      Variant A -- AGAINST the trend: previous day UP but gap DOWN today
      (or previous day DOWN but gap UP) -- bet the gap fades back,
      re-asserting the prior trend.
      Variant B -- WITH the trend: previous day UP and gap UP too (or
      previous day DOWN and gap DOWN too) -- bet on an initial dip/bounce
      right after the open that then resumes in the gap's own direction.
  - CONFIRMATION TRIGGER: within the first REJECTION_WINDOW_MIN minutes
    after the open, look for an actual rejection candle (engulfing/pinbar)
    in the expected direction -- this is what actually fires the entry.
  - HIGHER-TIMEFRAME CONFIRMATION: the 4-HOUR trend (EMA9/21) must also
    read the SAME direction as the trade at entry. Uses plain wall-clock
    4-hour resampling (NSE hours don't divide evenly into 4h blocks) --
    flagged in case it needs adjusting later.
  - TARGET: 15% of the day's gap size in points. Once hit, switches to a
    4-point trailing stop to let it run further if the move continues.
  - INITIAL STOP: the rejection candle's own extreme point (small buffer
    beyond its high/low) -- a structure-based stop, my own design choice
    since none was specified for the initial risk.
  - Same-day only -- exits at market close if neither target nor stop hit.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years, per request
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

GAP_ATR_MIN_MULT = 0.3
GAP_ATR_MAX_MULT = 2.0
DAILY_ATR_PERIOD = 14
REJECTION_WINDOW_MIN = 15  # first 15 minutes after open to find the confirming candle
TARGET_GAP_PCT = 0.15      # 15% of the gap as the target
TRAIL_PTS = 4              # per your instruction, trailing after target hit
STOP_BUFFER_PTS = 1.0      # small buffer beyond the rejection candle's extreme
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
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return daily


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
    """True higher-high/lower-low confirmation, not just a green/red candle."""
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


def is_bearish_engulfing(o, c, j):
    if j < 1:
        return False
    return (c[j] < o[j] and c[j-1] > o[j-1] and min(o[j], c[j]) <= min(o[j-1], c[j-1])
            and max(o[j], c[j]) >= max(o[j-1], c[j-1]))


def is_bullish_pinbar(o, h, l, c, j, ratio=PINBAR_WICK_RATIO):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and lw >= ratio*body and lw > uw


def is_bearish_pinbar(o, h, l, c, j, ratio=PINBAR_WICK_RATIO):
    body = abs(c[j]-o[j])
    lw = min(o[j], c[j]) - l[j]
    uw = h[j] - max(o[j], c[j])
    return body > 0 and uw >= ratio*body and uw > lw


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    daily = pd.DataFrame({
        "open":  [100, 101, 105, 103],
        "high":  [102, 108, 106, 107],
        "low":   [99,  100, 102, 101],
        "close": [101, 107, 104, 106],
    })
    assert classify_daily_trend(daily, 2) == "UP", "day1 close>open and high>day0 high -> UP"
    daily2 = pd.DataFrame({
        "open":  [100, 101, 95, 97],
        "high":  [101, 102, 96, 98],
        "low":   [98,  95,  90, 92],
        "close": [99,  95,  91, 90],
    })
    assert classify_daily_trend(daily2, 2) == "DOWN"

    o = [100, 108]; c = [108, 100]
    assert is_bearish_engulfing(o, c, 1) is True
    assert is_bullish_engulfing(o, c, 1) is False
    print("All self-checks passed.\n")


def detect_gap_signals_for_day(day_df, prev_trend, gap, gap_ok, trend_4h_at_open):
    """Returns (variantA_signal_or_None, variantB_signal_or_None) -- at most
    one entry per variant per day, found within the first REJECTION_WINDOW_MIN
    minutes."""
    if prev_trend not in ("UP", "DOWN") or not gap_ok or gap == 0:
        return None, None
    gap_dir = "UP" if gap > 0 else "DOWN"

    expected_a = None  # against-trend fade
    if prev_trend == "UP" and gap_dir == "DOWN":
        expected_a = "LONG"
    elif prev_trend == "DOWN" and gap_dir == "UP":
        expected_a = "SHORT"

    expected_b = None  # with-trend continuation
    if prev_trend == "UP" and gap_dir == "UP":
        expected_b = "LONG"
    elif prev_trend == "DOWN" and gap_dir == "DOWN":
        expected_b = "SHORT"

    o, h, l, c = day_df["open"].tolist(), day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    n = min(len(day_df), REJECTION_WINDOW_MIN)

    # trend_4h reads "UP"/"DOWN" (from ema_trend_series), but expected_a/expected_b are
    # "LONG"/"SHORT" -- map before comparing, otherwise these never match.
    side_to_trend_word = {"LONG": "UP", "SHORT": "DOWN"}

    sig_a, sig_b = None, None
    for j in range(1, n):
        if sig_a is None and expected_a is not None and trend_4h_at_open == side_to_trend_word[expected_a]:
            if expected_a == "LONG" and (is_bullish_engulfing(o, c, j) or is_bullish_pinbar(o, h, l, c, j)):
                sig_a = {"bar": j, "side": "LONG", "rej_low": l[j], "rej_high": h[j]}
            elif expected_a == "SHORT" and (is_bearish_engulfing(o, c, j) or is_bearish_pinbar(o, h, l, c, j)):
                sig_a = {"bar": j, "side": "SHORT", "rej_low": l[j], "rej_high": h[j]}
        if sig_b is None and expected_b is not None and trend_4h_at_open == side_to_trend_word[expected_b]:
            if expected_b == "LONG" and (is_bullish_engulfing(o, c, j) or is_bullish_pinbar(o, h, l, c, j)):
                sig_b = {"bar": j, "side": "LONG", "rej_low": l[j], "rej_high": h[j]}
            elif expected_b == "SHORT" and (is_bearish_engulfing(o, c, j) or is_bearish_pinbar(o, h, l, c, j)):
                sig_b = {"bar": j, "side": "SHORT", "rej_low": l[j], "rej_high": h[j]}
    return sig_a, sig_b


def simulate_gap_trade(day_df, signal, gap_size):
    if signal is None:
        return None
    closes, highs, lows = day_df["close"].tolist(), day_df["high"].tolist(), day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    j0 = signal["bar"]
    side = signal["side"]
    entry = closes[j0]
    target_pts = abs(gap_size) * TARGET_GAP_PCT
    if side == "LONG":
        stop = signal["rej_low"] - STOP_BUFFER_PTS
        target = entry + target_pts
    else:
        stop = signal["rej_high"] + STOP_BUFFER_PTS
        target = entry - target_pts

    trailing_active = False
    stop_level = None
    for j in range(j0 + 1, n):
        c, h, l = closes[j], highs[j], lows[j]
        if side == "LONG":
            if trailing_active:
                if c <= stop_level:
                    return _trade(idx, j0, j, side, entry, c, "TRAIL")
                stop_level = max(stop_level, l - TRAIL_PTS)
            else:
                if c <= stop:
                    return _trade(idx, j0, j, side, entry, c, "STOP")
                elif c >= target:
                    trailing_active = True
                    stop_level = l - TRAIL_PTS
        else:
            if trailing_active:
                if c >= stop_level:
                    return _trade(idx, j0, j, side, entry, c, "TRAIL")
                stop_level = min(stop_level, h + TRAIL_PTS)
            else:
                if c >= stop:
                    return _trade(idx, j0, j, side, entry, c, "STOP")
                elif c <= target:
                    trailing_active = True
                    stop_level = h + TRAIL_PTS
    return _trade(idx, j0, n-1, side, entry, closes[-1], "EOD")


def _trade(idx, j0, j, side, entry, exitp, reason):
    pts = (exitp - entry) if side == "LONG" else (entry - exitp)
    return {"entry_time": idx[j0], "exit_time": idx[j], "side": side, "entry": entry,
            "exit": exitp, "points": pts, "reason": reason}


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


def report(trades, label, filename):
    if not trades:
        print(f"\n--- {label}: no trades. ---")
        return None
    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    wins = tdf[tdf["points"] > 0]
    print(f"\n===== {label} =====")
    print(f"Total: {len(tdf)}  Win%: {100*len(wins)/len(tdf):.1f}  Net pts: {tdf['points'].sum():.2f}  "
          f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")
    for side in ["LONG", "SHORT"]:
        g = tdf[tdf["side"] == side]
        if len(g):
            print(f"  {side}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")
    tdf.to_csv(filename, index=False)
    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    parts = [f"{wd[:3]}={len(tdf[tdf['weekday']==wd])}/net{tdf[tdf['weekday']==wd]['points'].sum():.0f}"
             for wd in WEEKDAY_ORDER if len(tdf[tdf["weekday"] == wd])]
    print("  By weekday:", ", ".join(parts))
    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    sess = [(s, len(tdf[tdf["session"]==s]), tdf[tdf["session"]==s]["points"].sum())
            for s in SESSIONS+["Outside usual session"] if len(tdf[tdf["session"]==s])]
    sess.sort(key=lambda x: x[2])
    if sess:
        print(f"  Worst session: {sess[0]}   Best session: {sess[-1]}")
    rupee_pnl = tdf["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    print(f"  Money sim: Rs {STARTING_CAPITAL:,} -> Rs {capital_curve.iloc[-1]:,.2f}  (max DD Rs {max_dd:,.2f})",
          " WIPES OUT" if (capital_curve <= 0).any() else "")
    mid = len(tdf)//2
    h1 = rupee_pnl.iloc[:mid].sum(); h2 = rupee_pnl.iloc[mid:].sum()
    print(f"  Stability: H1(n={mid}) Rs{h1:,.2f}  H2(n={len(tdf)-mid}) Rs{h2:,.2f}"
          f"{'  -- THIN SAMPLE' if len(tdf)<20 else ''}")
    return {"label": label, "trades": len(tdf), "win_rate": 100*len(wins)/len(tdf),
            "net_points": tdf["points"].sum(), "final_capital": capital_curve.iloc[-1], "max_dd": max_dd}


def main():
    run_self_checks()
    print("=== Overnight Gap Strategy -- Variant A (against-trend fade) vs Variant B (with-trend) -- 2 YEARS ===\n")

    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned.")
        return

    daily = build_daily_bars(df)
    daily_atr = compute_daily_atr(daily)
    daily_dates = daily.index.date.tolist()

    df["trend_4h"] = None
    htf4h = dlb.ema_trend_series(dlb.resample_ohlc(df, "4h")["close"], dlb.TREND_FAST_EMA, dlb.TREND_SLOW_EMA)
    combined_idx = df.index.union(htf4h.index)
    aligned = htf4h.reindex(combined_idx).sort_index().ffill()
    df["trend_4h"] = aligned.reindex(df.index)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    trades_a, trades_b = [], []
    gap_fill_records = []  # every qualifying gap day, regardless of whether the entry trigger fired
    for day in days_covered:
        if day not in daily_dates:
            continue
        i = daily_dates.index(day)
        prev_trend = classify_daily_trend(daily, i)
        if prev_trend not in ("UP", "DOWN") or i == 0:
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

        if gap_ok and gap != 0:
            # How much of the gap actually got closed at ANY point in the day (using the
            # day's real high/low, not just the close) -- independent of whether our
            # rejection-candle trigger happened to fire. gap_dir=="UP" means the gap
            # closes if price moves DOWN from the open; "DOWN" means it closes moving UP.
            gap_dir = "UP" if gap > 0 else "DOWN"
            day_open_actual = day_df["open"].iloc[0]
            if gap_dir == "UP":
                fill_pts = day_open_actual - day_df["low"].min()
            else:
                fill_pts = day_df["high"].max() - day_open_actual
            fill_pct = max(0.0, fill_pts) / abs(gap)
            variant_type = "AGAINST_TREND" if (
                (prev_trend == "UP" and gap_dir == "DOWN") or (prev_trend == "DOWN" and gap_dir == "UP")
            ) else "WITH_TREND"
            gap_fill_records.append({"day": day, "gap": gap, "fill_pct": fill_pct, "variant_type": variant_type})

        sig_a, sig_b = detect_gap_signals_for_day(day_df, prev_trend, gap, gap_ok, trend_4h_at_open)
        if sig_a is not None:
            t = simulate_gap_trade(day_df, sig_a, gap)
            if t:
                trades_a.append(t)
        if sig_b is not None:
            t = simulate_gap_trade(day_df, sig_b, gap)
            if t:
                trades_b.append(t)

    res_a = report(trades_a, "Variant A -- AGAINST trend (gap fade)", "gap_fade_variantA_2yr_trades.csv")
    res_b = report(trades_b, "Variant B -- WITH trend (continuation)", "gap_fade_variantB_2yr_trades.csv")

    print("\n\n===================== GAP FILL STATISTICS (asked for) =====================")
    print("How often the gap actually got closed by at least X%, at ANY point during the day --")
    print("using the day's real high/low, independent of whether our entry trigger fired.\n")
    gf = pd.DataFrame(gap_fill_records)
    if gf.empty:
        print("No qualifying gap days found.")
    else:
        thresholds = [0.10, 0.15, 0.20, 0.25]
        for scope_name, scope_df in [("ALL qualifying gap days", gf),
                                      ("AGAINST-TREND gap days only (Variant A's gaps)",
                                       gf[gf["variant_type"] == "AGAINST_TREND"]),
                                      ("WITH-TREND gap days only (Variant B's gaps)",
                                       gf[gf["variant_type"] == "WITH_TREND"])]:
            n = len(scope_df)
            print(f"-- {scope_name} (n={n}) --")
            if n == 0:
                print("  (none)")
                continue
            for th in thresholds:
                pct_reaching = 100 * (scope_df["fill_pct"] >= th).mean()
                print(f"  Reached >= {int(th*100)}% gap fill: {pct_reaching:.1f}% of days "
                      f"({int((scope_df['fill_pct']>=th).sum())}/{n})")
            print(f"  Average fill: {100*scope_df['fill_pct'].mean():.1f}%   "
                  f"Median fill: {100*scope_df['fill_pct'].median():.1f}%")
            print()

    print("\n===================== SUMMARY =====================")
    print(f"{'Variant':<32}{'Trades':>8}{'Win%':>7}{'NetPts':>9}{'End Rs':>11}{'MaxDD Rs':>11}")
    for r in (res_a, res_b):
        if r:
            print(f"{r['label']:<32}{r['trades']:>8}{r['win_rate']:>6.1f}%{r['net_points']:>9.2f}"
                  f"{r['final_capital']:>11,.2f}{r['max_dd']:>11,.2f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
