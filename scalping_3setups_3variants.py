"""
3 Core Scalping Setups (same doc as before: Trend Pullback Continuation,
Reversal/Liquidity Sweep, Breakout & Retest) -- re-tested with 3 trade
management categories instead of 2, per your latest instruction:

  1. Fixed_1_1:        10-pt stop, 10-pt target (1:1), flat exit at target.
  2. Fixed_1_2:         10-pt stop, 20-pt target (1:2), flat exit at target.
  3. Trail_after_1_5:   10-pt stop, 15-pt target (1:1.5), then switch to a
                        3-pt candle trailing stop instead of closing flat.

Same 10-point initial stop across all 3 categories, so this is an
apples-to-apples comparison of "what to do once a trade is open" rather
than a comparison of different risk sizes. That's 3 setups x 3 variants =
9 backtests in this one run.

Entry logic for all 3 setups is UNCHANGED from the original 3-setup test
(scalping_3setups_backtest.py) -- Setup 1 uses the same 5m/15m trend filter
plus <=50% pullback plus engulfing/pin-bar trigger; Setup 2 uses the same
sweep (>=1pt past a swing point, closes back inside) plus next-candle
confirmation; Setup 3 uses the same 8+ bar/1xATR consolidation, strong
breakout, and retest-within-20-bars logic. (Note: this is the ORIGINAL
Setup 2 definition, not the zone-touch-count v2 refinement built afterward
for Setup 2 alone -- this run is specifically about testing new trade
management on the original 3-setup framework.)

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py /
any other concept script, including the Setup 2 v1/v2 refinement scripts.

NOTE ON OUTPUT: with 9 backtests in one run, per-trade timing tables and
day-by-day logs are left out of the printed output to keep it readable --
each of the 9 result sets is saved to its own CSV with full entry/exit
timing.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    attach_htf_trend,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

SWING_N = 3
ATR_PERIOD = 14
PULLBACK_MAX_RETRACE = 0.5
MIN_SWEEP_PTS = 1.0
CONSOLIDATION_BARS = 8
CONSOLIDATION_ATR_MULT = 1.0
BREAKOUT_BODY_ATR_MULT = 0.3
RETEST_MAX_AGE = 20
PINBAR_WICK_RATIO = 2.0

INITIAL_STOP_PTS = 10
TRAIL_PTS = 3

# (variant name, target points, trailing after target?)
MANAGEMENT_VARIANTS = [
    ("Fixed_1_1", 10, False),
    ("Fixed_1_2", 20, False),
    ("Trail_after_1_5", 15, True),
]

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1
ASSUMED_DELTA = 0.6

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def fetch_historical(from_date, to_date):
    print(f"Requesting Nifty 1-min candles from {from_date} to {to_date}...")
    resp = dhan.intraday_minute_data(
        security_id=UNDERLYING_SECURITY_ID,
        exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=UNDERLYING_INSTRUMENT_TYPE,
        from_date=from_date,
        to_date=to_date,
    )
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print("\nDhan returned an error:")
        print(resp)
        raise SystemExit(1)
    data = resp.get("data", resp)
    df = pd.DataFrame({
        "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "timestamp": data["timestamp"],
    })
    df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    return df.set_index("time").sort_index()


def fetch_historical_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        chunks.append(fetch_historical(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = []
    for i in range(len(trs)):
        window = trs[max(0, i - period + 1):i + 1]
        atr.append(sum(window) / len(window))
    return atr


def find_confirmed_swings(highs, lows, n=SWING_N):
    length = len(highs)
    swing_high, swing_low = {}, {}
    for i in range(n, length - n):
        window_h = highs[i - n:i + n + 1]
        if highs[i] == max(window_h):
            swing_high[i] = highs[i]
        window_l = lows[i - n:i + n + 1]
        if lows[i] == min(window_l):
            swing_low[i] = lows[i]
    return swing_high, swing_low


def is_bullish_engulfing(opens, closes, j):
    if j < 1:
        return False
    return (closes[j] > opens[j] and closes[j - 1] < opens[j - 1] and
            min(opens[j], closes[j]) <= min(opens[j - 1], closes[j - 1]) and
            max(opens[j], closes[j]) >= max(opens[j - 1], closes[j - 1]))


def is_bearish_engulfing(opens, closes, j):
    if j < 1:
        return False
    return (closes[j] < opens[j] and closes[j - 1] > opens[j - 1] and
            min(opens[j], closes[j]) <= min(opens[j - 1], closes[j - 1]) and
            max(opens[j], closes[j]) >= max(opens[j - 1], closes[j - 1]))


def is_bullish_pinbar(opens, highs, lows, closes, j):
    o, h, l, c = opens[j], highs[j], lows[j], closes[j]
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body <= 0:
        return False
    return lower_wick >= PINBAR_WICK_RATIO * body and lower_wick > upper_wick


def is_bearish_pinbar(opens, highs, lows, closes, j):
    o, h, l, c = opens[j], highs[j], lows[j], closes[j]
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body <= 0:
        return False
    return upper_wick >= PINBAR_WICK_RATIO * body and upper_wick > lower_wick


# ---------------- Setup 1: Trend Pullback Continuation ----------------
def detect_setup1_signals(day_df, trend_5m, trend_15m):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    n = len(day_df)
    swing_high, swing_low = find_confirmed_swings(highs, lows, SWING_N)
    sorted_sh = sorted(swing_high.items())
    sorted_sl = sorted(swing_low.items())
    sh_ptr, sl_ptr = 0, 0
    confirmed_highs, confirmed_lows = [], []

    signals = []
    up_leg, down_leg = None, None
    up_armed, down_armed = False, False
    up_pullback_low, down_pullback_high = None, None

    for j in range(n):
        while sh_ptr < len(sorted_sh) and sorted_sh[sh_ptr][0] + SWING_N <= j:
            _, sw_price = sorted_sh[sh_ptr]
            confirmed_highs.append(sw_price)
            if confirmed_lows:
                up_leg = {"low": confirmed_lows[-1], "high": sw_price}
                up_armed = True
                up_pullback_low = None
            sh_ptr += 1
        while sl_ptr < len(sorted_sl) and sorted_sl[sl_ptr][0] + SWING_N <= j:
            _, sw_price = sorted_sl[sl_ptr]
            confirmed_lows.append(sw_price)
            if confirmed_highs:
                down_leg = {"high": confirmed_highs[-1], "low": sw_price}
                down_armed = True
                down_pullback_high = None
            sl_ptr += 1

        trend_ok_up = trend_5m[j] == "UP" and trend_15m[j] == "UP"
        trend_ok_down = trend_5m[j] == "DOWN" and trend_15m[j] == "DOWN"

        o, h, l, c = opens[j], highs[j], lows[j], closes[j]

        if up_armed and up_leg and trend_ok_up:
            leg_range = up_leg["high"] - up_leg["low"]
            if leg_range > 0:
                if l < up_leg["low"]:
                    up_armed = False
                else:
                    if up_pullback_low is None or l < up_pullback_low:
                        up_pullback_low = l
                    retrace = (up_leg["high"] - up_pullback_low) / leg_range
                    if retrace <= PULLBACK_MAX_RETRACE and c < up_leg["high"]:
                        if is_bullish_engulfing(opens, closes, j) or is_bullish_pinbar(opens, highs, lows, closes, j):
                            signals.append({"bar": j, "side": "LONG"})
                            up_armed = False

        if down_armed and down_leg and trend_ok_down:
            leg_range = down_leg["high"] - down_leg["low"]
            if leg_range > 0:
                if h > down_leg["high"]:
                    down_armed = False
                else:
                    if down_pullback_high is None or h > down_pullback_high:
                        down_pullback_high = h
                    retrace = (down_pullback_high - down_leg["low"]) / leg_range
                    if retrace <= PULLBACK_MAX_RETRACE and c > down_leg["low"]:
                        if is_bearish_engulfing(opens, closes, j) or is_bearish_pinbar(opens, highs, lows, closes, j):
                            signals.append({"bar": j, "side": "SHORT"})
                            down_armed = False

    return signals


# ---------------- Setup 2: Reversal / Liquidity Sweep (original v1 definition) ----------------
def detect_setup2_signals(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    n = len(day_df)
    swing_high, swing_low = find_confirmed_swings(highs, lows, SWING_N)
    sorted_sh = sorted(swing_high.items())
    sorted_sl = sorted(swing_low.items())
    sh_ptr, sl_ptr = 0, 0
    last_sh_price, last_sl_price = None, None

    signals = []
    pending_bull_sweep_close = None
    pending_bear_sweep_close = None

    for j in range(n):
        while sh_ptr < len(sorted_sh) and sorted_sh[sh_ptr][0] + SWING_N <= j:
            _, last_sh_price = sorted_sh[sh_ptr]
            sh_ptr += 1
        while sl_ptr < len(sorted_sl) and sorted_sl[sl_ptr][0] + SWING_N <= j:
            _, last_sl_price = sorted_sl[sl_ptr]
            sl_ptr += 1

        o, h, l, c = opens[j], highs[j], lows[j], closes[j]

        if pending_bull_sweep_close is not None:
            if c > pending_bull_sweep_close:
                signals.append({"bar": j, "side": "LONG"})
            pending_bull_sweep_close = None
        if pending_bear_sweep_close is not None:
            if c < pending_bear_sweep_close:
                signals.append({"bar": j, "side": "SHORT"})
            pending_bear_sweep_close = None

        if last_sl_price is not None and l < last_sl_price - MIN_SWEEP_PTS and c > last_sl_price:
            pending_bull_sweep_close = c
        if last_sh_price is not None and h > last_sh_price + MIN_SWEEP_PTS and c < last_sh_price:
            pending_bear_sweep_close = c

    return signals


# ---------------- Setup 3: Breakout and Retest ----------------
def detect_setup3_signals(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    n = len(day_df)
    atr = compute_atr(highs, lows, closes, ATR_PERIOD)

    signals = []
    pending_breakout = None

    for j in range(n):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]

        if j >= CONSOLIDATION_BARS:
            window_h = highs[j - CONSOLIDATION_BARS:j]
            window_l = lows[j - CONSOLIDATION_BARS:j]
            cons_high, cons_low = max(window_h), min(window_l)
            in_consolidation = (cons_high - cons_low) <= CONSOLIDATION_ATR_MULT * atr[j]
            body = abs(c - o)
            if in_consolidation and pending_breakout is None:
                if c > cons_high and body > BREAKOUT_BODY_ATR_MULT * atr[j]:
                    pending_breakout = {"level": cons_high, "dir": "up", "bar": j}
                elif c < cons_low and body > BREAKOUT_BODY_ATR_MULT * atr[j]:
                    pending_breakout = {"level": cons_low, "dir": "down", "bar": j}

        if pending_breakout is not None:
            if j - pending_breakout["bar"] > RETEST_MAX_AGE:
                pending_breakout = None
            else:
                if pending_breakout["dir"] == "up":
                    if l <= pending_breakout["level"] and c > pending_breakout["level"]:
                        signals.append({"bar": j, "side": "LONG"})
                        pending_breakout = None
                else:
                    if h >= pending_breakout["level"] and c < pending_breakout["level"]:
                        signals.append({"bar": j, "side": "SHORT"})
                        pending_breakout = None

    return signals


# ---------------- Shared exit management: 10-pt stop, configurable target + trailing ----------------
def simulate_with_signals(day_df, entry_signals, target_pts, trailing_enabled):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)

    signals_by_bar = {}
    for s in entry_signals:
        signals_by_bar.setdefault(s["bar"], s["side"])

    position = None
    trades = []

    for j in range(n):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]

        if position is not None:
            exited, reason = False, None
            if position["side"] == "LONG":
                if position["trailing_active"]:
                    if c <= position["stop_level"]:
                        exited, reason = True, "TRAIL"
                    else:
                        position["stop_level"] = max(position["stop_level"], l - TRAIL_PTS)
                else:
                    if c <= position["stop"]:
                        exited, reason = True, "STOP"
                    elif c >= position["target"]:
                        if trailing_enabled:
                            position["trailing_active"] = True
                            position["stop_level"] = l - TRAIL_PTS
                        else:
                            exited, reason = True, "TARGET"
            else:
                if position["trailing_active"]:
                    if c >= position["stop_level"]:
                        exited, reason = True, "TRAIL"
                    else:
                        position["stop_level"] = min(position["stop_level"], h + TRAIL_PTS)
                else:
                    if c >= position["stop"]:
                        exited, reason = True, "STOP"
                    elif c <= position["target"]:
                        if trailing_enabled:
                            position["trailing_active"] = True
                            position["stop_level"] = h + TRAIL_PTS
                        else:
                            exited, reason = True, "TARGET"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j], "side": position["side"],
                               "entry": position["entry"], "exit": c, "points": pts, "reason": reason})
                position = None

        if position is None and j in signals_by_bar:
            side = signals_by_bar[j]
            entry = c
            if side == "LONG":
                stop = entry - INITIAL_STOP_PTS
                target = entry + target_pts
            else:
                stop = entry + INITIAL_STOP_PTS
                target = entry - target_pts
            position = {"side": side, "entry": entry, "entry_time": idx[j], "stop": stop, "target": target,
                        "trailing_active": False, "stop_level": None}

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
                       "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD"})

    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        start_s, end_s = label.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        if dt.time(sh, sm) <= t < dt.time(eh, em):
            return label
    return "Outside usual session"


def report(trades, label, filename):
    if not trades:
        print(f"\n--- {label}: no trades. ---")
        return None
    trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print(f"\n===== {label} =====")
    print(f"Total trades:  {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}")

    trades_df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    parts = []
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g) == 0:
            continue
        parts.append(f"{wd[:3]}={len(g)}/net{g['points'].sum():.0f}")
    print("By weekday:", ", ".join(parts))

    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    sess_stats = []
    for s in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == s]
        if len(g) == 0:
            continue
        sess_stats.append((s, len(g), g["points"].sum()))
    sess_stats.sort(key=lambda x: x[2])
    if sess_stats:
        print(f"Worst session: {sess_stats[0]}   Best session: {sess_stats[-1]}")

    rupee_pnl = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    final_capital = capital_curve.iloc[-1]
    print(f"Money sim: start Rs{STARTING_CAPITAL:,} -> end Rs{final_capital:,.2f}  "
          f"(P&L Rs{final_capital-STARTING_CAPITAL:,.2f})  worst drawdown Rs{max_dd:,.2f}")
    if (capital_curve <= 0).any():
        print("  WARNING: this variant wipes out the starting capital at some point.")

    return {
        "label": label, "trades": len(trades_df), "win_rate": 100*len(wins)/len(trades_df),
        "net_points": trades_df["points"].sum(), "final_capital": final_capital,
        "max_dd": max_dd,
    }


def main():
    print("=== 3 Core Scalping Setups -- 3 management categories (1:1, 1:2, trail-after-1.5), 1 year ===")
    print(f"All variants share a fixed {INITIAL_STOP_PTS}-pt initial stop; only the target/trailing differ.\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    print("Computing 5m/15m trend context for Setup 1 (one-time, vectorized)...")
    df = attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    setup1_signals_by_day, setup2_signals_by_day, setup3_signals_by_day = {}, {}, {}
    day_dfs = {}
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < max(2 * SWING_N + 5, CONSOLIDATION_BARS + 5):
            continue
        day_dfs[day] = day_df
        trend_5m = day_df["trend_5m"].tolist() if "trend_5m" in day_df.columns else ["NONE"] * len(day_df)
        trend_15m = day_df["trend_15m"].tolist() if "trend_15m" in day_df.columns else ["NONE"] * len(day_df)
        setup1_signals_by_day[day] = detect_setup1_signals(day_df, trend_5m, trend_15m)
        setup2_signals_by_day[day] = detect_setup2_signals(day_df)
        setup3_signals_by_day[day] = detect_setup3_signals(day_df)

    results = []
    setups = [
        ("Setup1_TrendPullback", setup1_signals_by_day),
        ("Setup2_LiquiditySweep", setup2_signals_by_day),
        ("Setup3_BreakoutRetest", setup3_signals_by_day),
    ]
    for setup_name, signals_by_day in setups:
        for variant_name, target_pts, trailing_enabled in MANAGEMENT_VARIANTS:
            all_trades = []
            for day, day_df in day_dfs.items():
                signals = signals_by_day.get(day, [])
                if not signals:
                    continue
                all_trades.extend(simulate_with_signals(day_df, signals, target_pts, trailing_enabled))
            label = f"{setup_name} | {variant_name}"
            filename = f"scalp3v_{setup_name}_{variant_name}_trades.csv"
            res = report(all_trades, label, filename)
            if res:
                results.append(res)

    print("\n\n===================== SUMMARY -- ALL 9 BACKTESTS =====================")
    print(f"{'Setup / Variant':<42}{'Trades':>8}{'Win %':>9}{'Net Pts':>10}{'End Rs':>12}{'Max DD Rs':>12}")
    for r in results:
        print(f"{r['label']:<42}{r['trades']:>8}{r['win_rate']:>8.1f}%{r['net_points']:>10.2f}"
              f"{r['final_capital']:>12,.2f}{r['max_dd']:>12,.2f}")
    print("=======================================================================================")


if __name__ == "__main__":
    main()
