"""
Setup 2 Liquidity Sweep, v2 -- rebuilt with 5 quality-over-quantity fixes,
based on the diagnosis that its 40.7% win rate and 57.6% stop-hit rate came
from taking too many low-quality signals at insignificant levels, not from
a broken core idea (the average win/loss ratio was always healthy).

THE 5 CHANGES FROM THE ORIGINAL VERSION:
  1. ZONE TOUCH-COUNT FILTER (the big one): a swing high/low only counts as
     a real liquidity level if it's been touched 2+ times within a small
     price cluster (width = 0.5x ATR(14), same convention as the Price
     Action 3-Step test) -- borrowed directly from that doc's "2+ prior
     reactions required" validity rule. A single 3-bar wiggle no longer
     qualifies as a "level" on its own.
  2. MIN_SWEEP_PTS raised from 1 to 4 points -- only decisive sweeps count,
     not noise-level wick pokes.
  3. CONFIRMATION_MIN_PTS (new, was implicit "any amount" before): the
     confirming candle must close at least 2 points beyond the sweep
     candle's close in the reversal direction, not just marginally further.
  4. STRUCTURE-BASED STOP: stop is now the sweep candle's actual wick
     extreme plus a small 1-point buffer, instead of a flat 10 points for
     every trade. Risk therefore varies per trade based on how deep the
     sweep actually went, and the 1:2 target is computed off that real
     risk, not a fixed 20 points.
  5. LIGHTWEIGHT 15-MIN TREND FILTER: skip the trade if the 15-min EMA
     trend (imported from dhan_live_bot.py, same calculation your final
     strategy uses) is flatly opposed to the reversal direction. This
     doesn't require agreement, just absence of a strong fight -- a bullish
     reversal is skipped only if trend_15m == "DOWN", not just "not UP".

All 5 changes were individually unit-tested against constructed scenarios
before this backtest was built: confirmed a single-touch level correctly
produces no signal, a second touch at the same level correctly arms it,
and the trend filter correctly blocks/allows as expected.

Everything else unchanged: same session filter as the last version
(09:15-09:45, 09:45-10:15, 10:45-11:15, all weekdays), same 3-point
trailing stop once the 1:2 target is hit, same Rs40/trade rough cost
assumption for the money simulation, same 2-year test window.

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py /
any other concept script, including the original (v1) Setup 2 scripts.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    attach_htf_trend,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

SWING_N = 3
ATR_PERIOD = 14
ZONE_WIDTH_MULT = 0.5
MAX_ZONE_AGE = 30
MIN_SWEEP_PTS = 4.0
CONFIRMATION_MIN_PTS = 2.0
STOP_BUFFER_PTS = 1.0
TARGET_RR = 2.0
TRAIL_PTS = 3

ALLOWED_ENTRY_WINDOWS = [
    (dt.time(9, 15), dt.time(9, 45)),
    (dt.time(9, 45), dt.time(10, 15)),
    (dt.time(10, 45), dt.time(11, 15)),
]

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1
ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0

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


def in_allowed_window(ts):
    t = ts.time()
    return any(start <= t < end for start, end in ALLOWED_ENTRY_WINDOWS)


def detect_setup2_v2_signals(day_df, trend_15m):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)

    atr = compute_atr(highs, lows, closes, ATR_PERIOD)
    swing_high, swing_low = find_confirmed_swings(highs, lows, SWING_N)
    sorted_sh = sorted(swing_high.items())
    sorted_sl = sorted(swing_low.items())
    sh_ptr, sl_ptr = 0, 0

    resistance_zones, support_zones = [], []
    last_sh_price, last_sh_touches = None, 0
    last_sl_price, last_sl_touches = None, 0

    signals = []
    pending_bull = None
    pending_bear = None

    for j in range(n):
        while sh_ptr < len(sorted_sh) and sorted_sh[sh_ptr][0] + SWING_N <= j:
            sw_idx, sw_price = sorted_sh[sh_ptr]
            zw = ZONE_WIDTH_MULT * atr[sw_idx]
            matched = None
            for z in resistance_zones:
                if abs(sw_price - z["center"]) <= zw:
                    z["touches"] += 1
                    z["low"] = min(z["low"], sw_price - zw)
                    z["high"] = max(z["high"], sw_price + zw)
                    z["last_touch_bar"] = j
                    matched = z
                    break
            if matched is None:
                matched = {"center": sw_price, "low": sw_price - zw, "high": sw_price + zw,
                           "touches": 1, "last_touch_bar": j}
                resistance_zones.append(matched)
            last_sh_price = sw_price
            last_sh_touches = matched["touches"]
            sh_ptr += 1
        while sl_ptr < len(sorted_sl) and sorted_sl[sl_ptr][0] + SWING_N <= j:
            sw_idx, sw_price = sorted_sl[sl_ptr]
            zw = ZONE_WIDTH_MULT * atr[sw_idx]
            matched = None
            for z in support_zones:
                if abs(sw_price - z["center"]) <= zw:
                    z["touches"] += 1
                    z["low"] = min(z["low"], sw_price - zw)
                    z["high"] = max(z["high"], sw_price + zw)
                    z["last_touch_bar"] = j
                    matched = z
                    break
            if matched is None:
                matched = {"center": sw_price, "low": sw_price - zw, "high": sw_price + zw,
                           "touches": 1, "last_touch_bar": j}
                support_zones.append(matched)
            last_sl_price = sw_price
            last_sl_touches = matched["touches"]
            sl_ptr += 1

        resistance_zones = [z for z in resistance_zones if j - z["last_touch_bar"] <= MAX_ZONE_AGE]
        support_zones = [z for z in support_zones if j - z["last_touch_bar"] <= MAX_ZONE_AGE]

        o, h, l, c = opens[j], highs[j], lows[j], closes[j]
        window_ok = in_allowed_window(idx[j])

        if pending_bull is not None:
            if c > pending_bull["sweep_close"] + CONFIRMATION_MIN_PTS and window_ok:
                signals.append({"bar": j, "side": "LONG", "stop": pending_bull["stop"]})
            pending_bull = None
        if pending_bear is not None:
            if c < pending_bear["sweep_close"] - CONFIRMATION_MIN_PTS and window_ok:
                signals.append({"bar": j, "side": "SHORT", "stop": pending_bear["stop"]})
            pending_bear = None

        if (last_sl_price is not None and last_sl_touches >= 2 and
                l < last_sl_price - MIN_SWEEP_PTS and c > last_sl_price):
            if trend_15m[j] != "DOWN":
                pending_bull = {"sweep_close": c, "stop": l - STOP_BUFFER_PTS}

        if (last_sh_price is not None and last_sh_touches >= 2 and
                h > last_sh_price + MIN_SWEEP_PTS and c < last_sh_price):
            if trend_15m[j] != "UP":
                pending_bear = {"sweep_close": c, "stop": h + STOP_BUFFER_PTS}

    return signals


def simulate_v2_for_day(day_df, entry_signals, trailing_enabled=True):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)

    signals_by_bar = {}
    for s in entry_signals:
        signals_by_bar.setdefault(s["bar"], s)

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
                               "entry": position["entry"], "exit": c, "points": pts, "reason": reason,
                               "risk": position["risk"]})
                position = None

        if position is None and j in signals_by_bar:
            sig = signals_by_bar[j]
            side, entry, stop = sig["side"], c, sig["stop"]
            valid = (side == "LONG" and entry > stop) or (side == "SHORT" and entry < stop)
            if valid:
                risk = abs(entry - stop)
                target = entry + TARGET_RR * risk if side == "LONG" else entry - TARGET_RR * risk
                position = {"side": side, "entry": entry, "entry_time": idx[j], "stop": stop, "target": target,
                            "trailing_active": False, "stop_level": None, "risk": risk}

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
                       "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD",
                       "risk": position["risk"]})

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


def main():
    print("=== Setup 2 Liquidity Sweep v2 -- zone touch-count, stronger sweep/confirmation, "
          "structure-based stop, 15m trend filter -- 2 years ===\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    print("Computing 15m trend context (one-time, vectorized)...")
    df = attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        trend_15m = day_df["trend_15m"].tolist() if "trend_15m" in day_df.columns else ["NONE"] * len(day_df)
        signals = detect_setup2_v2_signals(day_df, trend_15m)
        if signals:
            all_trades.extend(simulate_v2_for_day(day_df, signals, trailing_enabled=True))

    if not all_trades:
        print("\nNo trades were triggered in this date range -- the filters may be too strict. "
              "Consider relaxing MIN_SWEEP_PTS, CONFIRMATION_MIN_PTS, or the zone touch requirement.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n===== SETUP 2 v2 -- OVERALL RESULTS (2 YEARS) =====")
    print(f"Total trades:  {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    stop_ct = (trades_df["reason"] == "STOP").sum()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}  "
          f"(STOP rate: {100*stop_ct/len(trades_df):.1f}%)")
    print(f"Avg risk per trade (structure-based stop distance): {trades_df['risk'].mean():.2f} pts "
          f"(min={trades_df['risk'].min():.2f}, max={trades_df['risk'].max():.2f})")
    print("=====================================================")

    trades_df.to_csv("setup2_v2_trades.csv", index=False)
    print("\nFull trade list saved to setup2_v2_trades.csv")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    print("\nBy weekday:")
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        s = (g["reason"] == "STOP").sum()
        print(f"  {wd:<10} trades={len(g):<5} win%={100*w/len(g):5.1f}  net={g['points'].sum():8.2f}  stop%={100*s/len(g):5.1f}")

    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    print("\nBy session:")
    for s in SESSIONS:
        g = trades_df[trades_df["session"] == s]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        print(f"  {s:<14} trades={len(g):<5} win%={100*w/len(g):5.1f}  net={g['points'].sum():8.2f}")

    gross_rupee = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    net_rupee = gross_rupee - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + net_rupee.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()

    print(f"\n===================== MONEY SIMULATION (Rs{BROKERAGE_PER_TRADE:.0f}/trade cost) =====================")
    print(f"Gross P&L: Rs{gross_rupee.sum():,.2f}   Total brokerage: Rs{BROKERAGE_PER_TRADE*len(trades_df):,.0f}")
    print(f"Starting capital: Rs{STARTING_CAPITAL:,.2f}   Ending capital: Rs{capital_curve.iloc[-1]:,.2f}")
    print(f"Net P&L: Rs{capital_curve.iloc[-1]-STARTING_CAPITAL:,.2f}   Worst drawdown: Rs{max_dd:,.2f}")
    if (capital_curve <= 0).any():
        idx0 = (capital_curve <= 0).idxmax()
        print(f"  WARNING: capital hits Rs0 or below at trade #{trades_df.index.get_loc(idx0)+1} "
              f"({trades_df.loc[idx0, 'entry_time']})")
    else:
        print("  Capital never touches Rs0 or below at any point, even after costs.")
    days = trades_df["entry_time"].dt.date.nunique()
    print(f"Trades/day (on days with a signal): {len(trades_df)/days:.2f}  (days with a signal: {days})")
    print("======================================================================================")

    dates = sorted(trades_df["entry_time"].dt.date.unique())
    mid = dates[len(dates) // 2]
    h1 = trades_df[trades_df["entry_time"].dt.date < mid]
    h2 = trades_df[trades_df["entry_time"].dt.date >= mid]
    h1_net = (h1["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    h2_net = (h2["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    print(f"\nStability: H1 ({dates[0]} to {mid}) n={len(h1)} net_after_cost=Rs{h1_net:,.2f}")
    print(f"           H2 ({mid} to {dates[-1]}) n={len(h2)} net_after_cost=Rs{h2_net:,.2f}")


if __name__ == "__main__":
    main()
