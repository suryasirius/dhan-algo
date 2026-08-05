"""
Price Action Trading -- 3-Step Strategy (Structure -> Zone -> Trigger),
adapted from the uploaded doc into a rule-based, automatable form for Nifty
1-min index data. Standalone test -- does NOT touch dhan_live_bot.py /
backtest.py / FINAL_*.py / the Strike Momentum or SMC scripts.

HOW THE DOC'S RULES WERE ADAPTED (read this before trusting results):
  - Structure (trend filter): swing highs/lows via 3-bar fractal detection
    (the doc's own suggested starting value), confirmed with a 3-bar lag (no
    lookahead). Trend = "up" once the latest confirmed swing high AND swing
    low are BOTH higher than the previous ones (a real HH+HL), "down" when
    both are lower (LH+LL), otherwise "range" -- and no new trades are taken
    in a range, per the doc's "missing any one of the 3 = no trade" rule.
  - Zones (level filter): built from confirmed swing highs (resistance) and
    swing lows (support). Zone width = 0.5 x 14-period ATR, exactly as the
    doc specifies. New swings within an existing zone's width get merged
    into it and increment its "touches" counter; a zone only becomes usable
    once touches >= 2, per the doc's "2+ prior reactions" validity rule.
    Zones are discarded if untouched for 30 bars (doc suggests 20-50).
  - Trigger (entry filter): only two of the doc's three trigger types were
    implemented as objective, codeable rules -- Engulfing (candle body
    fully swallows the prior candle's body, opposite colour) and Pin Bar
    (wick >= 2x body, per the doc's own ratio). "Break-and-retest" was left
    out because it overlaps heavily with what the SMC script already tests
    and is harder to define without double-counting -- documented gap, not
    a silent omission.
  - Entry: on the close of the trigger candle, only when price is
    overlapping a valid (2+ touch) zone in the direction of the confirmed
    trend (support zone for longs, resistance zone for shorts).
  - Stop loss: just beyond the zone/wick (the lower of the trigger candle's
    low and the zone's lower edge, for longs -- mirrored for shorts), per
    the doc's rule 5.
  - Target / trail: the doc specifies "next structure level" OR a minimum
    1:2 R:R as the starting point. To keep management consistent with the
    other concepts tested in this project, this script uses the fixed 1:2
    R:R as the target, and once price closes at or beyond that level,
    switches to the same 4-point candle trailing stop used in the SMC test
    (only ever tightens, never loosens) instead of taking profit outright --
    flagging this clearly in case you wanted a hard exit at 1:2 instead.
  - Structure/zones/trend are all recomputed FRESH each trading day (no
    carry across days), matching how every other concept in this project is
    backtested and matching real intraday options trading (no overnight
    positions).

Money simulation uses the same approximation as the rest of this project:
real historical option premiums for expired weekly contracts aren't
retrievable, so index point moves are scaled by an assumed delta (0.6) and
the current Nifty lot size (65) to estimate rupee P&L. Not real premium data.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE

FROM_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

SWING_N = 3
ATR_PERIOD = 14
ZONE_WIDTH_MULT = 0.5
MAX_ZONE_AGE = 30
TARGET_RR = 2.0
TRAIL_PTS = 4
PINBAR_WICK_RATIO = 2.0

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


def simulate_day_pa3(day_df):
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

    confirmed_highs, confirmed_lows = [], []
    resistance_zones, support_zones = [], []
    trend = None

    position = None
    trades = []

    for j in range(n):
        while sh_ptr < len(sorted_sh) and sorted_sh[sh_ptr][0] + SWING_N <= j:
            sw_idx, sw_price = sorted_sh[sh_ptr]
            confirmed_highs.append(sw_price)
            zw = ZONE_WIDTH_MULT * atr[sw_idx]
            matched = False
            for z in resistance_zones:
                if abs(sw_price - z["center"]) <= zw:
                    z["touches"] += 1
                    z["low"] = min(z["low"], sw_price - zw)
                    z["high"] = max(z["high"], sw_price + zw)
                    z["last_touch_bar"] = j
                    matched = True
                    break
            if not matched:
                resistance_zones.append({"center": sw_price, "low": sw_price - zw, "high": sw_price + zw,
                                          "touches": 1, "last_touch_bar": j})
            sh_ptr += 1
        while sl_ptr < len(sorted_sl) and sorted_sl[sl_ptr][0] + SWING_N <= j:
            sw_idx, sw_price = sorted_sl[sl_ptr]
            confirmed_lows.append(sw_price)
            zw = ZONE_WIDTH_MULT * atr[sw_idx]
            matched = False
            for z in support_zones:
                if abs(sw_price - z["center"]) <= zw:
                    z["touches"] += 1
                    z["low"] = min(z["low"], sw_price - zw)
                    z["high"] = max(z["high"], sw_price + zw)
                    z["last_touch_bar"] = j
                    matched = True
                    break
            if not matched:
                support_zones.append({"center": sw_price, "low": sw_price - zw, "high": sw_price + zw,
                                       "touches": 1, "last_touch_bar": j})
            sl_ptr += 1

        if len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2:
            if confirmed_highs[-1] > confirmed_highs[-2] and confirmed_lows[-1] > confirmed_lows[-2]:
                trend = "up"
            elif confirmed_highs[-1] < confirmed_highs[-2] and confirmed_lows[-1] < confirmed_lows[-2]:
                trend = "down"
            else:
                trend = "range"

        resistance_zones = [z for z in resistance_zones if j - z["last_touch_bar"] <= MAX_ZONE_AGE]
        support_zones = [z for z in support_zones if j - z["last_touch_bar"] <= MAX_ZONE_AGE]

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
                        position["trailing_active"] = True
                        position["stop_level"] = l - TRAIL_PTS
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
                        position["trailing_active"] = True
                        position["stop_level"] = h + TRAIL_PTS
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j], "side": position["side"],
                               "entry": position["entry"], "exit": c, "points": pts, "reason": reason,
                               "trigger": position["trigger"]})
                position = None

        if position is None and j >= 1:
            if trend == "up":
                for z in support_zones:
                    if z["touches"] >= 2 and l <= z["high"] and h >= z["low"]:
                        trig = None
                        if is_bullish_engulfing(opens, closes, j):
                            trig = "Engulfing"
                        elif is_bullish_pinbar(opens, highs, lows, closes, j):
                            trig = "PinBar"
                        if trig:
                            entry = c
                            stop = min(l, z["low"])
                            if entry > stop:
                                risk = entry - stop
                                position = {"side": "LONG", "entry": entry, "entry_time": idx[j], "stop": stop,
                                            "target": entry + TARGET_RR * risk, "trailing_active": False,
                                            "stop_level": None, "trigger": trig}
                            break
            elif trend == "down":
                for z in resistance_zones:
                    if z["touches"] >= 2 and h >= z["low"] and l <= z["high"]:
                        trig = None
                        if is_bearish_engulfing(opens, closes, j):
                            trig = "Engulfing"
                        elif is_bearish_pinbar(opens, highs, lows, closes, j):
                            trig = "PinBar"
                        if trig:
                            entry = c
                            stop = max(h, z["high"])
                            if entry < stop:
                                risk = stop - entry
                                position = {"side": "SHORT", "entry": entry, "entry_time": idx[j], "stop": stop,
                                            "target": entry - TARGET_RR * risk, "trailing_active": False,
                                            "stop_level": None, "trigger": trig}
                            break

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
                       "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD",
                       "trigger": position["trigger"]})

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
    print("=== Price Action 3-Step (Structure -> Zone -> Trigger) -- adapted for Nifty 1-min, standalone, 1 year ===")
    print(f"Fixed 1:{int(TARGET_RR)} R:R, then {TRAIL_PTS}-pt candle trailing stop after target is hit.\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 2 * SWING_N + 5:
            continue
        all_trades.extend(simulate_day_pa3(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n===== PRICE ACTION 3-STEP -- OVERALL BACKTEST RESULTS (1 YEAR) =====")
    print(f"Total trades:       {len(trades_df)}")
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}")
    print(f"Win rate:           {100 * len(wins) / len(trades_df):.1f}%")
    print(f"Avg win (pts):      {wins['points'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss (pts):     {losses['points'].mean():.2f}" if len(losses) else "Avg loss: n/a")
    print(f"Net points:         {trades_df['points'].sum():.2f}")
    print(f"Exit reasons:       {trades_df['reason'].value_counts().to_dict()}")
    print(f"Trigger breakdown:  {trades_df['trigger'].value_counts().to_dict()}")
    print(f"Side breakdown:     {trades_df['side'].value_counts().to_dict()}")
    print("=======================================================================")

    trades_df.to_csv("price_action_3step_trades.csv", index=False)
    print("\nFull trade list saved to price_action_3step_trades.csv")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)

    print("\n===================== 30-MINUTE SESSION BREAKDOWN (ALL DAYS) =====================")
    print(f"{'Session':<16}{'Trades':>8}{'Wins':>8}{'Win %':>9}{'Net Pts':>12}{'Avg Pts':>10}")
    for session in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == session]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        print(f"{session:<16}{len(g):>8}{w:>8}{100*w/len(g):>8.1f}%{g['points'].sum():>12.2f}{g['points'].mean():>10.2f}")
    print("====================================================================================")

    print("\n===================== 30-MINUTE SESSION BREAKDOWN, BY WEEKDAY =====================")
    for wd in WEEKDAY_ORDER:
        wd_df = trades_df[trades_df["weekday"] == wd]
        if wd_df.empty:
            continue
        print(f"\n-- {wd} ({len(wd_df)} trades, net {wd_df['points'].sum():.2f}) --")
        print(f"{'Session':<16}{'Trades':>8}{'Wins':>8}{'Win %':>9}{'Net Pts':>12}")
        for session in SESSIONS + ["Outside usual session"]:
            g = wd_df[wd_df["session"] == session]
            if len(g) == 0:
                continue
            w = (g["points"] > 0).sum()
            print(f"{session:<16}{len(g):>8}{w:>8}{100*w/len(g):>8.1f}%{g['points'].sum():>12.2f}")
    print("\n=====================================================================================")

    print("\n===================== DAY-BY-DAY SUMMARY =====================")
    trades_df["trade_date"] = trades_df["entry_time"].dt.date
    for trade_date, day_group in trades_df.groupby("trade_date"):
        wd = day_group["entry_time"].iloc[0].day_name()
        print(f"{trade_date} ({wd}): {len(day_group)} trade(s), "
              f"net={day_group['points'].sum():.2f}, "
              f"first entry={day_group['entry_time'].iloc[0].strftime('%H:%M:%S')}, "
              f"last exit={day_group['exit_time'].iloc[-1].strftime('%H:%M:%S')}")
    print("================================================================")

    print("\n===================== EVERY TRADE, WITH FULL TIMING =====================")
    print(f"{'Entry time':<20}{'Exit time':<20}{'Side':<7}{'Weekday':<10}{'Points':>8}  {'Trigger':<10}{'Reason'}")
    for _, row in trades_df.iterrows():
        print(f"{str(row['entry_time']):<20}{str(row['exit_time']):<20}{row['side']:<7}"
              f"{row['entry_time'].day_name():<10}{row['points']:>8.2f}  {row['trigger']:<10}{row['reason']}")
    print("============================================================================")

    rupee_pnl = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    final_capital = capital_curve.iloc[-1]

    print("\n===================== MONEY SIMULATION (APPROXIMATE) =====================")
    print("NOTE: index points scaled by an assumed 0.6 delta and current lot size --")
    print("this is NOT real historical option premium data.")
    print(f"Starting capital:     Rs {STARTING_CAPITAL:,.2f}")
    print(f"Lot size used:        {LOT_SIZE}, {LOTS_PER_TRADE} lot/trade")
    print(f"Ending capital:       Rs {final_capital:,.2f}")
    print(f"Total P&L:            Rs {final_capital - STARTING_CAPITAL:,.2f}")
    print(f"Return on capital:    {100 * (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL:.1f}%")
    print(f"Worst drawdown:       Rs {max_dd:,.2f}")
    if (capital_curve <= 0).any():
        wipeout_trade = int((capital_curve <= 0).idxmax()) + 1
        print(f"WARNING: capital hits Rs 0 or below around trade #{wipeout_trade} of {len(trades_df)}.")
    print("============================================================================")


if __name__ == "__main__":
    main()
