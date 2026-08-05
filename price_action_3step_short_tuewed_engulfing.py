"""
Narrowed version of the Price Action 3-Step (Structure -> Zone -> Trigger)
concept, based on the full standalone backtest results (755 trades, 1 year):
  - SHORT-only: SHORT trades were net +384.25 while LONG was net -571.65
    (similar win rates, but LONG's losers were heavier relative to its
    winners).
  - Tuesday/Wednesday only: the two strongest weekdays (+276.95 and +237.65)
    vs Friday's -475.85 and Thursday's -214.60.
  - Engulfing trigger only: Engulfing was ~breakeven (+6.30 net) while Pin
    Bar was the drag (-193.70 net, worse win rate, and 2/3 of all trades).

All three filters combined here to see if the combination clears the
strategy into positive territory, or whether the sample gets too thin to
trust. Same 1:2 R:R + 4-pt candle trailing stop after target, same zone/
structure logic as the full version. Standalone -- does not touch
dhan_live_bot.py / backtest.py / FINAL_*.py / the full price_action_3step
script or any other concept script.
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

# ---- Narrowing filters, based on the full backtest's breakdown ----
ALLOWED_WEEKDAYS = {1, 2}  # Tuesday=1, Wednesday=2
ALLOWED_SIDE = "SHORT"     # only take the resistance-zone / bearish-trigger side
ALLOWED_TRIGGER = "Engulfing"  # drop Pin Bar entries entirely

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


def is_bearish_engulfing(opens, closes, j):
    if j < 1:
        return False
    return (closes[j] < opens[j] and closes[j - 1] > opens[j - 1] and
            min(opens[j], closes[j]) <= min(opens[j - 1], closes[j - 1]) and
            max(opens[j], closes[j]) >= max(opens[j - 1], closes[j - 1]))


def simulate_day_pa3_short_engulfing(day_df):
    """SHORT-only, Engulfing-trigger-only version of the day simulator.
    (LONG-side / support-zone / pin-bar logic intentionally removed --
    this file only ever looks for bearish resistance-zone engulfing setups.)"""
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
    resistance_zones = []
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
            sl_ptr += 1

        if len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2:
            if confirmed_highs[-1] > confirmed_highs[-2] and confirmed_lows[-1] > confirmed_lows[-2]:
                trend = "up"
            elif confirmed_highs[-1] < confirmed_highs[-2] and confirmed_lows[-1] < confirmed_lows[-2]:
                trend = "down"
            else:
                trend = "range"

        resistance_zones = [z for z in resistance_zones if j - z["last_touch_bar"] <= MAX_ZONE_AGE]

        o, h, l, c = opens[j], highs[j], lows[j], closes[j]

        if position is not None:
            exited, reason = False, None
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
                pts = position["entry"] - c
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j], "side": "SHORT",
                               "entry": position["entry"], "exit": c, "points": pts, "reason": reason,
                               "trigger": "Engulfing"})
                position = None

        if position is None and j >= 1 and trend == "down":
            for z in resistance_zones:
                if z["touches"] >= 2 and h >= z["low"] and l <= z["high"]:
                    if is_bearish_engulfing(opens, closes, j):
                        entry = c
                        stop = max(h, z["high"])
                        if entry < stop:
                            risk = stop - entry
                            position = {"entry": entry, "entry_time": idx[j], "stop": stop,
                                        "target": entry - TARGET_RR * risk, "trailing_active": False,
                                        "stop_level": None}
                        break

    if position is not None:
        c = closes[-1]
        pts = position["entry"] - c
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1], "side": "SHORT",
                       "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD",
                       "trigger": "Engulfing"})

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
    print("=== Price Action 3-Step -- NARROWED: SHORT-only + Tue/Wed-only + Engulfing-only, 1 year ===\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    allowed_days = [d for d in days_covered if d.weekday() in ALLOWED_WEEKDAYS]
    print(f"Of those, {len(allowed_days)} are Tuesday/Wednesday.")

    all_trades = []
    for day in allowed_days:
        day_df = df[df.index.date == day]
        if len(day_df) < 2 * SWING_N + 5:
            continue
        all_trades.extend(simulate_day_pa3_short_engulfing(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n===== NARROWED PRICE ACTION 3-STEP -- RESULTS (SHORT + TUE/WED + ENGULFING, 1 YEAR) =====")
    print(f"Total trades:       {len(trades_df)}")
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}")
    print(f"Win rate:           {100 * len(wins) / len(trades_df):.1f}%")
    print(f"Avg win (pts):      {wins['points'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss (pts):     {losses['points'].mean():.2f}" if len(losses) else "Avg loss: n/a")
    print(f"Net points:         {trades_df['points'].sum():.2f}")
    print(f"Exit reasons:       {trades_df['reason'].value_counts().to_dict()}")
    print("===========================================================================================")

    trades_df.to_csv("price_action_3step_short_tuewed_engulfing_trades.csv", index=False)
    print("\nFull trade list saved to price_action_3step_short_tuewed_engulfing_trades.csv")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)

    print("\n===================== 30-MINUTE SESSION BREAKDOWN =====================")
    print(f"{'Session':<16}{'Trades':>8}{'Wins':>8}{'Win %':>9}{'Net Pts':>12}{'Avg Pts':>10}")
    for session in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == session]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        print(f"{session:<16}{len(g):>8}{w:>8}{100*w/len(g):>8.1f}%{g['points'].sum():>12.2f}{g['points'].mean():>10.2f}")
    print("=========================================================================")

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
    print(f"{'Entry time':<20}{'Exit time':<20}{'Weekday':<10}{'Points':>8}  {'Reason'}")
    for _, row in trades_df.iterrows():
        print(f"{str(row['entry_time']):<20}{str(row['exit_time']):<20}"
              f"{row['entry_time'].day_name():<10}{row['points']:>8.2f}  {row['reason']}")
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
