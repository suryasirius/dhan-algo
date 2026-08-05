"""
Smart Money Concepts (SMC) strategy, adapted from the uploaded doc into a
rule-based, automatable form for Nifty 1-min index data. Standalone test --
does NOT touch dhan_live_bot.py / backtest.py / FINAL_*.py / the Strike
Momentum scripts.

HOW THE ORIGINAL SMC RULES WERE ADAPTED (read this before trusting results):
  - Swing highs/lows: fractal detection, N=3 candles either side (the doc's
    own suggested starting value), confirmed with a 3-bar lag (no lookahead).
  - Structure: BOS = close breaks the last known swing point in the current
    trend direction. CHoCH = close breaks it against the current trend
    (first reversal signal). Both are tracked, CHoCH just resets the trend.
  - Liquidity sweep: a candle's wick pokes past the last swing high/low but
    the candle CLOSES back on the original side -- exactly as the doc
    defines it.
  - Order Block: after a sweep is followed (within 40 bars) by a structure
    break in the sweep's reversal direction, we scan backward from the
    breakout for the last opposite-colour candle -- that's the OB zone. If
    none is found, we fall back to a 3-candle Fair Value Gap in the same
    impulsive leg, per the doc's FVG definition.
  - Premium/Discount filter: the OB/FVG zone must sit in the lower half of
    the sweep-to-breakout range for longs (discount), upper half for shorts
    (premium) -- otherwise the setup is discarded, per the doc's rule 6.
  - Entry: price retraces back into the zone and the same candle closes
    back in the trade direction (this collapses the doc's separate "LTF
    confirmation" step onto the same 1-min candle, since we only have one
    timeframe of data here -- documented simplification, not a change of
    intent).
  - Stop loss: beyond the liquidity sweep wick (the doc's first stop option).
  - Target / trail (per your instruction, overriding the doc's "next
    liquidity pool" target): fixed 1:2 risk:reward. Once price closes at or
    beyond the 1:2 target, switch to a candle-based trailing stop of 4
    points (current candle's low/high minus/plus 4pts), which only ever
    moves in the trade's favour, never loosens.
  - Structure/swings/zones are all recomputed FRESH each trading day (no
    carry across days) -- matches how the rest of this project's concepts
    are backtested, and matches real intraday options trading where
    positions are squared off by end of day, not held overnight.
  - Max zone age: an unconfirmed OB/FVG is discarded if not retested within
    30 bars (doc suggests 20-50 as a starting range).

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
MAX_ZONE_AGE = 30
SWEEP_LOOKBACK = 40
TARGET_RR = 2.0
TRAIL_PTS = 4

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


def simulate_day_smc(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)

    swing_high, swing_low = find_confirmed_swings(highs, lows, SWING_N)
    sorted_sh = sorted(swing_high.items())
    sorted_sl = sorted(swing_low.items())
    sh_ptr, sl_ptr = 0, 0
    last_sh_price = None
    last_sl_price = None

    trend = None
    pending_long_zones, pending_short_zones = [], []
    recent_bullish_sweep, recent_bearish_sweep = None, None

    position = None
    trades = []

    for j in range(n):
        while sh_ptr < len(sorted_sh) and sorted_sh[sh_ptr][0] + SWING_N <= j:
            _, last_sh_price = sorted_sh[sh_ptr]
            sh_ptr += 1
        while sl_ptr < len(sorted_sl) and sorted_sl[sl_ptr][0] + SWING_N <= j:
            _, last_sl_price = sorted_sl[sl_ptr]
            sl_ptr += 1

        ts = idx[j]
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
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts, "side": position["side"],
                    "entry": position["entry"], "exit": c, "points": pts, "reason": reason,
                    "setup": position["setup"],
                })
                position = None

        if last_sl_price is not None and l < last_sl_price and c > last_sl_price:
            recent_bullish_sweep = j
        if last_sh_price is not None and h > last_sh_price and c < last_sh_price:
            recent_bearish_sweep = j

        if last_sh_price is not None and c > last_sh_price:
            if recent_bullish_sweep is not None and (j - recent_bullish_sweep) <= SWEEP_LOOKBACK:
                ob = None
                for k in range(j - 1, recent_bullish_sweep - 1, -1):
                    if closes[k] < opens[k]:
                        ob = (min(opens[k], closes[k]), max(opens[k], closes[k]))
                        break
                if ob is None:
                    for k in range(recent_bullish_sweep, j - 2):
                        if highs[k] < lows[k + 2]:
                            ob = (highs[k], lows[k + 2])
                            break
                if ob is not None:
                    anchor_low = lows[recent_bullish_sweep]
                    discount_level = (anchor_low + c) / 2
                    zone_low, zone_high = ob
                    if zone_high <= discount_level:
                        pending_long_zones.append({
                            "zone_low": zone_low, "zone_high": zone_high,
                            "stop_price": anchor_low, "expiry_bar": j + MAX_ZONE_AGE,
                        })
                recent_bullish_sweep = None
            trend = "up"

        if last_sl_price is not None and c < last_sl_price:
            if recent_bearish_sweep is not None and (j - recent_bearish_sweep) <= SWEEP_LOOKBACK:
                ob = None
                for k in range(j - 1, recent_bearish_sweep - 1, -1):
                    if closes[k] > opens[k]:
                        ob = (min(opens[k], closes[k]), max(opens[k], closes[k]))
                        break
                if ob is None:
                    for k in range(recent_bearish_sweep, j - 2):
                        if lows[k] > highs[k + 2]:
                            ob = (highs[k + 2], lows[k])
                            break
                if ob is not None:
                    anchor_high = highs[recent_bearish_sweep]
                    premium_level = (anchor_high + c) / 2
                    zone_low, zone_high = ob
                    if zone_low >= premium_level:
                        pending_short_zones.append({
                            "zone_low": zone_low, "zone_high": zone_high,
                            "stop_price": anchor_high, "expiry_bar": j + MAX_ZONE_AGE,
                        })
                recent_bearish_sweep = None
            trend = "down"

        pending_long_zones = [z for z in pending_long_zones if z["expiry_bar"] >= j]
        pending_short_zones = [z for z in pending_short_zones if z["expiry_bar"] >= j]

        if position is None:
            entered = False
            for z in pending_long_zones:
                if l <= z["zone_high"] and c >= z["zone_low"] and c > z["stop_price"]:
                    entry, stop = c, z["stop_price"]
                    risk = entry - stop
                    position = {"side": "LONG", "entry": entry, "entry_time": ts, "stop": stop,
                                "target": entry + TARGET_RR * risk, "trailing_active": False,
                                "stop_level": None, "setup": "OB/FVG-long"}
                    pending_long_zones = []
                    entered = True
                    break
            if not entered:
                for z in pending_short_zones:
                    if h >= z["zone_low"] and c <= z["zone_high"] and c < z["stop_price"]:
                        entry, stop = c, z["stop_price"]
                        risk = stop - entry
                        position = {"side": "SHORT", "entry": entry, "entry_time": ts, "stop": stop,
                                    "target": entry - TARGET_RR * risk, "trailing_active": False,
                                    "stop_level": None, "setup": "OB/FVG-short"}
                        pending_short_zones = []
                        break

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({
            "entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
            "entry": position["entry"], "exit": c, "points": pts, "reason": "EOD",
            "setup": position["setup"],
        })

    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        start_s, end_s = label.split("-")
        sh, sm = int(start_s[:2]), int(start_s[2:])
        eh, em = int(end_s[:2]), int(end_s[2:])
        start_t = dt.time(sh, sm)
        end_t = dt.time(eh, em)
        if start_t <= t < end_t:
            return label
    return "Outside usual session"


def main():
    print("=== SMC (Smart Money Concepts) strategy -- adapted for Nifty 1-min, standalone, 1 year ===")
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
        all_trades.extend(simulate_day_smc(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n===== SMC STRATEGY -- OVERALL BACKTEST RESULTS (1 YEAR) =====")
    print(f"Total trades:       {len(trades_df)}")
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}")
    print(f"Win rate:           {100 * len(wins) / len(trades_df):.1f}%")
    print(f"Avg win (pts):      {wins['points'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss (pts):     {losses['points'].mean():.2f}" if len(losses) else "Avg loss: n/a")
    print(f"Net points:         {trades_df['points'].sum():.2f}")
    print(f"Exit reasons:       {trades_df['reason'].value_counts().to_dict()}")
    print(f"Setup breakdown:    {trades_df['setup'].value_counts().to_dict()}")
    print("===============================================================")

    trades_df.to_csv("smc_trades.csv", index=False)
    print("\nFull trade list saved to smc_trades.csv")

    # ---- 30-minute session breakdown, split by weekday (same shape as prior reports) ----
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

    # ---- day-by-day summary ----
    print("\n===================== DAY-BY-DAY SUMMARY =====================")
    trades_df["trade_date"] = trades_df["entry_time"].dt.date
    for trade_date, day_group in trades_df.groupby("trade_date"):
        wd = day_group["entry_time"].iloc[0].day_name()
        print(f"{trade_date} ({wd}): {len(day_group)} trade(s), "
              f"net={day_group['points'].sum():.2f}, "
              f"first entry={day_group['entry_time'].iloc[0].strftime('%H:%M:%S')}, "
              f"last exit={day_group['exit_time'].iloc[-1].strftime('%H:%M:%S')}")
    print("================================================================")

    # ---- every trade, with full timing ----
    print("\n===================== EVERY TRADE, WITH FULL TIMING =====================")
    print(f"{'Entry time':<20}{'Exit time':<20}{'Side':<7}{'Weekday':<10}{'Points':>8}  {'Reason'}")
    for _, row in trades_df.iterrows():
        print(f"{str(row['entry_time']):<20}{str(row['exit_time']):<20}{row['side']:<7}"
              f"{row['entry_time'].day_name():<10}{row['points']:>8.2f}  {row['reason']}")
    print("============================================================================")

    # ---- money simulation ----
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
