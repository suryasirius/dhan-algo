"""
Setup 2 (Reversal / Liquidity Sweep) ONLY, refined -- Setup 1 and Setup 3 are
dropped entirely per your instruction (Setup 1 showed a real capital wipeout
despite a positive end balance, and Setup 3 barely fired at all). This is
the strongest performer from the 3-setup scalping test, so it gets two
refinements before you trust it further:

  1. FULL TRADE-BY-TRADE CAPITAL CURVE -- not just start/end capital like
     before, but the running balance after every single trade, so we can see
     exactly where the deepest drawdown happens and confirm there's no
     hidden wipeout (Setup 2 didn't show one in the last test, but this
     makes it visible and checkable rather than trusting a summary number).
  2. ROUGH BROKERAGE/COST DEDUCTION -- Setup 2 fires ~10 times a day, and the
     original backtest charged nothing per trade. This version deducts a
     flat approximate round-trip cost per trade (BROKERAGE_PER_TRADE below)
     to see whether the edge survives real transaction costs. The number
     used is a rough placeholder for a discount broker's flat/near-flat
     per-order options charge plus STT/exchange/GST -- NOT your exact Dhan
     cost sheet. Adjust BROKERAGE_PER_TRADE below if you know your real
     effective cost per round trip and re-run.

Same entry logic as before (unchanged, already verified): a wick sweeps past
the last confirmed swing high/low by >=1 point and closes back inside, then
the very next candle closes further in the reversal direction -- entry on
that second candle's close. Same shared management: 10-pt initial stop,
20-pt (1:2) target, and two variants -- flat exit at target, or switch to a
3-pt candle trailing stop after target is hit.

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py /
the full 3-setup script.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE

FROM_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

SWING_N = 3
MIN_SWEEP_PTS = 1.0

INITIAL_STOP_PTS = 10
TARGET_PTS = 20  # 1:2 given a 10-pt initial stop
TRAIL_PTS = 3

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1
ASSUMED_DELTA = 0.6

# ---- Rough approximate cost per round-trip trade (entry + exit combined). ----
# Placeholder only -- adjust to your real effective Dhan options cost per
# round trip if you know it (brokerage + STT + exchange txn charges + GST +
# SEBI fee + stamp duty, all added up for one buy + one sell).
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


def simulate_with_signals(day_df, entry_signals, trailing_enabled):
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
                target = entry + TARGET_PTS
            else:
                stop = entry + INITIAL_STOP_PTS
                target = entry - TARGET_PTS
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
        return

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

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    parts = []
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        parts.append(f"{wd[:3]}:{len(g)}/{100*w/len(g):.0f}%/{g['points'].sum():.0f}")
    print("By weekday (n/win%/net):", " | ".join(parts))

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

    # ---- GROSS money sim (no costs), same as before, for comparison ----
    gross_rupee = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    gross_curve = STARTING_CAPITAL + gross_rupee.cumsum()
    gross_dd = (gross_curve - gross_curve.cummax()).min()
    print(f"\nGROSS (no brokerage): end Rs{gross_curve.iloc[-1]:,.2f}  "
          f"P&L Rs{gross_curve.iloc[-1]-STARTING_CAPITAL:,.2f}  worst drawdown Rs{gross_dd:,.2f}")
    if (gross_curve <= 0).any():
        idx0 = (gross_curve <= 0).idxmax()
        print(f"  WARNING: gross curve hits Rs0 or below at trade #{trades_df.index.get_loc(idx0)+1}")

    # ---- NET money sim (with rough per-trade brokerage) + full running curve ----
    net_rupee = gross_rupee - BROKERAGE_PER_TRADE
    trades_df["rupee_pnl_gross"] = gross_rupee
    trades_df["brokerage"] = BROKERAGE_PER_TRADE
    trades_df["rupee_pnl_net"] = net_rupee
    net_curve = STARTING_CAPITAL + net_rupee.cumsum()
    trades_df["running_capital"] = net_curve
    net_dd = (net_curve - net_curve.cummax()).min()
    lowest_point_idx = net_curve.idxmin()
    lowest_point_trade_num = trades_df.index.get_loc(lowest_point_idx) + 1
    lowest_point_value = net_curve.loc[lowest_point_idx]
    lowest_point_date = trades_df.loc[lowest_point_idx, "entry_time"]

    print(f"\nNET (after Rs{BROKERAGE_PER_TRADE:.0f}/trade assumed cost, "
          f"Rs{BROKERAGE_PER_TRADE*len(trades_df):,.0f} total over {len(trades_df)} trades):")
    print(f"  End Rs{net_curve.iloc[-1]:,.2f}   P&L Rs{net_curve.iloc[-1]-STARTING_CAPITAL:,.2f}   "
          f"worst drawdown Rs{net_dd:,.2f}")
    print(f"  Lowest point in the whole run: Rs{lowest_point_value:,.2f} at trade #{lowest_point_trade_num} "
          f"({lowest_point_date})")
    if (net_curve <= 0).any():
        idx0 = (net_curve <= 0).idxmax()
        print(f"  WARNING: NET curve (with costs) hits Rs0 or below at trade #{trades_df.index.get_loc(idx0)+1} "
              f"({trades_df.loc[idx0, 'entry_time']})")
    else:
        print("  Capital never touches Rs0 or below at any point, even after costs.")

    trades_df.to_csv(filename, index=False)
    print(f"\nFull trade-by-trade list (including running_capital column) saved to {filename}")


def main():
    print("=== Setup 2 ONLY (Reversal / Liquidity Sweep) -- refined, standalone, 1 year ===")
    print(f"Shared management: {INITIAL_STOP_PTS}-pt initial stop, {TARGET_PTS}-pt (1:2) target, "
          f"then either flat exit or a {TRAIL_PTS}-pt trail.")
    print(f"Rough assumed cost per round-trip trade: Rs{BROKERAGE_PER_TRADE:.0f} (adjust BROKERAGE_PER_TRADE if you know your real cost).\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    signals_by_day = {}
    day_dfs = {}
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 2 * SWING_N + 5:
            continue
        day_dfs[day] = day_df
        signals_by_day[day] = detect_setup2_signals(day_df)

    for variant_name, trailing_enabled in [("Fixed_1_2", False), ("Trail_after_1_2", True)]:
        all_trades = []
        for day, day_df in day_dfs.items():
            signals = signals_by_day.get(day, [])
            if not signals:
                continue
            all_trades.extend(simulate_with_signals(day_df, signals, trailing_enabled))
        label = f"Setup2_LiquiditySweep | {variant_name}"
        filename = f"scalp_setup2_refined_{variant_name}_trades.csv"
        report(all_trades, label, filename)


if __name__ == "__main__":
    main()
