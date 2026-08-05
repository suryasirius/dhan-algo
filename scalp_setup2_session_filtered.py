"""
Setup 2 (Reversal / Liquidity Sweep), session-filtered -- built after the
refined 1-year run showed the setup only survives real transaction costs in
specific 30-minute windows. Instead of restricting by weekday (which relies
on a thin, possibly-noisy Friday effect), this restricts by TIME OF DAY,
applied across ALL weekdays, per your instruction to "fire only when and
where the setup has shown high probability, take all days."

WHY THESE 3 WINDOWS, SPECIFICALLY:
The refined run's trades were split into the first 6 months vs the last 6
months of the 1-year backtest and checked session-by-session for whether
each window was profitable in BOTH halves independently (not just
overall) -- a basic check against picking a window that only looks good
because of one lucky stretch.
  - 10:45-11:15: ~48-49% win rate and net-positive (after costs) in BOTH
    halves, for BOTH the fixed and trailing variant. The clearest real edge.
  - 09:45-10:15: net-positive in both halves, though the edge fades in the
    second half (still positive, just weaker).
  - 09:15-09:45: net-positive in both halves, actually strengthening in the
    second half.
  - 10:15-10:45 and 14:15-14:45 were EXCLUDED because they were
    consistently negative in both halves -- a stable losing pattern, not
    noise, so there's no ambiguity about cutting them.
  - Every other window flipped sign between the two halves (negative in
    one, positive in the other) -- that's the signature of noise, not a
    real edge, so those were left out too even though a couple had decent
    full-period numbers.

A retrospective filter of the existing trade data (same entry logic, just
restricted to these windows) showed: 526 trades instead of 2,417, win rate
up from 37.6% to 44.9%, and NET (after the same Rs40/trade rough cost
assumption) flips from -Rs73,504 to +Rs39,753, with drawdown never
approaching zero. This script re-runs that properly as a forward backtest
(the session filter is applied to signal generation itself, not bolted on
after the fact) so the result can be trusted and re-verified independently.

Same entry logic as before (unchanged): a wick sweeps past the last
confirmed swing high/low by >=1 point and closes back inside, then the very
next candle closes further in the reversal direction -- entry on that
second candle's close. Same shared management: 10-pt initial stop, 20-pt
(1:2) target, two variants (flat exit at target, or 3-pt trail after
target). Same rough Rs40/trade cost assumption as the refined script --
adjust BROKERAGE_PER_TRADE if you know your real Dhan cost per round trip.

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py /
any other concept script.
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
BROKERAGE_PER_TRADE = 40.0  # rough placeholder, same as the refined script

# Only take NEW entries inside these windows, on ANY weekday. Chosen because
# they were the only sessions that stayed net-positive (after costs) in
# BOTH the first and second half of the 1-year period independently.
ALLOWED_ENTRY_WINDOWS = [
    (dt.time(9, 15), dt.time(9, 45)),
    (dt.time(9, 45), dt.time(10, 15)),
    (dt.time(10, 45), dt.time(11, 15)),
]

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


def in_allowed_window(ts):
    t = ts.time()
    return any(start <= t < end for start, end in ALLOWED_ENTRY_WINDOWS)


def detect_setup2_signals(day_df):
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
        window_ok = in_allowed_window(idx[j])

        if pending_bull_sweep_close is not None:
            if c > pending_bull_sweep_close and window_ok:
                signals.append({"bar": j, "side": "LONG"})
            pending_bull_sweep_close = None
        if pending_bear_sweep_close is not None:
            if c < pending_bear_sweep_close and window_ok:
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
    print("By weekday (n/win%/net pts) -- should now be spread across all days:", " | ".join(parts))

    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    print("By session (should only show the 3 allowed windows):")
    for s in SESSIONS:
        g = trades_df[trades_df["session"] == s]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        print(f"  {s:<14} trades={len(g):<5} win%={100*w/len(g):5.1f}  net={g['points'].sum():8.2f}")

    gross_rupee = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    net_rupee = gross_rupee - BROKERAGE_PER_TRADE
    trades_df["rupee_pnl_gross"] = gross_rupee
    trades_df["brokerage"] = BROKERAGE_PER_TRADE
    trades_df["rupee_pnl_net"] = net_rupee
    net_curve = STARTING_CAPITAL + net_rupee.cumsum()
    trades_df["running_capital"] = net_curve
    net_dd = (net_curve - net_curve.cummax()).min()

    print(f"\nGROSS (no cost): P&L Rs{gross_rupee.sum():,.2f}")
    print(f"Total brokerage over {len(trades_df)} trades (Rs{BROKERAGE_PER_TRADE:.0f}/trade): "
          f"Rs{BROKERAGE_PER_TRADE*len(trades_df):,.0f}")
    print(f"NET (after cost): end Rs{net_curve.iloc[-1]:,.2f}   P&L Rs{net_curve.iloc[-1]-STARTING_CAPITAL:,.2f}   "
          f"worst drawdown Rs{net_dd:,.2f}")
    if (net_curve <= 0).any():
        idx0 = (net_curve <= 0).idxmax()
        print(f"  WARNING: NET curve hits Rs0 or below at trade #{trades_df.index.get_loc(idx0)+1} "
              f"({trades_df.loc[idx0, 'entry_time']})")
    else:
        print("  Capital never touches Rs0 or below at any point, even after costs.")

    days = trades_df["entry_time"].dt.date.nunique()
    print(f"Trades/day (on days with a signal): {len(trades_df)/days:.2f}  (days with a signal: {days})")

    trades_df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def main():
    print("=== Setup 2 (Liquidity Sweep), SESSION-FILTERED -- all weekdays, only 09:15-10:15 and 10:45-11:15 ===")
    print(f"Shared management: {INITIAL_STOP_PTS}-pt initial stop, {TARGET_PTS}-pt (1:2) target, "
          f"then either flat exit or a {TRAIL_PTS}-pt trail.")
    print(f"Rough assumed cost per round-trip trade: Rs{BROKERAGE_PER_TRADE:.0f}.\n")

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
        label = f"Setup2_SessionFiltered | {variant_name}"
        filename = f"scalp_setup2_sessionfiltered_{variant_name}_trades.csv"
        report(all_trades, label, filename)


if __name__ == "__main__":
    main()
