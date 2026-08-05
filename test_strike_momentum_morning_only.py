"""
Same Strike Momentum concept as test_strike_momentum.py, but with ONE change:
new entries are only allowed between 09:15 and 10:15 (the session that carried
almost all of the profit -- 09:15-09:45 was +359.34 on its own, while the
11:15-12:45 window was the biggest drag at -213.42 combined).

Once a trade is entered inside that window, it's still managed (stop/trail)
for as long as it stays open, even past 10:15 -- only NEW entries are
restricted to the window. All other rules are identical to the original
test. Standalone script -- does not touch dhan_live_bot.py / backtest.py /
FINAL_*.py.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE

FROM_DATE = (dt.date.today() - dt.timedelta(days=182)).strftime("%Y-%m-%d")  # ~6 months
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

DELTA = 0.6
OPEN_INVALIDATE_PTS = 5
INITIAL_STOP_PTS = 10
TRAIL_PTS = 3

# Only take NEW entries inside this window -- based on the 30-min session
# breakdown showing 09:15-09:45 alone carried the whole strategy's profit.
ENTRY_WINDOW_START = dt.time(9, 15)
ENTRY_WINDOW_END = dt.time(10, 15)

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1


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


def simulate_day(day_df):
    trades = []
    day_open_close = day_df["close"].iloc[0]

    def synth(price):
        return DELTA * (price - day_open_close)

    invalidated = False
    position = None

    for i in range(1, len(day_df)):
        close_p = synth(day_df["close"].iloc[i])
        low_p = synth(day_df["low"].iloc[i])
        ts = day_df.index[i]

        if low_p < -OPEN_INVALIDATE_PTS:
            invalidated = True

        if position is not None:
            exited, reason = False, None
            if position["trailing_active"]:
                if close_p <= position["stop_level"]:
                    exited, reason = True, "TRAIL"
                else:
                    position["stop_level"] = max(position["stop_level"], low_p - TRAIL_PTS)
            else:
                if close_p <= position["stop_level"]:
                    exited, reason = True, "STOP"
                elif close_p - position["entry"] > 0:
                    position["trailing_active"] = True
                    position["stop_level"] = low_p - TRAIL_PTS

            if exited:
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "entry": position["entry"], "exit": close_p,
                    "points": close_p - position["entry"], "reason": reason,
                })
                position = None

        in_entry_window = ENTRY_WINDOW_START <= ts.time() < ENTRY_WINDOW_END
        if position is None and not invalidated and close_p > 0 and in_entry_window:
            position = {"entry": close_p, "entry_time": ts,
                        "trailing_active": False, "stop_level": close_p - INITIAL_STOP_PTS}

    if position is not None:
        last_close = synth(day_df["close"].iloc[-1])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": day_df.index[-1],
            "entry": position["entry"], "exit": last_close,
            "points": last_close - position["entry"], "reason": "EOD",
        })

    return trades


def main():
    print("=== Strike Momentum concept -- MORNING-ONLY entries (09:15-10:15), 1-min, ~6 months ===")
    print(f"Synthetic premium: DELTA={DELTA} applied to real Nifty underlying moves ")
    print("(no real historical option premium available for expired weekly contracts).\n")

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
        if len(day_df) < 10:
            continue
        all_trades.extend(simulate_day(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n========== STRIKE MOMENTUM (MORNING-ONLY) -- BACKTEST RESULTS ==========")
    print(f"Total trades:       {len(trades_df)}")
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}")
    print(f"Win rate:           {100 * len(wins) / len(trades_df):.1f}%")
    print(f"Avg win (pts):      {wins['points'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss (pts):     {losses['points'].mean():.2f}" if len(losses) else "Avg loss: n/a")
    print(f"Net points:         {trades_df['points'].sum():.2f}")
    print(f"Exit reasons:       {trades_df['reason'].value_counts().to_dict()}")
    print("==========================================================================")

    trades_df.to_csv("strike_momentum_morning_only_trades.csv", index=False)
    print("\nFull trade list saved to strike_momentum_morning_only_trades.csv")

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    rupee_pnl = trades_df["points"] * LOT_SIZE * LOTS_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    final_capital = capital_curve.iloc[-1]

    print("\n===================== MONEY SIMULATION (APPROXIMATE) =====================")
    print("NOTE: premium points here are already delta-scaled from the underlying --")
    print("this is not real historical option premium.")
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
