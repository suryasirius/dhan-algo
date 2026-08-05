"""
COMBINED backtest -- runs the FINAL strategy (concepts 2 & 4, Tue/Wed only,
5-pt stop / 10-pt target, 5m+15m trend filter) AND the new Strike Momentum
concept (Mon/Wed/Fri only, first 30 minutes only, 10-pt stop switching to
3-pt candle trail) side by side over 1 year of data, as two independent
strategies sharing the same underlying feed. Reports each separately AND
combined.

IMPORTANT OPERATIONAL NOTE: on Wednesdays, BOTH strategies are allowed to
trade at the same time (Concepts 2/4 trade Tue/Wed; Strike Momentum trades
Mon/Wed/Fri). That means on Wednesdays you could have two option positions
open simultaneously, needing separate capital/margin for each -- this
backtest treats them as sharing one combined capital pool, which is the
realistic case if you're trading both from the same account.

This script does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py --
it's a standalone combined test to decide whether to actually fold Strike
Momentum into the live bot.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    compute_latest_signal, SL_POINTS, TP_POINTS, MAX_BARS_IN_TRADE, DISABLED_CONCEPTS,
    attach_htf_trend, REQUIRE_HTF_TREND_MATCH, ALLOWED_WEEKDAYS as CONCEPTS_ALLOWED_WEEKDAYS,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

# ---- Concepts 2 & 4 (the locked FINAL strategy) use SL_POINTS/TP_POINTS/etc
# imported directly from dhan_live_bot.py above -- always in sync with FINAL_*.py.

# ---- Strike Momentum concept (new, being tested for inclusion) ----
SM_DELTA = 0.6
SM_OPEN_INVALIDATE_PTS = 5
SM_INITIAL_STOP_PTS = 10
SM_TRAIL_PTS = 3
SM_ENTRY_WINDOW_START = dt.time(9, 15)
SM_ENTRY_WINDOW_END = dt.time(9, 45)
SM_ALLOWED_WEEKDAYS = {0, 2, 4}  # Monday, Wednesday, Friday

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1
CONCEPTS_ASSUMED_DELTA = 0.6  # same approximation used throughout for concepts 2/4

CONCEPT_NAMES = {2: "2 Compression spike (sell)", 4: "4 Rollover (sell)"}


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


# ------------------------- Concepts 2 & 4 (FINAL strategy) -------------------------
def simulate_concepts_for_day(day_df):
    trades = []
    position = None

    for i in range(len(day_df)):
        visible = day_df.iloc[:i + 1]
        sig = compute_latest_signal(visible)
        if sig is None:
            continue
        price = day_df["close"].iloc[i]
        ts = day_df.index[i]

        if position is not None:
            exited = False
            if position["side"] == "LONG":
                hard_stop = position["entry"] - SL_POINTS
                target = position["entry"] + TP_POINTS
                if price <= hard_stop:
                    exit_reason, exited = "STOP", True
                elif price >= target:
                    exit_reason, exited = "TARGET", True
                elif position["bars"] >= MAX_BARS_IN_TRADE:
                    exit_reason, exited = "TIME", True
            else:
                hard_stop = position["entry"] + SL_POINTS
                target = position["entry"] - TP_POINTS
                if price >= hard_stop:
                    exit_reason, exited = "STOP", True
                elif price <= target:
                    exit_reason, exited = "TARGET", True
                elif position["bars"] >= MAX_BARS_IN_TRADE:
                    exit_reason, exited = "TIME", True

            if exited:
                pnl_points = (price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - price)
                trades.append({
                    "strategy": "Concepts_2_4", "entry_time": position["entry_time"], "exit_time": ts,
                    "side": position["side"], "points": pnl_points, "reason": exit_reason,
                    "concept_id": position["concept_id"],
                })
                position = None
            else:
                position["bars"] += 1

        if position is None:
            if sig["final_buy"]:
                position = {"side": "LONG", "entry": price, "bars": 0, "entry_time": ts,
                            "concept_id": sig["concept_id"]}
            elif sig["final_sell"]:
                position = {"side": "SHORT", "entry": price, "bars": 0, "entry_time": ts,
                            "concept_id": sig["concept_id"]}

    return trades


# ------------------------- Strike Momentum concept -------------------------
def simulate_strike_momentum_for_day(day_df):
    trades = []
    day_open_close = day_df["close"].iloc[0]

    def synth(price):
        return SM_DELTA * (price - day_open_close)

    invalidated = False
    position = None

    for i in range(1, len(day_df)):
        close_p = synth(day_df["close"].iloc[i])
        low_p = synth(day_df["low"].iloc[i])
        ts = day_df.index[i]

        if low_p < -SM_OPEN_INVALIDATE_PTS:
            invalidated = True

        if position is not None:
            exited, reason = False, None
            if position["trailing_active"]:
                if close_p <= position["stop_level"]:
                    exited, reason = True, "TRAIL"
                else:
                    position["stop_level"] = max(position["stop_level"], low_p - SM_TRAIL_PTS)
            else:
                if close_p <= position["stop_level"]:
                    exited, reason = True, "STOP"
                elif close_p - position["entry"] > 0:
                    position["trailing_active"] = True
                    position["stop_level"] = low_p - SM_TRAIL_PTS

            if exited:
                trades.append({
                    "strategy": "StrikeMomentum", "entry_time": position["entry_time"], "exit_time": ts,
                    "side": "LONG", "points": close_p - position["entry"], "reason": reason,
                    "concept_id": None,
                })
                position = None

        in_window = SM_ENTRY_WINDOW_START <= ts.time() < SM_ENTRY_WINDOW_END
        if position is None and not invalidated and close_p > 0 and in_window:
            position = {"entry": close_p, "entry_time": ts,
                        "trailing_active": False, "stop_level": close_p - SM_INITIAL_STOP_PTS}

    if position is not None:
        last_close = synth(day_df["close"].iloc[-1])
        trades.append({
            "strategy": "StrikeMomentum", "entry_time": position["entry_time"], "exit_time": day_df.index[-1],
            "side": "LONG", "points": last_close - position["entry"], "reason": "EOD",
            "concept_id": None,
        })

    return trades


def summarize(trades_df, label):
    if trades_df.empty:
        print(f"\n{label}: no trades.")
        return
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]
    print(f"\n--- {label} ---")
    print(f"Trades: {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}")


def main():
    print("=== COMBINED backtest: Concepts 2&4 (Tue/Wed) + Strike Momentum (Mon/Wed/Fri, 09:15-09:45) ===")
    print("1 year of Nifty 1-min data.\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    print("Computing 5m/15m trend context for Concepts 2&4 (one-time, vectorized)...")
    df = attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    concept_trades, sm_trades = [], []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        wd = day.weekday()
        if wd in CONCEPTS_ALLOWED_WEEKDAYS:
            concept_trades.extend(simulate_concepts_for_day(day_df))
        if wd in SM_ALLOWED_WEEKDAYS:
            sm_trades.extend(simulate_strike_momentum_for_day(day_df))

    concept_df = pd.DataFrame(concept_trades)
    sm_df = pd.DataFrame(sm_trades)

    print("\n================================ RESULTS BY STRATEGY ================================")
    summarize(concept_df, "Concepts 2 & 4 (Tue/Wed, fixed 1:2)")
    if not concept_df.empty:
        for cid, g in concept_df.groupby("concept_id"):
            wr = 100 * (g["points"] > 0).mean()
            print(f"    {CONCEPT_NAMES.get(cid, cid):<28} trades={len(g):<4} win rate={wr:5.1f}%  net={g['points'].sum():8.2f}")
    summarize(sm_df, "Strike Momentum (Mon/Wed/Fri, 09:15-09:45)")
    print("=======================================================================================")

    if concept_df.empty and sm_df.empty:
        print("\nNo trades at all in this date range.")
        return

    # Convert each to rupees using each strategy's own scaling, then combine
    # chronologically for one shared capital curve.
    if not concept_df.empty:
        concept_df["rupee_pnl"] = concept_df["points"] * CONCEPTS_ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    if not sm_df.empty:
        sm_df["rupee_pnl"] = sm_df["points"] * LOT_SIZE * LOTS_PER_TRADE

    combined = pd.concat([concept_df, sm_df], ignore_index=True) if (not concept_df.empty and not sm_df.empty) \
        else (concept_df if sm_df.empty else sm_df)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    combined.to_csv("combined_1year_trades.csv", index=False)

    capital_curve = STARTING_CAPITAL + combined["rupee_pnl"].cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    final_capital = capital_curve.iloc[-1]

    print("\n===================== COMBINED MONEY SIMULATION (APPROXIMATE) =====================")
    print("NOTE: Concepts 2&4 use a 0.6-delta approximation on underlying points;")
    print("Strike Momentum's points are already delta-scaled synthetic premium.")
    print("Neither is real historical option premium data. Both share ONE combined")
    print(f"Rs {STARTING_CAPITAL:,} capital pool here -- in reality, Wednesdays could need")
    print("separate margin for two simultaneously open positions (one per strategy).")
    print(f"\nTotal combined trades:  {len(combined)}")
    print(f"Starting capital:       Rs {STARTING_CAPITAL:,.2f}")
    print(f"Ending capital:         Rs {final_capital:,.2f}")
    print(f"Total P&L:              Rs {final_capital - STARTING_CAPITAL:,.2f}")
    print(f"Return on capital:      {100 * (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL:.1f}%")
    print(f"Worst drawdown:         Rs {max_dd:,.2f}")
    if (capital_curve <= 0).any():
        wipeout_trade = int((capital_curve <= 0).idxmax()) + 1
        print(f"WARNING: capital hits Rs 0 or below around trade #{wipeout_trade} of {len(combined)}.")
    print("=====================================================================================")
    print("\nFull combined trade list saved to combined_1year_trades.csv")


if __name__ == "__main__":
    main()