import sys
import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    compute_latest_signal, SL_POINTS, TP_POINTS, MAX_BARS_IN_TRADE, DISABLED_CONCEPTS,
    attach_htf_trend, REQUIRE_HTF_TREND_MATCH,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=182)).strftime("%Y-%m-%d")  # ~6 months
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

# ===================== MONEY SIMULATION (APPROXIMATE) =====================
# We do NOT have real historical option premiums for these dates -- expired
# weekly contracts drop out of Dhan's instrument master, so there's no way
# to pull back exactly what the ITM Call/Put actually cost on a past date.
# Instead we approximate: option premium move = underlying's point move x
# an assumed delta. ITM_STRIKE_OFFSET=0 means the nearest ITM strike (one
# strike ITM, not deep ITM), so ~0.6 delta is a reasonable estimate -- real
# fills will differ from this.
STARTING_CAPITAL = 10_000
LOT_SIZE = 65          # Nifty lot size, revised Jan 2026 (was 75 before that)
ASSUMED_DELTA = 0.6    # approx delta for a one-strike ITM option
LOTS_PER_TRADE = 1      # sized to fit inside STARTING_CAPITAL at typical ITM premiums
# ============================================================================


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
        sys.exit(1)
    data = resp.get("data", resp)
    df = pd.DataFrame({
        "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "timestamp": data["timestamp"],
    })
    # Dhan's timestamp is plain UTC seconds -- add 5:30 to get real IST
    # market-clock time (otherwise every candle looks 5.5 hours earlier
    # than it actually happened, e.g. a 2:45pm IST trade shows as 9:15am).
    df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    df = df.set_index("time").sort_index()
    return df


def fetch_historical_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    """Dhan only allows 90 days of minute data per request, so for ranges
    longer than that we split into chunks and stitch the results together."""
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()

    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        chunks.append(fetch_historical(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + dt.timedelta(days=1)

    if not chunks:
        return pd.DataFrame()
    full = pd.concat(chunks)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full


def simulate_trades_for_day(day_df):
    """Fixed stop / fixed target, no trailing -- SL_POINTS stop, TP_POINTS target."""
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
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "side": position["side"], "entry": position["entry"], "exit": price,
                    "points": pnl_points, "reason": exit_reason,
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


CONCEPT_NAMES = {
    1: "1 Compression spike (buy)", 2: "2 Compression spike (sell)",
    3: "3 Rollover (buy)", 4: "4 Rollover (sell)",
    5: "5 Failed-spike reversal (buy)", 6: "6 Failed-spike reversal (sell)",
    7: "7 Leg-2 confirm (buy)", 8: "8 Leg-2 confirm (sell)",
    9: "9 Weakening domes (sell)",
}

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def classify_session(ts):
    t = ts.time()
    if dt.time(9, 15) <= t < dt.time(11, 0):
        return "1 Morning   (9:15-11:00)"
    elif dt.time(11, 0) <= t < dt.time(14, 0):
        return "2 Midday    (11:00-14:00)"
    elif dt.time(14, 0) <= t <= dt.time(15, 30):
        return "3 Closing   (14:00-15:30)"
    return "4 Outside usual session"


def print_breakdown(trades_df, group_col, title, order=None):
    print(f"\n===================== {title} =====================")
    groups = trades_df.groupby(group_col)
    keys = order if order else sorted(groups.groups.keys())
    for key in keys:
        if key not in groups.groups:
            continue
        group = groups.get_group(key)
        wr = 100 * (group["points"] > 0).mean()
        reason_pct = 100 * group["reason"].value_counts(normalize=True)
        stop_pct = reason_pct.get("STOP", 0.0)
        target_pct = reason_pct.get("TARGET", 0.0)
        time_pct = reason_pct.get("TIME", 0.0)
        print(f"{str(key):<26} trades={len(group):<4} win rate={wr:5.1f}%  "
              f"stop-hit={stop_pct:5.1f}%  target-hit={target_pct:5.1f}%  time-exit={time_pct:5.1f}%  "
              f"avg={group['points'].mean():6.2f}  net={group['points'].sum():7.2f}")
    print("=" * 60)


def main():
    if DISABLED_CONCEPTS:
        names = ", ".join(CONCEPT_NAMES.get(cid, str(cid)) for cid in sorted(DISABLED_CONCEPTS))
        print(f"Concepts turned OFF for this run: {names}\n")
    if REQUIRE_HTF_TREND_MATCH:
        print("5m + 15m trend filter is ON -- signals only fire when both agree with the trade direction.\n")

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    print("Computing 5m/15m trend context across the full history (one-time, vectorized)...")
    df = attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        all_trades.extend(simulate_trades_for_day(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n================ BACKTEST RESULTS ================")
    print(f"Total trades:       {len(trades_df)}")
    print(f"Wins / Losses:      {len(wins)} / {len(losses)}")
    print(f"Win rate:           {100 * len(wins) / len(trades_df):.1f}%")
    print(f"Avg win (points):   {wins['points'].mean():.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss (points):  {losses['points'].mean():.2f}" if len(losses) else "Avg loss: n/a")
    print(f"Net points:         {trades_df['points'].sum():.2f}")
    print(f"Exit reasons:       {trades_df['reason'].value_counts().to_dict()}")
    print("====================================================")

    print("\n===================== WIN RATE / STOP RATE / TARGET RATE BY CONCEPT =====================")
    for cid, group in trades_df.groupby("concept_id"):
        name = CONCEPT_NAMES.get(cid, f"Unknown ({cid})")
        wr = 100 * (group["points"] > 0).mean()
        reason_pct = 100 * group["reason"].value_counts(normalize=True)
        stop_pct = reason_pct.get("STOP", 0.0)
        target_pct = reason_pct.get("TARGET", 0.0)
        time_pct = reason_pct.get("TIME", 0.0)
        print(f"{name:<38} trades={len(group):<4} win rate={wr:5.1f}%  "
              f"stop-hit={stop_pct:5.1f}%  target-hit={target_pct:5.1f}%  time-exit={time_pct:5.1f}%  "
              f"avg={group['points'].mean():6.2f}  net={group['points'].sum():7.2f}")
    print("===========================================================================================")

    trades_df["session"] = trades_df["entry_time"].apply(classify_session)
    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()

    print_breakdown(trades_df, "session", "WIN RATE BY SESSION")
    print_breakdown(trades_df, "weekday", "WIN RATE BY DAY OF WEEK", order=WEEKDAY_ORDER)

    trades_df.to_csv("backtest_trades.csv", index=False)
    print("\nFull trade list saved to backtest_trades.csv")

    # ---------------- Money simulation (approximate, see note above) ----------------
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    rupee_pnl = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    running_peak = capital_curve.cummax()
    drawdown = capital_curve - running_peak
    max_drawdown = drawdown.min()
    final_capital = capital_curve.iloc[-1]

    print("\n===================== MONEY SIMULATION (APPROXIMATE) =====================")
    print("NOTE: option premiums are estimated from underlying point moves using")
    print(f"an assumed delta of {ASSUMED_DELTA} -- not real historical option prices.")
    print(f"Starting capital:     Rs {STARTING_CAPITAL:,.2f}")
    print(f"Lot size used:        {LOT_SIZE} (Nifty, Jan-2026 revision), {LOTS_PER_TRADE} lot/trade")
    print(f"Ending capital:       Rs {final_capital:,.2f}")
    print(f"Total P&L:            Rs {final_capital - STARTING_CAPITAL:,.2f}")
    print(f"Return on capital:    {100 * (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL:.1f}%")
    print(f"Worst drawdown:       Rs {max_drawdown:,.2f}")
    if (capital_curve <= 0).any():
        wipeout_trade = int((capital_curve <= 0).idxmax()) + 1
        print(f"WARNING: capital hits Rs 0 or below around trade #{wipeout_trade} of {len(trades_df)} "
              f"-- Rs {STARTING_CAPITAL:,} was not enough to survive this sequence at {LOTS_PER_TRADE} lot/trade.")
    print("============================================================================")


if __name__ == "__main__":
    main()
