"""
Re-verification: Concepts 2 & 4, but with the Tuesday/Wednesday weekday
restriction TURNED OFF -- runs on ALL 5 weekdays -- over 2 years of data,
to double-check whether Tue/Wed is still the right call now that we have a
longer, fresher dataset than when that restriction was originally chosen.
Everything else identical to the locked FINAL strategy: 5-pt stop / 10-pt
target fixed 1:2, 5m+15m trend filter, concepts 2 & 4 only.

HOW THE WEEKDAY OVERRIDE WORKS: dhan_live_bot.py's compute_latest_signal()
reads the module-level ALLOWED_WEEKDAYS constant to decide whether to fire
a signal on a given day. Rather than editing that file (which would affect
the locked FINAL strategy), this script imports the module and overwrites
dhan_live_bot.ALLOWED_WEEKDAYS to include all 5 weekdays BEFORE calling
compute_latest_signal -- so this test uses the exact same signal logic,
just without the day-of-week gate. dhan_live_bot.py itself is never
modified on disk.

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py /
any other concept script. Separate from and does not affect
combined_final_strategy_2yr_v2.py.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

# Override the weekday gate to allow ALL days, without touching the file.
dlb.ALLOWED_WEEKDAYS = {0, 1, 2, 3, 4}  # Monday-Friday

dhan = dlb.dhan
UNDERLYING_SECURITY_ID = dlb.UNDERLYING_SECURITY_ID
UNDERLYING_EXCHANGE_SEGMENT = dlb.UNDERLYING_EXCHANGE_SEGMENT
UNDERLYING_INSTRUMENT_TYPE = dlb.UNDERLYING_INSTRUMENT_TYPE
SL_POINTS = dlb.SL_POINTS
TP_POINTS = dlb.TP_POINTS
MAX_BARS_IN_TRADE = dlb.MAX_BARS_IN_TRADE

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
LOTS_PER_TRADE = 1
ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0

CONCEPT_NAMES = {2: "2 Compression spike (sell)", 4: "4 Rollover (sell)"}
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


def simulate_concepts_for_day(day_df):
    trades = []
    position = None
    for i in range(len(day_df)):
        visible = day_df.iloc[:i + 1]
        sig = dlb.compute_latest_signal(visible)
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
    print("=== Concepts 2 & 4 -- ALL WEEKDAYS (Tue/Wed restriction removed for this test), 2 years ===\n")
    assert dlb.ALLOWED_WEEKDAYS == {0, 1, 2, 3, 4}, "Weekday override did not take effect"

    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    print("Computing 5m/15m trend context (one-time, vectorized)...")
    df = dlb.attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        all_trades.extend(simulate_concepts_for_day(day_df))

    if not all_trades:
        print("\nNo trades were triggered in this date range.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print("\n===== CONCEPTS 2 & 4, ALL WEEKDAYS -- OVERALL RESULTS (2 YEARS) =====")
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
    for cid, g in trades_df.groupby("concept_id"):
        wr = 100 * (g["points"] > 0).mean()
        print(f"    {CONCEPT_NAMES.get(cid, cid):<28} trades={len(g):<4} win rate={wr:5.1f}%  net={g['points'].sum():8.2f}")
    print("========================================================================")

    trades_df.to_csv("concepts_2_4_all_days_trades.csv", index=False)
    print("\nFull trade list saved to concepts_2_4_all_days_trades.csv")

    # ---- THE KEY OUTPUT: full weekday breakdown, so Tue/Wed can be compared
    # against Mon/Thu/Fri directly on this fresh 2-year dataset ----
    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    print("\n===================== BY WEEKDAY (the main comparison) =====================")
    print(f"{'Weekday':<12}{'Trades':>8}{'Wins':>7}{'Win %':>9}{'Net Pts':>12}{'Avg Pts':>10}{'Stop %':>9}")
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        s = (g["reason"] == "STOP").sum()
        print(f"{wd:<12}{len(g):>8}{w:>7}{100*w/len(g):>8.1f}%{g['points'].sum():>12.2f}{g['points'].mean():>10.2f}{100*s/len(g):>8.1f}%")
    print("==============================================================================")

    print("\n===================== 30-MINUTE SESSION BREAKDOWN, BY WEEKDAY =====================")
    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    for wd in WEEKDAY_ORDER:
        wd_df = trades_df[trades_df["weekday"] == wd]
        if wd_df.empty:
            continue
        print(f"\n-- {wd} ({len(wd_df)} trades, net {wd_df['points'].sum():.2f}) --")
        for session in SESSIONS + ["Outside usual session"]:
            g = wd_df[wd_df["session"] == session]
            if len(g) == 0:
                continue
            w = (g["points"] > 0).sum()
            print(f"  {session:<16}{len(g):>6}  win%={100*w/len(g):5.1f}  net={g['points'].sum():8.2f}")
    print("\n=====================================================================================")

    # ---- money sim: full all-days version, and a "what if only Tue/Wed" comparison ----
    rupee_pnl = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
    net_pnl = rupee_pnl - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + net_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()

    print("\n===================== MONEY SIMULATION -- ALL WEEKDAYS (with Rs40/trade cost) =====================")
    print(f"Starting capital:     Rs {STARTING_CAPITAL:,.2f}")
    print(f"Ending capital:       Rs {capital_curve.iloc[-1]:,.2f}")
    print(f"Total P&L:            Rs {capital_curve.iloc[-1] - STARTING_CAPITAL:,.2f}")
    print(f"Worst drawdown:       Rs {max_dd:,.2f}")
    if (capital_curve <= 0).any():
        print("  WARNING: capital hits Rs0 or below at some point.")
    else:
        print("  Capital never touches Rs0 or below.")

    tw_only = trades_df[trades_df["weekday"].isin(["Tuesday", "Wednesday"])]
    tw_rupee = tw_only["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE - BROKERAGE_PER_TRADE
    tw_cap = STARTING_CAPITAL + tw_rupee.sort_index().cumsum()
    print(f"\nFor comparison -- Tue/Wed ONLY subset of this same 2-year run (i.e. what the locked FINAL "
          f"strategy would have produced): {len(tw_only)} trades, net {tw_only['points'].sum():.2f} pts, "
          f"end capital Rs{tw_cap.iloc[-1]:,.2f}")
    print("======================================================================================================")


if __name__ == "__main__":
    main()
