"""
Concepts 2&4 -- Standalone 6-Month Re-Test, Previous Rules Unchanged
=======================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.
No OI, no other concept mixed in (per your instruction not to combine OI
with anything else) -- this is exactly the already-proven, locked strategy
(dlb.compute_latest_signal, with its own existing ALLOWED_WEEKDAYS={1,2}
Tue/Wed restriction, SL_POINTS=5/TP_POINTS=10 fixed 1:2, REQUIRE_HTF_TREND_
MATCH=True, DISABLED_CONCEPTS as already configured), just re-run fresh on
the most recent 6 months of data to see if it still holds up.

Nothing about dhan_live_bot.py's settings is overridden here -- this test
uses whatever is CURRENTLY saved in your dhan_config.py-driven dhan_live_bot.py
on your machine, exactly as the live bot would use it.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=182)).strftime("%Y-%m-%d")  # ~6 months
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 10_000
LOT_SIZE = 65
ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp)
        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
            "volume": data["volume"], "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def simulate_day(day_df):
    """Same pattern as every other Concepts 2&4 backtest in this project --
    expanding-window calls to dlb.compute_latest_signal(), then SL/TP
    management using the locked SL_POINTS/TP_POINTS/MAX_BARS_IN_TRADE."""
    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)
    trades = []
    position = None
    for i in range(30, n):
        c = closes[i]
        if position is not None:
            exited, reason = False, None
            if position["side"] == "LONG":
                if c <= position["entry"] - dlb.SL_POINTS:
                    exited, reason = True, "STOP"
                elif c >= position["entry"] + dlb.TP_POINTS:
                    exited, reason = True, "TARGET"
            else:
                if c >= position["entry"] + dlb.SL_POINTS:
                    exited, reason = True, "STOP"
                elif c <= position["entry"] - dlb.TP_POINTS:
                    exited, reason = True, "TARGET"
            if not exited and (i - position["entry_bar"]) >= dlb.MAX_BARS_IN_TRADE:
                exited, reason = True, "TIME_EXIT"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[i],
                               "side": position["side"], "entry": position["entry"], "exit": c,
                               "points": pts, "reason": reason, "concept_id": position["concept_id"]})
                position = None

        if position is None:
            sig = dlb.compute_latest_signal(day_df.iloc[:i+1])
            if sig is None:
                continue
            if sig["final_buy"]:
                position = {"side": "LONG", "entry": c, "entry_time": idx[i], "entry_bar": i,
                            "concept_id": sig.get("concept_id")}
            elif sig["final_sell"]:
                position = {"side": "SHORT", "entry": c, "entry_time": idx[i], "entry_bar": i,
                            "concept_id": sig.get("concept_id")}

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1],
                       "side": position["side"], "entry": position["entry"], "exit": c,
                       "points": pts, "reason": "EOD", "concept_id": position["concept_id"]})
    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


def main():
    print("=== Concepts 2&4 -- standalone 6-month re-test, rules UNCHANGED ===")
    print(f"ALLOWED_WEEKDAYS currently set to: {dlb.ALLOWED_WEEKDAYS}  "
          f"(should be {{1, 2}} = Tue/Wed if nothing's been changed)\n")

    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned.")
        return
    df = dlb.attach_htf_trend(df)

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 40:
            continue
        all_trades.extend(simulate_day(day_df))

    if not all_trades:
        print("No trades generated -- check ALLOWED_WEEKDAYS / DISABLED_CONCEPTS in dhan_live_bot.py.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("concepts_2_4_6month_trades.csv", index=False)

    wins = tdf[tdf["points"] > 0]
    print(f"Total trades: {len(tdf)}   Win rate: {100*len(wins)/len(tdf):.1f}%   "
          f"Net points: {tdf['points'].sum():.2f}")
    print(f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")
    print(f"By concept_id: {tdf.groupby('concept_id')['points'].agg(['count','sum']).to_dict()}")

    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    print("\n--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    print("\n--- By 30-min session ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, win {100*(g['points']>0).mean():.1f}%")

    rupee_pnl = tdf["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    print(f"\n--- Money simulation ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {capital_curve.iloc[-1]:,.2f}   "
          f"Max DD Rs {(capital_curve-capital_curve.cummax()).min():,.2f}")
    if (capital_curve <= 0).any():
        print("WARNING: wipes out at some point.")
    else:
        print("Never wipes out.")

    mid = len(tdf)//2
    h1 = (tdf.iloc[:mid]["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    h2 = (tdf.iloc[mid:]["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    print(f"\n--- Stability check ---\nFirst half: {mid} trades, net Rs {h1:,.2f}")
    print(f"Second half: {len(tdf)-mid} trades, net Rs {h2:,.2f}")


if __name__ == "__main__":
    main()
