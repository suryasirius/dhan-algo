"""
OI + Multi-Timeframe Trend Confirmation -- Standalone Re-Test
=================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.
NOT combined with Concepts 2&4, Strike Momentum, or anything else, per your
instruction -- this is the OI concept entirely on its own, same rules as
originally built and rejected (30m/15m/5m/1m trend agreement + OI buildup
regime + volume confirmation, fixed 1:2 management SL=5/TP=10).

DATA LIMITATION (same one already found and confirmed, not new): you asked
for 6 months, but NIFTY futures contracts roll monthly and expired ones
drop out of Dhan's instrument list, so real historical OI only goes back to
whenever the CURRENT contract started trading -- confirmed at ~90 calendar
days in every prior check. This script requests a wide window and simply
reports the real range Dhan actually returns, same as before.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

REQUEST_DAYS_BACK = 182  # ~6 months requested -- Dhan will return whatever it actually has
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=REQUEST_DAYS_BACK)).strftime("%Y-%m-%d")

OI_CHANGE_LOOKBACK_MIN = 15
AGGR_VOL_LOOKBACK = 20
SL_POINTS = dlb.SL_POINTS
TP_POINTS = dlb.TP_POINTS
MAX_BARS_IN_TRADE = dlb.MAX_BARS_IN_TRADE
TREND_FAST_EMA = dlb.TREND_FAST_EMA
TREND_SLOW_EMA = dlb.TREND_SLOW_EMA

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


def fetch_index_chunked(from_date, to_date, chunk_days=85):
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


def find_current_nifty_future():
    df = dlb.load_instrument_master()
    futs = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "FUT")
    ].copy()
    if futs.empty:
        raise SystemExit("No NIFTY futures contract found.")
    futs["SM_EXPIRY_DATE"] = pd.to_datetime(futs["SM_EXPIRY_DATE"])
    row = futs.sort_values("SM_EXPIRY_DATE").iloc[0]
    return row["SECURITY_ID"], row["DISPLAY_NAME"], row["SM_EXPIRY_DATE"].date()


def fetch_futures_oi_chunked(security_id, from_date, to_date, chunk_days=85):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=str(security_id), exchange_segment="NSE_FNO", instrument_type="FUTIDX",
            from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"), oi=True,
        )
        if isinstance(resp, dict) and resp.get("status") == "failure":
            print(f"  [{cur} to {chunk_end}] skipped: {resp.get('remarks', resp)}")
            cur = chunk_end + dt.timedelta(days=1)
            continue
        data = resp.get("data", resp)
        if isinstance(data, dict) and "open_interest" in data and len(data.get("timestamp", [])) > 0:
            d = pd.DataFrame({"oi": data["open_interest"], "timestamp": data["timestamp"]})
            d["time"] = pd.to_datetime(d["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            chunks.append(d.set_index("time").sort_index())
            print(f"  [{cur} to {chunk_end}] got {len(d)} candles.")
        else:
            print(f"  [{cur} to {chunk_end}] no OI in response.")
        cur = chunk_end + dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["oi"])
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def classify_oi_regime(price_change, oi_change):
    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    if price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    if price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    if price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    return "NEUTRAL"


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert classify_oi_regime(10, 5000) == "LONG_BUILDUP"
    assert classify_oi_regime(-10, 5000) == "SHORT_BUILDUP"
    assert classify_oi_regime(10, -5000) == "SHORT_COVERING"
    assert classify_oi_regime(-10, -5000) == "LONG_UNWINDING"
    print("All self-checks passed.\n")


def simulate_day(day_df):
    closes = day_df["close"].tolist()
    vols = day_df["volume"].tolist()
    trend_30m = day_df["trend_30m"].tolist()
    trend_15m = day_df["trend_15m"].tolist()
    trend_5m = day_df["trend_5m"].tolist()
    trend_1m = day_df["trend_1m"].tolist()
    oi_regime = day_df["oi_regime"].tolist()
    idx = day_df.index
    n = len(day_df)
    trades = []
    position = None
    prev_trend_1m = None
    for j in range(n):
        c = closes[j]
        if position is not None:
            exited, reason = False, None
            if position["side"] == "LONG":
                if c <= position["entry"] - SL_POINTS:
                    exited, reason = True, "STOP"
                elif c >= position["entry"] + TP_POINTS:
                    exited, reason = True, "TARGET"
            else:
                if c >= position["entry"] + SL_POINTS:
                    exited, reason = True, "STOP"
                elif c <= position["entry"] - TP_POINTS:
                    exited, reason = True, "TARGET"
            if not exited and (j - position["entry_bar"]) >= MAX_BARS_IN_TRADE:
                exited, reason = True, "TIME_EXIT"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j],
                               "side": position["side"], "entry": position["entry"], "exit": c,
                               "points": pts, "reason": reason})
                position = None
        if position is None:
            fresh_up = trend_1m[j] == "UP" and prev_trend_1m != "UP"
            fresh_down = trend_1m[j] == "DOWN" and prev_trend_1m != "DOWN"
            avg_vol_window = vols[max(0, j - AGGR_VOL_LOOKBACK):j]
            avg_vol = sum(avg_vol_window)/len(avg_vol_window) if avg_vol_window else 0
            vol_ok = avg_vol > 0 and vols[j] >= avg_vol
            all_up = trend_30m[j] == "UP" and trend_15m[j] == "UP" and trend_5m[j] == "UP"
            all_down = trend_30m[j] == "DOWN" and trend_15m[j] == "DOWN" and trend_5m[j] == "DOWN"
            if all_up and fresh_up and oi_regime[j] == "LONG_BUILDUP" and vol_ok:
                position = {"side": "LONG", "entry": c, "entry_time": idx[j], "entry_bar": j}
            elif all_down and fresh_down and oi_regime[j] == "SHORT_BUILDUP" and vol_ok:
                position = {"side": "SHORT", "entry": c, "entry_time": idx[j], "entry_bar": j}
        prev_trend_1m = trend_1m[j]
    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({"entry_time": position["entry_time"], "exit_time": idx[-1],
                       "side": position["side"], "entry": position["entry"], "exit": c,
                       "points": pts, "reason": "EOD"})
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
    run_self_checks()
    print("=== OI Standalone Concept -- re-test, NOT combined with anything else ===\n")

    fut_id, fut_symbol, fut_expiry = find_current_nifty_future()
    print(f"Using {fut_symbol} (id={fut_id}, expiry={fut_expiry})\n")

    print(f"Requesting {REQUEST_DAYS_BACK} days back for OI (you asked for 6 months) -- "
          f"Dhan will only actually return what's real:")
    oi_df = fetch_futures_oi_chunked(fut_id, FROM_DATE, TO_DATE)
    if oi_df.empty:
        print("No OI data returned at all.")
        return
    oi_start, oi_end = oi_df.index.min(), oi_df.index.max()
    print(f"\n*** REAL usable OI window: {oi_start} to {oi_end} "
          f"({(oi_end-oi_start).days} calendar days) -- NOT 6 months, per the contract-rollover limit. ***\n")

    price_df = fetch_index_chunked((oi_start - dt.timedelta(days=dlb.HTF_CONTEXT_DAYS)).strftime("%Y-%m-%d"),
                                    oi_end.strftime("%Y-%m-%d"))
    df = price_df.copy()
    for rule, col in (("30min", "trend_30m"), ("15min", "trend_15m"), ("5min", "trend_5m")):
        htf = dlb.ema_trend_series(dlb.resample_ohlc(df, rule)["close"], TREND_FAST_EMA, TREND_SLOW_EMA)
        combined_idx = df.index.union(htf.index)
        aligned = htf.reindex(combined_idx).sort_index().ffill()
        df[col] = aligned.reindex(df.index)
    df["trend_1m"] = dlb.ema_trend_series(df["close"], TREND_FAST_EMA, TREND_SLOW_EMA)

    df = df.join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()
    df = df.dropna(subset=["oi"])
    oi_change = df["oi"].diff(OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]
    df = df.dropna(subset=["trend_30m", "trend_15m", "trend_5m", "trend_1m"])

    days_covered = sorted(set(df.index.date))
    print(f"Final usable dataset: {len(df)} candles, {len(days_covered)} days: "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 30:
            continue
        all_trades.extend(simulate_day(day_df))

    if not all_trades:
        print("No trades generated in this window.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("oi_standalone_6month_trades.csv", index=False)

    wins = tdf[tdf["points"] > 0]
    print(f"Total: {len(tdf)}   Win rate: {100*len(wins)/len(tdf):.1f}%   Net points: {tdf['points'].sum():.2f}")
    print(f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")
    print(f"By side: {tdf.groupby('side')['points'].agg(['count','sum']).to_dict()}")

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

    rupee_pnl = tdf["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    print(f"\n--- Money simulation ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {capital_curve.iloc[-1]:,.2f}   "
          f"Max DD Rs {(capital_curve-capital_curve.cummax()).min():,.2f}")
    print("WARNING: wipes out at some point." if (capital_curve <= 0).any() else "Never wipes out.")

    mid = len(tdf)//2
    h1 = (tdf.iloc[:mid]["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    h2 = (tdf.iloc[mid:]["points"]*ASSUMED_DELTA*LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    print(f"\n--- Stability check ---\nFirst half: {mid} trades, net Rs {h1:,.2f}")
    print(f"Second half: {len(tdf)-mid} trades, net Rs {h2:,.2f}")


if __name__ == "__main__":
    main()
