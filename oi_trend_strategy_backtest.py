"""
OI + Multi-Timeframe Trend Confirmation Strategy -- Nifty 1-min
=================================================================
Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py.

Your idea, translated into rules:
  - "OI data big on one side supports the long trend" -> classify each bar
    using the classic price+OI quadrant read on NIFTY FUTURES open interest:
        price up  + OI up   = LONG BUILDUP   (fresh buying -- real conviction)
        price down+ OI up   = SHORT BUILDUP  (fresh selling -- real conviction)
        price up  + OI down = SHORT COVERING (weaker -- shorts just exiting)
        price down+ OI down = LONG UNWINDING (weaker -- longs just exiting)
    Only LONG BUILDUP / SHORT BUILDUP count as real OI confirmation here --
    the "covering"/"unwinding" cases are explicitly NOT treated as fresh
    conviction, since that's the whole point of using OI instead of price
    alone.
  - "30min, 15min, 5min long trend, then short trend 1min also follow" ->
    ALL FOUR timeframes' EMA(9/21) trend reading must agree before a trade
    is allowed. The 1-minute chart's OWN trend flipping to match the higher
    timeframes is what triggers the actual entry (not just being in
    agreement passively).
  - "volume also support this theory" -> the entry bar's volume must be
    above its own trailing 20-bar average.

All four conditions (30m+15m+5m trend agree, 1m trend just flipped to match,
OI regime confirms same direction, volume above average) must be true
together for an entry -- otherwise no trade.

Exits: SAME fixed 1:2 management as the already-proven Concepts 2&4
(SL_POINTS=5, TP_POINTS=10, MAX_BARS_IN_TRADE=15), imported directly from
dhan_live_bot.py so this is judged on the identical yardstick as everything
else in this project, not a new made-up risk profile.

DATA LIMITATION (confirmed via diagnostic, not guessed): NIFTY futures
contracts roll monthly and expired ones drop out of Dhan's instrument list,
so real historical OI is only available back to whenever the CURRENT
contract started trading -- confirmed to be ~90 calendar days here, not the
full year used for other concepts in this project. Results below should be
read with that smaller sample in mind.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, load_instrument_master, resample_ohlc, ema_trend_series,
    UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    SL_POINTS, TP_POINTS, MAX_BARS_IN_TRADE, TREND_FAST_EMA, TREND_SLOW_EMA,
)

REQUEST_DAYS_BACK = 120  # ask for plenty -- Dhan will just return what's actually available
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
FROM_DATE = (dt.date.today() - dt.timedelta(days=REQUEST_DAYS_BACK)).strftime("%Y-%m-%d")

OI_CHANGE_LOOKBACK_MIN = 15   # rolling window (minutes) used to classify the OI regime
AGGR_VOL_LOOKBACK = 20        # trailing bars for the "volume supports it" average
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 10_000
LOT_SIZE = 65
ASSUMED_DELTA = 0.6

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------- Data fetching ----------------

def fetch_index_chunked(from_date, to_date, chunk_days=85):
    """Same pattern as every other script in this project -- Nifty INDEX
    1-min price data, used for the actual trend/entry logic. Includes
    volume too (the index feed does carry real volume, confirmed earlier
    in this project)."""
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dhan.intraday_minute_data(
            security_id=UNDERLYING_SECURITY_ID,
            exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=UNDERLYING_INSTRUMENT_TYPE,
            from_date=cur.strftime("%Y-%m-%d"),
            to_date=chunk_end.strftime("%Y-%m-%d"),
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
    df = load_instrument_master()
    futs = df[
        (df["UNDERLYING_SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (df["EXCH_ID"].astype(str).str.upper() == "NSE")
        & (df["INSTRUMENT_TYPE"].astype(str).str.upper() == "FUT")
    ].copy()
    if futs.empty:
        raise SystemExit("No NIFTY futures contract found in the instrument master.")
    futs["SM_EXPIRY_DATE"] = pd.to_datetime(futs["SM_EXPIRY_DATE"])
    row = futs.sort_values("SM_EXPIRY_DATE").iloc[0]
    return row["SECURITY_ID"], row["DISPLAY_NAME"], row["SM_EXPIRY_DATE"].date()


def fetch_futures_oi_chunked(security_id, from_date, to_date, chunk_days=85):
    """NIFTY FUTURES 1-min OHLCV + real Open Interest (oi=True). Dhan may
    reject the portion of the requested range from before this contract
    existed -- that's printed explicitly per chunk (not silently skipped),
    and we just use whatever chunks actually succeeded."""
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment="NSE_FNO",
            instrument_type="FUTIDX",
            from_date=cur.strftime("%Y-%m-%d"),
            to_date=chunk_end.strftime("%Y-%m-%d"),
            oi=True,
        )
        if isinstance(resp, dict) and resp.get("status") == "failure":
            print(f"  [{cur} to {chunk_end}] Dhan returned an error, skipping this chunk: "
                  f"{resp.get('remarks', resp)}")
            cur = chunk_end + dt.timedelta(days=1)
            continue
        data = resp.get("data", resp)
        if isinstance(data, dict) and "open_interest" in data and len(data.get("timestamp", [])) > 0:
            df = pd.DataFrame({
                "oi": data["open_interest"], "fut_volume": data["volume"],
                "timestamp": data["timestamp"],
            })
            df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            chunks.append(df.set_index("time").sort_index())
            print(f"  [{cur} to {chunk_end}] got {len(df)} candles.")
        else:
            print(f"  [{cur} to {chunk_end}] no open_interest/timestamp in response -- raw content: "
                  f"{list(data.keys()) if isinstance(data, dict) else str(data)[:300]}")
        cur = chunk_end + dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["oi", "fut_volume"])
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


# ---------------- OI regime classification (unit-tested below) ----------------

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
    print("Running self-checks on OI regime classification (synthetic data)...")
    assert classify_oi_regime(10, 5000) == "LONG_BUILDUP"
    assert classify_oi_regime(-10, 5000) == "SHORT_BUILDUP"
    assert classify_oi_regime(10, -5000) == "SHORT_COVERING"
    assert classify_oi_regime(-10, -5000) == "LONG_UNWINDING"
    assert classify_oi_regime(0, 5000) == "NEUTRAL"
    assert classify_oi_regime(10, 0) == "NEUTRAL"
    print("All self-checks passed.\n")


# ---------------- Day simulation ----------------

def simulate_day(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
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
                stop = position["entry"] - SL_POINTS
                target = position["entry"] + TP_POINTS
                if c <= stop:
                    exited, reason = True, "STOP"
                elif c >= target:
                    exited, reason = True, "TARGET"
                elif (j - position["entry_bar"]) >= MAX_BARS_IN_TRADE:
                    exited, reason = True, "TIME_EXIT"
            else:
                stop = position["entry"] + SL_POINTS
                target = position["entry"] - TP_POINTS
                if c >= stop:
                    exited, reason = True, "STOP"
                elif c <= target:
                    exited, reason = True, "TARGET"
                elif (j - position["entry_bar"]) >= MAX_BARS_IN_TRADE:
                    exited, reason = True, "TIME_EXIT"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({"entry_time": position["entry_time"], "exit_time": idx[j],
                               "side": position["side"], "entry": position["entry"], "exit": c,
                               "points": pts, "reason": reason})
                position = None

        if position is None:
            fresh_flip_up = trend_1m[j] == "UP" and prev_trend_1m != "UP"
            fresh_flip_down = trend_1m[j] == "DOWN" and prev_trend_1m != "DOWN"

            avg_vol_window = vols[max(0, j - AGGR_VOL_LOOKBACK):j]
            avg_vol = sum(avg_vol_window) / len(avg_vol_window) if avg_vol_window else 0
            volume_supports = avg_vol > 0 and vols[j] >= avg_vol

            all_up = (trend_30m[j] == "UP" and trend_15m[j] == "UP" and trend_5m[j] == "UP")
            all_down = (trend_30m[j] == "DOWN" and trend_15m[j] == "DOWN" and trend_5m[j] == "DOWN")

            if all_up and fresh_flip_up and oi_regime[j] == "LONG_BUILDUP" and volume_supports:
                position = {"side": "LONG", "entry": c, "entry_time": idx[j], "entry_bar": j}
            elif all_down and fresh_flip_down and oi_regime[j] == "SHORT_BUILDUP" and volume_supports:
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
        start_s, end_s = label.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        if dt.time(sh, sm) <= t < dt.time(eh, em):
            return label
    return "Outside usual session"


def main():
    run_self_checks()

    print("=== OI + Multi-Timeframe Trend Confirmation Strategy ===\n")

    print("Finding the current NIFTY futures contract...")
    fut_security_id, fut_symbol, fut_expiry = find_current_nifty_future()
    print(f"Using {fut_symbol} (security_id={fut_security_id}, expiry={fut_expiry})\n")

    print("Fetching NIFTY futures OI history (this determines our real usable window)...")
    oi_df = fetch_futures_oi_chunked(fut_security_id, FROM_DATE, TO_DATE)
    if oi_df.empty:
        print("\nNo OI data returned at all across the whole requested range -- see the "
              "per-chunk messages above for the exact reason. Stopping here.")
        return
    oi_start, oi_end = oi_df.index.min(), oi_df.index.max()
    print(f"\nReal OI data available from {oi_start} to {oi_end} "
          f"({(oi_end - oi_start).days} calendar days) -- this is the actual backtest window, "
          f"not the full year used for other concepts in this project.\n")

    print("Fetching NIFTY index price data for the same window...")
    price_df = fetch_index_chunked(oi_start.strftime("%Y-%m-%d"), oi_end.strftime("%Y-%m-%d"))

    print("Computing 30m/15m/5m/1m trend context...")
    df = price_df.copy()
    for rule, col in (("30min", "trend_30m"), ("15min", "trend_15m"), ("5min", "trend_5m")):
        htf_trend = ema_trend_series(resample_ohlc(df, rule)["close"], TREND_FAST_EMA, TREND_SLOW_EMA)
        combined_index = df.index.union(htf_trend.index)
        aligned = htf_trend.reindex(combined_index).sort_index().ffill()
        df[col] = aligned.reindex(df.index)
    df["trend_1m"] = ema_trend_series(df["close"], TREND_FAST_EMA, TREND_SLOW_EMA)

    print("Joining futures OI onto the index price series (aligned by minute, forward-filled)...")
    df = df.join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()
    df = df.dropna(subset=["oi"])

    oi_change = df["oi"].diff(OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]

    df = df.dropna(subset=["trend_30m", "trend_15m", "trend_5m", "trend_1m"])
    if df.empty:
        print("No usable rows after joining OI + trend context -- stopping here.")
        return

    days_covered = sorted(set(df.index.date))
    print(f"\nFinal usable dataset: {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < 30:
            continue
        all_trades.extend(simulate_day(day_df))

    if not all_trades:
        print("No trades were generated at all. The 30m+15m+5m+1m-agreement + OI-buildup + "
              "volume filters stacked together may be too strict for this (short) window.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv("oi_trend_strategy_trades.csv", index=False)
    print(f"Saved {len(trades_df)} trades to oi_trend_strategy_trades.csv\n")

    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]
    print(f"Total trades: {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}")
    print(f"By side: {trades_df.groupby('side')['points'].agg(['count','sum']).to_dict()}")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    print("\n--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    print("\n--- By 30-min session ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    rupee_pnl_net = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    trades_df["running_capital"] = STARTING_CAPITAL + rupee_pnl_net.cumsum()
    trades_df.to_csv("oi_trend_strategy_trades.csv", index=False)

    final_capital = trades_df["running_capital"].iloc[-1]
    max_dd = (trades_df["running_capital"] - trades_df["running_capital"].cummax()).min()
    print(f"\n--- Money simulation (Rs {STARTING_CAPITAL:,} start, Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage) ---")
    print(f"Ending capital: Rs {final_capital:,.2f}   Max drawdown: Rs {max_dd:,.2f}")
    if (trades_df["running_capital"] <= 0).any():
        wipeout_idx = trades_df[trades_df["running_capital"] <= 0].index[0]
        print(f"WARNING: capital wipes out at trade #{wipeout_idx+1} ({trades_df.loc[wipeout_idx,'entry_time']})")
    else:
        print("Never wipes out across the full period.")

    mid = len(trades_df) // 2
    h1, h2 = trades_df.iloc[:mid], trades_df.iloc[mid:]
    h1_net = (h1["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    h2_net = (h2["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE).sum()
    print(f"\n--- Stability check (NOTE: short window, take this with real caution) ---")
    print(f"First half:  {len(h1)} trades, net Rs {h1_net:,.2f}")
    print(f"Second half: {len(h2)} trades, net Rs {h2_net:,.2f}")


if __name__ == "__main__":
    main()
