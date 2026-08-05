"""
COMBINED standalone backtest v2 -- Setup 2 Liquidity Sweep DROPPED per your
decision: despite the highest raw return of the three candidates, it had
the worst win rate (40.7%) and worst stop-loss hit rate (57.6%) of
everything that wasn't already struck down, and it was the strategy
directly implicated in the combined pool's negative-equity dip in the
prior 2-year test (capital went to -Rs2,629 around trade #664, April 2025,
before recovering). Keeping only the two candidates that actually meet all
three of your criteria together (win probability, return, and NOT hitting
stop-loss often):

  1. Concepts 2 & 4 (Tue/Wed only, 5-pt stop / 10-pt target fixed 1:2,
     5m+15m trend filter) -- imported directly from dhan_live_bot.py so
     it's always in sync with FINAL_*.py. 43.9% win rate, 55.9% stop rate.
  2. Strike Momentum (Mon/Wed/Fri only, entries restricted to 09:15-09:45,
     10-pt stop switching to 3-pt candle trail). 43.8% win rate, and the
     best stop rate of anything tested (28.8% -- most trades exit via the
     trailing stop, not a hard loss).

This is a "realistic" pass in two senses:
  - A rough BROKERAGE_PER_TRADE cost (Rs40/round trip, adjust if you know
    your real Dhan cost) is deducted from every trade in the combined money
    simulation, same as before.
  - The combined capital curve is checked trade-by-trade for whether it
    EVER goes to zero or below, not just whether the final number is
    positive -- exactly the check that caught Setup 2's problem, now
    re-applied to confirm whether removing Setup 2 actually fixes it.

Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py.
This is still a test to decide whether to fold Strike Momentum into the
real FINAL files alongside Concepts 2&4 -- that integration has not
happened yet.

Money simulation is still approximate: real historical option premiums for
expired weekly contracts aren't retrievable, so index point moves are
scaled by an assumed delta (0.6) and the current Nifty lot size (65) --
except Strike Momentum, whose points are already delta-scaled synthetic
premium, so only the lot size is applied to it.
"""

import datetime as dt
import pandas as pd

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
    compute_latest_signal, SL_POINTS, TP_POINTS, MAX_BARS_IN_TRADE, DISABLED_CONCEPTS,
    attach_htf_trend, REQUIRE_HTF_TREND_MATCH, ALLOWED_WEEKDAYS as CONCEPTS_ALLOWED_WEEKDAYS,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85  # Dhan caps a single request at 90 days of minute data

CONCEPT_NAMES = {2: "2 Compression spike (sell)", 4: "4 Rollover (sell)"}

# ---- Strike Momentum ----
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
CONCEPTS_ASSUMED_DELTA = 0.6
BROKERAGE_PER_TRADE = 40.0  # rough placeholder -- applied to every trade, both strategies

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


# ------------------------- Concepts 2 & 4 -------------------------
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


# ------------------------- Strike Momentum -------------------------
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
        })
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


def report_strategy(trades, label, filename):
    if not trades:
        print(f"\n--- {label}: no trades. ---")
        return None
    trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]

    print(f"\n===== {label} =====")
    print(f"Trades: {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    stop_ct = (trades_df["reason"] == "STOP").sum()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}  "
          f"(STOP rate: {100*stop_ct/len(trades_df):.1f}%)")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)

    print(f"\n-- {label}: 30-MINUTE SESSION BREAKDOWN (all days) --")
    print(f"{'Session':<16}{'Trades':>8}{'Wins':>8}{'Win %':>9}{'Net Pts':>12}")
    for session in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == session]
        if len(g) == 0:
            continue
        w = (g["points"] > 0).sum()
        print(f"{session:<16}{len(g):>8}{w:>8}{100*w/len(g):>8.1f}%{g['points'].sum():>12.2f}")

    print(f"\n-- {label}: 30-MINUTE SESSION BREAKDOWN, BY WEEKDAY --")
    for wd in WEEKDAY_ORDER:
        wd_df = trades_df[trades_df["weekday"] == wd]
        if wd_df.empty:
            continue
        print(f"  {wd} ({len(wd_df)} trades, net {wd_df['points'].sum():.2f}):")
        for session in SESSIONS + ["Outside usual session"]:
            g = wd_df[wd_df["session"] == session]
            if len(g) == 0:
                continue
            w = (g["points"] > 0).sum()
            print(f"    {session:<16}{len(g):>6}  win%={100*w/len(g):5.1f}  net={g['points'].sum():8.2f}")

    trades_df.to_csv(filename, index=False)
    print(f"\nSaved to {filename}")
    return trades_df


def main():
    print("=== COMBINED FINAL STRATEGY v2 -- Concepts 2&4 + Strike Momentum ONLY (Setup 2 dropped), 2 YEARS ===\n")

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

    print("\n\n################ PER-STRATEGY RESULTS (points, no costs) ################")
    concept_df = report_strategy(concept_trades, "Concepts 2 & 4 (Tue/Wed, fixed 1:2)", "combined2yr_v2_concepts_trades.csv")
    if concept_df is not None:
        for cid, g in concept_df.groupby("concept_id"):
            wr = 100 * (g["points"] > 0).mean()
            print(f"    {CONCEPT_NAMES.get(cid, cid):<28} trades={len(g):<4} win rate={wr:5.1f}%  net={g['points'].sum():8.2f}")

    sm_df = report_strategy(sm_trades, "Strike Momentum (Mon/Wed/Fri, 09:15-09:45)", "combined2yr_v2_strikemomentum_trades.csv")

    if concept_df is None and sm_df is None:
        print("\nNo trades at all across either strategy in this date range.")
        return

    print("\n\n################ COMBINED MONEY SIMULATION (with rough per-trade costs) ################")
    parts = []
    if concept_df is not None:
        concept_df["rupee_pnl_gross"] = concept_df["points"] * CONCEPTS_ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE
        parts.append(concept_df[["entry_time", "rupee_pnl_gross"]].assign(strategy="Concepts_2_4"))
    if sm_df is not None:
        sm_df["rupee_pnl_gross"] = sm_df["points"] * LOT_SIZE * LOTS_PER_TRADE
        parts.append(sm_df[["entry_time", "rupee_pnl_gross"]].assign(strategy="StrikeMomentum"))

    combined = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    combined["brokerage"] = BROKERAGE_PER_TRADE
    combined["rupee_pnl_net"] = combined["rupee_pnl_gross"] - BROKERAGE_PER_TRADE
    net_curve = STARTING_CAPITAL + combined["rupee_pnl_net"].cumsum()
    combined["running_capital"] = net_curve   # computed BEFORE saving this time
    combined.to_csv("combined2yr_v2_ALL_trades.csv", index=False)

    print(f"Total combined trades across both strategies: {len(combined)}")
    print("Trades per strategy:", combined["strategy"].value_counts().to_dict())

    gross_curve = STARTING_CAPITAL + combined["rupee_pnl_gross"].cumsum()
    print(f"\nGROSS (no costs): end Rs{gross_curve.iloc[-1]:,.2f}   "
          f"P&L Rs{gross_curve.iloc[-1]-STARTING_CAPITAL:,.2f}   "
          f"worst drawdown Rs{(gross_curve-gross_curve.cummax()).min():,.2f}")

    net_dd = (net_curve - net_curve.cummax()).min()
    total_brokerage = BROKERAGE_PER_TRADE * len(combined)

    print(f"\nNET (after Rs{BROKERAGE_PER_TRADE:.0f}/trade assumed cost, Rs{total_brokerage:,.0f} total "
          f"over {len(combined)} trades):")
    print(f"  Starting capital: Rs{STARTING_CAPITAL:,.2f}")
    print(f"  Ending capital:   Rs{net_curve.iloc[-1]:,.2f}")
    print(f"  Total P&L:        Rs{net_curve.iloc[-1]-STARTING_CAPITAL:,.2f}")
    print(f"  Return on capital: {100*(net_curve.iloc[-1]-STARTING_CAPITAL)/STARTING_CAPITAL:.1f}%")
    print(f"  Worst drawdown:   Rs{net_dd:,.2f}")
    print(f"  Lowest point ever reached: Rs{net_curve.min():,.2f} "
          f"(trade #{int(net_curve.idxmin())+1}, {combined.loc[net_curve.idxmin(), 'entry_time']})")
    if (net_curve <= 0).any():
        idx0 = (net_curve <= 0).idxmax()
        print(f"  WARNING: NET combined capital hits Rs0 or below at trade #{combined.index.get_loc(idx0)+1} "
              f"({combined.loc[idx0, 'entry_time']}, strategy={combined.loc[idx0, 'strategy']})")
    else:
        print("  CONFIRMED: capital never touches Rs0 or below at any point, even after costs -- no wipeout.")

    # stability check
    combined["date"] = combined["entry_time"].dt.date
    dates = sorted(combined["date"].unique())
    mid = dates[len(dates) // 2]
    h1 = combined[combined["date"] < mid]
    h2 = combined[combined["date"] >= mid]
    print(f"\nStability: H1 ({dates[0]} to {mid}) n={len(h1)} net_after_cost=Rs{(h1['rupee_pnl_gross']-h1['brokerage']).sum():,.2f}")
    print(f"           H2 ({mid} to {dates[-1]}) n={len(h2)} net_after_cost=Rs{(h2['rupee_pnl_gross']-h2['brokerage']).sum():,.2f}")

    print("\nFull combined trade list (with running_capital) saved to combined2yr_v2_ALL_trades.csv")
    print("Individual strategy trade lists also saved (see filenames printed above).")


if __name__ == "__main__":
    main()
