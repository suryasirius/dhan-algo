"""
OI-Gated Concepts 2&4 -- Nifty 1-min
=================================================================
Standalone -- does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py.
Reuses compute_latest_signal / attach_htf_trend / DISABLED_CONCEPTS from
dhan_live_bot.py exactly as-is (same expanding-window per-bar call pattern
already used for the earlier "test all weekdays" check), only reading its
ALLOWED_WEEKDAYS attribute (overridden to all 5 days here, per your "full
day analysis" request -- the Tue/Wed restriction is a SEPARATE decision we
can re-apply afterward once we see the fresh breakdown).

Your request, translated into rules:
  "once the oi is favourable then final stragey can take trade accordingly"
  -> Concepts 2&4 generate the entry signal exactly as they already do
  (same trend filter, same concept logic, same weekday gate). On TOP of
  that, the trade is only actually taken if NIFTY futures OI confirms the
  same direction at that moment (LONG BUILDUP for a BUY, SHORT BUILDUP for
  a SELL -- reusing the same price+OI quadrant classification built and
  tested for the standalone OI concept). This is reported SIDE BY SIDE
  against the same signals WITHOUT the OI gate, on the identical date
  window, so the OI filter's real effect is isolated and not confused with
  a different backtest period.

  "1:1, 1:2, 1:5, full trailing, trail after 1:1, trail after 1:5" -> SIX
  management variants tested, all sharing the SAME 5-point stop (the
  locked strategy's own SL_POINTS) so this is an apples-to-apples
  comparison of exit management, not a new risk profile:
    Fixed_1_1              : 5pt stop / 5pt target, flat exit
    Fixed_1_2              : 5pt stop / 10pt target, flat exit (the ORIGINAL
                              locked management -- included as the baseline)
    Fixed_1_5               : 5pt stop / 25pt target, flat exit
    Trail_after_1_1         : 5pt stop, switches to a 3pt trail once profit
                              first reaches 5pts (1:1)
    Trail_after_1_5         : 5pt stop, switches to a 3pt trail once profit
                              first reaches 25pts (1:5)
    Full_Trailing_From_Start: 3pt trail active from the very first bar,
                              no fixed target at all

  "short and call separate analysis" -> LONG (Call) vs SHORT (Put) reported
  separately for every variant.

  "30 minutes time session analysis" / "full day analysis" -> the usual
  30-min session breakdown and weekday breakdown for every variant.

HARD DATA LIMITATION (already found and confirmed via diagnostic, not a
choice): NIFTY futures contracts roll monthly and expired ones drop out of
Dhan's instrument list entirely, so real historical OI is only available
back to whenever the CURRENT contract started trading -- confirmed to be
~90 calendar days, not the full year requested. This script uses whatever
real window Dhan actually returns (prints it explicitly) for BOTH the
gated and ungated comparison, so the comparison is fair even though it
isn't the full year. A true full-year test is only possible for the
UNGATED (no-OI) side, since that doesn't depend on futures OI at all --
flagged separately in the output if you want that run too.
"""

import datetime as dt
import pandas as pd

import dhan_live_bot as dlb

FROM_DATE_1YR = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OI_REQUEST_DAYS_BACK = 120
OI_CHANGE_LOOKBACK_MIN = 15
AGGR_MIN_WARMUP_BARS = 30  # skip the first few bars of each day (not enough context yet)

TRAIL_PTS = 3
SL_POINTS = dlb.SL_POINTS  # 5 -- reuse the locked strategy's own stop, not a new number

MANAGEMENT_VARIANTS = [
    # (name, target_pts_or_None, trail_after_target, full_trail_from_start)
    ("Fixed_1_1", SL_POINTS * 1, False, False),
    ("Fixed_1_2", SL_POINTS * 2, False, False),
    ("Fixed_1_5", SL_POINTS * 5, False, False),
    ("Trail_after_1_1", SL_POINTS * 1, True, False),
    ("Trail_after_1_5", SL_POINTS * 5, True, False),
    ("Full_Trailing_From_Start", None, False, True),
]

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


# ---------------- Data fetching ----------------

def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        resp = dlb.dhan.intraday_minute_data(
            security_id=dlb.UNDERLYING_SECURITY_ID,
            exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
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
    df = dlb.load_instrument_master()
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


def fetch_futures_oi_chunked(security_id, from_date, to_date, chunk_days=CHUNK_DAYS):
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
            df = pd.DataFrame({"oi": data["open_interest"], "timestamp": data["timestamp"]})
            df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
            chunks.append(df.set_index("time").sort_index())
            print(f"  [{cur} to {chunk_end}] got {len(df)} candles.")
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


# ---------------- Signal detection (shared, once per day) ----------------

def detect_concepts_signals_for_day(day_df):
    """Same expanding-window pattern as test_concepts_2_4_all_days.py --
    calls dlb.compute_latest_signal() on a growing slice of the day so we
    get a signal reading at every bar, not just the last one."""
    signals = []
    n = len(day_df)
    for i in range(AGGR_MIN_WARMUP_BARS, n):
        visible = day_df.iloc[:i + 1]
        sig = dlb.compute_latest_signal(visible)
        if sig is None:
            continue
        if sig["final_buy"]:
            signals.append({"bar": i, "side": "LONG"})
        elif sig["final_sell"]:
            signals.append({"bar": i, "side": "SHORT"})
    return signals


# ---------------- Shared exit management ----------------

def simulate_with_signals(day_df, entry_signals, target_pts, trail_after_target, full_trail_from_start):
    closes = day_df["close"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)

    signals_by_bar = {}
    for s in entry_signals:
        signals_by_bar.setdefault(s["bar"], s["side"])

    position = None
    trades = []

    for j in range(n):
        c = closes[j]
        h = highs[j]
        l = lows[j]

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
                    elif target_pts is not None and c >= position["entry"] + target_pts:
                        if trail_after_target:
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
                    elif target_pts is not None and c <= position["entry"] - target_pts:
                        if trail_after_target:
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
                stop = entry - SL_POINTS
                stop_level = (l - TRAIL_PTS) if full_trail_from_start else None
            else:
                stop = entry + SL_POINTS
                stop_level = (h + TRAIL_PTS) if full_trail_from_start else None
            position = {"side": side, "entry": entry, "entry_time": idx[j], "stop": stop,
                        "trailing_active": full_trail_from_start, "stop_level": stop_level}

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
        return None
    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    wins = tdf[tdf["points"] > 0]
    losses = tdf[tdf["points"] <= 0]

    print(f"\n===== {label} =====")
    print(f"Total: {len(tdf)}  Win%: {100*len(wins)/len(tdf):.1f}  Net pts: {tdf['points'].sum():.2f}  "
          f"Exit reasons: {tdf['reason'].value_counts().to_dict()}")

    for side in ["LONG", "SHORT"]:
        g = tdf[tdf["side"] == side]
        if len(g):
            label_side = "LONG (Call)" if side == "LONG" else "SHORT (Put)"
            print(f"  {label_side}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    tdf.to_csv(filename, index=False)

    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    parts = []
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            parts.append(f"{wd[:3]}={len(g)}/net{g['points'].sum():.0f}")
    print("  By weekday:", ", ".join(parts))

    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    sess_stats = []
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            sess_stats.append((s, len(g), g["points"].sum()))
    sess_stats.sort(key=lambda x: x[2])
    if sess_stats:
        print(f"  Worst session: {sess_stats[0]}   Best session: {sess_stats[-1]}")

    rupee_pnl = tdf["points"] * ASSUMED_DELTA * LOT_SIZE - BROKERAGE_PER_TRADE
    capital_curve = STARTING_CAPITAL + rupee_pnl.cumsum()
    max_dd = (capital_curve - capital_curve.cummax()).min()
    final_capital = capital_curve.iloc[-1]
    print(f"  Money sim: Rs {STARTING_CAPITAL:,} -> Rs {final_capital:,.2f}  (max DD Rs {max_dd:,.2f})",
          "  WIPES OUT" if (capital_curve <= 0).any() else "")

    mid = len(tdf) // 2
    h1_net = rupee_pnl.iloc[:mid].sum()
    h2_net = rupee_pnl.iloc[mid:].sum()
    print(f"  Stability: H1(n={mid}) Rs{h1_net:,.2f}  H2(n={len(tdf)-mid}) Rs{h2_net:,.2f}"
          f"{'  -- SAMPLE TOO THIN TO TRUST' if len(tdf) < 20 else ''}")

    return {"label": label, "trades": len(tdf), "win_rate": 100*len(wins)/len(tdf),
            "net_points": tdf["points"].sum(), "final_capital": final_capital, "max_dd": max_dd,
            "h1_net": h1_net, "h2_net": h2_net}


def main():
    run_self_checks()

    print("=== OI-Gated Concepts 2&4 vs Ungated -- same window, 6 management variants ===")
    print("NOTE: testing ALL WEEKDAYS (not just Tue/Wed) per your 'full day analysis' request.\n")

    dlb.ALLOWED_WEEKDAYS = {0, 1, 2, 3, 4}
    assert dlb.ALLOWED_WEEKDAYS == {0, 1, 2, 3, 4}

    print("Finding the current NIFTY futures contract...")
    fut_security_id, fut_symbol, fut_expiry = find_current_nifty_future()
    print(f"Using {fut_symbol} (security_id={fut_security_id}, expiry={fut_expiry})\n")

    print("Fetching NIFTY futures OI history (this sets our real usable window)...")
    from_date_oi = (dt.date.today() - dt.timedelta(days=OI_REQUEST_DAYS_BACK)).strftime("%Y-%m-%d")
    oi_df = fetch_futures_oi_chunked(fut_security_id, from_date_oi, TO_DATE)
    if oi_df.empty:
        print("No OI data returned at all -- stopping here.")
        return
    oi_start, oi_end = oi_df.index.min(), oi_df.index.max()
    print(f"\nReal OI data available from {oi_start} to {oi_end} "
          f"({(oi_end - oi_start).days} calendar days).")
    print("*** This is the actual window for BOTH the gated and ungated comparison below --")
    print("*** NOT the full 1-year you asked for. A full-year run is only possible for the")
    print("*** ungated (no-OI) side of Concepts 2&4, since that alone doesn't need futures OI.\n")

    print("Fetching Nifty index price data + extra warmup history for trend context...")
    warmup_start = (oi_start - dt.timedelta(days=dlb.HTF_CONTEXT_DAYS)).strftime("%Y-%m-%d")
    price_df = fetch_index_chunked(warmup_start, oi_end.strftime("%Y-%m-%d"))
    price_df = dlb.attach_htf_trend(price_df)

    print("Joining futures OI onto the index price series...")
    df = price_df.join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()

    oi_change = df["oi"].diff(OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]

    test_df = df[df.index.date >= oi_start.date()]
    days_covered = sorted(set(test_df.index.date))
    print(f"\nTest window: {len(test_df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    print("Detecting Concepts 2&4 signals for every day (this is the slow part, please wait)...")
    all_raw_signals = {}
    for day in days_covered:
        day_df = df[df.index.date == day]  # use FULL df (with warmup) so trend context is real
        if len(day_df) < AGGR_MIN_WARMUP_BARS + 10:
            continue
        all_raw_signals[day] = detect_concepts_signals_for_day(day_df)
    total_raw = sum(len(v) for v in all_raw_signals.values())
    print(f"Done. {total_raw} raw Concepts 2&4 signals found across the test window.\n")

    results = []
    for gated in [False, True]:
        tag = "OI_GATED" if gated else "UNGATED"
        for variant_name, target_pts, trail_after, full_trail in MANAGEMENT_VARIANTS:
            all_trades = []
            for day in days_covered:
                day_df = df[df.index.date == day]
                raw_signals = all_raw_signals.get(day, [])
                if not raw_signals:
                    continue
                if gated:
                    signals = []
                    for s in raw_signals:
                        regime = day_df["oi_regime"].iloc[s["bar"]]
                        if (s["side"] == "LONG" and regime == "LONG_BUILDUP") or \
                           (s["side"] == "SHORT" and regime == "SHORT_BUILDUP"):
                            signals.append(s)
                else:
                    signals = raw_signals
                if not signals:
                    continue
                all_trades.extend(simulate_with_signals(day_df, signals, target_pts, trail_after, full_trail))

            label = f"{tag} | {variant_name}"
            filename = f"oi_gated_concepts_{tag}_{variant_name}_trades.csv"
            res = report(all_trades, label, filename)
            if res:
                results.append(res)

    print("\n\n===================== SUMMARY -- ALL VARIANTS =====================")
    print(f"{'Label':<32}{'Trades':>8}{'Win %':>8}{'NetPts':>9}{'End Rs':>11}{'MaxDD Rs':>11}")
    for r in results:
        print(f"{r['label']:<32}{r['trades']:>8}{r['win_rate']:>7.1f}%{r['net_points']:>9.2f}"
              f"{r['final_capital']:>11,.2f}{r['max_dd']:>11,.2f}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
