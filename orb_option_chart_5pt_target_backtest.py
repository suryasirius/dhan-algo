"""
Opening-Range (09:15-09:25) Breakout, Read Entirely Off the Option's Own
Premium Chart -- +5pt Target with Floor-Protected 2pt Trail, 3-min Time-Stop,
Initial-SL Sweep (3/4/5 pts), Multiple Entries/Day -- standalone backtest
=================================================================================

STRATEGY AS SPECIFIED BY THE USER:
  1. At market open, mark the High and Low of "the market" from 09:15 to
     09:25 (a 10-minute opening range) -- per the user's explicit
     instruction ("don't check the points in Nifty chart, entire strategy
     should be looked into the option chart"), this range is marked on an
     OPTION'S OWN premium chart, not the Nifty index.
  2. When price breaks the High and 2 consecutive 30-second candles close
     above it -> buy a CALL. When price breaks the Low and 2 consecutive
     30-second candles close below it -> buy a PUT.
  3. Target: +5 points. Before the target is ever reached, also exit if
     3 minutes pass without hitting it (time-stop) or the initial stop.
  4. Once unrealized profit ever reaches +5, the time-stop no longer
     applies; a trailing stop of 2 points activates, but the user was
     explicit that this trail must never lock in less than +5 -- their
     own example: peak profit of +7 -> trailing stop = +5 (7-2), i.e. the
     stop is max(entry+5, peak-2), not peak-2 unconditionally.
  5. Initial stop-loss is swept across 3, 4, and 5 points (three separate
     runs), all other rules unchanged.
  6. Multiple entries per day are allowed -- every fresh breakout+confirm
     of the SAME fixed (09:15-09:25) high/low can trigger a new trade,
     any number of times, sequentially (never overlapping positions).
  7. Report day-wise and 30-minute-session breakdowns.

TWO ASSUMPTIONS FLAGGED EXPLICITLY (the source instructions were ambiguous
on these two points -- proceeding with the most literal/simple reading
rather than blocking on it, per this project's standing practice of
documenting simplifications instead of guessing silently):

  (A) SINGLE REFERENCE CHART: the instructions say "the high" / "the low"
      (singular), not "each option's own high/low" -- so this build uses
      ONE reference series (the day's nearest-ITM CALL, fixed at 09:15)
      to mark the opening range AND detect both the high-break (-> buy
      that same CALL) and the low-break (-> buy the day's nearest-ITM
      PUT, a different instrument, priced on its own proxy premium
      series). If the actual intent was two independent charts (the CALL
      watching its own high, the PUT separately watching its own high,
      as in the earlier "ITM Option Breakout" strategy), that's a
      one-line change (see `opening_range()` / `scan_and_simulate_day()`)
      -- flag it and this can be rebuilt.

  (B) 30-SECOND CANDLES: Dhan's historical intraday API (the only real
      data source this project has) only returns 1-MINUTE bars -- there
      is no sub-minute historical endpoint. Requiring "2 consecutive
      30-second closes beyond the level" is therefore not directly
      testable. The closest faithful proxy is a SINGLE 1-minute bar
      closing beyond the level (one minute = two 30-second sub-periods),
      used here as CONFIRM_ON_CLOSE. The stricter alternative -- 2 full
      *1-minute* bars (i.e. testing a 2-minute hold, not 1) -- would
      overstate the confirmation requirement and change the strategy's
      character; it's available as a toggle (see CONFIRM_BARS) if wanted.

DATA-ACCESS NOTE: this sandbox's network egress cannot reach api.dhan.co
(tested directly -- blocked with a 403 at the proxy level), even though
the C:\\dhan algo project and its real credentials are connected. The
self-checks and smoke test below run entirely on synthetic, fabricated
data and prove the logic is correct; they do NOT require network access
and have been run successfully in-sandbox. The REAL 6-month result can
only be produced by running this file where Dhan connectivity works --
i.e. on your own machine:

    cd "C:\\dhan algo"
    python orb_option_chart_5pt_target_backtest.py

Standalone -- does not modify dhan_live_bot.py or any other file.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

# ------------------------------- CONFIG -------------------------------
BACKTEST_MONTHS = 6
FROM_DATE = (dt.date.today() - dt.timedelta(days=int(BACKTEST_MONTHS * 30.44))).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OPEN_RANGE_START = dt.time(9, 15)
OPEN_RANGE_END = dt.time(9, 25)          # 10-minute opening range, per instructions
MIN_OR_BARS = 8                          # need most of the 10 expected 1-min bars present

CONFIRM_BARS = 1                         # see docstring note (B) -- 1 bar = closest proxy to "2x30sec"
TARGET_PTS = 5.0
TRAIL_PTS = 2.0
TIME_STOP_MINUTES = 3
INITIAL_SL_SWEEP = [3.0, 4.0, 5.0]

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0


# ------------------------------- SHARED HELPERS (same conventions as the rest of this project) -------------------------------

def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


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


def build_premium_series(day_df, strike, day_open_spot, side):
    """Standard proxy premium model used throughout this project: intrinsic
    at open + a flat assumed time value, moves scaled by a flat assumed
    delta, floored at 0. Documented simplification -- real option-chain
    minute data isn't available (see project-level notes)."""
    idx_open = day_df["open"].to_numpy()
    idx_high = day_df["high"].to_numpy()
    idx_low = day_df["low"].to_numpy()
    idx_close = day_df["close"].to_numpy()
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        mo, mh, ml, mc = (idx_open - day_open_spot, idx_high - day_open_spot,
                          idx_low - day_open_spot, idx_close - day_open_spot)
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        mo, mh, ml, mc = (day_open_spot - idx_open, day_open_spot - idx_low,
                          day_open_spot - idx_high, day_open_spot - idx_close)
    base = intrinsic0 + ASSUMED_TIME_VALUE
    p_open = np.maximum(base + mo * ASSUMED_DELTA, 0.0)
    p_high = np.maximum(base + mh * ASSUMED_DELTA, 0.0)
    p_low = np.maximum(base + ml * ASSUMED_DELTA, 0.0)
    p_close = np.maximum(base + mc * ASSUMED_DELTA, 0.0)
    return pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close}, index=day_df.index)


# ------------------------------- STRATEGY-SPECIFIC LOGIC -------------------------------

def opening_range(ref_df):
    """High/Low of the reference option's own premium chart, 09:15-09:25.
    Returns (or_high, or_low, n_bars) -- n_bars lets the caller skip days
    with too much missing data in the window."""
    times = ref_df.index.time
    mask = (times >= OPEN_RANGE_START) & (times < OPEN_RANGE_END)
    window = ref_df[mask]
    if len(window) < MIN_OR_BARS:
        return None, None, len(window)
    return window["high"].max(), window["low"].min(), len(window)


def simulate_trade_target_trail(trade_df, entry_idx, entry_price, sl_pts):
    """Manages one trade on whichever premium series it was actually taken
    on (CE's own series for a call, PE's own series for a put).

    Rules (exactly as specified):
      - Before profit ever reaches +TARGET_PTS: hard stop at entry-sl_pts;
        also force-exit if TIME_STOP_MINUTES elapse with neither the stop
        nor the target hit.
      - The instant unrealized profit (this bar's high) first reaches
        +TARGET_PTS, the time-stop is permanently disabled and a trailing
        stop activates: stop = max(entry+TARGET_PTS, peak-TRAIL_PTS) --
        i.e. it can never lock in less than the +5 target itself (the
        user's own worked example: peak=+7 -> stop=+5, not peak-2
        unconditionally).
      - Conservative intrabar ordering (this project's standing
        convention): each bar's LOW is checked against the stop level as
        it stood BEFORE this bar, before that same bar's HIGH is allowed
        to trigger the target or ratchet the trailing stop for the next
        bar -- OHLC bars don't reveal which happened first intrabar, so
        the worst case is assumed.
      - EOD square-off fallback.
    """
    idx = trade_df.index
    n = len(trade_df)
    entry_time = idx[entry_idx]
    deadline = entry_time + pd.Timedelta(minutes=TIME_STOP_MINUTES)
    stop_level = entry_price - sl_pts
    target_touched = False
    peak = entry_price

    for j in range(entry_idx, n):
        bar = trade_df.iloc[j]
        t = idx[j]

        # 1) conservative: this bar's low vs. the stop level as it stood BEFORE this bar
        if bar["low"] <= stop_level:
            reason = "TRAIL_SL" if target_touched else "INITIAL_SL"
            return stop_level, reason, t, j

        # 2) EOD fallback
        if t.time() >= EOD_SQUAREOFF:
            return bar["close"], "EOD", t, j

        # 3) time-stop -- only while the target has never been touched
        if not target_touched and t >= deadline:
            return bar["close"], "TIME_STOP", t, j

        # 4) only now let this bar's high touch the target / ratchet the trail (for next bar)
        if not target_touched:
            if bar["high"] >= entry_price + TARGET_PTS:
                target_touched = True
                peak = max(peak, bar["high"])
                stop_level = max(entry_price + TARGET_PTS, peak - TRAIL_PTS)
        else:
            if bar["high"] > peak:
                peak = bar["high"]
            stop_level = max(entry_price + TARGET_PTS, peak - TRAIL_PTS)

    last = trade_df.iloc[-1]
    return last["close"], "EOD", idx[-1], n - 1


def scan_and_simulate_day(ce_df, pe_df, or_high, or_low, sl_pts):
    """Scans bars after the opening range on the REFERENCE chart (ce_df,
    see assumption A in the module docstring). A confirmed break above
    or_high buys the CALL (on ce_df's own series); a confirmed break below
    or_low buys the PUT (on pe_df's own series). Confirmation = CONFIRM_BARS
    consecutive closes beyond the level (see assumption B). Execution lag:
    fill happens at the NEXT bar's open after the signal bar's close, never
    the signal bar's own close (this project's standing look-ahead fix).
    After a trade exits, scanning resumes from the very next bar, so the
    SAME fixed opening range can trigger further trades later the same day
    (multiple entries/day, as specified)."""
    trades = []
    idx = ce_df.index
    n = len(ce_df)
    start_j = None
    for k in range(n):
        if idx[k].time() >= OPEN_RANGE_END:
            start_j = k
            break
    if start_j is None:
        return trades

    j = start_j
    while j < n - 1:
        t = idx[j].time()
        if t >= EOD_SQUAREOFF:
            break

        bar = ce_df.iloc[j]
        signal_side = None
        if bar["close"] > or_high:
            signal_side = "CE"
        elif bar["close"] < or_low:
            signal_side = "PE"

        if signal_side is None:
            j += 1
            continue

        # CONFIRM_BARS consecutive closes beyond the level, ending at bar j
        if CONFIRM_BARS > 1:
            ok = True
            for back in range(1, CONFIRM_BARS):
                if j - back < start_j:
                    ok = False
                    break
                prev_close = ce_df["close"].iloc[j - back]
                if signal_side == "CE" and prev_close <= or_high:
                    ok = False
                    break
                if signal_side == "PE" and prev_close >= or_low:
                    ok = False
                    break
            if not ok:
                j += 1
                continue

        fill_idx = j + 1
        if fill_idx >= n:
            break

        trade_df = ce_df if signal_side == "CE" else pe_df
        entry_price = trade_df["open"].iloc[fill_idx]
        exit_price, reason, exit_time, exit_j = simulate_trade_target_trail(
            trade_df, fill_idx, entry_price, sl_pts
        )
        trades.append({
            "side": signal_side, "signal_time": idx[j], "entry_time": trade_df.index[fill_idx],
            "exit_time": exit_time, "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2), "exit_reason": reason,
            "pnl_pts": round(float(exit_price - entry_price), 2),
        })
        j = max(exit_j + 1, fill_idx + 1)

    return trades


# ------------------------------- METRICS / REPORTING -------------------------------

def compute_metrics(tdf, starting_capital=STARTING_CAPITAL):
    if tdf.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0, "cagr_pct": 0.0}
    pnl = tdf["rupee_pnl"].to_numpy()
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_pct = 100.0 * len(wins) / n
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_rs = pnl.sum()
    equity = starting_capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd_rs = dd.max()
    sqn = (pnl.mean() / pnl.std() * math.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    dates = pd.to_datetime(tdf["date"])
    years = max((dates.max() - dates.min()).days, 1) / 365.25
    cagr_pct = 0.0
    if (starting_capital + net_rs) > 0:
        cagr_pct = 100.0 * (((starting_capital + net_rs) / starting_capital) ** (1.0 / years) - 1.0)
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def daily_table(tdf):
    if tdf.empty:
        return pd.DataFrame()
    rows = []
    for d, g in tdf.groupby("date"):
        rows.append({
            "date": d, "n_trades": len(g), "ce": int((g["side"] == "CE").sum()),
            "pe": int((g["side"] == "PE").sum()), "net_pts": g["pnl_pts"].sum(),
            "win_pct": round(100 * (g["pnl_pts"] > 0).mean(), 1), "net_rs": g["rupee_pnl"].sum(),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def session_table(tdf):
    if tdf.empty:
        return pd.DataFrame()
    et = pd.to_datetime(tdf["entry_time"])

    def bucket(ts):
        minutes_from_open = (ts.hour * 60 + ts.minute) - (9 * 60 + 15)
        b = max(minutes_from_open, 0) // 30
        start_min = 9 * 60 + 15 + b * 30
        return f"{start_min // 60:02d}:{start_min % 60:02d}"

    tmp = tdf.copy()
    tmp["session"] = et.apply(bucket)
    rows = []
    for s, g in tmp.groupby("session"):
        rows.append({
            "session_start": s, "n": len(g), "win_pct": round(100 * (g["pnl_pts"] > 0).mean(), 1),
            "avg_pts": round(g["pnl_pts"].mean(), 2), "net_rs": round(g["rupee_pnl"].sum(), 2),
        })
    return pd.DataFrame(rows).sort_values("session_start").reset_index(drop=True)


def diagnostics(tdf, total_days):
    if tdf.empty:
        print("  no trades generated")
        return
    days_with_trade = tdf["date"].nunique()
    ce = int((tdf["side"] == "CE").sum())
    pe = int((tdf["side"] == "PE").sum())
    multi = tdf.groupby("date").size()
    multi_days = int((multi > 1).sum())
    print(f"  fires on {100*days_with_trade/total_days:.1f}% of days ({days_with_trade}/{total_days}) | "
          f"trades={len(tdf)} (CE={ce} PE={pe}) | days with >1 entry: {multi_days} | "
          f"exit reasons: {tdf['exit_reason'].value_counts().to_dict()}")


# ------------------------------- SELF-CHECKS (synthetic data, no network) -------------------------------

def run_self_checks():
    print("Running self-checks (synthetic data, no network access needed)...")

    # --- opening_range: basic high/low over a 10-bar window ---
    idx = pd.date_range("2025-01-06 09:15", periods=15, freq="1min")
    highs = [102, 103, 101, 104, 100, 99, 105, 103, 102, 101] + [110] * 5
    lows = [h - 1 for h in highs]
    df = pd.DataFrame({"open": lows, "high": highs, "low": lows, "close": highs}, index=idx)
    or_high, or_low, n_bars = opening_range(df)
    assert or_high == 105 and or_low == 98 and n_bars == 10, (or_high, or_low, n_bars)
    print("  opening_range basic high/low check ok")

    # --- opening_range: too few bars in the window -> None ---
    df_short = df.iloc[:5]
    or_high2, or_low2, n_bars2 = opening_range(df_short)
    assert or_high2 is None and n_bars2 == 5
    print("  opening_range insufficient-data check ok")

    # --- simulate_trade_target_trail: hard stop, target never reached ---
    idx_t = pd.date_range("2025-01-06 09:26", periods=10, freq="1min")
    prem = pd.DataFrame({
        "open": [150, 149, 148, 147, 146, 146, 146, 146, 146, 146],
        "high": [150, 149.5, 148.5, 147.5, 146.5, 146, 146, 146, 146, 146],
        "low": [150, 148, 147, 146, 145, 146, 146, 146, 146, 146],
        "close": [150, 149, 148, 147, 146, 146, 146, 146, 146, 146],
    }, index=idx_t)
    exit_price, reason, exit_t, _ = simulate_trade_target_trail(prem, 0, entry_price=150, sl_pts=3)
    assert reason == "INITIAL_SL" and abs(exit_price - 147.0) < 1e-6, (exit_price, reason)
    print("  simulate_trade_target_trail hard-stop check ok (entry 150, sl=3 -> exit 147)")

    # --- simulate_trade_target_trail: time-stop before target ever reached ---
    prem_t = pd.DataFrame({
        "open": [150, 151, 150.5, 151, 150.8, 150.5],
        "high": [150, 151.5, 151, 151.5, 151, 150.8],
        "low": [150, 150.5, 150, 150.5, 150.5, 150.2],
        "close": [150, 151, 150.5, 151, 150.8, 150.5],
    }, index=idx_t[:6])
    exit_price_t, reason_t, exit_t2, _ = simulate_trade_target_trail(prem_t, 0, entry_price=150, sl_pts=5)
    # deadline = entry_time + 3min = idx_t[3]; bar index 3's time >= deadline -> TIME_STOP there
    assert reason_t == "TIME_STOP" and exit_t2 == idx_t[3], (reason_t, exit_t2)
    print("  simulate_trade_target_trail time-stop-before-target check ok")

    # --- THE USER'S OWN WORKED EXAMPLE: entry=150, target touched at peak=157
    #     (entry+7) -> trailing stop must protect exactly +5 (155), not
    #     157-2=155 coincidentally equal here by construction matching their
    #     own numeric example (peak 7 -> stop 5) ---
    prem2 = pd.DataFrame({
        "open": [150, 153, 157, 154, 153],
        "high": [150, 154, 157, 155, 154],
        "low": [150, 152, 153, 153.5, 153],
        "close": [150, 153, 156, 154, 153.5],
    }, index=idx_t[:5])
    exit_price2, reason2, exit_t3, _ = simulate_trade_target_trail(prem2, 0, entry_price=150, sl_pts=5)
    # bar2 high=157 -> target touched (150+5=155 cleared), peak=157, stop=max(155,157-2=155)=155
    # bar3 low=153.5 <= 155 -> exit at 155, TRAIL_SL
    assert reason2 == "TRAIL_SL" and abs(exit_price2 - 155.0) < 1e-6, (exit_price2, reason2)
    print("  simulate_trade_target_trail floor-protected trail check ok (matches user's own +7->+5 example)")

    # --- floor logic when peak goes well past target: trail must NOT dip
    #     below entry+5 even as peak-2 rises above it ---
    prem3 = pd.DataFrame({
        "open": [150, 158, 165, 162, 161],
        "high": [150, 159, 166, 163, 162],
        "low": [150, 157, 158, 161, 160.5],
        "close": [150, 158, 165, 162, 161],
    }, index=idx_t[:5])
    exit_price3, reason3, _, _ = simulate_trade_target_trail(prem3, 0, entry_price=150, sl_pts=5)
    # bar1 high=159 -> target touched, peak=159, stop=max(155,157)=157
    # bar2 low=158 > 157 ok; high=166 -> peak=166, stop=max(155,164)=164
    # bar3 low=161 <= 164 -> exit at 164, TRAIL_SL
    assert reason3 == "TRAIL_SL" and abs(exit_price3 - 164.0) < 1e-6, (exit_price3, reason3)
    print("  simulate_trade_target_trail trail-exceeds-floor check ok")

    # --- conservative intrabar ordering: a bar that dips through the OLD
    #     stop AND makes a big new high in the same bar must exit at the
    #     OLD level, not the ratcheted one ---
    prem4 = pd.DataFrame({
        "open": [150, 156, 156, 160, 160],
        "high": [150, 157, 170, 161, 161],
        "low": [150, 155, 150, 159, 159],
        "close": [150, 156, 160, 160, 160],
    }, index=idx_t[:5])
    exit_price4, reason4, exit_t4, _ = simulate_trade_target_trail(prem4, 0, entry_price=150, sl_pts=5)
    # bar1 high=157 -> target touched (>=155), peak=157, stop=max(155,155)=155
    # bar2 low=150 <= 155 -> exit at OLD stop 155, TRAIL_SL (must NOT ratchet to 168 first)
    assert reason4 == "TRAIL_SL" and abs(exit_price4 - 155.0) < 1e-6 and exit_t4 == idx_t[2], (
        exit_price4, reason4, exit_t4
    )
    print("  simulate_trade_target_trail conservative intrabar-ordering check ok")

    # --- scan_and_simulate_day: multiple entries in one day (a CE trade
    #     that exits, then a later fresh breakout of the SAME or_high
    #     triggers a second CE trade) ---
    idx_d = pd.date_range("2025-01-06 09:15", periods=30, freq="1min")
    ce_close = [100] * 10 + [106, 106, 106, 96, 96, 96, 96, 96] + [106, 106, 106, 96, 96, 96, 96, 96, 96, 96, 96, 96]
    ce_close = ce_close[:30]
    ce_high = [c + 0.5 for c in ce_close]
    ce_low = [c - 0.5 for c in ce_close]
    ce_open = [100] + ce_close[:-1]
    ce_df_test = pd.DataFrame({"open": ce_open, "high": ce_high, "low": ce_low, "close": ce_close}, index=idx_d)
    pe_df_test = ce_df_test.copy()  # doesn't matter for this CE-only synthetic check
    or_high_t, or_low_t = 105.0, 95.0
    day_trades = scan_and_simulate_day(ce_df_test, pe_df_test, or_high_t, or_low_t, sl_pts=3)
    assert len(day_trades) >= 2, f"expected at least 2 trades (multiple entries), got {len(day_trades)}"
    assert all(tr["side"] == "CE" for tr in day_trades)
    print(f"  scan_and_simulate_day multiple-entries-per-day check ok ({len(day_trades)} trades found)")

    # --- scan_and_simulate_day: a low-break buys the PUT on pe_df, not ce_df ---
    ce_close2 = [100] * 10 + [90, 90, 90] + [90] * 17
    ce_high2 = [c + 0.5 for c in ce_close2]
    ce_low2 = [c - 0.5 for c in ce_close2]
    ce_open2 = [100] + ce_close2[:-1]
    ce_df2 = pd.DataFrame({"open": ce_open2, "high": ce_high2, "low": ce_low2, "close": ce_close2}, index=idx_d)
    pe_close2 = [50] * 30  # a clearly different price path so we can prove the PUT fill used pe_df, not ce_df
    pe_df2 = pd.DataFrame({"open": pe_close2, "high": [p + 1 for p in pe_close2],
                            "low": [p - 1 for p in pe_close2], "close": pe_close2}, index=idx_d)
    day_trades2 = scan_and_simulate_day(ce_df2, pe_df2, or_high_t, or_low_t, sl_pts=3)
    assert len(day_trades2) >= 1 and day_trades2[0]["side"] == "PE"
    assert abs(day_trades2[0]["entry_price"] - 50) < 1.5, (
        f"PE trade must be priced off pe_df (~50), not ce_df (~90); got {day_trades2[0]['entry_price']}"
    )
    print("  scan_and_simulate_day low-break-buys-PUT-on-its-own-series check ok")

    print("All self-checks passed.\n")


# ------------------------------- SMOKE TEST (synthetic multi-day data, no network) -------------------------------

def run_smoke_test():
    print("Running full smoke test on synthetic multi-day data (no network access needed)...")
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2025-01-06", periods=12)
    all_synth = []
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:15", f"{d.date()} 15:29", freq="1min")
        spot0 = 22000 + rng.normal(0, 50)
        walk = spot0 + np.cumsum(rng.normal(0, 3, size=len(idx)))
        highs = walk + rng.uniform(0.5, 2.5, size=len(idx))
        lows = walk - rng.uniform(0.5, 2.5, size=len(idx))
        opens = walk + rng.normal(0, 0.5, size=len(idx))
        closes = walk
        all_synth.append(pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx))
    synth_index_df = pd.concat(all_synth).sort_index()

    total_days = len(days)
    for sl_pts in INITIAL_SL_SWEEP:
        all_trades = []
        for day in sorted(set(synth_index_df.index.date)):
            day_df = synth_index_df[synth_index_df.index.date == day]
            if len(day_df) < 30:
                continue
            day_open_spot = day_df["open"].iloc[0]
            ce_strike = itm_call_strike(day_open_spot)
            pe_strike = itm_put_strike(day_open_spot)
            ce_df = build_premium_series(day_df, ce_strike, day_open_spot, "CE")
            pe_df = build_premium_series(day_df, pe_strike, day_open_spot, "PE")
            or_high, or_low, n_bars = opening_range(ce_df)
            if or_high is None:
                continue
            day_trades = scan_and_simulate_day(ce_df, pe_df, or_high, or_low, sl_pts)
            for tr in day_trades:
                tr["date"] = day
                lot = lot_size_for_date(day)
                tr["rupee_pnl"] = round(tr["pnl_pts"] * lot - BROKERAGE_PER_TRADE, 2)
                all_trades.append(tr)
        tdf = pd.DataFrame(all_trades)
        m = compute_metrics(tdf) if not tdf.empty else None
        print(f"  SL={sl_pts}pts: {len(tdf)} synthetic trades generated across {total_days} synthetic days"
              + (f" | n={m['n']} win%={m['win_pct']} PF={m['pf']}" if m else " (none fired -- fine for random-walk synthetic data)"))
    print("Smoke test completed without errors.\n")


# ------------------------------- REAL BACKTEST (needs live Dhan connectivity -- run this file locally) -------------------------------

def run_one_sl(full_df, days_covered, sl_pts):
    all_trades = []
    for day in days_covered:
        day_df = full_df[full_df.index.date == day]
        if len(day_df) < 30:
            continue
        day_open_spot = day_df["open"].iloc[0]
        ce_strike = itm_call_strike(day_open_spot)
        pe_strike = itm_put_strike(day_open_spot)
        ce_df = build_premium_series(day_df, ce_strike, day_open_spot, "CE")
        pe_df = build_premium_series(day_df, pe_strike, day_open_spot, "PE")
        or_high, or_low, n_bars = opening_range(ce_df)
        if or_high is None:
            continue
        day_trades = scan_and_simulate_day(ce_df, pe_df, or_high, or_low, sl_pts)
        lot = lot_size_for_date(day)
        for tr in day_trades:
            tr["date"] = day
            tr["ce_strike"] = ce_strike
            tr["pe_strike"] = pe_strike
            tr["or_high"] = round(float(or_high), 2)
            tr["or_low"] = round(float(or_low), 2)
            tr["lot_size"] = lot
            tr["rupee_pnl"] = round(tr["pnl_pts"] * lot - BROKERAGE_PER_TRADE, 2)
            all_trades.append(tr)
    return pd.DataFrame(all_trades)


def main():
    run_self_checks()
    run_smoke_test()

    print("=" * 78)
    print(f"REAL BACKTEST -- {BACKTEST_MONTHS} months, {FROM_DATE} to {TO_DATE}")
    print("Requires live Dhan connectivity. If this errors with a connection/proxy")
    print("issue, run this file on the machine where dhan_live_bot.py normally works.")
    print("=" * 78)
    raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    # 1-minute bars are the finest Dhan provides -- see docstring note (B)
    full_df = raw
    days_covered = sorted(set(full_df.index.date))
    total_days = len(days_covered)
    print(f"Loaded {len(full_df)} 1-min bars across {total_days} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    summary_rows = []
    for sl_pts in INITIAL_SL_SWEEP:
        print("-" * 78)
        print(f"INITIAL SL = {sl_pts} points (target +{TARGET_PTS}, trail {TRAIL_PTS} after target, "
              f"floor-protected, {TIME_STOP_MINUTES}-min time-stop before target)")
        print("-" * 78)
        tdf = run_one_sl(full_df, days_covered, sl_pts)
        if tdf.empty:
            print("  No trades generated.\n")
            summary_rows.append({"sl_pts": sl_pts, "n": 0})
            continue

        m = compute_metrics(tdf)
        print(f"  Overall: n={m['n']} win%={m['win_pct']} PF={m['pf']} net_Rs={m['net_rs']:,} "
              f"maxDD_Rs={m['max_dd_rs']:,} SQN={m['sqn']} CAGR%={m['cagr_pct']}")
        diagnostics(tdf, total_days)

        dtab = daily_table(tdf)
        stab = session_table(tdf)
        tag = f"sl{int(sl_pts)}"
        tdf.to_csv(f"orb_option_chart_5pt_{tag}_trades.csv", index=False)
        dtab.to_csv(f"orb_option_chart_5pt_{tag}_daily.csv", index=False)
        stab.to_csv(f"orb_option_chart_5pt_{tag}_sessions.csv", index=False)
        print(f"  Saved: orb_option_chart_5pt_{tag}_trades.csv / _daily.csv / _sessions.csv")

        print("  30-min session breakdown:")
        for _, r in stab.iterrows():
            print(f"    {r['session_start']}: n={r['n']} win%={r['win_pct']} avg_pts={r['avg_pts']} net_Rs={r['net_rs']:,}")

        by_reason = tdf.groupby("exit_reason")["rupee_pnl"].agg(["count", "sum"])
        print("  By exit reason:")
        for reason, row in by_reason.iterrows():
            print(f"    {reason}: n={int(row['count'])} net_Rs={row['sum']:,.2f}")

        mid_date = pd.to_datetime(tdf["date"]).min() + (pd.to_datetime(tdf["date"]).max() - pd.to_datetime(tdf["date"]).min()) / 2
        dts = pd.to_datetime(tdf["date"])
        fh, sh = tdf[dts <= mid_date], tdf[dts > mid_date]
        m1, m2 = compute_metrics(fh), compute_metrics(sh)
        print(f"  Stability (first half vs second half by date): "
              f"n={m1['n']} net_Rs={m1['net_rs']:,} PF={m1['pf']}  |  n={m2['n']} net_Rs={m2['net_rs']:,} PF={m2['pf']}")
        print()

        summary_rows.append({"sl_pts": sl_pts, **m})

    print("=" * 78)
    print("SUMMARY ACROSS INITIAL-SL SWEEP")
    print("=" * 78)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
