"""
FIXED-STRIKE OPENING-RANGE BREAKOUT -- 5-POINT TARGET (Nilus concept)
=================================================================
Backtest of the uploaded draft ("Fixed-Strike Opening-Range Breakout --
5-Point Target Strategy", Nilus, 29 Jul 2026), with the open questions
the doc itself flagged resolved as follows (confirmed with you):

  - STOP-LOSS (the doc had none): 10-point fixed initial stop, matching
    every other strategy in this project.
  - STRIKE SELECTION: ATM at market open (nearest 50 to Nifty's opening
    spot), fixed for the whole day -- matches the doc's "one strike,
    fixed for the day" design.
  - DIRECTION: CE-only, exactly as written (no mirrored PE off a
    low-breakout).
  - RE-ENTRY: one trade per day only -- once today's target or stop is
    hit (or if no trade triggers at all), done for the day.

RULES, AS CODED:
  1. On the strike's OWN premium chart (proxy-premium model, same
     convention as every other script here), mark the opening range from
     the first 5 one-minute candles (09:15-09:19, bars 0-4): Strike High
     = max high, Strike Low = min low of those 5 bars. (Strike Low is
     computed for completeness/reference but isn't used by this CE-only
     version.)
  2. From bar 5 (09:20) onward, watch for a bar whose CLOSE (not high --
     the doc's own clarification) exceeds Strike High. That's the
     confirmed breakout bar.
  3. Entry fills at the NEXT bar's open (signal confirms on a bar's
     close, fill is the next bar's open -- the standing look-ahead-
     safety rule used everywhere else in this project; the doc didn't
     specify exact fill timing, so this fills the gap explicitly rather
     than assuming same-bar fill).
  4. Exit: 5-point fixed target OR 10-point fixed stop, whichever hits
     first -- no trailing, one target, one exit, per the doc. Conservative
     intrabar ordering: a bar's LOW is checked against the stop BEFORE its
     HIGH is allowed to hit the target (standing project rule -- OHLC
     bars don't reveal which happened first within the bar).
  5. One trade per day. If the high is never broken, no trade that day.

Run:
    cd "C:\\dhan algo"
    python nifty_orb_fixed_strike_5pt_target_backtest.py --smoke-test
    python nifty_orb_fixed_strike_5pt_target_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd

import nifty_paper_trade_top3_live as top3
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== config ===============================
FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OPENING_RANGE_BARS = 5   # 09:15-09:19 (bars 0-4)
TARGET_PTS, STOP_PTS = 5.0, 10.0
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000
EOD_SQUAREOFF = dt.time(15, 20)

TRADES_CSV = "nifty_orb_fixed_strike_5pt_trades.csv"


# ============================== data fetch ===================================
def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    """Standard 1-min Nifty index fetch, same pattern as every other script
    in this project (no volume needed here -- this strategy's filter is
    purely price-based)."""
    import dhan_live_bot as dlb
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
            "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def atm_strike(spot, step=50):
    return round(spot / step) * step


# =============================================================================
# Day-level simulation
# =============================================================================
def detect_orb_signal_bar(premium_df, strike_high):
    """Returns the first bar index >= OPENING_RANGE_BARS whose CLOSE (not
    high) exceeds strike_high, or None if never broken that day."""
    closes = premium_df["close"].to_numpy()
    for i in range(OPENING_RANGE_BARS, len(premium_df)):
        if closes[i] > strike_high:
            return i
    return None


def step_fixed_target_stop(pos, premium_df):
    """Conservative ordering: this bar's low is checked against the stop
    BEFORE its high is allowed to hit the target."""
    n = len(premium_df)
    for j in range(pos["last_checked_idx"] + 1, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= pos["stop_level"]:
            return {"exit_price": pos["stop_level"], "reason": "STOP", "exit_time": premium_df.index[j], "exit_idx": j}
        if bar["high"] >= pos["target_level"]:
            return {"exit_price": pos["target_level"], "reason": "TARGET", "exit_time": premium_df.index[j], "exit_idx": j}
        if t >= EOD_SQUAREOFF:
            return {"exit_price": bar["close"], "reason": "EOD", "exit_time": premium_df.index[j], "exit_idx": j}
        pos["last_checked_idx"] = j
    return None


def run_orb_standalone_for_day(day_df):
    """One trade per day, CE-only. Returns a list with 0 or 1 trades."""
    n = len(day_df)
    if n < OPENING_RANGE_BARS + 2:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    strike_high = premium_df["high"].iloc[:OPENING_RANGE_BARS].max()
    strike_low = premium_df["low"].iloc[:OPENING_RANGE_BARS].min()

    sig_bar = detect_orb_signal_bar(premium_df, strike_high)
    if sig_bar is None:
        return []
    exec_idx = sig_bar + 1
    if exec_idx >= n:
        return []

    entry_price = premium_df["open"].iloc[exec_idx]
    pos = {"target_level": entry_price + TARGET_PTS, "stop_level": entry_price - STOP_PTS,
           "last_checked_idx": exec_idx - 1}
    exit_info = step_fixed_target_stop(pos, premium_df)
    if exit_info is None:
        exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                     "exit_time": premium_df.index[-1], "exit_idx": n - 1}
    pnl_pts = exit_info["exit_price"] - entry_price
    rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
    return [{
        "strategy": "ORB_FixedStrike_5pt", "side": "CE", "strike": strike,
        "strike_high": strike_high, "strike_low": strike_low,
        "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    }]


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()  # re-verify the shared premium-building primitive

    assert atm_strike(24255) == 24250 and atm_strike(24280) == 24300
    print("  atm_strike rounding ok")

    # --- opening-range high/low + confirmed-close-only entry (touch, not close, must NOT fire) ---
    idx = pd.date_range("2025-01-06 09:15", periods=10, freq="1min")
    # bars 0-4: opening range, high maxes at 110. bar5: touches 115 intra-bar (high) but CLOSES at 108 -- must NOT fire.
    # bar6: closes at 112 -- confirmed break, must fire here.
    opens_ = [100, 102, 104, 103, 105, 106, 109, 112, 112, 112]
    highs = [105, 106, 108, 107, 110, 115, 113, 113, 113, 113]
    lows = [98, 100, 102, 101, 103, 105, 108, 111, 111, 111]
    closes = [102, 104, 103, 105, 106, 108, 112, 112, 112, 112]
    df = pd.DataFrame({"open": opens_, "high": highs, "low": lows, "close": closes}, index=idx)
    strike_high = max(highs[:5])  # 110
    assert strike_high == 110
    sig = detect_orb_signal_bar(df, strike_high)
    assert sig == 6, f"expected confirmed close-break at bar 6 (bar 5 only touched intra-bar, didn't close above), got {sig}"
    print("  opening-range high + confirmed-CLOSE-only entry (intra-bar touch alone does not fire) ok")

    # --- conservative ordering: stop must win if a bar's range spans both stop and target ---
    idx2 = pd.date_range("2025-01-06 09:15", periods=3, freq="1min")
    prem = pd.DataFrame({
        "open": [150, 150, 150], "high": [150, 156, 156], "low": [150, 139, 139], "close": [150, 145, 145],
    }, index=idx2)  # bar1's range spans both target (155) and stop (140)
    pos = {"target_level": 155, "stop_level": 140, "last_checked_idx": 0}
    exit_info = step_fixed_target_stop(pos, prem)
    assert exit_info["reason"] == "STOP" and exit_info["exit_price"] == 140, (
        f"expected conservative STOP to win when a bar spans both levels, got {exit_info}"
    )
    print("  conservative ordering: stop wins over target when a bar's range spans both ok")

    # --- no trade if the high is never broken ---
    idx3 = pd.date_range("2025-01-06 09:15", periods=8, freq="1min")
    flat = pd.DataFrame({
        "open": [100] * 8, "high": [105, 106, 104, 103, 107, 106, 105, 104],
        "low": [98] * 8, "close": [100] * 8,
    }, index=idx3)
    sig_none = detect_orb_signal_bar(flat, max(flat["high"].iloc[:5]))
    assert sig_none is None, f"expected no signal when the opening high is never closed above, got {sig_none}"
    print("  no trade when the opening-range high is never broken ok")

    print("All self-checks passed.\n")


# =============================================================================
# Diagnostics
# =============================================================================
def compute_pf_sqn(trades_df):
    pnl = trades_df["rupee_pnl"]
    n = len(pnl)
    if n == 0:
        return 0.0, 0.0, 0
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    std = pnl.std(ddof=1) if n > 1 else 0.0
    sqn = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    return pf, sqn, n


def run_full_pipeline(df1m, label="FULL 2-YEAR BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(run_orb_standalone_for_day(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades generated at all -- the opening-range high may rarely get broken with a "
              "confirmed close, or the 5-bar opening window needs revisiting.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    if traded_days / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days -- worth checking this hasn't degenerated into "
              "something looser than intended (same artifact pattern seen elsewhere in this project). ***")

    win_by_reason = trades_df["reason"].value_counts()
    print(f"  Exit reason breakdown: {win_by_reason.to_dict()}")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ ORB FIXED-STRIKE 5PT -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. (Calibration: real clean results look like PF ~1.4-1.7, SQN ~2.5-3.) ***")
    # sanity check on the doc's own implied win-rate expectation: with a 1:2 reward:risk (5 target / 10
    # stop), breakeven win rate is 66.7% -- worth testing that claim directly (checklist item 10).
    breakeven_wr = STOP_PTS / (STOP_PTS + TARGET_PTS) * 100
    print(f"  (For reference: with a {TARGET_PTS}pt target / {STOP_PTS}pt stop, breakeven win rate is "
          f"{breakeven_wr:.1f}% -- actual win rate above needs to beat that, not just be >50%.)")

    # in-sample / out-of-sample split
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    dates = sorted(trades_df["date"].unique())
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        h1 = trades_df[trades_df["date"] < mid]
        h2 = trades_df[trades_df["date"] >= mid]
        pf1, sqn1, n1 = compute_pf_sqn(h1)
        pf2, sqn2, n2 = compute_pf_sqn(h2)
        print(f"\nOOS split: H1 ({dates[0].date()} to {mid.date()}) n={n1} PF={pf1:.2f} SQN={sqn1:.2f} "
              f"net=Rs{h1['rupee_pnl'].sum():,.2f}")
        print(f"           H2 ({mid.date()} to {dates[-1].date()}) n={n2} PF={pf2:.2f} SQN={sqn2:.2f} "
              f"net=Rs{h2['rupee_pnl'].sum():,.2f}")

    # capital adequacy
    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (Rs{STARTING_CAPITAL:,} starting capital): Rs{worst_dd:,.2f}")
    if abs(worst_dd) >= STARTING_CAPITAL:
        print("  *** CAPITAL ADEQUACY WARNING: worst drawdown approaches/exceeds starting capital. ***")
    if (equity_curve <= 0).any():
        print("  WARNING: equity touches Rs0 or below at some point in this trade sequence.")
    else:
        print("  Equity never touches Rs0 or below across this trade sequence.")

    # full institutional metrics
    daily_pnl = build_daily_pnl(trades_df.rename(columns={"date": "date"}))
    m = full_metrics(daily_pnl, STARTING_CAPITAL)
    mc = monte_carlo_reshuffle(trades_df, STARTING_CAPITAL)
    print(f"\n################ FULL INSTITUTIONAL METRICS ################")
    print(f"CAGR: {m['cagr_pct']}%   Sharpe (active days): {m['sharpe_active_days']}   "
          f"Sharpe (full-time capital): {m['sharpe_fulltime']}")
    print(f"Sortino (active days): {m['sortino_active_days']}   Sortino (full-time capital): {m['sortino_fulltime']}")
    print(f"Max DD: {m['max_dd_pct']}%   Max DD duration: {m['max_dd_duration_days']} trading days")
    print(f"VaR95 (daily): {m['var95_pct']}%   CVaR95 (daily): {m['cvar95_pct']}%")
    print(f"t-stat: {m['t_stat']}   p-value: {m['p_value']} "
          f"({'statistically significant at 5%' if m['p_value'] < 0.05 else 'NOT statistically significant at 5%'})")
    print(f"Monte Carlo (5000 reshuffles): wipeout probability {mc['pct_paths_wipeout']}%   "
          f"below-50%-capital probability {mc['pct_paths_below_50pct_capital']}%")
    print(f"Final equity: Rs{m['final_equity']:,.2f}")

    return {"trades": trades_df, "metrics": m, "monte_carlo": mc}


# =============================================================================
# Smoke test / main
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). This failure is specifically about "
              "reaching the Dhan API from wherever this was just run -- re-run on the machine with a "
              "working Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)")
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data.")


def main():
    run_self_checks()
    print(f"=== ORB FIXED-STRIKE 5PT TARGET -- 2-YEAR REAL BACKTEST ===")
    print("Fetching Nifty index 1-min data (2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    if df1m.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}")
    run_full_pipeline(df1m)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main()
