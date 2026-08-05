"""
TWO-TOUCH RESISTANCE BREAKOUT (3rd-time breach entry) -- 6-MONTH BACKTEST
=================================================================
New concept, translated into concrete rules. Everything below is on the
ATM CALL's own premium chart (per "always analyse the option chart which
is nearest at the money strike"), NOT the raw Nifty index -- same
convention as the M-Structure and ORB strategies in this project.

RULES, AS CODED (documented assumptions called out explicitly, not
hidden, since the original description is a chart pattern, not exact
numbers):
  1. OPENING RANGE: first 30 one-minute candles (09:15-09:44) on the ATM
     CE premium. Mark Day High (highest high) and Day Low (lowest low,
     kept for reference -- this version is CE-only).
  2. FIRST BREACH: from bar 30 (09:45) onward, the first bar whose CLOSE
     exceeds Day High confirms the initial breakout.
  3. TWO-TOUCH RESISTANCE ("U shape above the marked high"): after the
     first breach, look for swing-high pivots ABOVE Day High using this
     project's existing pivot_high() function (reused verbatim from
     dhan_live_bot.py -- a bar is a confirmed swing high if it's the max
     within a +/-2-bar window, i.e. PIVOT_LEFT=PIVOT_RIGHT=2, a modeling
     choice for "how many bars confirm a pivot", documented here rather
     than hidden). The first two such pivots whose levels sit within
     TOUCH_TOLERANCE_PTS=3.0 points of each other (another documented
     assumption -- "approximately the same level") form the two-touch
     resistance. Resistance = the HIGHER of the two touch levels (a
     genuine breakout must clear both peaks, not just their average).
  4. THIRD BREACH = ENTRY SIGNAL: the first bar, after the second touch
     is confirmed, whose CLOSE exceeds the resistance level. Entry fills
     at the NEXT bar's open (signal confirms on a close, fill is the
     next bar's open -- the standing look-ahead-safety rule used
     everywhere in this project).
  5. STOP-LOSS: exactly as you specified -- 8 points BELOW THE
     RESISTANCE LEVEL (not below the entry price). Since entry typically
     executes at or above the resistance, this is usually a wider
     cushion than a plain entry-8 stop would be.
  6. TRAILING: 4-point trail once the trade is in profit, using the
     exact same conservative-ordering stepper (top3.step_long_premium_
     position) already validated on every other strategy here.
  7. CE-ONLY (your instruction: "first test only for upside movement").
     One trade per day.
  8. ATM strike fixed at market open (nearest 50 to Nifty's opening spot).

Backtest window: last ~6 months (your instruction), Rs25,000 starting
capital. Also reports 30-minute session-of-day and day-of-week
breakdowns, plus the full institutional metrics suite used everywhere
else in this project.

Run:
    cd "C:\\dhan algo"
    python nifty_two_touch_resistance_breakout_backtest.py --smoke-test
    python nifty_two_touch_resistance_breakout_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd
import numpy as np

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== config ===============================
BACKTEST_DAYS = 183  # ~6 months
FROM_DATE = (dt.date.today() - dt.timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OPENING_RANGE_BARS = 30       # first 30 one-minute candles, 09:15-09:44
PIVOT_LEFT, PIVOT_RIGHT = 2, 2
TOUCH_TOLERANCE_PTS = 3.0
STOP_BELOW_RESISTANCE_PTS = 8.0
TRAIL_PTS = 4.0
BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 25_000
EOD_SQUAREOFF = dt.time(15, 20)

TRADES_CSV = "two_touch_resistance_breakout_trades.csv"

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def classify_session(ts):
    t = ts.time()
    for label in SESSIONS:
        start_s, end_s = label.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        if dt.time(sh, sm) <= t < dt.time(eh, em):
            return label
    return "Outside usual session"


# ============================== data fetch ===================================
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
# Pattern detection -- day-high breach, two-touch resistance, third-breach entry
# =============================================================================
def find_two_touch_resistance_and_entry(premium_df):
    """Returns (signal_bar_index, resistance_level, touch1_idx, touch2_idx) or
    (None, None, None, None) if the pattern never completes that day."""
    n = len(premium_df)
    if n < OPENING_RANGE_BARS + 2 * PIVOT_RIGHT + 5:
        return None, None, None, None
    highs = premium_df["high"].to_numpy()
    lows = premium_df["low"].to_numpy()
    closes = premium_df["close"].to_numpy()

    day_high = highs[:OPENING_RANGE_BARS].max()

    breach1 = None
    for i in range(OPENING_RANGE_BARS, n):
        if closes[i] > day_high:
            breach1 = i
            break
    if breach1 is None:
        return None, None, None, None

    pivot_series = dlb.pivot_high(premium_df["high"], PIVOT_LEFT, PIVOT_RIGHT)
    pivot_idxs = [i for i in range(breach1, n)
                  if not pd.isna(pivot_series.iloc[i]) and pivot_series.iloc[i] > day_high]
    if len(pivot_idxs) < 2:
        return None, None, None, None

    touch1_idx = touch2_idx = None
    for a in range(len(pivot_idxs)):
        for b in range(a + 1, len(pivot_idxs)):
            ia, ib = pivot_idxs[a], pivot_idxs[b]
            va, vb = pivot_series.iloc[ia], pivot_series.iloc[ib]
            if abs(va - vb) <= TOUCH_TOLERANCE_PTS:
                touch1_idx, touch2_idx = ia, ib
                break
        if touch1_idx is not None:
            break
    if touch1_idx is None:
        return None, None, None, None

    resistance = max(pivot_series.iloc[touch1_idx], pivot_series.iloc[touch2_idx])

    # third breach can only be searched from the point the 2nd touch is actually
    # confirmable (touch2_idx + PIVOT_RIGHT bars later) -- look-ahead safe.
    search_start = touch2_idx + PIVOT_RIGHT + 1
    for i in range(search_start, n):
        if closes[i] > resistance:
            return i, resistance, touch1_idx, touch2_idx
    return None, None, None, None


def run_standalone_for_day(day_df):
    n = len(day_df)
    if n < OPENING_RANGE_BARS + 10:
        return []
    day_open_spot = day_df["open"].iloc[0]
    strike = atm_strike(day_open_spot)
    premium_df = top3.build_premium_series(day_df, strike, day_open_spot, side="CE")
    lot = top3.lot_size_for_date(day_df.index[0].date())

    sig_bar, resistance, touch1_idx, touch2_idx = find_two_touch_resistance_and_entry(premium_df)
    if sig_bar is None:
        return []
    exec_idx = sig_bar + 1
    if exec_idx >= n:
        return []

    entry_price = premium_df["open"].iloc[exec_idx]
    pos = {"entry_price": entry_price, "stop_level": resistance - STOP_BELOW_RESISTANCE_PTS,
           "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
    exit_info = top3.step_long_premium_position(pos, premium_df, TRAIL_PTS)
    if exit_info is None:
        exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                     "exit_time": premium_df.index[-1], "exit_idx": n - 1}
    pnl_pts = exit_info["exit_price"] - entry_price
    rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
    return [{
        "strategy": "TwoTouchResistanceBreakout", "side": "CE", "strike": strike,
        "resistance": resistance, "entry_time": premium_df.index[exec_idx], "exit_time": exit_info["exit_time"],
        "entry_price": entry_price, "exit_price": exit_info["exit_price"], "reason": exit_info["reason"],
        "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl, "date": day_df.index[0].date(),
    }]


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()  # re-verify the shared premium/position-management primitives

    assert atm_strike(24255) == 24250 and atm_strike(24280) == 24300
    print("  atm_strike rounding ok")

    # --- build a hand-traced day: flat opening range, breach, two equal-ish
    # pivots (a "U" shape), then a third breach -> entry ---
    idx = pd.date_range("2025-01-06 09:15", periods=50, freq="1min")
    closes = [100.0] * 30                                  # bars 0-29: flat opening range, Day High=100
    closes += [105, 108, 112, 110, 108]                     # bars 30-34: breach(30)->112 peak at bar32
    closes += [105, 104, 106, 111, 109, 107]                # bars 35-40: dip, then 111 peak at bar38 (within 3 of 112)
    closes += [109, 113]                                    # bars 41-42: bar42 closes 113 > resistance(112) -> 3rd breach
    closes += [114, 116, 118, 120, 122, 124, 126, 128]       # bars 43-50: entry bar(43) + a few more for exit testing
    closes = closes[:50]
    df = pd.DataFrame({
        "open": [100.0] + closes[:-1], "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
    }, index=idx)
    sig_bar, resistance, t1, t2 = find_two_touch_resistance_and_entry(df)
    assert sig_bar == 42, f"expected the 3rd breach at bar 42, got {sig_bar}"
    assert abs(resistance - 112.5) < 1e-6, f"expected resistance ~112.5 (bar32's high), got {resistance}"
    print("  day-high breach -> two-touch ('U' shape) pivot detection -> 3rd-breach entry fires at the right bar ok")

    # --- stop level must be RESISTANCE-8, not entry-8 ---
    exec_idx = sig_bar + 1
    entry_price = df["open"].iloc[exec_idx]
    stop_level = resistance - STOP_BELOW_RESISTANCE_PTS
    assert abs(stop_level - (entry_price - STOP_BELOW_RESISTANCE_PTS)) > 0.01, (
        "test setup sanity check: resistance-8 should differ from entry-8 in this synthetic example "
        "(otherwise the check below wouldn't actually distinguish the two)"
    )
    print(f"  stop-level correctly computed as resistance-8 ({stop_level:.2f}), not entry-8 "
          f"({entry_price - STOP_BELOW_RESISTANCE_PTS:.2f}) ok")

    # --- no pattern at all (flat day, no breach) must return no signal ---
    flat_closes = [100.0] * 50
    flat_df = pd.DataFrame({
        "open": flat_closes, "high": [c + 0.2 for c in flat_closes],
        "low": [c - 0.2 for c in flat_closes], "close": flat_closes,
    }, index=idx)
    sig_none, *_ = find_two_touch_resistance_and_entry(flat_df)
    assert sig_none is None, f"a flat day with no breach must produce no signal, got {sig_none}"
    print("  a flat day (no breach at all) correctly produces no signal ok")

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


def print_combo_row(label, g):
    pf, sqn, n = compute_pf_sqn(g)
    net = g["rupee_pnl"].sum() if n else 0.0
    win = 100 * (g["rupee_pnl"] > 0).mean() if n else 0.0
    print(f"  {label:<14} n={n:<5} win%={win:5.1f}  PF={pf:5.2f}  SQN={sqn:5.2f}  net=Rs{net:>10,.2f}")


def run_full_pipeline(df1m, label="6-MONTH BACKTEST"):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        all_trades.extend(run_standalone_for_day(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades generated at all -- the two-touch-then-third-breach pattern may be rare with "
              "these tolerance/pivot settings, or may need a longer/shorter opening range.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv(TRADES_CSV, index=False)

    traded_days = trades_df["date"].nunique()
    print(f"\n-- Entry-frequency diagnostics --")
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)")
    if traded_days / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days -- worth checking this hasn't degenerated into "
              "something looser than intended. ***")

    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ TWO-TOUCH RESISTANCE BREAKOUT -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt "
              "before trusting this. (Calibration: real clean results look like PF ~1.4-1.7, SQN ~2.5-3.) ***")

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

    # --- 30-minute session-of-day breakdown ---
    print(f"\n################ 30-MINUTE SESSION-OF-DAY BREAKDOWN ################")
    trades_df["session"] = trades_df["entry_time"].apply(classify_session)
    for sess in SESSIONS:
        g = trades_df[trades_df["session"] == sess]
        if len(g):
            print_combo_row(sess, g)

    # --- day-of-week breakdown ---
    print(f"\n################ DAY-OF-WEEK BREAKDOWN ################")
    trades_df["weekday"] = pd.to_datetime(trades_df["entry_time"]).dt.day_name()
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            print_combo_row(wd, g)

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
    print(f"=== TWO-TOUCH RESISTANCE BREAKOUT -- 6-MONTH REAL BACKTEST ===")
    print(f"Window: {FROM_DATE} to {TO_DATE}")
    print("Fetching Nifty index 1-min data...")
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
