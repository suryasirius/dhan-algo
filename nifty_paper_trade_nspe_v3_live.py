"""
Live PAPER (mock, NO real money / NO real orders) -- NSPE v3 (Nifty
Statistical Projection Engine, pure-trail, full-window-agreement)
=================================================================
IMPORTANT STANDING CAUTION, carried over from the real 2-year backtest:
this strategy's default combo (stop_frac=0.20, trail_frac=0.06) showed
PF=2.40/SQN=4.04 over 2 years, and its recent-6-month slice AND its
best-by-SQN grid combo both tripped this project's PF>2.5/SQN>5 red-flag
rule. Only 2 of 234 trades ever hit the initial stop -- meaning there's
almost no real data yet on what a genuine loss looks like in this
system. This is EXACTLY why it's being run as PAPER (mock) trading, not
real capital: to collect real, observed stop-loss behavior going
forward, not to trade this on size. Do not fund this live yet.

One trade per day: NSPE's day-runner (v3.run_standalone_for_day) scans
today's data from market open for the first bar where ALL FOUR lookback
windows (4/10/20/50-day) agree the market has broken the blended
projection level, with gap/volatility gates. Once found, it's locked in
for the day and managed with the standard incremental position stepper
(top3.step_long_premium_position) -- same as every other live script
here -- using a dynamic (ATR-scaled) stop/trail, default combo
stop_frac=0.20/trail_frac=0.06.

Does NOT touch nifty_statistical_projection_v3_backtest.py or any other
locked file. Imports v3/v1/top3/base and reuses their logic verbatim.

Log file: paper_trade_nspe_v3_log_<date>.csv

Run:
    cd "C:\\dhan algo"
    python nifty_paper_trade_nspe_v3_live.py --smoke-test
    python nifty_paper_trade_nspe_v3_live.py
"""

import sys
import csv
import os
import time
import datetime as dt

import nifty_paper_trade_top3_live as top3
import nifty_two_touch_resistance_breakout_backtest as base
import nifty_statistical_projection_backtest as v1
import nifty_statistical_projection_v3_backtest as v3

LOG_FILE_PREFIX = "paper_trade_nspe_v3_log_"
LOG_FIELDS = ["logged_at", "strategy", "action", "side", "strike", "bar_time",
              "price", "exit_reason", "pnl_pts", "rupee_pnl", "lot_size", "confidence"]
HISTORY_DAYS = 90  # enough calendar days for the 50-trading-day window + buffer


def _log_file_for(day):
    return f"{LOG_FILE_PREFIX}{day.isoformat()}.csv"


def _append_row(log_file, row):
    is_new = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    v3.run_self_checks()
    print("  v3.run_self_checks() re-verified ok (NSPE v3: full-agreement, pure-trail)")
    print("All self-checks passed.\n")


def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching real recent Nifty data, replaying most recent trading day ===\n")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    try:
        raw = base.fetch_index_chunked(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}\n")
        print("All self-checks above still passed. Re-run on the machine with a working Dhan "
              "connection before trusting the smoke test fully.")
        return
    if raw.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    days = sorted(set(raw.index.date))
    if len(days) < 2:
        print("Not enough days returned to smoke test (need at least 2).")
        return
    last_day = days[-1]
    daily_ohlc = v1.compute_daily_ohlc(raw[raw.index.date < last_day])
    day_df = raw[raw.index.date == last_day].copy()
    print(f"Replaying {last_day} ({len(day_df)} 1-min bars, {len(daily_ohlc)} prior days of history)\n")

    ctx = v3.run_standalone_for_day(day_df, daily_ohlc)
    if ctx is None:
        print("No NSPE v3 signal on this day (expected most days -- entry frequency is ~47% in the "
              "real 2-year backtest). No crash, self-checks passed -- that's what this test confirms.")
        return

    stop_pts = v3.DEFAULT_STOP_FRAC * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
    trail_pts = v3.DEFAULT_TRAIL_FRAC * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
    pos = {"entry_price": ctx["entry_price"], "stop_level": ctx["entry_price"] - stop_pts,
           "peak": ctx["entry_price"], "trailing_active": False, "last_checked_idx": ctx["exec_idx"] - 1}
    exit_info = top3.step_long_premium_position(pos, ctx["premium_df"], trail_pts)
    print(f"Signal: {ctx['side']}@{ctx['strike']}  confidence={ctx['confidence']*100:.0f}%  "
          f"entry={ctx['entry_price']:.2f}  stop_pts={stop_pts:.2f}  trail_pts={trail_pts:.2f}")
    if exit_info:
        print(f"Exit: {exit_info['reason']} @ {exit_info['exit_price']:.2f}")
    else:
        print("Position still open at end of available data (normal for a same-day replay).")
    print("\nSmoke test complete -- no crashes, all self-checks passed, ran end-to-end on real data. "
          "Nothing was written to the trade log CSV (smoke test only).")


def is_market_open():
    now = dt.datetime.now().time()
    return top3.MARKET_OPEN <= now <= top3.MARKET_CLOSE


def main_live():
    run_self_checks()
    today = dt.date.today()
    log_file = _log_file_for(today)
    print("=== LIVE PAPER TRADE (MOCK, NO REAL ORDERS) -- NSPE v3 ===")
    print(f"Log file: {log_file}")
    print(f"Polling every {top3.POLL_SECONDS}s during market hours "
          f"({top3.MARKET_OPEN}-{top3.MARKET_CLOSE} IST)\n")
    print("*** REMINDER: this strategy has open red flags from its 2-year backtest (near-zero "
          "stop-out rate, PF crossing 2.5 on some cuts). Paper trading only -- do not fund live. ***\n")

    ctx = None
    pos = None
    day_done = False

    while True:
        now = dt.datetime.now()
        if now.date() != today:
            today = now.date()
            log_file = _log_file_for(today)
            ctx, pos, day_done = None, None, False
            print(f"\nNew trading day: {today}. Log file: {log_file}")

        if not is_market_open():
            print(f"{now.strftime('%H:%M:%S')} Market closed, sleeping 60s...")
            time.sleep(60)
            continue

        try:
            to_date = today.strftime("%Y-%m-%d")
            from_date = (today - dt.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
            raw = base.fetch_index_chunked(from_date, to_date)
        except Exception:
            print("Failed to fetch candles, retrying...")
            time.sleep(top3.POLL_SECONDS)
            continue

        day_df = raw[raw.index.date == today].copy()
        if day_df.empty:
            print(f"{now.strftime('%H:%M:%S')} No candles for today yet, sleeping {top3.POLL_SECONDS}s...")
            time.sleep(top3.POLL_SECONDS)
            continue
        if day_done:
            print(f"{now.strftime('%H:%M:%S')} Today's one trade is already done (NSPE v3 is one-trade-"
                  f"per-day) -- idling until tomorrow, still polling every {top3.POLL_SECONDS}s.")
            time.sleep(top3.POLL_SECONDS)
            continue
        daily_ohlc = v1.compute_daily_ohlc(raw[raw.index.date < today])

        if ctx is None:
            print(f"{now.strftime('%H:%M:%S')} bars={len(day_df)} spot={day_df['close'].iloc[-1]:.2f}  "
                  f"NSPE-v3[searching...]")
            ctx = v3.run_standalone_for_day(day_df, daily_ohlc)
            if ctx is not None:
                stop_pts = v3.DEFAULT_STOP_FRAC * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
                trail_pts = v3.DEFAULT_TRAIL_FRAC * ctx["blended_avg_range"] * top3.ASSUMED_DELTA
                pos = {"entry_price": ctx["entry_price"], "stop_level": ctx["entry_price"] - stop_pts,
                       "peak": ctx["entry_price"], "trailing_active": False,
                       "last_checked_idx": ctx["exec_idx"] - 1}
                ctx["stop_pts"], ctx["trail_pts"] = stop_pts, trail_pts
                _append_row(log_file, {
                    "logged_at": now.isoformat(timespec="seconds"), "strategy": "NSPE_v3", "action": "ENTRY",
                    "side": ctx["side"], "strike": ctx["strike"], "bar_time": ctx["premium_df"].index[ctx["exec_idx"]],
                    "price": round(ctx["entry_price"], 2), "exit_reason": "", "pnl_pts": "", "rupee_pnl": "",
                    "lot_size": ctx["lot"], "confidence": f"{ctx['confidence']*100:.0f}%",
                })
                print(f"  [NSPE_v3] ENTRY {ctx['side']}@{ctx['strike']} premium={ctx['entry_price']:.2f} "
                      f"confidence={ctx['confidence']*100:.0f}% stop_pts={stop_pts:.2f} trail_pts={trail_pts:.2f}")
        else:
            day_open_spot = day_df["open"].iloc[0]
            premium_df = top3.build_premium_series(day_df, ctx["strike"], day_open_spot, side=ctx["side"])
            exit_info = top3.step_long_premium_position(pos, premium_df, ctx["trail_pts"])
            print(f"{now.strftime('%H:%M:%S')} bars={len(day_df)}  NSPE-v3[OPEN {ctx['side']}@{ctx['strike']}]")
            if exit_info:
                pnl_pts = exit_info["exit_price"] - ctx["entry_price"]
                rupee_pnl = pnl_pts * ctx["lot"] - v3.BROKERAGE_PER_TRADE
                _append_row(log_file, {
                    "logged_at": now.isoformat(timespec="seconds"), "strategy": "NSPE_v3", "action": "EXIT",
                    "side": ctx["side"], "strike": ctx["strike"], "bar_time": exit_info["exit_time"],
                    "price": round(exit_info["exit_price"], 2), "exit_reason": exit_info["reason"],
                    "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
                    "lot_size": ctx["lot"], "confidence": f"{ctx['confidence']*100:.0f}%",
                })
                print(f"  [NSPE_v3] EXIT {ctx['side']}@{ctx['strike']} @ {exit_info['exit_price']:.2f} "
                      f"({exit_info['reason']}) pnl_Rs={rupee_pnl:,.2f}")
                day_done = True  # one trade per day, per the backtest spec

        time.sleep(top3.POLL_SECONDS)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main_live()
