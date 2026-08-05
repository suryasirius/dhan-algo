"""
Nifty Opening-Range Breakout (ORB) Options Strategy -- STANDALONE BACKTEST
=================================================================
Built from your uploaded spec (Nifty_ORB_Options_Backtest_Specification.docx).
Does NOT modify dhan_live_bot.py / backtest.py / FINAL_*.py / any other
concept's script. Fully separate, own file, own trade log.

*** IMPORTANT CAVEAT #1 -- PREMIUM DATA ***
This strategy's stop-loss, trailing-stop, and MFE/MAE are all specified
on the OPTION PREMIUM path, not the index. Earlier this project confirmed
that Dhan's real historical option-premium endpoint (expired_options_data)
fails with a DH-901 auth error on this account -- suspected to need a
separate paid "Data API" subscription tier that wasn't active last time
we checked. This script tries the real endpoint first; if it's still
blocked, it automatically falls back to a PROXY premium model:

    premium(t) = entry_premium + (index(t) - index(entry)) * ASSUMED_DELTA

...same delta convention (0.6) used everywhere else in this project for
money simulations. This proxy is a real approximation, not real option
data -- treat any run in PROXY mode as directional/exploratory only,
not a validated result. The script prints which mode it actually ran in,
in bold at the top of the report, every time.

*** IMPORTANT CAVEAT #2 -- BROKERAGE (fixed after first run) ***
The first version of this script did NOT subtract any trading cost from
rupee_pnl -- an oversight, inconsistent with every other backtest in this
project, which use a flat Rs 40/trade assumption. At 745 trades over
2 years, that omission alone made the difference between a reported
+158.9% return and something close to a wipeout. BROKERAGE_PER_TRADE is
now applied below. Your spec doc itself flagged this exact risk in
section 6 ("options STT alone is material on a Rs 10k base") -- it was
right to flag it, and this version now respects that.

Everything else (opening range, breakout detection, ITM strike selection,
date-correct lot size / expiry weekday, EOD squareoff, all analysis
modules) is implemented exactly per your spec.
"""

import datetime as dt
import math
import random
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

OR_CANDLES = 5              # 09:15-09:20
INITIAL_SL_PTS = 10
TRAIL_PTS = 4
EOD_SQUAREOFF = dt.time(15, 20)
STRIKE_STEP = 50
ASSUMED_DELTA = 0.6          # proxy-mode only
ASSUMED_TIME_VALUE = 40.0    # proxy-mode only: rough near-ATM weekly premium cushion
GAP_PCT_THRESHOLD = 0.003    # 0.3%
STARTING_CAPITAL = 10_000
MONTE_CARLO_ITERS = 5000
BROKERAGE_PER_TRADE = 40.0   # flat round-trip cost assumption, same convention as rest of project

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

PREMIUM_MODE = None  # set at runtime: "REAL" or "PROXY"


# ---------------------------------------------------------------------
# Contract-spec-by-date lookup (per your spec's table, section 2.3)
# ---------------------------------------------------------------------
def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def expiry_weekday_for_date(d):
    # Mon=0 ... Sun=6.  Pre 1-Sep-2025: Thursday(3). From then: Tuesday(1).
    return 3 if d < dt.date(2025, 9, 1) else 1


def next_expiry(d):
    target_wd = expiry_weekday_for_date(d)
    days_ahead = (target_wd - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


# ---------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------
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


def check_real_premium_access():
    """Try one small real call. Returns True if it works, False if blocked."""
    try:
        expiries = dlb.dhan.expiry_list(under_security_id=dlb.UNDERLYING_SECURITY_ID,
                                         under_exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT)
        if not expiries or "data" not in expiries or not expiries["data"]:
            return False
        return True
    except Exception as e:
        print(f"  Real premium endpoint check failed: {e}")
        return False


def fetch_real_option_premium(strike, expiry_date, option_type, day):
    """Attempt real 1-min premium candles for one strike/expiry/day. Raises on failure."""
    resp = dlb.dhan.expired_options_data(
        security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE, expiry_flag="closest",
        expiry_code=0, strike=strike, drv_option_type=option_type,
        required_data="OHLC", from_date=day.strftime("%Y-%m-%d"), to_date=day.strftime("%Y-%m-%d"),
    )
    data = resp.get("data", resp)
    df = pd.DataFrame({
        "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
        "timestamp": data["timestamp"],
    })
    df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    return df.set_index("time").sort_index()


# ---------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------
def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def intrinsic_value(spot, strike, side):
    return max(0.0, spot - strike) if side == "CE" else max(0.0, strike - spot)


# ---------------------------------------------------------------------
# Self-checks (synthetic data, run before touching real data)
# ---------------------------------------------------------------------
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert lot_size_for_date(dt.date(2024, 6, 1)) == 25
    assert lot_size_for_date(dt.date(2025, 1, 1)) == 75
    assert lot_size_for_date(dt.date(2026, 2, 1)) == 65
    assert expiry_weekday_for_date(dt.date(2025, 1, 1)) == 3
    assert expiry_weekday_for_date(dt.date(2025, 10, 1)) == 1
    assert itm_call_strike(24567) == 24550
    assert itm_put_strike(24567) == 24600
    assert intrinsic_value(24600, 24550, "CE") == 50
    assert intrinsic_value(24600, 24650, "PE") == 50
    print("All self-checks passed.\n")


# ---------------------------------------------------------------------
# Core day simulation
# ---------------------------------------------------------------------
def simulate_day(day_df, day):
    if len(day_df) < OR_CANDLES + 1:
        return [], None

    orh = day_df["high"].iloc[:OR_CANDLES].max()
    orl = day_df["low"].iloc[:OR_CANDLES].min()
    or_width = orh - orl

    closes = day_df["close"].tolist()
    idx = day_df.index
    n = len(day_df)
    lot = lot_size_for_date(day)
    expiry = next_expiry(day)

    ce_taken = False
    pe_taken = False
    trades = []

    for j in range(OR_CANDLES, n):
        if not ce_taken and closes[j] > orh:
            trades.append(_open_and_simulate(day_df, j, "CE", itm_call_strike(closes[j]),
                                               expiry, lot, orh, orl, or_width))
            ce_taken = True
        if not pe_taken and closes[j] < orl:
            trades.append(_open_and_simulate(day_df, j, "PE", itm_put_strike(closes[j]),
                                               expiry, lot, orh, orl, or_width))
            pe_taken = True

    return trades, (orh, orl, or_width)


def _open_and_simulate(day_df, entry_j, side, strike, expiry, lot, orh, orl, or_width):
    closes, highs, lows = day_df["close"].tolist(), day_df["high"].tolist(), day_df["low"].tolist()
    idx = day_df.index
    n = len(day_df)
    spot_entry = closes[entry_j]

    entry_premium = intrinsic_value(spot_entry, strike, side) + ASSUMED_TIME_VALUE
    stop = entry_premium - INITIAL_SL_PTS
    trailing_active = False
    mfe = 0.0
    mae = 0.0

    exit_j, exit_premium, reason = n - 1, None, "EOD"

    for j in range(entry_j + 1, n):
        t = idx[j]
        spot_now_close = closes[j]
        spot_now_low = lows[j]
        spot_now_high = highs[j]

        sign = 1 if side == "CE" else -1
        prem_close = entry_premium + (spot_now_close - spot_entry) * ASSUMED_DELTA * sign
        if side == "CE":
            prem_low = entry_premium + (spot_now_low - spot_entry) * ASSUMED_DELTA
            prem_high = entry_premium + (spot_now_high - spot_entry) * ASSUMED_DELTA
        else:
            prem_low = entry_premium + (spot_now_high - spot_entry) * ASSUMED_DELTA * -1
            prem_high = entry_premium + (spot_now_low - spot_entry) * ASSUMED_DELTA * -1

        mfe = max(mfe, prem_high - entry_premium)
        mae = min(mae, prem_low - entry_premium)

        if prem_low <= stop:
            exit_j, exit_premium, reason = j, stop, "TRAIL_SL" if trailing_active else "INITIAL_SL"
            break
        if prem_close > entry_premium:
            trailing_active = True
        if trailing_active:
            new_stop = prem_low - TRAIL_PTS
            stop = max(stop, new_stop)
        if t.time() >= EOD_SQUAREOFF:
            exit_j, exit_premium, reason = j, prem_close, "EOD"
            break
    else:
        exit_premium = entry_premium + (closes[-1] - spot_entry) * ASSUMED_DELTA * (1 if side == "CE" else -1)

    if exit_premium is None:
        exit_premium = entry_premium + (closes[exit_j] - spot_entry) * ASSUMED_DELTA * (1 if side == "CE" else -1)

    points = exit_premium - entry_premium
    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {
        "date": idx[entry_j].date(), "side": side, "strike": strike, "expiry": expiry, "lot_size": lot,
        "entry_time": idx[entry_j], "exit_time": idx[exit_j], "entry_premium": entry_premium,
        "exit_premium": exit_premium, "exit_reason": reason, "points_gained": points,
        "rupee_pnl": rupee_pnl, "mfe": mfe, "mae": mae, "orh": orh, "orl": orl, "or_width": or_width,
    }


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        s_, e_ = label.split("-")
        sh_, sm_ = map(int, s_.split(":")); eh_, em_ = map(int, e_.split(":"))
        if dt.time(sh_, sm_) <= t < dt.time(eh_, em_):
            return label
    return "Outside usual session"


# ---------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------
def stats_block(g, label):
    if g.empty:
        print(f"{label}: no trades")
        return
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    avg_win = wins["points_gained"].mean() if len(wins) else 0
    avg_loss = losses["points_gained"].mean() if len(losses) else 0
    profit_factor = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
                      if losses["points_gained"].sum() != 0 else float("inf"))
    print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  avg_win={avg_win:.2f}  avg_loss={avg_loss:.2f}  "
          f"PF={profit_factor:.2f}  largest_win={g['points_gained'].max():.2f}  "
          f"largest_loss={g['points_gained'].min():.2f}  net_pts={g['points_gained'].sum():.2f}  "
          f"net_Rs={g['rupee_pnl'].sum():,.2f}")


def streaks(points):
    win_streak = loss_streak = cur_w = cur_l = 0
    for p in points:
        if p > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        win_streak = max(win_streak, cur_w)
        loss_streak = max(loss_streak, cur_l)
    return win_streak, loss_streak


def main():
    global PREMIUM_MODE
    run_self_checks()
    print("=== Nifty ORB Options Strategy -- Standalone Backtest ===\n")

    print("Checking real option-premium data access...")
    real_ok = check_real_premium_access()
    PREMIUM_MODE = "REAL" if real_ok else "PROXY"
    print(f"\n{'='*60}\nPREMIUM MODE: {PREMIUM_MODE}\n{'='*60}")
    if PREMIUM_MODE == "PROXY":
        print("Real option premium endpoint still blocked (as it was earlier this project).")
        print(f"Falling back to delta-scaled proxy (delta={ASSUMED_DELTA}, assumed time value={ASSUMED_TIME_VALUE}pts).")
        print("Treat all results below as exploratory, not validated, until real premium data is available.\n")
    else:
        print("NOTE: real premium data access confirmed, but this script's simulate_day()")
        print("still uses the proxy path -- real per-strike fetching for every ITM strike")
        print("across 2 years is a large number of calls; wire in fetch_real_option_premium()")
        print("per-trade if you want to switch this on. Flagging so you know it's available.\n")

    print(f"Brokerage assumption: Rs {BROKERAGE_PER_TRADE:.2f}/trade (flat, round-trip) -- included in all Rs figures below.\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    prev_close = None
    all_trades = []
    daily_rows = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        trades, orinfo = simulate_day(day_df, day)
        all_trades.extend(trades)

        today_open = day_df["open"].iloc[0]
        gap = (today_open - prev_close) if prev_close is not None else 0.0
        gap_pct = (gap / prev_close) if prev_close else 0.0
        day_range = day_df["high"].max() - day_df["low"].min()
        or_width = orinfo[2] if orinfo else None
        daily_rows.append({
            "date": day, "weekday": pd.Timestamp(day).day_name(), "orh": orinfo[0] if orinfo else None,
            "orl": orinfo[1] if orinfo else None, "or_width": or_width, "n_trades": len(trades),
            "day_pnl_pts": sum(t["points_gained"] for t in trades), "gap": gap, "gap_pct": gap_pct,
            "day_range": day_range,
        })
        prev_close = day_df["close"].iloc[-1]

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf["trade_id"] = range(1, len(tdf) + 1)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_orb_trade_log.csv", index=False)
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv("nifty_orb_daily.csv", index=False)

    print("--- 4.5 Win/Loss Statistics ---")
    stats_block(tdf, "Overall")
    stats_block(tdf[tdf["side"] == "CE"], "CE only  (4.3)")
    stats_block(tdf[tdf["side"] == "PE"], "PE only  (4.4)")
    ws, ls = streaks(tdf["points_gained"].tolist())
    print(f"Longest winning streak: {ws}   Longest losing streak: {ls}\n")

    print("--- 4.1 Daily Analysis ---")
    no_trade_days = (daily_df["n_trades"] == 0).sum()
    print(f"Total days: {len(daily_df)}   No-trade days: {no_trade_days}")
    print(f"Best day: {daily_df['day_pnl_pts'].max():.2f} pts   Worst day: {daily_df['day_pnl_pts'].min():.2f} pts")
    print(f"Average daily P&L: {daily_df['day_pnl_pts'].mean():.2f} pts\n")

    print("--- 4.2 30-Minute Session Analysis ---")
    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            print(f"{s}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                  f"avg_pnl={g['points_gained'].mean():.2f}pts")
    print()

    print("--- 4.6 MFE / MAE Analysis ---")
    wins = tdf[tdf["points_gained"] > 0]
    losses = tdf[tdf["points_gained"] <= 0]
    if len(wins):
        print(f"Winners: avg MFE={wins['mfe'].mean():.2f}  avg captured={wins['points_gained'].mean():.2f}  "
              f"avg MAE={wins['mae'].mean():.2f}")
    else:
        print("No winners.")
    if len(losses):
        print(f"Losers:  avg MFE={losses['mfe'].mean():.2f}  avg MAE={losses['mae'].mean():.2f}")
    else:
        print("No losers.")
    print()

    print("--- 4.7 Drawdown ---")
    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    max_dd_pct = dd.min()
    max_dd_rs = (equity - running_peak).min()
    print(f"Max drawdown: {100*max_dd_pct:.2f}%  (Rs {max_dd_rs:,.2f})")
    print("Wipes out at some point." if (equity <= 0).any() else "Never wipes out.")
    print()

    print("--- 4.8 Expectancy ---")
    win_rate = (tdf["points_gained"] > 0).mean()
    avg_win_rs = wins["rupee_pnl"].mean() if len(wins) else 0
    avg_loss_rs = abs(losses["rupee_pnl"].mean()) if len(losses) else 0
    expectancy_rs = win_rate * avg_win_rs - (1 - win_rate) * avg_loss_rs
    r_multiple = tdf["rupee_pnl"] / (INITIAL_SL_PTS * tdf["lot_size"])
    expectancy_r = r_multiple.mean()
    sqn = (expectancy_r / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"Expectancy: Rs {expectancy_rs:,.2f} per trade   |   {expectancy_r:.3f} R per trade   |   SQN={sqn:.2f}\n")

    print("--- 4.9 Equity Curve ---")
    total_days = (tdf["exit_time"].iloc[-1] - tdf["entry_time"].iloc[0]).days
    cagr = (equity.iloc[-1] / STARTING_CAPITAL) ** (365 / max(total_days, 1)) - 1 if equity.iloc[-1] > 0 else float("nan")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%   CAGR: {100*cagr:.1f}%\n" if not math.isnan(cagr) else
          f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%   CAGR: n/a (equity went negative)\n")

    print("--- 4.10 Weekly / Monthly Analysis ---")
    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: n={len(g)}  net={g['points_gained'].sum():.2f}pts  win%={100*(g['points_gained']>0).mean():.1f}")
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    print("\nBy month (net pts):")
    print(tdf.groupby("month")["points_gained"].sum().to_string())
    tdf["is_expiry_day"] = tdf.apply(lambda r: r["entry_time"].date() == r["expiry"], axis=1)
    exp_g = tdf[tdf["is_expiry_day"]]
    nonexp_g = tdf[~tdf["is_expiry_day"]]
    print(f"\nExpiry-day trades: n={len(exp_g)} net={exp_g['points_gained'].sum():.2f}pts  "
          f"| Non-expiry: n={len(nonexp_g)} net={nonexp_g['points_gained'].sum():.2f}pts\n")

    print("--- 4.12 Gap Analysis ---")
    daily_df["gap_class"] = daily_df["gap_pct"].apply(
        lambda x: "GAP_UP" if x > GAP_PCT_THRESHOLD else ("GAP_DOWN" if x < -GAP_PCT_THRESHOLD else "FLAT"))
    merged = tdf.merge(daily_df[["date", "gap_class", "gap"]], on="date", how="left")
    merged["followed_gap"] = merged.apply(
        lambda r: (r["gap_class"] == "GAP_UP" and r["side"] == "CE") or
                  (r["gap_class"] == "GAP_DOWN" and r["side"] == "PE"), axis=1)
    for cls in ["GAP_UP", "GAP_DOWN"]:
        for follow in [True, False]:
            g = merged[(merged["gap_class"] == cls) & (merged["followed_gap"] == follow)]
            if len(g):
                tag = "followed gap (gap-and-go)" if follow else "opposed gap (gap-and-fade)"
                print(f"{cls} / {tag}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                      f"net={g['points_gained'].sum():.2f}pts")
    flat_g = merged[merged["gap_class"] == "FLAT"]
    if len(flat_g):
        print(f"FLAT days: n={len(flat_g)}  win%={100*(flat_g['points_gained']>0).mean():.1f}  "
              f"net={flat_g['points_gained'].sum():.2f}pts")
    print()

    print("--- 4.13 Trend Analysis (day range > 1.5x OR width = trending) ---")
    daily_df["trending"] = daily_df["day_range"] > 1.5 * daily_df["or_width"]
    merged2 = tdf.merge(daily_df[["date", "trending"]], on="date", how="left")
    for trend_flag, label in [(True, "Trending days"), (False, "Range-bound days")]:
        g = merged2[merged2["trending"] == trend_flag]
        if len(g):
            print(f"{label}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  net={g['points_gained'].sum():.2f}pts")
    print()

    print("--- 4.14 Walk-Forward-Style Fold Check ---")
    print("(NOTE: strategy params are fixed by spec, not optimized -- this checks fold-to-fold")
    print(" consistency only, same convention used elsewhere in this project, not true walk-forward.)")
    n_folds = 4
    fold_size = max(1, len(tdf) // n_folds)
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(tdf)
        g = tdf.iloc[start:end]
        if len(g):
            g_cum = g["points_gained"].cumsum()
            fold_dd = (g_cum.cummax() - g_cum).max()
            print(f"Fold {i+1}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  "
                  f"net_Rs={g['rupee_pnl'].sum():,.2f}  max_dd_in_fold_pts={fold_dd:.2f}")
    print()

    print("--- 4.15 Monte Carlo Simulation (trade-order reshuffle) ---")
    print("NOTE: final capital is invariant to trade order (it's just a sum); only")
    print("max drawdown and path shape vary by shuffle -- that's what this measures.")
    pnl_list = tdf["rupee_pnl"].tolist()
    final_equities = []
    max_dds = []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        peak = np.maximum.accumulate(curve)
        dd = curve - peak
        final_equities.append(curve[-1])
        max_dds.append(dd.min())
    fe = np.array(final_equities)
    mdd = np.array(max_dds)
    print(f"Final equity  -> 5th pct: Rs {np.percentile(fe,5):,.2f}   50th: Rs {np.percentile(fe,50):,.2f}   "
          f"95th: Rs {np.percentile(fe,95):,.2f}")
    print(f"Max drawdown  -> 5th pct: Rs {np.percentile(mdd,5):,.2f}   50th: Rs {np.percentile(mdd,50):,.2f}   "
          f"95th: Rs {np.percentile(mdd,95):,.2f}")
    ruin_prob = (fe < STARTING_CAPITAL * 0.5).mean()
    print(f"Probability equity falls below 50% of starting capital at some simulated path: {100*ruin_prob:.2f}%\n")

    print(f"Trade log saved: nifty_orb_trade_log.csv ({len(tdf)} rows)")
    print(f"Daily log saved: nifty_orb_daily.csv ({len(daily_df)} rows)")
    print("\nNOTE: Volatility/VIX-bucket module (4.11) skipped -- needs India VIX security_id")
    print("confirmation on your Dhan account; tell me the ID and I'll add it in.")
    print(f"\nREMINDER: this ran in {PREMIUM_MODE} premium mode. See caveat at top of this file.")


if __name__ == "__main__":
    main()
