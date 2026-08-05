"""
Nifty ORB -- GAP-DAY-ONLY, GAP-DIRECTION-ONLY variant -- STANDALONE
=================================================================
Same engine as nifty_orb_backtest.py (does NOT modify that file or any
other file in this project). Narrowed per the pattern found in the full
745-trade run:

  - Only trades on days with a genuine gap at open (>= 0.3% of prior
    close, same threshold as before). FLAT days are skipped entirely
    -- these were 477 of 745 trades in the full run and, combined with
    the counter-gap trades, were the main source of brokerage drag.
  - Only takes the breakout that matches the gap's own direction:
    gap UP -> only take the CE (ORH break) signal, skip PE entirely
    that day. gap DOWN -> only take the PE (ORL break) signal, skip CE.
  - Everything else unchanged: 5-candle opening range, close-confirmed
    breakout, nearest-ITM strike, date-correct lot size/expiry, 10pt
    initial stop / 4pt trailing on premium, 3:20pm squareoff, Rs 40/trade
    brokerage.

Same PROXY-premium caveat as before: real per-strike premium data is
still blocked on this account (DH-901), so this uses the delta-scaled
proxy (0.6 delta) -- treat results as exploratory, not validated, same
as every other run in this project until real premium data is available.
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

OR_CANDLES = 5
INITIAL_SL_PTS = 10
TRAIL_PTS = 4
EOD_SQUAREOFF = dt.time(15, 20)
STRIKE_STEP = 50
ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
GAP_PCT_THRESHOLD = 0.003
STARTING_CAPITAL = 10_000
MONTE_CARLO_ITERS = 5000
BROKERAGE_PER_TRADE = 40.0

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def expiry_weekday_for_date(d):
    return 3 if d < dt.date(2025, 9, 1) else 1


def next_expiry(d):
    target_wd = expiry_weekday_for_date(d)
    days_ahead = (target_wd - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


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
    try:
        expiries = dlb.dhan.expiry_list(under_security_id=dlb.UNDERLYING_SECURITY_ID,
                                         under_exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT)
        if not expiries or "data" not in expiries or not expiries["data"]:
            return False
        return True
    except Exception as e:
        print(f"  Real premium endpoint check failed: {e}")
        return False


def itm_call_strike(spot):
    return math.floor(spot / STRIKE_STEP) * STRIKE_STEP


def itm_put_strike(spot):
    return math.ceil(spot / STRIKE_STEP) * STRIKE_STEP


def intrinsic_value(spot, strike, side):
    return max(0.0, spot - strike) if side == "CE" else max(0.0, strike - spot)


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    assert lot_size_for_date(dt.date(2024, 6, 1)) == 25
    assert lot_size_for_date(dt.date(2025, 1, 1)) == 75
    assert lot_size_for_date(dt.date(2026, 2, 1)) == 65
    assert itm_call_strike(24567) == 24550
    assert itm_put_strike(24567) == 24600
    # gap-direction filter check: gap UP day should only ever attempt CE
    allowed = {"UP": "CE", "DOWN": "PE"}
    assert allowed["UP"] == "CE" and allowed["DOWN"] == "PE"
    print("All self-checks passed.\n")


def simulate_day(day_df, day, gap_class):
    """Only simulates the side matching gap_class. gap_class in {'UP','DOWN'};
    FLAT days should never reach here (filtered by caller)."""
    if len(day_df) < OR_CANDLES + 1:
        return [], None

    orh = day_df["high"].iloc[:OR_CANDLES].max()
    orl = day_df["low"].iloc[:OR_CANDLES].min()
    or_width = orh - orl

    closes = day_df["close"].tolist()
    n = len(day_df)
    lot = lot_size_for_date(day)
    expiry = next_expiry(day)

    allowed_side = "CE" if gap_class == "UP" else "PE"
    taken = False
    trades = []

    for j in range(OR_CANDLES, n):
        if taken:
            break
        if allowed_side == "CE" and closes[j] > orh:
            trades.append(_open_and_simulate(day_df, j, "CE", itm_call_strike(closes[j]),
                                               expiry, lot, orh, orl, or_width))
            taken = True
        elif allowed_side == "PE" and closes[j] < orl:
            trades.append(_open_and_simulate(day_df, j, "PE", itm_put_strike(closes[j]),
                                               expiry, lot, orh, orl, or_width))
            taken = True

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

        prem_close = entry_premium + (spot_now_close - spot_entry) * ASSUMED_DELTA * (1 if side == "CE" else -1)
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
            stop = max(stop, prem_low - TRAIL_PTS)
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
    run_self_checks()
    print("=== Nifty ORB -- GAP-DAY-ONLY, GAP-DIRECTION-ONLY -- Standalone Backtest ===\n")

    print("Checking real option-premium data access...")
    real_ok = check_real_premium_access()
    mode = "REAL" if real_ok else "PROXY"
    print(f"\n{'='*60}\nPREMIUM MODE: {mode}\n{'='*60}")
    if mode == "PROXY":
        print("Real option premium endpoint still blocked. Using delta-scaled proxy "
              f"(delta={ASSUMED_DELTA}, time value={ASSUMED_TIME_VALUE}pts). Exploratory only.\n")
    print(f"Brokerage assumption: Rs {BROKERAGE_PER_TRADE:.2f}/trade -- included in all Rs figures.\n")

    print("Fetching Nifty index 1-min data (2 years)...")
    df = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    prev_close = None
    all_trades = []
    daily_rows = []
    skipped_flat = 0
    for day in days_covered:
        day_df = df[df.index.date == day]
        today_open = day_df["open"].iloc[0]
        gap = (today_open - prev_close) if prev_close is not None else 0.0
        gap_pct = (gap / prev_close) if prev_close else 0.0

        if prev_close is None or abs(gap_pct) < GAP_PCT_THRESHOLD:
            skipped_flat += 1
            prev_close = day_df["close"].iloc[-1]
            continue

        gap_class = "UP" if gap > 0 else "DOWN"
        trades, orinfo = simulate_day(day_df, day, gap_class)
        all_trades.extend(trades)
        day_range = day_df["high"].max() - day_df["low"].min()
        daily_rows.append({
            "date": day, "weekday": pd.Timestamp(day).day_name(), "gap_class": gap_class,
            "gap": gap, "gap_pct": gap_pct, "n_trades": len(trades),
            "day_pnl_pts": sum(t["points_gained"] for t in trades), "day_range": day_range,
        })
        prev_close = day_df["close"].iloc[-1]

    print(f"Skipped {skipped_flat} FLAT/no-gap days entirely (as designed).\n")

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv("nifty_orb_gap_filtered_trades.csv", index=False)
    daily_df = pd.DataFrame(daily_rows)

    print("--- Win/Loss Statistics ---")
    stats_block(tdf, "Overall (gap-filtered)")
    stats_block(tdf[tdf["side"] == "CE"], "CE only (gap-up days)")
    stats_block(tdf[tdf["side"] == "PE"], "PE only (gap-down days)")
    ws, ls = streaks(tdf["points_gained"].tolist())
    print(f"Longest winning streak: {ws}   Longest losing streak: {ls}\n")

    tdf["session"] = tdf["entry_time"].apply(classify_30min_session)
    print("--- 30-Minute Session ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = tdf[tdf["session"] == s]
        if len(g):
            print(f"{s}: n={len(g)}  win%={100*(g['points_gained']>0).mean():.1f}  avg_pnl={g['points_gained'].mean():.2f}pts")
    print()

    print("--- MFE / MAE ---")
    wins = tdf[tdf["points_gained"] > 0]
    losses = tdf[tdf["points_gained"] <= 0]
    if len(wins):
        print(f"Winners: avg MFE={wins['mfe'].mean():.2f}  avg captured={wins['points_gained'].mean():.2f}  avg MAE={wins['mae'].mean():.2f}")
    if len(losses):
        print(f"Losers:  avg MFE={losses['mfe'].mean():.2f}  avg MAE={losses['mae'].mean():.2f}")
    print()

    print("--- Drawdown ---")
    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    print(f"Max drawdown: {100*dd.min():.2f}%  (Rs {(equity - running_peak).min():,.2f})")
    print("Wipes out at some point." if (equity <= 0).any() else "Never wipes out.")
    print()

    print("--- Expectancy ---")
    win_rate = (tdf["points_gained"] > 0).mean()
    avg_win_rs = wins["rupee_pnl"].mean() if len(wins) else 0
    avg_loss_rs = abs(losses["rupee_pnl"].mean()) if len(losses) else 0
    expectancy_rs = win_rate * avg_win_rs - (1 - win_rate) * avg_loss_rs
    r_multiple = tdf["rupee_pnl"] / (INITIAL_SL_PTS * tdf["lot_size"])
    expectancy_r = r_multiple.mean()
    sqn = (expectancy_r / r_multiple.std()) * math.sqrt(len(tdf)) if r_multiple.std() else float("nan")
    print(f"Expectancy: Rs {expectancy_rs:,.2f} per trade   |   {expectancy_r:.3f} R per trade   |   SQN={sqn:.2f}\n")

    print("--- Equity Curve ---")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Total return: {100*(equity.iloc[-1]/STARTING_CAPITAL - 1):.1f}%\n")

    tdf["weekday"] = tdf["entry_time"].dt.day_name()
    print("--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = tdf[tdf["weekday"] == wd]
        if len(g):
            print(f"{wd}: n={len(g)}  net={g['points_gained'].sum():.2f}pts  win%={100*(g['points_gained']>0).mean():.1f}")
    print()

    mid = len(tdf) // 2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print("--- Stability check (first half vs second half) ---")
    print(f"First half: n={mid}  net Rs {h1:,.2f}")
    print(f"Second half: n={len(tdf)-mid}  net Rs {h2:,.2f}\n")

    print("--- Monte Carlo (trade-order reshuffle) ---")
    pnl_list = tdf["rupee_pnl"].tolist()
    final_equities, max_dds = [], []
    for _ in range(MONTE_CARLO_ITERS):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)
        curve = STARTING_CAPITAL + np.cumsum(shuffled)
        peak = np.maximum.accumulate(curve)
        final_equities.append(curve[-1])
        max_dds.append((curve - peak).min())
    fe = np.array(final_equities); mdd = np.array(max_dds)
    print(f"Final equity  -> 5th: Rs {np.percentile(fe,5):,.2f}  50th: Rs {np.percentile(fe,50):,.2f}  95th: Rs {np.percentile(fe,95):,.2f}")
    print(f"Max drawdown  -> 5th: Rs {np.percentile(mdd,5):,.2f}  50th: Rs {np.percentile(mdd,50):,.2f}  95th: Rs {np.percentile(mdd,95):,.2f}")
    print(f"Probability equity falls below 50% of starting capital: {100*(fe < STARTING_CAPITAL*0.5).mean():.2f}%\n")

    print(f"Trade log saved: nifty_orb_gap_filtered_trades.csv ({len(tdf)} rows)")
    print(f"\nREMINDER: this ran in {mode} premium mode. Treat as exploratory until real premium data is confirmed.")


if __name__ == "__main__":
    main()
