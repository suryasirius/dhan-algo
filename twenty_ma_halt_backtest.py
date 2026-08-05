"""
Oliver Velez "20 MA Halt" Bottom/Top-Picking Strategy -- Nifty ITM Options
=================================================================
STANDALONE -- does not modify any other file in this project.

Scope for this first pass, per the doc's own "Notes for Implementation"
(and matching the phased approach already used for the Moving-Average
Price-Action method): SINGLE-HALT version only (Section 6's two-halt
add-on is explicitly deferred -- validate the simple version first).
Both directions (bottom-picking LONG and mirror top-picking SHORT) are
built and reported SEPARATELY, per the doc's own recommendation.
Timeframe: 2-minute Nifty bars (the doc's stated primary teaching
timeframe, and this project's established intraday scope).

No Greeks/delta modeling -- raw index points x lot size, same
convention as the current Velez final version. Rs 40/trade brokerage.
Date-correct lot size / expiry (same table as every other options
script this project).

Setup steps implemented (LONG / bottom-picking version described,
SHORT is the exact mirror):
  1. Elongated decline: at least MIN_DECLINE_BARS of the last
     DECLINE_LOOKBACK_BARS bars show price below a declining 20MA.
     "Space" is measured two ways during that decline -- Space 1 (20MA
     vs 200MA distance) and Space 2 (price vs 20MA distance) -- each
     trade is logged as dual_space or single_space so that filter can
     be evaluated afterward, per the doc's own suggestion to test it
     rather than assume it.
  2. Sear-through: a close above the 20MA by a minimum buffer (not a
     mere touch), following the decline.
  3. Halt: the pullback that follows must be caught by the 20MA within
     a window of bars -- a close back above 20MA without 2 consecutive
     closes back below it (the "leaning on it like ropes" allowance).
     2 consecutive closes below invalidates the setup entirely.
  4. Entry: the instant a green bar's close breaks the high of the most
     recent red bar (or small red-bar cluster) before the halt.
  5. Management: stop = pullback low - buffer, exited only on 2
     consecutive closes below it (not a single wick). Target = the
     200MA level AT THE TIME OF ENTRY (a fixed price, described in the
     doc as a minimum objective, not guaranteed). EOD squareoff applies
     since this executes as same-day options, per this project's rules.
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
BAR_RULE = "2min"

SMA_FAST, SMA_SLOW = 20, 200
AVG_RANGE_LOOKBACK = 20

DECLINE_LOOKBACK_BARS = 30
MIN_DECLINE_BARS = 15
SPACE1_ATR_MULT = 1.5     # 20MA-vs-200MA distance, in avg_range units
SPACE2_ATR_MULT = 1.0     # price-vs-20MA distance, in avg_range units
SEAR_BUFFER_PTS = 2.0     # minimum close-through distance past the 20MA

HALT_WINDOW_BARS = 10     # bars allowed to find the halt after sear-through
TOUCH_TOLERANCE_PTS = 2.0 # "near/slightly pierce" the 20MA counts as a touch

STOP_BUFFER_PTS = 1.0
TWO_BAR_RULE = 2          # consecutive closes beyond a level required to invalidate/exit
EOD_SQUAREOFF = dt.time(15, 20)

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0
MONTE_CARLO_ITERS = 5000

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


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


def resample_2min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def compute_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    df["sma200"] = df["close"].rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["range"] = df["high"] - df["low"]
    df["avg_range"] = df["range"].shift(1).rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK).mean()
    return df


def is_green(o, c):
    return c > o


def is_red(o, c):
    return c < o


def find_marker_high(opens, closes, highs, i_halt):
    """Most recent red bar (or small combined red-bar cluster) high
    before the halt bar, scanning backward."""
    j = i_halt - 1
    while j >= 0 and not is_red(opens[j], closes[j]):
        j -= 1
    return j


def check_elongated_decline(closes, sma20, sma200, avg_ranges, i, lookback, min_bars):
    """Returns (ok, dual_space, single_space) for a decline ending just
    before index i (the candidate sear-through bar)."""
    start = max(0, i - lookback)
    bearish_count = 0
    max_space2 = 0.0
    space1_ok_seen = False
    for k in range(start, i):
        if pd.isna(sma20[k]) or pd.isna(sma200[k]) or pd.isna(avg_ranges[k]):
            continue
        if closes[k] < sma20[k] and sma20[k] < sma200[k]:
            bearish_count += 1
            avgr = avg_ranges[k]
            if avgr and avgr > 0:
                gap2 = sma20[k] - closes[k]
                max_space2 = max(max_space2, gap2 / avgr)
                gap1 = sma200[k] - sma20[k]
                if gap1 / avgr >= SPACE1_ATR_MULT:
                    space1_ok_seen = True
    ok = bearish_count >= min_bars
    space2_ok = max_space2 >= SPACE2_ATR_MULT
    dual_space = ok and space1_ok_seen and space2_ok
    single_space = ok and (space1_ok_seen or space2_ok) and not dual_space
    return ok, dual_space, single_space


def detect_long_setups(day_df, prior_context_df):
    """Scans one day's bars for LONG (bottom-picking) 20MA-halt setups.
    prior_context_df supplies enough bars before today for decline/space
    lookback (continuous indicator series, not reset per day)."""
    full = pd.concat([prior_context_df, day_df]) if len(prior_context_df) else day_df
    full = full[~full.index.duplicated(keep="last")].sort_index()
    opens = full["open"].tolist()
    highs = full["high"].tolist()
    lows = full["low"].tolist()
    closes = full["close"].tolist()
    sma20 = full["sma20"].tolist()
    sma200 = full["sma200"].tolist()
    avg_ranges = full["avg_range"].tolist()
    idx = full.index
    n = len(full)

    today_start = len(prior_context_df)
    signals = []

    for i in range(max(today_start, DECLINE_LOOKBACK_BARS + 5), n):
        if idx[i].date() != day_df.index[0].date() if len(day_df) else True:
            pass
        if idx[i].date() not in set(day_df.index.date):
            continue
        if pd.isna(sma20[i]) or pd.isna(sma200[i]) or pd.isna(sma20[i - 1]):
            continue
        # sear-through: first bar closing above the buffered 20MA line, having
        # been at/under that buffered line the bar before (a real close-through,
        # not merely a lagging-MA artifact of an already-completed crossover)
        if not (closes[i] > sma20[i] + SEAR_BUFFER_PTS and closes[i - 1] <= sma20[i - 1] + SEAR_BUFFER_PTS):
            continue
        ok, dual_space, single_space = check_elongated_decline(
            closes, sma20, sma200, avg_ranges, i, DECLINE_LOOKBACK_BARS, MIN_DECLINE_BARS)
        if not ok:
            continue

        # Single continuous forward scan covering halt-confirmation AND the
        # entry trigger together. pullback_low keeps extending for as long as
        # the setup remains alive (right up to the entry bar), so the stop
        # level used later always reflects the true lowest point reached --
        # not a stale reading frozen at the moment the halt first appeared.
        # 2 CONSECUTIVE closes below the 20MA invalidates the setup at any
        # point, even after an initial halt looked confirmed.
        below_streak = 0
        touched = False
        halt_confirmed = False
        halt_bar = None
        pullback_low = None
        marker_high = None
        entry_bar = None
        for j in range(i + 1, min(i + 1 + 2 * HALT_WINDOW_BARS, n)):
            if not halt_confirmed and (j - i) > HALT_WINDOW_BARS:
                break  # never got a halt within the allowed window -- abandon
            if lows[j] <= sma20[j] + TOUCH_TOLERANCE_PTS:
                touched = True
            if touched:
                pullback_low = lows[j] if pullback_low is None else min(pullback_low, lows[j])
            if closes[j] < sma20[j]:
                below_streak += 1
                if below_streak >= TWO_BAR_RULE:
                    halt_bar, halt_confirmed, entry_bar = None, False, None
                    break
            else:
                below_streak = 0
                if touched and not halt_confirmed:
                    halt_confirmed = True
                    halt_bar = j
                    marker_j = find_marker_high(opens, closes, highs, halt_bar)
                    if marker_j < 0:
                        halt_confirmed = False
                        break
                    marker_high = highs[marker_j]
            if halt_confirmed and j >= halt_bar and is_green(opens[j], closes[j]) and closes[j] > marker_high:
                entry_bar = j
                break
        if entry_bar is None or pullback_low is None:
            continue
        if idx[entry_bar].date() not in set(day_df.index.date):
            continue

        signals.append({
            "bar": entry_bar, "side": "LONG", "pullback_low": pullback_low,
            "target": sma200[entry_bar], "dual_space": dual_space, "single_space": single_space,
        })

    seen = set()
    deduped = []
    for s in signals:
        if s["bar"] in seen:
            continue
        seen.add(s["bar"])
        deduped.append(s)
    return deduped, full, today_start


def detect_short_setups(day_df, prior_context_df):
    """Mirror of detect_long_setups for SHORT (top-picking)."""
    full = pd.concat([prior_context_df, day_df]) if len(prior_context_df) else day_df
    full = full[~full.index.duplicated(keep="last")].sort_index()
    opens = full["open"].tolist()
    highs = full["high"].tolist()
    lows = full["low"].tolist()
    closes = full["close"].tolist()
    sma20 = full["sma20"].tolist()
    sma200 = full["sma200"].tolist()
    avg_ranges = full["avg_range"].tolist()
    idx = full.index
    n = len(full)

    today_start = len(prior_context_df)
    signals = []

    for i in range(max(today_start, DECLINE_LOOKBACK_BARS + 5), n):
        if idx[i].date() not in set(day_df.index.date):
            continue
        if pd.isna(sma20[i]) or pd.isna(sma200[i]) or pd.isna(sma20[i - 1]):
            continue
        if not (closes[i] < sma20[i] - SEAR_BUFFER_PTS and closes[i - 1] >= sma20[i - 1] - SEAR_BUFFER_PTS):
            continue

        start = max(0, i - DECLINE_LOOKBACK_BARS)
        bullish_count = 0
        max_space2 = 0.0
        space1_ok_seen = False
        for k in range(start, i):
            if pd.isna(sma20[k]) or pd.isna(sma200[k]) or pd.isna(avg_ranges[k]):
                continue
            if closes[k] > sma20[k] and sma20[k] > sma200[k]:
                bullish_count += 1
                avgr = avg_ranges[k]
                if avgr and avgr > 0:
                    gap2 = closes[k] - sma20[k]
                    max_space2 = max(max_space2, gap2 / avgr)
                    gap1 = sma20[k] - sma200[k]
                    if gap1 / avgr >= SPACE1_ATR_MULT:
                        space1_ok_seen = True
        ok = bullish_count >= MIN_DECLINE_BARS
        space2_ok = max_space2 >= SPACE2_ATR_MULT
        dual_space = ok and space1_ok_seen and space2_ok
        single_space = ok and (space1_ok_seen or space2_ok) and not dual_space
        if not ok:
            continue

        above_streak = 0
        touched = False
        halt_confirmed = False
        halt_bar = None
        pullback_high = None
        marker_low = None
        entry_bar = None
        for j in range(i + 1, min(i + 1 + 2 * HALT_WINDOW_BARS, n)):
            if not halt_confirmed and (j - i) > HALT_WINDOW_BARS:
                break
            if highs[j] >= sma20[j] - TOUCH_TOLERANCE_PTS:
                touched = True
            if touched:
                pullback_high = highs[j] if pullback_high is None else max(pullback_high, highs[j])
            if closes[j] > sma20[j]:
                above_streak += 1
                if above_streak >= TWO_BAR_RULE:
                    halt_bar, halt_confirmed, entry_bar = None, False, None
                    break
            else:
                above_streak = 0
                if touched and not halt_confirmed:
                    halt_confirmed = True
                    halt_bar = j
                    j2 = halt_bar - 1
                    while j2 >= 0 and not is_green(opens[j2], closes[j2]):
                        j2 -= 1
                    if j2 < 0:
                        halt_confirmed = False
                        break
                    marker_low = lows[j2]
            if halt_confirmed and j >= halt_bar and is_red(opens[j], closes[j]) and closes[j] < marker_low:
                entry_bar = j
                break
        if entry_bar is None or pullback_high is None:
            continue
        if idx[entry_bar].date() not in set(day_df.index.date):
            continue

        signals.append({
            "bar": entry_bar, "side": "SHORT", "pullback_high": pullback_high,
            "target": sma200[entry_bar], "dual_space": dual_space, "single_space": single_space,
        })

    seen = set()
    deduped = []
    for s in signals:
        if s["bar"] in seen:
            continue
        seen.add(s["bar"])
        deduped.append(s)
    return deduped, full, today_start


def simulate_trade(full_df, sig, day):
    closes = full_df["close"].tolist()
    idx = full_df.index
    n = len(full_df)
    j0 = sig["bar"]
    side = sig["side"]
    entry_price = closes[j0]
    lot = lot_size_for_date(day)

    if side == "LONG":
        stop = sig["pullback_low"] - STOP_BUFFER_PTS
        target = sig["target"]
        option_side = "CE"
    else:
        stop = sig["pullback_high"] + STOP_BUFFER_PTS
        target = sig["target"]
        option_side = "PE"

    below_streak = 0
    exit_j, exit_price, reason = n - 1, closes[-1], "EOD"
    for j in range(j0 + 1, n):
        t = idx[j]
        if t.date() != idx[j0].date():
            exit_j, exit_price, reason = j - 1, closes[j - 1], "EOD"
            break
        c = closes[j]
        if side == "LONG":
            if c < stop:
                below_streak += 1
                if below_streak >= TWO_BAR_RULE:
                    exit_j, exit_price, reason = j, c, "STOP"
                    break
            else:
                below_streak = 0
            if c >= target:
                exit_j, exit_price, reason = j, target, "TARGET"
                break
        else:
            if c > stop:
                below_streak += 1
                if below_streak >= TWO_BAR_RULE:
                    exit_j, exit_price, reason = j, c, "STOP"
                    break
            else:
                below_streak = 0
            if c <= target:
                exit_j, exit_price, reason = j, target, "TARGET"
                break
        if t.time() >= EOD_SQUAREOFF:
            exit_j, exit_price, reason = j, c, "EOD"
            break
    else:
        pass

    if side == "LONG":
        points = exit_price - entry_price
        strike = itm_call_strike(entry_price)
    else:
        points = entry_price - exit_price
        strike = itm_put_strike(entry_price)

    rupee_pnl = points * lot - BROKERAGE_PER_TRADE
    return {
        "date": day, "side": option_side, "strike": strike, "lot_size": lot,
        "entry_time": idx[j0], "exit_time": idx[exit_j], "index_entry": entry_price,
        "index_exit": exit_price, "exit_reason": reason, "points_gained": points,
        "rupee_pnl": rupee_pnl, "dual_space": sig["dual_space"], "single_space": sig["single_space"],
    }


def run_self_checks():
    print("Running self-checks (synthetic data)...")
    # bars: 0 green(100->102), 1 red(103->98), 2 green(99->101), 3 red(102->100), halt at 4
    opens = [100, 103, 99, 102, 101]
    closes = [102, 98, 101, 100, 103]
    highs = [103, 104, 102, 103, 104]
    j = find_marker_high(opens, closes, highs, 4)
    assert j == 3, f"expected nearest red bar at idx 3, got {j}"
    print("  find_marker_high basic check ok")

    closes = [110, 108, 106, 104, 102, 100, 100.5, 101, 103, 108]
    sma20 = [112]*5 + [104, 103.5, 103, 103, 103]
    sma200 = [115]*10
    avg_ranges = [3.0]*10
    ok, dual, single = check_elongated_decline(closes, sma20, sma200, avg_ranges, 5, 5, 3)
    assert ok is True
    print("  check_elongated_decline basic check ok")

    # End-to-end synthetic day: elongated decline -> sharp sear-through ->
    # single-bar halt at the 20MA -> entry -> rally through to TARGET.
    # Verifies the full LONG and SHORT pipelines, the stop level landing on
    # the correct side of entry, and that a broken (2-consecutive-closes-
    # below-MA) pullback correctly produces NO signal at all.
    ctx_rng = pd.date_range("2026-01-04 09:15", periods=210, freq="2min")
    rs = np.random.RandomState(0)
    ctx_close = np.full(len(ctx_rng), 200.0) + rs.normal(0, 0.05, len(ctx_rng))
    ctx_open = np.roll(ctx_close, 1); ctx_open[0] = ctx_close[0]
    ctx_df = pd.DataFrame({
        "open": ctx_open, "high": np.maximum(ctx_open, ctx_close) + 0.2,
        "low": np.minimum(ctx_open, ctx_close) - 0.2, "close": ctx_close, "volume": 100,
    }, index=ctx_rng)
    rng = pd.date_range("2026-01-05 09:15", periods=150, freq="2min")
    n_bars = len(rng)

    def _build_day(direction, dip_depth=4.0):
        sign = 1 if direction == "LONG" else -1
        close = np.zeros(n_bars)
        close[0] = 200
        for i in range(1, 60):
            close[i] = close[i - 1] - sign * 0.65
        base = close[59]
        for i in range(60, 63):
            close[i] = base + sign * (i - 59) * 3.0
        sear = close[62]
        close[63] = sear - sign * dip_depth
        for i in range(64, n_bars):
            close[i] = close[63] + sign * (i - 63) * 1.2
        open_ = np.roll(close, 1); open_[0] = close[0]
        wick = 0.6 if direction == "LONG" else 0.3
        wick_hi = 0.3 if direction == "LONG" else 0.6
        return pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) + wick_hi,
            "low": np.minimum(open_, close) - wick, "close": close, "volume": 100,
        }, index=rng)

    for direction, detect_fn in [("LONG", detect_long_setups), ("SHORT", detect_short_setups)]:
        full_raw = pd.concat([ctx_df, _build_day(direction)])
        full_ind = compute_indicators(full_raw)
        ctx_ind = full_ind[full_ind.index.date == ctx_rng[0].date()]
        day_ind = full_ind[full_ind.index.date == rng[0].date()]
        sigs, full_df, _ = detect_fn(day_ind, ctx_ind)
        assert len(sigs) >= 1, f"{direction} synthetic halt setup should have produced a signal"
        entry_price = full_df["close"].tolist()[sigs[0]["bar"]]
        if direction == "LONG":
            assert sigs[0]["pullback_low"] < entry_price, "LONG stop basis must sit below entry"
        else:
            assert sigs[0]["pullback_high"] > entry_price, "SHORT stop basis must sit above entry"
        t = simulate_trade(full_df, sigs[0], rng[0].date())
        assert t["points_gained"] > 0, f"{direction} synthetic winner scenario should be a winner"
        assert t["exit_reason"] == "TARGET"
    print("  end-to-end LONG/SHORT halt+entry+target synthetic check ok")

    # Broken pullback (2 consecutive closes below the 20MA) must invalidate --
    # no trade, not a bad trade.
    close = np.zeros(n_bars)
    close[0] = 200
    for i in range(1, 60):
        close[i] = close[i - 1] - 0.65
    base = close[59]
    for i in range(60, 63):
        close[i] = base + (i - 59) * 3.0
    sear = close[62]
    for i in range(63, 70):
        close[i] = sear - (i - 62) * 3.5
    for i in range(70, n_bars):
        close[i] = close[69]
    open_ = np.roll(close, 1); open_[0] = close[0]
    broken_day = pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3, "close": close, "volume": 100,
    }, index=rng)
    full_raw = pd.concat([ctx_df, broken_day])
    full_ind = compute_indicators(full_raw)
    ctx_ind = full_ind[full_ind.index.date == ctx_rng[0].date()]
    day_ind = full_ind[full_ind.index.date == rng[0].date()]
    inval_sigs, _, _ = detect_long_setups(day_ind, ctx_ind)
    assert len(inval_sigs) == 0, "2-consecutive-closes-below-20MA must invalidate the setup"
    print("  broken-pullback invalidation check ok")

    print("All self-checks passed.\n")


def stats_block(g, label):
    if g.empty:
        print(f"{label}: no trades")
        return
    wins = g[g["points_gained"] > 0]
    losses = g[g["points_gained"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["points_gained"].sum() / abs(losses["points_gained"].sum())
          if losses["points_gained"].sum() != 0 else float("inf"))
    print(f"{label}: n={len(g)}  win%={100*win_rate:.1f}  net_pts={g['points_gained'].sum():.2f}  "
          f"PF={pf:.2f}  net_Rs={g['rupee_pnl'].sum():,.2f}")


def run_direction(df2m, days_covered, detect_fn, label):
    all_trades = []
    for idx_day, day in enumerate(days_covered):
        day_df = df2m[df2m.index.date == day]
        if day_df.empty:
            continue
        context_start = max(0, idx_day - 3)
        context_days = days_covered[context_start:idx_day]
        prior_context_df = df2m[np.isin(df2m.index.date, context_days)] if context_days else df2m.iloc[0:0]
        sigs, full_df, today_start = detect_fn(day_df, prior_context_df)
        open_bar = -1
        for sig in sigs:
            if sig["bar"] <= open_bar:
                continue
            t = simulate_trade(full_df, sig, day)
            all_trades.append(t)
            open_bar = full_df.index.get_loc(t["exit_time"])
    return pd.DataFrame(all_trades)


def report(tdf, label):
    print(f"\n=== {label} ===")
    if tdf.empty:
        print("No trades generated.")
        return
    tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = STARTING_CAPITAL + tdf["rupee_pnl"].cumsum()
    tdf.to_csv(f"twenty_ma_halt_{label.lower()}_trades.csv", index=False)

    stats_block(tdf, "Overall")
    stats_block(tdf[tdf["dual_space"]], "Dual-space only")
    stats_block(tdf[tdf["single_space"]], "Single-space only")
    print()
    for reason, g in tdf.groupby("exit_reason"):
        stats_block(g, reason)

    equity = tdf["running_equity"]
    running_peak = equity.cummax()
    max_dd = (equity - running_peak).min()
    print(f"\nMax drawdown: Rs {max_dd:,.2f}   {'WIPES OUT' if (equity<=0).any() else 'never wipes out'}")
    print(f"Start Rs {STARTING_CAPITAL:,} -> End Rs {equity.iloc[-1]:,.2f}   "
          f"Return: {100*(equity.iloc[-1]/STARTING_CAPITAL-1):.1f}%")

    mid = len(tdf)//2
    h1 = tdf["rupee_pnl"].iloc[:mid].sum()
    h2 = tdf["rupee_pnl"].iloc[mid:].sum()
    print(f"Stability: first half n={mid} Rs{h1:,.2f}  |  second half n={len(tdf)-mid} Rs{h2:,.2f}")
    print(f"Trade log saved: twenty_ma_halt_{label.lower()}_trades.csv ({len(tdf)} rows)")


def main():
    run_self_checks()
    print("=== 20 MA Halt Bottom/Top-Picking -- Nifty ITM Options (single-halt, both directions) ===\n")
    print("No Greeks/delta modeling -- raw index points x lot size. Rs 40/trade brokerage.\n")

    print("Fetching Nifty index 1-min data (2 years) and resampling to 2-min bars...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    df2m = resample_2min_per_day(df1m)
    df2m = compute_indicators(df2m)
    days_covered = sorted(set(df2m.index.date))
    print(f"Got {len(df2m)} 2-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    print("Scanning for LONG (bottom-picking) setups...")
    long_trades = run_direction(df2m, days_covered, detect_long_setups, "LONG")
    print(f"  {len(long_trades)} trades.\n")

    print("Scanning for SHORT (top-picking) setups...")
    short_trades = run_direction(df2m, days_covered, detect_short_setups, "SHORT")
    print(f"  {len(short_trades)} trades.\n")

    report(long_trades, "LONG")
    report(short_trades, "SHORT")

    print("\nNOTE: this is the single-halt version only (Section 6's two-halt add-on deferred, per "
          "the doc's own recommendation to validate the simple version first). Dual-space vs "
          "single-space breakdown is reported above so that filter can be evaluated with real numbers "
          "rather than assumed.")


if __name__ == "__main__":
    main()
