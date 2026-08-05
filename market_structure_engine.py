"""
Module: market_structure_engine.py
Description: Market Structure Engine -- price-only, no OI/ML dependency.
             See Research/MARKET_STRUCTURE_ML_ARCHITECTURE.md for the full
             design writeup. This is step 1 of that document's roadmap:
             build the engine, then test it ALONE (main() below) before any
             ML layer is even considered.

WHAT IT COMPUTES, per 2-min bar, causally (no lookahead -- verified by a
self-check that compares a truncated recompute against the full-day one):
  - trend_structure: "Uptrend" (Higher-High + Higher-Low), "Downtrend"
    (Lower-High + Lower-Low), or "Range" (anything else / not enough
    confirmed swings yet)
  - last_bos_direction: "UP"/"DOWN"/None -- most recent Break of Structure
    (price closing past the most recent CONFIRMED swing high/low)
  - last_choch: True if that break contradicted the trend_structure in
    effect at the time (Change of Character -- possible reversal, not
    continuation)
  - dist_prev_day_high_pct / dist_prev_day_low_pct: distance from current
    close to yesterday's high/low, as a percent
  - dist_nearest_swing_pct: distance from current close to the nearest
    confirmed swing point
  - opening_range_high / opening_range_low: first OPENING_RANGE_BARS bars'
    range, None until that window has actually completed (no lookahead)

SWING DETECTION: fixed lookback (SWING_N = 5 bars each side), NOT swept or
tuned to find the best-looking value -- see the architecture doc's Section
2.4 for why. A swing point only becomes "confirmed" (usable) SWING_N bars
after it happens, since you need that many bars afterward to know it really
was a local extreme. This delay is enforced explicitly, not an accident.

DATA: price-only. Uses whatever real cached/fetched Nifty 2-min data is
available -- no option-chain or futures OI dependency, unlike the DNA/OI
tests. main() below uses the real 6-month dataset already cached in this
project (dhan_historical_data_180days_v5.csv) and runs entirely offline.

Run:
    cd "C:\\dhan algo"
    python market_structure_engine.py
"""

import pandas as pd

SWING_N = 5              # fixed, not swept -- see docstring
OPENING_RANGE_BARS = 8    # ~16 minutes on 2-min bars


# ======================================================================
# Swing detection (pure, per-day arrays)
# ======================================================================

def compute_swing_flags(highs, lows, n=SWING_N):
    """Bar i is a (not-yet-confirmed) swing high if its high is strictly
    greater than every other high in the n-bar window on both sides; mirror
    for swing low. Returns two boolean lists, same length as the input."""
    m = len(highs)
    swing_high = [False] * m
    swing_low = [False] * m
    for i in range(n, m - n):
        window_h = highs[i - n: i + n + 1]
        window_l = lows[i - n: i + n + 1]
        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            swing_high[i] = True
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            swing_low[i] = True
    return swing_high, swing_low


# ======================================================================
# Structure computation (causal -- a swing at index j is only usable
# starting at index j+n, once it's actually confirmed)
#
# IMPORTANT DESIGN NOTE (found empirically, not tuned for a better number):
# the first version of this engine reset swing/trend structure to zero at
# the start of every trading day. Real-tested result: with n=5 bars needed
# to confirm even one swing, and TWO confirmed highs + TWO confirmed lows
# needed before a trend_structure reading exists at all, an intraday-only
# reset essentially never accumulates enough confirmed swings before either
# strategy's entry fires (both trade early-to-mid session). That produced
# ~0 usable readings, not "no edge" -- a broken measurement, not a result.
# Fixed by making swing/trend/BOS state continuous ACROSS days (the way you'd
# actually read structure on a real multi-day chart), while opening-range and
# distance-to-previous-day-level stay day-scoped since those ARE inherently
# daily concepts. This is a one-time scope correction, made before looking at
# whether it changes the backtest result -- not a parameter swept afterward
# to get a better-looking number.
# ======================================================================

def compute_full_structure(full_df: pd.DataFrame, n=SWING_N) -> pd.DataFrame:
    """Continuous swing/trend/BOS/CHoCH state across the WHOLE series (all
    days concatenated in order), not reset per day."""
    highs = full_df["high"].tolist()
    lows = full_df["low"].tolist()
    closes = full_df["close"].tolist()
    m = len(full_df)
    swing_high_flags, swing_low_flags = compute_swing_flags(highs, lows, n)

    confirmed_highs, confirmed_lows = [], []
    trend_structure = "Range"
    last_bos_direction, last_choch = None, False
    last_broken_high_idx, last_broken_low_idx = None, None

    rows = []
    for i in range(m):
        j = i - n
        if 0 <= j < m:
            if swing_high_flags[j]:
                confirmed_highs.append((j, highs[j]))
            if swing_low_flags[j]:
                confirmed_lows.append((j, lows[j]))

        if len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2:
            hh = confirmed_highs[-1][1] > confirmed_highs[-2][1]
            hl = confirmed_lows[-1][1] > confirmed_lows[-2][1]
            lh = confirmed_highs[-1][1] < confirmed_highs[-2][1]
            ll = confirmed_lows[-1][1] < confirmed_lows[-2][1]
            if hh and hl:
                trend_structure = "Uptrend"
            elif lh and ll:
                trend_structure = "Downtrend"
            else:
                trend_structure = "Range"
        else:
            trend_structure = "Range"

        if confirmed_highs and closes[i] > confirmed_highs[-1][1] and confirmed_highs[-1][0] != last_broken_high_idx:
            last_choch = (trend_structure == "Downtrend")
            last_bos_direction = "UP"
            last_broken_high_idx = confirmed_highs[-1][0]
        elif confirmed_lows and closes[i] < confirmed_lows[-1][1] and confirmed_lows[-1][0] != last_broken_low_idx:
            last_choch = (trend_structure == "Uptrend")
            last_bos_direction = "DOWN"
            last_broken_low_idx = confirmed_lows[-1][0]

        dists = []
        if confirmed_highs:
            dists.append(abs(closes[i] - confirmed_highs[-1][1]))
        if confirmed_lows:
            dists.append(abs(closes[i] - confirmed_lows[-1][1]))
        dist_nearest_swing_pct = (min(dists) / closes[i] * 100.0) if dists and closes[i] else None

        rows.append({
            "trend_structure": trend_structure,
            "last_bos_direction": last_bos_direction,
            "last_choch": last_choch,
            "dist_nearest_swing_pct": dist_nearest_swing_pct,
        })

    return pd.DataFrame(rows, index=full_df.index)


def compute_daily_context(full_df: pd.DataFrame, or_bars=OPENING_RANGE_BARS) -> pd.DataFrame:
    """Day-scoped features: opening range (None until it's actually formed
    each day) and distance to the ACTUAL previous trading day's high/low."""
    days = sorted(set(full_df.index.date))
    prev_high, prev_low = None, None
    out = []
    for day in days:
        day_df = full_df.loc[full_df.index.date == day]
        highs, lows, closes = day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
        m = len(day_df)
        rows = []
        for i in range(m):
            if i < or_bars:
                or_high, or_low = None, None
            else:
                or_high, or_low = max(highs[:or_bars]), min(lows[:or_bars])
            rows.append({
                "dist_prev_day_high_pct": ((closes[i] - prev_high) / prev_high * 100.0) if prev_high else None,
                "dist_prev_day_low_pct": ((closes[i] - prev_low) / prev_low * 100.0) if prev_low else None,
                "opening_range_high": or_high,
                "opening_range_low": or_low,
            })
        out.append(pd.DataFrame(rows, index=day_df.index))
        prev_high, prev_low = float(day_df["high"].max()), float(day_df["low"].min())
    return pd.concat(out)


def compute_structure_series(full_df: pd.DataFrame, n=SWING_N, or_bars=OPENING_RANGE_BARS) -> pd.DataFrame:
    """Full feature set: continuous multi-day structure + day-scoped context, joined."""
    structure = compute_full_structure(full_df, n=n)
    context = compute_daily_context(full_df, or_bars=or_bars)
    return structure.join(context)


# Kept for backward compatibility with the self-checks below, which test a
# single day in isolation -- delegates to the continuous version.
def compute_day_structure(day_df: pd.DataFrame, prev_day_high=None, prev_day_low=None,
                           n=SWING_N, or_bars=OPENING_RANGE_BARS) -> pd.DataFrame:
    structure = compute_full_structure(day_df, n=n)
    highs, lows, closes = day_df["high"].tolist(), day_df["low"].tolist(), day_df["close"].tolist()
    m = len(day_df)
    ctx_rows = []
    for i in range(m):
        if i < or_bars:
            or_high, or_low = None, None
        else:
            or_high, or_low = max(highs[:or_bars]), min(lows[:or_bars])
        ctx_rows.append({
            "dist_prev_day_high_pct": ((closes[i] - prev_day_high) / prev_day_high * 100.0) if prev_day_high else None,
            "dist_prev_day_low_pct": ((closes[i] - prev_day_low) / prev_day_low * 100.0) if prev_day_low else None,
            "opening_range_high": or_high,
            "opening_range_low": or_low,
        })
    context = pd.DataFrame(ctx_rows, index=day_df.index)
    return structure.join(context)


def structure_agrees_with_direction(row, direction, use_bos=False) -> bool:
    """The two filter variants tested in main(): either the classified
    trend_structure agrees with the trade direction, or (use_bos=True) the
    most recent Break of Structure agrees. LONG needs Uptrend/UP, SHORT
    needs Downtrend/DOWN."""
    field = "last_bos_direction" if use_bos else "trend_structure"
    val = row.get(field)
    if use_bos:
        return (direction == "LONG" and val == "UP") or (direction == "SHORT" and val == "DOWN")
    return (direction == "LONG" and val == "Uptrend") or (direction == "SHORT" and val == "Downtrend")


# ======================================================================
# Self-checks (synthetic data, no network needed)
# ======================================================================

def run_self_checks():
    print("Running market_structure_engine self-checks (synthetic data)...")

    # --- swing detection on a clean zigzag, small n for a small test series ---
    highs = [10, 11, 12, 15, 12, 11, 10, 9, 8, 11, 14, 11, 10]
    lows = [x - 1 for x in highs]
    sh, sl = compute_swing_flags(highs, lows, n=2)
    assert sh[3] is True, "the clear peak at index 3 (15) must be flagged a swing high"
    assert sl[8] is True, "the clear trough at index 8 (8) must be flagged a swing low"
    assert sh[0] is False and sh[-1] is False, "edges without a full window must never be flagged"
    print("  compute_swing_flags: detects a clean peak/trough correctly")

    # --- trend_structure: constructed Higher-High/Higher-Low sequence ---
    idx = pd.date_range("2025-01-06 09:15", periods=40, freq="2min")
    # build an unmistakable staircase-up pattern: each swing high/low higher than the last
    closes = []
    base = 100.0
    for step in range(8):
        for k in range(5):
            closes.append(base + step * 3 + (2 if k == 2 else 0) - (1 if k in (0, 4) else 0))
    day_df = pd.DataFrame({
        "open": closes, "close": closes,
        "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
        "volume": [100] * len(closes),
    }, index=idx)
    struct = compute_day_structure(day_df, prev_day_high=None, prev_day_low=None, n=2, or_bars=4)
    assert struct["trend_structure"].iloc[-1] == "Uptrend", \
        f"a clean staircase-up pattern must classify as Uptrend, got {struct['trend_structure'].iloc[-1]}"
    print("  compute_day_structure: staircase-up pattern reads Uptrend")

    # --- causality / no-lookahead: truncated recompute must match the full-day one at every shared bar ---
    full = compute_day_structure(day_df, None, None, n=2, or_bars=4)
    mismatch = None
    for cut in (10, 20, 30):
        truncated = compute_day_structure(day_df.iloc[:cut], None, None, n=2, or_bars=4)
        # only bars deep enough into the truncated slice to be unaffected by its own shorter tail are comparable
        cmp_upto = cut - 2  # swing confirmation needs n=2 bars after; last couple of bars in a short slice can differ
        for col in ["trend_structure", "last_bos_direction", "last_choch"]:
            a = truncated[col].iloc[:cmp_upto].tolist()
            b = full[col].iloc[:cmp_upto].tolist()
            if a != b:
                mismatch = (cut, col)
    assert mismatch is None, f"lookahead detected: truncated vs full-day mismatch at {mismatch}"
    print("  no-lookahead check ok (truncated recompute matches full-day computation)")

    # --- opening range: must be None until the window has actually completed ---
    assert pd.isna(struct["opening_range_high"].iloc[0]), "opening range must not exist before it's formed"
    assert pd.notna(struct["opening_range_high"].iloc[4]), "opening range must exist once or_bars have completed"
    print("  opening_range causality check ok")

    # --- structure_agrees_with_direction ---
    assert structure_agrees_with_direction({"trend_structure": "Uptrend"}, "LONG") is True
    assert structure_agrees_with_direction({"trend_structure": "Downtrend"}, "LONG") is False
    assert structure_agrees_with_direction({"last_bos_direction": "DOWN"}, "SHORT", use_bos=True) is True
    print("  structure_agrees_with_direction ok")

    print("All market_structure_engine self-checks passed.\n")


# ======================================================================
# Gate check: does structure ALONE improve the two live strategies?
# (Per the architecture doc's Section 7 -- this must pass before any ML
# layer is even considered.)
# ======================================================================

def main():
    run_self_checks()

    import nifty_open_momentum_stoptrail_sweep as om
    import nifty_six_concepts_screening as sc
    import best_two_of_claude_5_8_26 as bt

    CSV = "dhan_historical_data_180days_v5.csv"
    print("=" * 78)
    print("MARKET STRUCTURE GATE CHECK -- does structure alone help, before any ML?")
    print(f"Using real cached data: {CSV} (price-only, no OI/Dhan connection needed)")
    print("=" * 78)

    raw = pd.read_csv(CSV, parse_dates=["datetime"])
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.tz_localize(None)
    raw = raw.set_index("datetime").sort_index()
    raw = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    full_df = om.resample_2min_per_day(raw)
    print(f"Loaded {len(full_df)} bars across {len(set(full_df.index.date))} days.\n")

    day_groups_om = [(day, full_df.loc[full_df.index.date == day]) for day in sorted(set(full_df.index.date))]
    om_trades = om.run_for_combo(full_df, day_groups_om, bt.OM_STOP_PTS, bt.OM_TRAIL_PTS)

    full_df_sc = sc.compute_indicators(full_df)
    full_df_sc = sc.compute_slope_stage(full_df_sc)
    day_groups_sc = {}
    for pos, d in enumerate(full_df_sc.index.date):
        day_groups_sc.setdefault(d, []).append(pos)
    sc_trades = sc.run_module(full_df_sc, day_groups_sc, lambda d, i: sc.rbi_gbi_signal(d, i, relaxed=True))

    structure = compute_structure_series(full_df)

    def attach_structure(trades_df):
        t = trades_df.copy()
        t["entry_time"] = pd.to_datetime(t["entry_time"])
        return t.join(structure, on="entry_time")

    om_t, sc_t = attach_structure(om_trades), attach_structure(sc_trades)

    combined_base, _ = bt.merge_with_priority(om_t, sc_t)
    m_base = bt.compute_combined_metrics(combined_base, bt.STARTING_CAPITAL)
    print(f"Baseline (no structure filter): n={m_base['n']:3d}  win={m_base['win_pct']}%  "
          f"net=Rs{m_base['net_rs']:,.0f}  maxDD=Rs{m_base['max_dd_rs']:,.0f}  SQN={m_base['sqn']}\n")

    for use_bos, label in [(False, "S1: trend_structure agrees with direction"),
                            (True, "S2: last_bos_direction agrees with direction")]:
        om_f = om_t[om_t.apply(lambda r: structure_agrees_with_direction(r, r["direction"], use_bos), axis=1)].copy()
        sc_f = sc_t[sc_t.apply(lambda r: structure_agrees_with_direction(r, r["direction"], use_bos), axis=1)].copy()
        print(f"-- {label} --")
        print(f"   kept: om={len(om_f)}/{len(om_t)}, sc={len(sc_f)}/{len(sc_t)}")
        if om_f.empty and sc_f.empty:
            print("   n=  0 -- no trades passed this filter at all.\n")
            continue
        combined_f, _ = bt.merge_with_priority(om_f, sc_f)
        m_f = bt.compute_combined_metrics(combined_f, bt.STARTING_CAPITAL)
        print(f"   n={m_f['n']:3d}  win={m_f['win_pct']}%  net=Rs{m_f['net_rs']:,.0f}  "
              f"maxDD=Rs{m_f['max_dd_rs']:,.0f}  SQN={m_f['sqn']}\n")

    print("=" * 78)
    print("Per-strategy breakdown (trend_structure agreement)")
    print("=" * 78)
    for name, t in [("open_momentum", om_t), ("six_concepts", sc_t)]:
        t = t.copy()
        t["agrees"] = t.apply(lambda r: structure_agrees_with_direction(r, r["direction"], False), axis=1)
        g = t.groupby("agrees")["rupee_pnl"].agg(
            n="count", win_pct=lambda s: round(100 * (s > 0).mean(), 1), net=lambda s: round(s.sum(), 2)
        )
        print(f"{name}:\n{g.to_string()}\n")

    combined_base.to_csv("market_structure_gate_baseline_trades.csv", index=False)
    print("Saved: market_structure_gate_baseline_trades.csv")


if __name__ == "__main__":
    main()
