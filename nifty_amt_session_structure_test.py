"""
Session-structure statistical tests -- "Order Flow Scalping: Live Session
Refinements" companion doc
=================================================================
That doc layers detail on top of the core AMT model
(nifty_amt_orderflow_backtest.py): the AAA setup (Section 1), protection
levels (Section 4), and conditional scale-in sizing (Section 3) all
require real print-level/footprint order-flow data to detect faithfully
(repeated absorption at a level) -- the same limitation already flagged
in the core model's own docstring, and admitted directly in this
companion doc's own Section 13.

The loss-limit staging, win-rate/R:R trade-off, exit-discipline, and
dynamic position-sizing refinements (Sections 5, 6, 7, 10) all need REAL
TRADES from the AAA trigger to test meaningfully -- and that trigger
currently produces ZERO trades on 2 years of real Nifty data (see
nifty_amt_diagnose_zero_trades.py's gate-funnel result: the aggression-
proxy gate lets through only 0.7% of eligible bars). Fixing that gate with
a real, data-driven AGGRESSION_VOL_MULT (STEP 4b in that diagnostic) is
still pending -- those sections stay blocked until then.

BUT the companion doc's own Section 13 calls out TWO claims as "the most
specific, falsifiable additions" worth prioritizing, and both are testable
on plain OHLCV data RIGHT NOW, with zero dependency on the stalled AAA
trigger:

  A. Section 9 -- Opening Range Breakout (ORB) non-revisit stat: once
     price has clearly moved away from one side of an early opening
     range, does that side of the range tend to NOT get revisited later
     in the session? (The doc cites a specific high percentage figure
     without stating it precisely enough to reuse -- this measures the
     real number on Nifty directly instead of assuming the source's
     figure transfers.)
  B. Section 8 -- "Power Hour" / P-shape session structure: does Nifty
     show a genuine SECOND expansion phase (higher realized range) late
     in the session, after a mid-session lull, distinct from the first-
     hour expansion -- or is the "P-shape" claim specific to the source's
     own instrument/session and not present here?

This script tests both directly against 2 years of real Nifty 1-min data.
No trades, no P&L, no capital -- this is a pure market-structure
statistical test, not a strategy backtest.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "1min"

ORB_MINUTES = 15                 # doc's stated range: "first 15-30 minutes"
ORB_CONFIRM_MARGIN_PTS = 10.0    # how far beyond the ORB counts as a "clear directional move away from it"

PHASE1_END = dt.time(10, 30)     # initial expansion window (09:15-10:30)
PHASE2_END = dt.time(14, 0)      # mid-session consolidation window (10:30-14:00)
PHASE3_END = dt.time(15, 20)     # "power hour" window (14:00-15:20, EOD squareoff convention)


def fetch_index_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        last_err, df = None, None
        for attempt in range(2):
            try:
                resp = dlb.dhan.intraday_minute_data(
                    security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                    instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                    from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
                )
                if not isinstance(resp, dict):
                    raise RuntimeError(f"non-dict response: {str(resp)[:300]}")
                data = resp.get("data", resp)
                if not isinstance(data, dict) or "open" not in data:
                    raise RuntimeError(f"response missing OHLC keys: {str(resp)[:300]}")
                df = pd.DataFrame({
                    "open": data["open"], "high": data["high"], "low": data["low"], "close": data["close"],
                    "volume": data["volume"], "timestamp": data["timestamp"],
                })
                df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
                df = df.set_index("time").sort_index()
                break
            except Exception as e:
                last_err = e
                if attempt == 0:
                    import time as _t
                    _t.sleep(2)
        if df is None:
            raise RuntimeError(f"fetch failed for chunk {cur} to {chunk_end}: {last_err}")
        chunks.append(df)
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def resample_1min_per_day(df):
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


# =============================================================================
# A. Opening Range Breakout non-revisit test (companion doc Section 9)
# =============================================================================
def opening_range(day_df, minutes=ORB_MINUTES):
    day_start = day_df.index[0]
    cutoff = (day_start + dt.timedelta(minutes=minutes)).time()
    orb_window = day_df[day_df.index.time < cutoff]
    if orb_window.empty:
        return None
    return {"high": orb_window["high"].max(), "low": orb_window["low"].min(), "cutoff": cutoff}


def test_orb_revisit_for_day(day_df, orb, margin=ORB_CONFIRM_MARGIN_PTS):
    """Returns a list of 0-2 result dicts: {'side': 'LOW'|'HIGH', 'revisited': bool}.
    LOW-side test only runs if price later clearly confirmed a move AWAY from the
    low (closed beyond orb_high+margin); HIGH-side test mirrors this for the low."""
    after = day_df[day_df.index.time >= orb["cutoff"]]
    if after.empty:
        return []
    closes = after["close"].to_numpy()
    lows = after["low"].to_numpy()
    highs = after["high"].to_numpy()
    results = []

    up_confirm = np.where(closes > orb["high"] + margin)[0]
    if len(up_confirm) > 0:
        i0 = up_confirm[0]
        revisited = bool((lows[i0 + 1:] <= orb["low"]).any()) if i0 + 1 < len(lows) else False
        results.append({"side": "LOW", "revisited": revisited})

    down_confirm = np.where(closes < orb["low"] - margin)[0]
    if len(down_confirm) > 0:
        i0 = down_confirm[0]
        revisited = bool((highs[i0 + 1:] >= orb["high"]).any()) if i0 + 1 < len(highs) else False
        results.append({"side": "HIGH", "revisited": revisited})

    return results


# =============================================================================
# B. Power Hour / P-shape session structure test (companion doc Section 8)
# =============================================================================
def phase_stats(day_df):
    p1 = day_df[day_df.index.time < PHASE1_END]
    p2 = day_df[(day_df.index.time >= PHASE1_END) & (day_df.index.time < PHASE2_END)]
    p3 = day_df[(day_df.index.time >= PHASE2_END) & (day_df.index.time < PHASE3_END)]
    if p1.empty or p2.empty or p3.empty:
        return None
    return {
        "phase1_range": p1["high"].max() - p1["low"].min(), "phase1_vol": p1["volume"].sum(),
        "phase2_range": p2["high"].max() - p2["low"].min(), "phase2_vol": p2["volume"].sum(),
        "phase3_range": p3["high"].max() - p3["low"].min(), "phase3_vol": p3["volume"].sum(),
    }


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_significance(diffs):
    diffs = np.array(diffs, dtype=float)
    n = len(diffs)
    if n < 2 or diffs.std(ddof=1) == 0:
        return {"n": n, "mean_diff": diffs.mean() if n else 0.0, "t_stat": 0.0, "p_value": 1.0}
    mean_d, sd_d = diffs.mean(), diffs.std(ddof=1)
    t_stat = mean_d / (sd_d / math.sqrt(n))
    p_value = 2 * (1 - norm_cdf(abs(t_stat)))
    return {"n": n, "mean_diff": mean_d, "t_stat": t_stat, "p_value": p_value}


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")

    idx = pd.date_range("2025-01-06 09:15", periods=30, freq="1min")
    closes = [22000.0] * 15 + [22030.0] * 15   # ORB window flat, then a clear up-move after
    day_df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes,
        "volume": [100] * 30,
    }, index=idx)
    orb = opening_range(day_df, minutes=15)
    assert abs(orb["high"] - 22001.0) < 1e-6 and abs(orb["low"] - 21999.0) < 1e-6
    print("  opening_range hand-traced check ok")

    # up-confirm happens (closes clear orb_high+10), and the low is NEVER revisited afterward -> not revisited
    results = test_orb_revisit_for_day(day_df, orb)
    assert len(results) == 1 and results[0]["side"] == "LOW" and results[0]["revisited"] is False
    print("  test_orb_revisit_for_day non-revisit case ok")

    # same setup, but this time price dips back down through the ORB low later -> revisited=True
    closes2 = [22000.0] * 15 + [22030.0] * 5 + [21990.0] * 10   # up-confirm, then a dip back below the ORB low
    day_df2 = pd.DataFrame({
        "open": closes2, "high": [c + 1 for c in closes2], "low": [c - 1 for c in closes2], "close": closes2,
        "volume": [100] * 30,
    }, index=idx)
    results2 = test_orb_revisit_for_day(day_df2, opening_range(day_df2, minutes=15))
    assert len(results2) == 1 and results2[0]["side"] == "LOW" and results2[0]["revisited"] is True
    print("  test_orb_revisit_for_day revisit case ok")

    # no clear directional move at all -> zero eligible results
    closes3 = [22000.0] * 30
    day_df3 = pd.DataFrame({
        "open": closes3, "high": [c + 1 for c in closes3], "low": [c - 1 for c in closes3], "close": closes3,
        "volume": [100] * 30,
    }, index=idx)
    results3 = test_orb_revisit_for_day(day_df3, opening_range(day_df3, minutes=15))
    assert results3 == []
    print("  test_orb_revisit_for_day no-eligible-move check ok")

    # --- phase_stats hand-traced ---
    idx_full = pd.date_range("2025-01-06 09:15", periods=375, freq="1min")   # 09:15 to 15:29 (375 min)
    closes_full = [22000.0] * 375
    # phase1 (09:15-10:30, first 75 bars): a 50pt range
    for k in range(75):
        closes_full[k] = 22000.0 + (50 if k == 30 else 0)
    # phase2 (10:30-14:00, bars 75-284): flat, tiny range
    # (already 22000 flat)
    # phase3 (14:00-15:20, bars 285-364): a 80pt range (bigger than phase1 and phase2 -- "power hour")
    for k in range(285, 365):
        closes_full[k] = 22000.0 + (80 if k == 300 else 0)
    day_full = pd.DataFrame({
        "open": closes_full, "high": [c + 1 for c in closes_full], "low": [c - 1 for c in closes_full],
        "close": closes_full, "volume": [100] * 375,
    }, index=idx_full)
    ps = phase_stats(day_full)
    assert ps is not None
    assert ps["phase1_range"] > 40 and ps["phase3_range"] > 70 and ps["phase2_range"] < 5, f"unexpected: {ps}"
    print("  phase_stats hand-traced check ok")

    # --- paired_significance sanity: a clear, consistent positive difference -> significant ---
    sig = paired_significance([5, 6, 4, 7, 5, 6, 5, 4, 6, 5])
    assert sig["p_value"] < 0.05, f"expected a significant result, got p={sig['p_value']}"
    # a set of differences centered on zero with high variance -> not significant
    sig2 = paired_significance([5, -5, 6, -6, 4, -4, 5, -5, 6, -6])
    assert sig2["p_value"] > 0.05, f"expected a non-significant result, got p={sig2['p_value']}"
    print("  paired_significance sanity check ok")

    print("All self-checks passed.\n")


def main():
    run_self_checks()
    print("=== Session-structure tests: ORB non-revisit (Sec.9) + Power Hour/P-shape (Sec.8) ===\n")
    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    try:
        raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    except Exception as e:
        print(f"\nCould not fetch real data: {e}\n")
        print("All self-checks above still passed. This failure is about reaching the Dhan API from wherever "
              "this just ran -- if you're seeing this on your own machine with a working Dhan connection, "
              "check your network/ACCESS_TOKEN and retry.")
        return
    df1m = resample_1min_per_day(raw)
    days = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

    # ---- A. ORB non-revisit ----
    print("=== A. Opening Range Breakout (ORB) non-revisit test ===")
    print(f"ORB window: first {ORB_MINUTES} min. 'Clear move away' = closing >= {ORB_CONFIRM_MARGIN_PTS}pts "
          f"beyond the opposite ORB extreme.\n")
    low_results, high_results = [], []
    for day in days:
        day_df = df1m[df1m.index.date == day]
        orb = opening_range(day_df)
        if orb is None:
            continue
        for r in test_orb_revisit_for_day(day_df, orb):
            (low_results if r["side"] == "LOW" else high_results).append(r["revisited"])

    for label, results in (("ORB LOW (after an up-move away from it)", low_results),
                            ("ORB HIGH (after a down-move away from it)", high_results)):
        if results:
            pct_not_revisited = 100.0 * (1 - sum(results) / len(results))
            print(f"{label}: n={len(results)} eligible days, "
                  f"NOT revisited later that session: {pct_not_revisited:.1f}%")
        else:
            print(f"{label}: no eligible days found")
    combined = low_results + high_results
    if combined:
        print(f"\nCombined (both sides): n={len(combined)}, "
              f"NOT revisited: {100.0*(1 - sum(combined)/len(combined)):.1f}%")

    # ---- B. Power Hour / P-shape ----
    print("\n=== B. Power Hour / P-shape session structure test ===")
    print(f"Phase 1 (initial expansion): 09:15-{PHASE1_END}   "
          f"Phase 2 (consolidation): {PHASE1_END}-{PHASE2_END}   "
          f"Phase 3 (power hour): {PHASE2_END}-{PHASE3_END}\n")
    all_phases = []
    for day in days:
        day_df = df1m[df1m.index.date == day]
        ps = phase_stats(day_df)
        if ps:
            all_phases.append(ps)

    if all_phases:
        pdf = pd.DataFrame(all_phases)
        print(f"n={len(pdf)} days")
        print(f"Mean range   -- phase1: {pdf['phase1_range'].mean():.2f}  phase2: {pdf['phase2_range'].mean():.2f}  "
              f"phase3: {pdf['phase3_range'].mean():.2f}")
        print(f"Median range -- phase1: {pdf['phase1_range'].median():.2f}  phase2: {pdf['phase2_range'].median():.2f}  "
              f"phase3: {pdf['phase3_range'].median():.2f}")
        print(f"Mean volume  -- phase1: {pdf['phase1_vol'].mean():,.0f}  phase2: {pdf['phase2_vol'].mean():,.0f}  "
              f"phase3: {pdf['phase3_vol'].mean():,.0f}")
        pct_phase3_gt_phase2 = 100.0 * (pdf["phase3_range"] > pdf["phase2_range"]).mean()
        pct_phase1_gt_phase2 = 100.0 * (pdf["phase1_range"] > pdf["phase2_range"]).mean()
        print(f"\n% of days phase3 range > phase2 range (power-hour re-expansion): {pct_phase3_gt_phase2:.1f}%")
        print(f"% of days phase1 range > phase2 range (initial expansion bigger than the lull): {pct_phase1_gt_phase2:.1f}%")

        sig_3v2 = paired_significance((pdf["phase3_range"] - pdf["phase2_range"]).tolist())
        print(f"\nPaired test, phase3-vs-phase2 range difference: mean_diff={sig_3v2['mean_diff']:.2f}pts  "
              f"t={sig_3v2['t_stat']:.2f}  p={sig_3v2['p_value']:.4f}  "
              f"({'significant' if sig_3v2['p_value'] < 0.05 else 'not significant'})")
    else:
        print("No days had all three phases available.")

    print("\nReminder: this is a market-structure statistical test, not a trade backtest -- no P&L, no capital. "
          "It tells you whether the claimed session shape/ORB behavior exists on Nifty, which informs whether "
          "it's worth building session-phase filters into the AAA trigger once that trigger itself is fixed.")


if __name__ == "__main__":
    main()
