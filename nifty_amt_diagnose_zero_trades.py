"""
Diagnostic: why did nifty_amt_orderflow_backtest.py produce zero trades?
=================================================================
The real 2-year run found zero trades at Rs25k (and apparently Rs200k too,
same "not working" result). Rather than guess which of the model's several
AND-gates (state classification, LVN proximity, aggression proxy, R:R
minimum) is starving the signal, this pulls a short recent window of real
data and counts how many bars pass EACH gate individually, in order.

Leading hypothesis: NIFTY the INDEX itself has no real traded volume --
only futures/options ON it trade -- so Dhan's "volume" field for the
underlying index (security_id=13) may come back as all zeros or some
meaningless placeholder. If so, the aggression-proxy gate (bar volume >=
3x trailing average volume) can never be satisfied, no matter what the
price is doing, which alone would explain zero trades at any capital
level. This script checks that FIRST, then walks the rest of the gates
regardless, so we get a real answer instead of a guess.

Reuses the real functions from nifty_amt_orderflow_backtest.py directly
(no logic duplicated/re-typed) so this diagnostic can't drift out of sync
with the actual backtest.

Run:
    cd "C:\\dhan algo"
    python nifty_amt_diagnose_zero_trades.py
"""

import datetime as dt
import pandas as pd
import numpy as np

import nifty_amt_orderflow_backtest as amt

DIAG_DAYS = 60   # short recent window -- fast, and recent data is what matters most anyway

to_date = dt.date.today().strftime("%Y-%m-%d")
from_date = (dt.date.today() - dt.timedelta(days=DIAG_DAYS)).strftime("%Y-%m-%d")

print(f"Fetching last {DIAG_DAYS} calendar days of real Nifty 1-min data ({from_date} to {to_date})...")
raw = amt.fetch_index_chunked(from_date, to_date)
df1m = amt.resample_1min_per_day(raw)
days = sorted(set(df1m.index.date))
print(f"Got {len(df1m)} 1-min bars across {len(days)} trading day(s): {days[0]} to {days[-1]}\n")

# ---- STEP 0: is the volume field even real? ----
vol = df1m["volume"]
print("=== STEP 0: volume field sanity ===")
print(f"min={vol.min()}  max={vol.max()}  mean={vol.mean():.2f}  "
      f"pct_zero_bars={100*(vol == 0).mean():.1f}%  pct_bars_below_20={100*(vol < 20).mean():.1f}%")
if vol.max() == 0:
    print(">>> CONFIRMED: Dhan's volume field for the Nifty INDEX (security_id=13) is ALWAYS ZERO. "
          "This alone fully explains zero trades -- the aggression-proxy gate (volume spike vs trailing "
          "average) can never fire if there's no real volume data to spike. NIFTY the index has no traded "
          "volume of its own (only its futures/options do); this model needs volume from a tradable "
          "instrument, not the index itself. See the fix suggestions at the bottom of this output.\n")
elif vol.std() == 0:
    print(">>> Volume is a nonzero CONSTANT across every bar -- also fatal for a volume-SPIKE detector, "
          "same root problem as all-zero volume.\n")
else:
    print("Volume field has real variation -- not the all-zero index-volume problem. Continuing to the "
          "other gates below.\n")

# ---- STEP 1-4: walk the gates bar-by-bar for one representative recent day ----
print("=== STEP 1-4: gate-by-gate bar counts (summed across all fetched days, trend+meanrev windows only) ===")
n_total_eligible = 0
n_state_imbalance = 0
n_at_lvn = 0
n_aggression = 0
n_state_and_lvn = 0
n_state_and_lvn_and_aggression = 0
vol_ratios = []   # this bar's volume / trailing average volume, for every eligible bar -- used to
                   # pick a data-driven AGGRESSION_VOL_MULT instead of guessing one

for day in days:
    day_df = df1m[df1m.index.date == day].copy()
    n = len(day_df)
    if n <= amt.PROFILE_LOOKBACK_BARS + amt.AGGRESSION_AVG_LOOKBACK:
        continue
    closes = day_df["close"].to_numpy()
    vols = day_df["volume"].to_numpy()
    times = day_df.index

    for j in range(amt.PROFILE_LOOKBACK_BARS, n):
        t = times[j].time()
        in_trend = amt.TREND_WINDOW[0] <= t < amt.TREND_WINDOW[1]
        in_meanrev = amt.MEANREV_WINDOW[0] <= t < amt.MEANREV_WINDOW[1]
        if not (in_trend or in_meanrev):
            continue
        n_total_eligible += 1

        window = day_df.iloc[j - amt.PROFILE_LOOKBACK_BARS:j]
        profile = amt.build_volume_profile(window)
        va = amt.compute_poc_value_area(profile)
        if va is None:
            continue
        lvns = amt.find_lvns(profile)
        avg_vol = vols[j - amt.AGGRESSION_AVG_LOOKBACK:j].mean()
        state = amt.classify_state(closes[j], va)
        aggression = amt.is_aggression_bar(vols[j], avg_vol)
        at_lvn = amt.near_lvn(closes[j], lvns)
        if avg_vol > 0:
            vol_ratios.append(vols[j] / avg_vol)

        is_imbalance = state in ("IMBALANCE_UP", "IMBALANCE_DOWN")
        if is_imbalance:
            n_state_imbalance += 1
        if at_lvn:
            n_at_lvn += 1
        if aggression:
            n_aggression += 1
        if is_imbalance and at_lvn:
            n_state_and_lvn += 1
        if is_imbalance and at_lvn and aggression:
            n_state_and_lvn_and_aggression += 1

print(f"Total eligible bars (in trend or meanrev window, profile available): {n_total_eligible}")
print(f"  ...classified IMBALANCE (either direction):     {n_state_imbalance}  "
      f"({100*n_state_imbalance/max(n_total_eligible,1):.1f}%)")
print(f"  ...near a validated LVN:                        {n_at_lvn}  "
      f"({100*n_at_lvn/max(n_total_eligible,1):.1f}%)")
print(f"  ...an aggression (volume-spike) bar:             {n_aggression}  "
      f"({100*n_aggression/max(n_total_eligible,1):.1f}%)")
print(f"  ...IMBALANCE *and* near an LVN:                  {n_state_and_lvn}  "
      f"({100*n_state_and_lvn/max(n_total_eligible,1):.1f}%)")
print(f"  ...IMBALANCE *and* near LVN *and* aggression:    {n_state_and_lvn_and_aggression}  "
      f"({100*n_state_and_lvn_and_aggression/max(n_total_eligible,1):.1f}%)  <- this is what a TREND signal needs\n")

# ---- STEP 4b: data-driven look at what AGGRESSION_VOL_MULT should actually be ----
print("=== STEP 4b: volume-ratio (this bar's volume / trailing average) distribution ===")
if vol_ratios:
    ratios = np.array(vol_ratios)
    for p in (50, 75, 90, 95, 99):
        print(f"  {p}th percentile ratio: {np.percentile(ratios, p):.2f}x   "
              f"(a mult of this value would flag ~{100-p}% of eligible bars as 'aggression')")
    print(f"  Current AGGRESSION_VOL_MULT={amt.AGGRESSION_VOL_MULT} flags "
          f"{100*(ratios >= amt.AGGRESSION_VOL_MULT).mean():.2f}% of bars -- compare to the ~10% selectivity "
          f"used for the LVN definition (bottom 10th percentile) to judge whether 3.0x is unusually strict.\n")

# ---- STEP 5: run the real detect_signals_for_day + count how many pass the R:R gate ----
print("=== STEP 5: full signal detection + R:R gate (reusing the real backtest function) ===")
n_signals = 0
n_passed_rr = 0
for day in days:
    day_df = df1m[df1m.index.date == day].copy()
    if len(day_df) <= amt.PROFILE_LOOKBACK_BARS + amt.AGGRESSION_AVG_LOOKBACK:
        continue
    day_open_spot = day_df["open"].iloc[0]
    sigs = amt.detect_signals_for_day(day_df)
    n_signals += len(sigs)
    for sig in sigs:
        j0 = sig["bar"]
        direction = sig["direction"]
        side = "CE" if direction == "LONG" else "PE"
        strike = amt.itm_call_strike(day_open_spot) if side == "CE" else amt.itm_put_strike(day_open_spot)
        sig_low, sig_high = day_df["low"].iloc[j0], day_df["high"].iloc[j0]
        index_stop_level = (sig_low - amt.STOP_BUFFER_PTS) if direction == "LONG" else (sig_high + amt.STOP_BUFFER_PTS)
        stop_level = amt.premium_at_index_level(index_stop_level, day_open_spot, strike, side)
        target_level = amt.premium_at_index_level(sig["target"], day_open_spot, strike, side)
        exec_idx = j0 + 1
        if exec_idx >= len(day_df):
            continue
        premium_df = amt.build_premium_series(day_df, strike, day_open_spot, side)
        entry_price = premium_df["open"].iloc[exec_idx]
        stop_distance = entry_price - stop_level
        rr = (target_level - entry_price) / stop_distance if stop_distance > 0 else float("nan")
        if stop_distance > 0 and rr >= amt.MIN_RR:
            n_passed_rr += 1
        print(f"  {day} bar={j0} model={sig['model']} dir={direction} entry={entry_price:.2f} "
              f"stop={stop_level:.2f} target={target_level:.2f} R:R={rr:.2f} "
              f"{'PASSES' if (stop_distance > 0 and rr >= amt.MIN_RR) else 'REJECTED (below 1:2.5)'}")

print(f"\nTotal raw signals found (before R:R gate): {n_signals}")
print(f"Signals that clear the minimum 1:{amt.MIN_RR} R:R: {n_passed_rr}")

print("\n=== DIAGNOSIS SUMMARY ===")
if vol.max() == 0:
    print("Root cause: Dhan's volume field for the Nifty index is always zero -- the aggression-proxy gate "
          "can never fire. Fix options: (a) run the aggression check against Nifty FUTURES volume instead of "
          "the index (futures actually trade and have real volume), or (b) drop the aggression gate entirely "
          "and test the volume-profile + LVN logic alone first, exactly as the source doc's own Section 12 "
          "suggests as a fallback. Both are real design changes, not tuning -- flagging for your call rather "
          "than picking one silently.")
elif n_state_and_lvn_and_aggression == 0:
    print("Root cause: state+LVN+aggression never align simultaneously in this window -- most likely the "
          "LVN_PROXIMITY_PTS=5.0 tolerance is too tight relative to BIN_SIZE_PTS=5.0 and typical Nifty moves, "
          "so 'beyond the value area' and 'near a validated LVN' are close to mutually exclusive as currently "
          "defined. Worth loosening LVN_PROXIMITY_PTS and/or BIN_SIZE_PTS and re-running this diagnostic before "
          "touching the full 2-year backtest again.")
elif n_signals > 0 and n_passed_rr == 0:
    print("Root cause: signals DO fire, but none clear the minimum 1:2.5 R:R -- the 3pt STOP_BUFFER_PTS is "
          "producing rewards-to-risk below the doc's own stated minimum. Worth checking the printed R:R values "
          "above to see how far short they fall.")
else:
    print("See the gate breakdown above to see exactly where the funnel narrows to zero.")
