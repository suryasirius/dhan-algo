"""
Order Flow / Auction Market Theory (AMT) Scalping Model -- adapted backtest
=================================================================
Source: user-provided doc (Fabio Valentino / Chart Fanatics interview,
paraphrased/structured). Two sub-models: TREND (imbalance, first-drive
entry) and MEAN-REVERSION (balance, second-drive-ONLY entry), both gated
by the doc's own 3-step sequence: Location (balance/imbalance) -> Validate
(low-volume-node) -> Trigger (aggression).

===========================================================================
WHAT COULD NOT BE BUILT FAITHFULLY -- READ THIS BEFORE TRUSTING ANY RESULT
===========================================================================
The doc's OWN Section 11/12 already say this model needs real order-flow/
footprint data (individual print sizes) to backtest faithfully, and
explicitly recommend backtesting the volume-profile + LVN logic alone
first with standard OHLCV data, then layering in an aggression filter
only if footprint data becomes available. That is exactly what this
script does -- with the aggression filter approximated, not skipped:

  1. NO REAL PRINT-LEVEL DATA. Dhan's intraday_minute_data only returns
     bar-level OHLC + a single aggregate volume number per bar -- never
     individual order/print sizes. "A single print exceeding a contract
     threshold" (the doc's literal aggression trigger) is INVISIBLE to
     this data source: a genuinely huge single print inside an otherwise
     quiet bar looks identical to no aggression at all. The best available
     substitute implemented here is a BAR-VOLUME-SPIKE PROXY: this bar's
     total volume >= AGGRESSION_VOL_MULT x its trailing average bar
     volume. This is a real, honest approximation, not the same signal --
     every result below is "volume-profile + LVN + volume-spike-proxy",
     NOT "true footprint/order-flow". Flagged loudly, not buried.
  2. NO NY/LONDON SESSION SPLIT. Nifty trades one continuous 09:15-15:30
     IST session -- there is no separate London session to test the
     mean-reversion model against. This script substitutes Nifty's OWN
     early higher-volatility window as a stand-in for "trend/NY conditions"
     and its own midday window as a stand-in for "balance/London
     conditions". This is an interpretive adaptation of the source's
     framing, not a literal implementation of it.
  3. "Ticks" and "contracts" are re-denominated onto Nifty index points and
     this project's existing proxy premium option model (intrinsic-at-open
     + assumed time value, moves scaled by a flat assumed delta), for
     consistency with every other strategy already tested in this project.

Everything else -- value-area/POC volume profile, LVN = bottom-10th-
percentile-of-volume bins, balance/imbalance via value-area breach,
second-drive-ONLY gating for mean-reversion, stop just beyond the
level, move-to-breakeven on first follow-through bar (no cushion),
target = previous balance area's POC, %-risk position sizing, daily
loss cap, and the compounding rule (only risk the day's already-banked
profit once positive) -- is implemented per the doc's own literal
"Backtest rule" bullets, not reinterpreted loosely.

No look-ahead: state/profile/LVNs at bar j are built from a trailing
window STRICTLY BEFORE bar j; the signal itself confirms on bar j's own
close/volume (matches this project's standard convention), and the
actual FILL happens on bar (j+1)'s open, one bar later -- same rule
applied to every other strategy tested in this project after the
look-ahead bug was found and fixed in two of them.

Conservative intrabar ordering: within any single bar, the STOP check
always happens before the TARGET check and before the breakeven-move
check -- i.e., a bar that could be read either way (stopped out or hit
target) is always resolved in the worse-case direction for the trade,
consistent with every other script in this project.
"""

import datetime as dt
import math
import random
import time
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

# ============================== CONFIG ======================================
FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
BAR_RULE = "1min"   # doc's stated entry timeframe

STRIKE_STEP = 50
EOD_SQUAREOFF = dt.time(15, 20)
SESSION_OPEN_SKIP_UNTIL = dt.time(9, 35)   # skip the doc's "15-20min liquidity-test window"

# Adapted session windows (see docstring point 2) -- NOT literal NY/London
TREND_WINDOW = (dt.time(9, 35), dt.time(11, 30))
MEANREV_WINDOW = (dt.time(12, 0), dt.time(14, 0))

PROFILE_LOOKBACK_BARS = 120     # trailing N-period volume profile (2 hours of 1-min bars)
BIN_SIZE_PTS = 5.0
VALUE_AREA_PCT = 0.70
LVN_PERCENTILE = 10
LVN_PROXIMITY_PTS = 5.0
MAX_CONSEC_IMBALANCE_AS_POKE = 10   # beyond this many consecutive bars, it's a real trend, not a failed poke

AGGRESSION_VOL_MULT = 3.0          # bar-volume-spike proxy (see docstring point 1 -- NOT a real print-size threshold)
AGGRESSION_AVG_LOOKBACK = 20

STOP_BUFFER_PTS = 3.0               # stand-in for "1-2 ticks beyond the level"
MIN_RR = 2.5

ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0
BROKERAGE_PER_TRADE = 40.0

PER_TRADE_RISK_PCT = 0.004     # 0.4%, middle of the doc's stated 0.25%-0.5%
DAILY_LOSS_CAP_PCT = 0.02
MONTE_CARLO_ITERS = 5000


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
    """Same chunked-fetch pattern used by every other script in this
    project. Adds a clear diagnostic + one retry per chunk on a
    transient/malformed response instead of crashing deep inside a
    DataFrame constructor -- the known 'fetch_index_chunked transient API
    error' lesson already in the playbook's bug catalog."""
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        last_err = None
        df = None
        for attempt in range(2):
            try:
                resp = dlb.dhan.intraday_minute_data(
                    security_id=dlb.UNDERLYING_SECURITY_ID, exchange_segment=dlb.UNDERLYING_EXCHANGE_SEGMENT,
                    instrument_type=dlb.UNDERLYING_INSTRUMENT_TYPE,
                    from_date=cur.strftime("%Y-%m-%d"), to_date=chunk_end.strftime("%Y-%m-%d"),
                )
                if not isinstance(resp, dict):
                    raise RuntimeError(f"non-dict response (likely a connection/proxy error surfaced as a "
                                        f"string): {str(resp)[:300]}")
                data = resp.get("data", resp)
                if not isinstance(data, dict) or "open" not in data:
                    raise RuntimeError(f"response missing expected OHLC keys -- likely an error payload, "
                                        f"not real data: {str(resp)[:300]}")
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
                    time.sleep(2)
        if df is None:
            raise RuntimeError(f"fetch_index_chunked failed for chunk {cur} to {chunk_end} after 2 attempts: {last_err}")
        chunks.append(df)
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


def resample_1min_per_day(df):
    """Dhan already returns ~1min bars; this just cleanly re-buckets and
    dedupes per day using the same origin-per-day convention as every
    other script here."""
    out = []
    for day in sorted(set(df.index.date)):
        day_df = df[df.index.date == day]
        r = day_df.resample(BAR_RULE, origin=day_df.index[0]).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        out.append(r)
    return pd.concat(out).sort_index()


def build_premium_series(day_df, strike, day_open_spot, side):
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


def premium_at_index_level(index_level, day_open_spot, strike, side):
    """Same base+move*delta formula as build_premium_series, evaluated at
    a single arbitrary index price -- used to translate an index-level
    stop/target (defined in underlying points, per the doc's own language)
    into the equivalent point on the option premium series."""
    if side == "CE":
        intrinsic0 = max(day_open_spot - strike, 0.0)
        move = index_level - day_open_spot
    else:
        intrinsic0 = max(strike - day_open_spot, 0.0)
        move = day_open_spot - index_level
    return max(intrinsic0 + ASSUMED_TIME_VALUE + move * ASSUMED_DELTA, 0.0)


# =============================================================================
# STEP 1 -- Location: volume profile, point of control, value area
# =============================================================================
def build_volume_profile(sub_df, bin_size=BIN_SIZE_PTS):
    """Bins each bar's FULL volume into the bin containing that bar's
    CLOSE (a necessary simplification -- OHLCV bars don't reveal how
    volume was distributed within the bar's own range, only tick/footprint
    data would)."""
    if sub_df.empty:
        return pd.Series(dtype=float)
    bins = (sub_df["close"] // bin_size) * bin_size
    return sub_df.groupby(bins)["volume"].sum().sort_index()


def compute_poc_value_area(profile, value_area_pct=VALUE_AREA_PCT):
    if profile.empty or profile.sum() <= 0:
        return None
    prices = list(profile.index)
    poc_pos = int(profile.values.argmax())
    poc_price = prices[poc_pos]
    total = profile.sum()
    target = total * value_area_pct
    lo = hi = poc_pos
    cum = profile.iloc[poc_pos]
    while cum < target and (lo > 0 or hi < len(prices) - 1):
        vol_below = profile.iloc[lo - 1] if lo > 0 else -1.0
        vol_above = profile.iloc[hi + 1] if hi < len(prices) - 1 else -1.0
        if vol_above >= vol_below:
            hi += 1
            cum += profile.iloc[hi]
        else:
            lo -= 1
            cum += profile.iloc[lo]
    return {"poc": poc_price, "va_low": prices[lo], "va_high": prices[hi]}


def find_lvns(profile, percentile=LVN_PERCENTILE):
    if profile.empty:
        return set()
    threshold = np.percentile(profile.values, percentile)
    return set(profile[profile <= threshold].index)


def classify_state(price, va):
    if va is None:
        return None
    if price > va["va_high"]:
        return "IMBALANCE_UP"
    if price < va["va_low"]:
        return "IMBALANCE_DOWN"
    return "BALANCE"


# =============================================================================
# STEP 3 -- Trigger: aggression proxy + LVN proximity
# =============================================================================
def is_aggression_bar(vol, avg_vol, mult=AGGRESSION_VOL_MULT):
    return (not pd.isna(avg_vol)) and avg_vol > 0 and vol >= mult * avg_vol


def near_lvn(price, lvn_set, tol=LVN_PROXIMITY_PTS):
    return any(abs(price - lvn) <= tol for lvn in lvn_set)


# =============================================================================
# Combined signal detection for one day -- TREND (first drive) +
# MEAN-REVERSION (second drive only)
# =============================================================================
def detect_signals_for_day(day_df):
    signals = []
    n = len(day_df)
    if n <= PROFILE_LOOKBACK_BARS + AGGRESSION_AVG_LOOKBACK:
        return signals
    closes = day_df["close"].to_numpy()
    opens = day_df["open"].to_numpy()
    vols = day_df["volume"].to_numpy()
    times = day_df.index

    prev_state = None
    regime_active = False
    consec_imbalance = 0
    upper_drives = 0
    lower_drives = 0

    for j in range(PROFILE_LOOKBACK_BARS, n):
        t = times[j].time()
        if t < SESSION_OPEN_SKIP_UNTIL:
            continue

        window = day_df.iloc[j - PROFILE_LOOKBACK_BARS:j]   # strictly prior bars -- no look-ahead
        profile = build_volume_profile(window)
        va = compute_poc_value_area(profile)
        if va is None:
            continue
        lvns = find_lvns(profile)
        avg_vol = vols[j - AGGRESSION_AVG_LOOKBACK:j].mean()
        state = classify_state(closes[j], va)
        aggression = is_aggression_bar(vols[j], avg_vol)
        at_lvn = near_lvn(closes[j], lvns)

        in_trend_window = TREND_WINDOW[0] <= t < TREND_WINDOW[1]
        in_meanrev_window = MEANREV_WINDOW[0] <= t < MEANREV_WINDOW[1]

        # ---- TREND MODEL: imbalance + at/near LVN + aggression WITH the move, first drive is fine ----
        if in_trend_window and state in ("IMBALANCE_UP", "IMBALANCE_DOWN") and at_lvn and aggression:
            direction = "LONG" if state == "IMBALANCE_UP" else "SHORT"
            directional_bar = (closes[j] > opens[j]) if direction == "LONG" else (closes[j] < opens[j])
            if directional_bar:
                signals.append({"bar": j, "model": "TREND", "direction": direction, "target": va["poc"]})
                prev_state = state
                continue

        # ---- MEAN-REVERSION MODEL: balance regime, second drive beyond a boundary only ----
        if state == "BALANCE":
            if not regime_active:
                regime_active, upper_drives, lower_drives = True, 0, 0
            consec_imbalance = 0
        elif state in ("IMBALANCE_UP", "IMBALANCE_DOWN") and regime_active:
            if prev_state != state:
                consec_imbalance = 1
                if state == "IMBALANCE_UP":
                    upper_drives += 1
                else:
                    lower_drives += 1
            else:
                consec_imbalance += 1
                if consec_imbalance > MAX_CONSEC_IMBALANCE_AS_POKE:
                    regime_active = False   # this has become a real trend, not a failed poke -- stop tracking

            if regime_active and in_meanrev_window and at_lvn and aggression:
                if state == "IMBALANCE_UP" and upper_drives == 2 and closes[j] < opens[j]:
                    signals.append({"bar": j, "model": "MEANREV", "direction": "SHORT", "target": va["poc"]})
                    regime_active = False
                elif state == "IMBALANCE_DOWN" and lower_drives == 2 and closes[j] > opens[j]:
                    signals.append({"bar": j, "model": "MEANREV", "direction": "LONG", "target": va["poc"]})
                    regime_active = False

        prev_state = state

    return signals


# =============================================================================
# Trade management -- conservative ordering: STOP before TARGET before
# breakeven-move, within any single bar
# =============================================================================
def simulate_trade(premium_df, entry_idx, entry_price, stop_level, target_level):
    stop = stop_level
    breakeven_moved = False
    n = len(premium_df)
    for j in range(entry_idx, n):
        bar = premium_df.iloc[j]
        t = premium_df.index[j].time()
        if bar["low"] <= stop:
            reason = "BREAKEVEN" if breakeven_moved else "STOP"
            return stop, reason, premium_df.index[j]
        if bar["high"] >= target_level:
            return target_level, "TARGET", premium_df.index[j]
        if t >= EOD_SQUAREOFF:
            return bar["close"], "EOD", premium_df.index[j]
        if not breakeven_moved and bar["close"] > entry_price:
            stop = max(stop, entry_price)
            breakeven_moved = True
    last = premium_df.iloc[-1]
    return last["close"], "EOD", premium_df.index[-1]


def size_position(equity, day_pnl_so_far, stop_distance_premium_pts, lot_size):
    """%-risk sizing with the doc's compounding rule: once the day is
    net positive, only the already-banked profit is put at risk on
    further trades, never the starting capital again that day."""
    if day_pnl_so_far > 0:
        risk_budget = min(PER_TRADE_RISK_PCT * equity, day_pnl_so_far)
    else:
        risk_budget = PER_TRADE_RISK_PCT * equity
    if stop_distance_premium_pts <= 0:
        return 0
    per_lot_risk = stop_distance_premium_pts * lot_size
    return max(int(risk_budget // per_lot_risk), 0)


# =============================================================================
# Self-checks
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # --- volume profile / POC / value area, hand-traced ---
    idx = pd.date_range("2025-01-06 09:15", periods=6, freq="1min")
    prof_df = pd.DataFrame({
        "close": [100, 100, 105, 105, 110, 110],
        "volume": [10, 10, 50, 50, 5, 5],
    }, index=idx)
    profile = build_volume_profile(prof_df, bin_size=5.0)
    assert profile.loc[100.0] == 20 and profile.loc[105.0] == 100 and profile.loc[110.0] == 10
    va = compute_poc_value_area(profile, value_area_pct=0.70)
    assert va["poc"] == 105.0
    # total=130, target=91 (70%); POC bin (105) alone already holds 100 >= 91, so the value area
    # is just the POC bin itself -- no expansion needed (standard market-profile behavior when a
    # single bin already dominates the distribution).
    assert va["va_low"] == 105.0 and va["va_high"] == 105.0, f"unexpected value area: {va}"
    print("  build_volume_profile / compute_poc_value_area hand-traced check ok")

    lvns = find_lvns(profile, percentile=50)  # median split for this tiny synthetic case
    assert 110.0 in lvns or 100.0 in lvns
    print("  find_lvns check ok")

    # a second profile with a wider value area, to test BALANCE (inside the range) vs IMBALANCE (outside it)
    profile2 = pd.Series({100.0: 30.0, 105.0: 40.0, 110.0: 20.0, 115.0: 10.0})
    va2 = compute_poc_value_area(profile2, value_area_pct=0.70)
    # total=100, target=70; POC=105 (40) alone < 70 -> expand to the bigger neighbor (100:30 > 110:20) -> cum=70, stop
    assert va2["poc"] == 105.0 and va2["va_low"] == 100.0 and va2["va_high"] == 105.0, f"unexpected: {va2}"
    assert classify_state(106, va2) == "IMBALANCE_UP"
    assert classify_state(99, va2) == "IMBALANCE_DOWN"
    assert classify_state(102, va2) == "BALANCE"
    print("  classify_state boundary check ok")

    assert is_aggression_bar(100, 20, mult=3.0) is True
    assert is_aggression_bar(50, 20, mult=3.0) is False
    assert near_lvn(101, {100.0}, tol=5.0) is True
    assert near_lvn(200, {100.0}, tol=5.0) is False
    print("  is_aggression_bar / near_lvn check ok")

    # --- premium_at_index_level sanity ---
    p = premium_at_index_level(22050, 22000, itm_put_strike(22000), side="PE")
    # PE: intrinsic0 = max(strike-22000,0); strike=ceil(22000/50)*50=22000 -> intrinsic0=0
    # move = 22000-22050 = -50 -> premium = 0+40+(-50*0.6) = 10.0
    assert abs(p - 10.0) < 1e-6, f"expected 10.0, got {p}"
    print("  premium_at_index_level check ok")

    # --- simulate_trade: conservative ordering (stop checked before target in the same bar) ---
    idx_t = pd.date_range("2025-01-06 09:15", periods=5, freq="1min")
    prem_ambiguous = pd.DataFrame({
        "open": [50, 50, 50, 50, 50],
        "high": [50, 52, 90, 60, 60],   # bar2's high would reach an aggressive target...
        "low": [50, 48, 40, 58, 58],    # ...but bar2's low ALSO breaches the stop at 42 -- must resolve to STOP
        "close": [50, 48, 60, 60, 60],  # bar1 closes BELOW entry so breakeven never triggers before bar2
    }, index=idx_t)
    exit_price, reason, _ = simulate_trade(prem_ambiguous, 0, entry_price=50, stop_level=42, target_level=80)
    assert reason == "STOP" and abs(exit_price - 42.0) < 1e-6, f"expected conservative STOP@42, got {exit_price},{reason}"
    print("  simulate_trade conservative stop-before-target ordering check ok")

    # --- simulate_trade: clean target hit ---
    prem_target = pd.DataFrame({
        "open": [50, 55, 60, 65, 70], "high": [50, 56, 61, 66, 85],
        "low": [50, 54, 59, 64, 69], "close": [50, 55, 60, 65, 80],
    }, index=idx_t)
    exit_price2, reason2, _ = simulate_trade(prem_target, 0, entry_price=50, stop_level=42, target_level=80)
    assert reason2 == "TARGET" and abs(exit_price2 - 80.0) < 1e-6
    print("  simulate_trade clean TARGET-hit check ok")

    # --- simulate_trade: breakeven move then stopped at entry (not the original stop) ---
    prem_be = pd.DataFrame({
        "open": [50, 55, 52, 48, 45], "high": [50, 56, 53, 49, 46],
        "low": [50, 54, 49.9, 47, 44], "close": [50, 55, 51, 48, 45],
    }, index=idx_t)
    exit_price3, reason3, _ = simulate_trade(prem_be, 0, entry_price=50, stop_level=42, target_level=200)
    # bar1 closes 55>50 -> breakeven moves stop to 50; bar2 low=49.9<=50 -> exit at 50, reason BREAKEVEN
    assert reason3 == "BREAKEVEN" and abs(exit_price3 - 50.0) < 1e-6, f"expected BREAKEVEN@50, got {exit_price3},{reason3}"
    print("  simulate_trade breakeven-move (no cushion) check ok")

    # --- size_position: risk budget, compounding rule, zero-lots skip ---
    lots1 = size_position(equity=25000, day_pnl_so_far=0, stop_distance_premium_pts=10, lot_size=65)
    # risk_budget = 0.004*25000=100; per_lot_risk=10*65=650 -> 100//650 = 0 lots (capital inadequate at this size)
    assert lots1 == 0, f"expected 0 lots (capital-inadequate case), got {lots1}"
    lots2 = size_position(equity=200000, day_pnl_so_far=0, stop_distance_premium_pts=10, lot_size=65)
    # risk_budget=800; per_lot_risk=650 -> 1 lot
    assert lots2 == 1, f"expected 1 lot, got {lots2}"
    lots3 = size_position(equity=200000, day_pnl_so_far=5000, stop_distance_premium_pts=10, lot_size=65)
    # compounding rule: risk_budget = min(800, 5000) = 800 -> still 1 lot (banked profit exceeds the %-risk cap here)
    assert lots3 == 1, f"expected 1 lot (banked profit exceeds %% cap, so %% cap still binds), got {lots3}"
    lots4 = size_position(equity=200000, day_pnl_so_far=300, stop_distance_premium_pts=10, lot_size=65)
    # compounding rule: risk_budget=min(800,300)=300 -> 300//650=0 lots (banked profit too small to size a lot)
    assert lots4 == 0, f"expected 0 lots (banked profit smaller than one lot's risk), got {lots4}"
    print("  size_position risk-budget / compounding-rule / zero-lot-skip check ok")

    # --- mean-reversion second-drive-only gating: synthetic day, first poke must NOT fire, second must ---
    # Day starts 09:15 so there's a long runway before the first evaluated bar (needs
    # PROFILE_LOOKBACK_BARS + AGGRESSION_AVG_LOOKBACK = 140 prior bars); pokes are timed to land
    # inside the 12:00-14:00 meanrev window (09:15 + 165min = 12:00).
    n_bars = 260
    idx_day = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="1min")
    base = 22000.0
    closes_syn = [base] * n_bars
    vols_syn = [100] * n_bars
    # sprinkle a handful of thin-volume visits to base+50 so a genuine low-volume node (LVN)
    # bin exists there BEFORE either poke happens (a thin "air pocket" the market passed
    # through quickly -- exactly what an LVN represents). Placed at index < PROFILE_LOOKBACK_BARS
    # (120) so the main detection loop (which starts evaluating at j=120) never processes these
    # bars AS a signal bar themselves (avoiding false drive-counter increments), while still
    # falling inside the trailing 120-bar window used to evaluate both pokes below (poke1's
    # window is bars[50:170), poke2's is bars[65:185) -- index 100-102 sits inside both).
    for k in (100, 101, 102):
        closes_syn[k] = base + 50
        vols_syn[k] = 1

    poke1 = 170   # 09:15 + 170min = 12:05, inside the meanrev window
    closes_syn[poke1] = base + 50
    vols_syn[poke1] = 100   # normal volume -> NOT aggression -> first drive must NOT fire

    poke2 = poke1 + 15   # 12:20, still inside the window, same boundary -> the 2nd drive
    closes_syn[poke2] = base + 50
    vols_syn[poke2] = 1000   # big volume spike -> aggression -> eligible to fire

    opens_syn = [base] * n_bars
    opens_syn[poke2] = base + 55   # red/reversing bar at the 2nd drive (close < open), required for the fade entry
    day_syn = pd.DataFrame({
        "open": opens_syn, "high": [max(o, c) + 1 for o, c in zip(opens_syn, closes_syn)],
        "low": [min(o, c) - 1 for o, c in zip(opens_syn, closes_syn)], "close": closes_syn, "volume": vols_syn,
    }, index=idx_day)

    sigs = detect_signals_for_day(day_syn)
    meanrev_sigs = [s for s in sigs if s["model"] == "MEANREV"]
    assert len(meanrev_sigs) == 1, f"expected exactly 1 MEANREV signal (from the 2nd drive only), got {len(meanrev_sigs)}"
    assert meanrev_sigs[0]["bar"] == poke2, f"expected the signal at the 2nd drive (bar {poke2}), got bar {meanrev_sigs[0]['bar']}"
    print("  mean-reversion second-drive-ONLY gating check ok (1st drive correctly ignored, 2nd drive fires)")

    print("All self-checks passed.\n")


def compute_metrics(trades_df, starting_capital):
    if trades_df.empty:
        return {"n": 0, "win_pct": 0.0, "pf": 0.0, "net_rs": 0.0, "max_dd_rs": 0.0, "sqn": 0.0, "cagr_pct": 0.0}
    pnl = trades_df["rupee_pnl"].to_numpy()
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
    max_dd_rs = (peak - equity).max()
    sqn = (pnl.mean() / pnl.std() * math.sqrt(n)) if n > 1 and pnl.std() > 0 else 0.0
    dates = pd.to_datetime(trades_df["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr_pct = 0.0
    if years > 0 and (starting_capital + net_rs) > 0:
        cagr_pct = 100.0 * (((starting_capital + net_rs) / starting_capital) ** (1.0 / years) - 1.0)
    return {"n": n, "win_pct": round(win_pct, 1), "pf": round(pf, 2) if pf != float("inf") else pf,
            "net_rs": round(net_rs, 2), "max_dd_rs": round(max_dd_rs, 2), "sqn": round(sqn, 2),
            "cagr_pct": round(cagr_pct, 2)}


def print_diagnostics(t, total_days):
    if t.empty:
        print("  no trades generated")
        return
    coverage = 100.0 * len(t) / total_days
    trend_n = int((t["model"] == "TREND").sum())
    meanrev_n = int((t["model"] == "MEANREV").sum())
    long_n = int((t["direction"] == "LONG").sum())
    short_n = int((t["direction"] == "SHORT").sum())
    print(f"  fires on {coverage:.1f}% of eligible days ({len(t)}/{total_days}) | "
          f"TREND={trend_n} MEANREV={meanrev_n} | LONG={long_n} SHORT={short_n}")


def run_backtest_for_capital(df1m, days, total_days, starting_capital, label):
    trades = []
    skipped_zero_lots = 0
    for day in days:
        day_df = df1m[df1m.index.date == day].copy()
        if len(day_df) < PROFILE_LOOKBACK_BARS + AGGRESSION_AVG_LOOKBACK + 5:
            continue
        day_open_spot = day_df["open"].iloc[0]
        signals = detect_signals_for_day(day_df)
        if not signals:
            continue

        equity = starting_capital
        day_pnl = 0.0
        open_trade_bar = -1
        lot = lot_size_for_date(day)

        for sig in signals:
            j0 = sig["bar"]
            if j0 <= open_trade_bar:
                continue
            if day_pnl <= -DAILY_LOSS_CAP_PCT * starting_capital:
                break   # daily loss cap hit -- stop taking new trades for the rest of the day

            exec_idx = j0 + 1
            if exec_idx >= len(day_df):
                continue

            direction = sig["direction"]
            side = "CE" if direction == "LONG" else "PE"
            strike = itm_call_strike(day_open_spot) if side == "CE" else itm_put_strike(day_open_spot)
            premium_df = build_premium_series(day_df, strike, day_open_spot, side)
            entry_price = premium_df["open"].iloc[exec_idx]

            sig_close = day_df["close"].iloc[j0]
            sig_low, sig_high = day_df["low"].iloc[j0], day_df["high"].iloc[j0]
            if direction == "LONG":
                index_stop_level = sig_low - STOP_BUFFER_PTS
            else:
                index_stop_level = sig_high + STOP_BUFFER_PTS
            stop_level = premium_at_index_level(index_stop_level, day_open_spot, strike, side)
            target_level = premium_at_index_level(sig["target"], day_open_spot, strike, side)

            stop_distance = entry_price - stop_level
            if stop_distance <= 0 or (target_level - entry_price) / stop_distance < MIN_RR:
                continue   # doesn't meet the doc's own minimum 1:2.5 R:R -- skip, per the doc's own rule

            lots = size_position(equity, day_pnl, stop_distance, lot)
            if lots <= 0:
                skipped_zero_lots += 1
                continue

            exit_price, reason, exit_time = simulate_trade(premium_df, exec_idx, entry_price, stop_level, target_level)
            pnl_pts = exit_price - entry_price
            rupee_pnl = pnl_pts * lot * lots - BROKERAGE_PER_TRADE
            day_pnl += rupee_pnl
            equity += rupee_pnl

            trades.append({
                "date": day, "model": sig["model"], "direction": direction, "side": side, "strike": strike,
                "signal_time": day_df.index[j0], "entry_time": day_df.index[exec_idx], "exit_time": exit_time,
                "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2), "exit_reason": reason,
                "lots": lots, "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
                "rr_planned": round((target_level - entry_price) / stop_distance, 2),
            })
            open_trade_bar = day_df.index.get_loc(exit_time)

    if not trades:
        print(f"[{label}] No trades generated (or none met the R:R / risk-sizing gates).")
        return None

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    tdf["running_equity"] = starting_capital + tdf["rupee_pnl"].cumsum()
    fname = f"nifty_amt_orderflow_trades_{label}.csv"
    tdf.to_csv(fname, index=False)

    m = compute_metrics(tdf, starting_capital)
    print(f"\n=== [{label}] Starting capital Rs {starting_capital:,} ===")
    print(f"n={m['n']}  win%={m['win_pct']}  PF={m['pf']}  net_Rs={m['net_rs']:,.2f}  "
          f"maxDD_Rs={m['max_dd_rs']:,.2f}  SQN={m['sqn']}  CAGR%={m['cagr_pct']}")
    print(f"Trades skipped because %-risk sizing rounded to 0 lots: {skipped_zero_lots}")
    print_diagnostics(tdf, total_days)

    for model_name, g in tdf.groupby("model"):
        mm = compute_metrics(g, starting_capital)
        print(f"  [{model_name}] n={mm['n']} win%={mm['win_pct']} PF={mm['pf']} net_Rs={mm['net_rs']:,.2f}")

    for reason, g in tdf.groupby("exit_reason"):
        print(f"  [exit={reason}] n={len(g)} net_Rs={g['rupee_pnl'].sum():,.2f}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh = tdf[dates <= mid]
    sh = tdf[dates > mid]
    m1, m2 = compute_metrics(fh, starting_capital), compute_metrics(sh, starting_capital)
    print(f"First half: n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']} SQN={m1['sqn']}  |  "
          f"Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']} SQN={m2['sqn']}")

    if m['pf'] != float("inf") and (m['pf'] > 2.5 or m['sqn'] > 5):
        print("  *** RED FLAG: PF>2.5 or SQN>5 -- treat as an extraordinary claim requiring extra scrutiny, "
              "NOT a validated edge, per this project's standing calibration. ***")

    print(f"Saved: {fname}")
    return tdf


def main():
    run_self_checks()
    print("=== Order Flow / AMT Scalping Model -- adapted backtest (see docstring for what's approximated) ===\n")
    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE})...")
    try:
        raw = fetch_index_chunked(FROM_DATE, TO_DATE)
    except Exception as e:
        print(f"\nCould not fetch real data: {e}\n")
        print("All self-checks above still passed (the logic itself is verified) -- this failure is about "
              "reaching the Dhan API from wherever this just ran. If you're seeing this on your own machine "
              "with a normally-working Dhan connection, check your network/ACCESS_TOKEN and retry.")
        return
    df1m = resample_1min_per_day(raw)
    days = sorted(set(df1m.index.date))
    total_days = len(days)
    print(f"Got {len(df1m)} 1-min bars across {total_days} trading day(s): {days[0]} to {days[-1]}\n")

    run_backtest_for_capital(df1m, days, total_days, starting_capital=25_000, label="Rs25k")
    run_backtest_for_capital(df1m, days, total_days, starting_capital=200_000, label="Rs200k")

    print("\nReminder: this uses a bar-volume-spike PROXY for 'aggression', not real footprint/print data, "
          "and an adapted (not literal) session split -- see the docstring at the top of this file before "
          "treating any result above as a validated edge.")


if __name__ == "__main__":
    main()
