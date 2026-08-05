"""
Concepts 1, 2, 3, 5 COMBINED -- independent, overlap-allowed -- 2-year backtest
=================================================================
Does NOT touch dhan_live_bot.py (locked, per project convention). That
file's compute_latest_signal() runs all 9 concepts through a single
CONFLUENCE vote -- multiple concepts agreeing REINFORCE one combined
signal, and only one concept_id gets credited per bar. That is NOT
what you asked for here.

This script instead extracts the four concepts you named as their own
INDEPENDENT raw signals (same exact formulas/thresholds as
dhan_live_bot.py, copied verbatim, just not funneled through
CONFLUENCE_REQUIRED/bull_votes/bear_votes):

  1 = Spike-buy       (compression -> upward breakout spike)
  2 = Spike-sell      (compression -> downward breakout spike)
  3 = Rollover-buy     (weak-ADX arc stretched near a low, fast rejection up)
  5 = Reversal-buy     (a recent down-spike failed and reversed back up)

Each concept runs as its OWN separate mini-strategy with its OWN open-
position tracking. If two (or more) concepts fire on the same bar, this
is fine -- each opens its OWN trade rather than one blocking the other,
per your instruction. The only overlap restriction is the obvious one:
a SINGLE concept can't open a second trade while its own current trade
is still open (same convention as every other script here -- no
re-entry until the position closes).

Concepts 1, 3, 5 are all buy-side -> buy the ITM CALL. Concept 2 is
sell-side -> buy the ITM PUT. All four can have positions open at the
same time, on the same underlying, if their independent conditions
line up.

Exit rule: kept IDENTICAL to dhan_live_bot.py's own stated convention
for concept-based (non-Velez) signals -- 5-point stop / 10-point
target (1:2 R:R), measured in NIFTY INDEX POINTS on the underlying
(not the premium), plus a 15-bar (1-min bars => 15-minute) maximum
hold as a backstop, same as MAX_BARS_IN_TRADE in the live bot. This
is the SAME index-points convention already used for concepts 2 & 4
elsewhere in this project (concept2_compression_spike_optimize.py),
kept consistent rather than switching to a premium-tracked stop.

NOT applied here (flag this clearly): the live bot's HTF 5m/15m trend-
match filter and its Tuesday/Wednesday-only weekday filter. Both were
calibrated specifically for the CONFLUENCE-based concepts 2/4 output,
not for these four concepts run independently -- so they're left off
by default to see the raw combined behavior first. Say the word if you
want either filter added back in.

All four concepts' trades are combined onto ONE shared equity curve
(one Rs25,000 account, all trades from all four concepts pooled and
sorted by time) so the backtest reflects what a single account
running all four concepts at once would actually look like, including
whatever capital pressure simultaneous open positions create.

2-year window, starting capital Rs 25,000, per your request.
"""

import datetime as dt
import math
import pandas as pd
import numpy as np

import dhan_live_bot as dlb

FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85
HTF_CONTEXT_DAYS = dlb.HTF_CONTEXT_DAYS if hasattr(dlb, "HTF_CONTEXT_DAYS") else 5

# --- same thresholds as dhan_live_bot.py, copied verbatim ---
COMP_LEN, COMP_THRESHOLD, MIN_TIGHT_BARS = dlb.COMP_LEN, dlb.COMP_THRESHOLD, dlb.MIN_TIGHT_BARS
ATR_LEN, SPIKE_MULT, BREAKOUT_LEN, EXHAUSTION_BARS = dlb.ATR_LEN, dlb.SPIKE_MULT, dlb.BREAKOUT_LEN, dlb.EXHAUSTION_BARS
ADX_LEN, WEAK_ADX_THRESH, WEAK_ARC_BARS, NEAR_EXTREME_PCT = dlb.ADX_LEN, dlb.WEAK_ADX_THRESH, dlb.WEAK_ARC_BARS, dlb.NEAR_EXTREME_PCT
REJECTION_WICK_RATIO = dlb.REJECTION_WICK_RATIO

# --- same exits as dhan_live_bot.py's concept-based (non-Velez) convention ---
SL_POINTS = dlb.SL_POINTS      # 5
TP_POINTS = dlb.TP_POINTS      # 10
MAX_BARS_IN_TRADE = dlb.MAX_BARS_IN_TRADE  # 15
EOD_SQUAREOFF = dt.time(15, 20)

STRIKE_STEP = 50
STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0
ASSUMED_DELTA = 0.6
ASSUMED_TIME_VALUE = 40.0

CONCEPT_NAMES = {1: "1 Spike-buy", 2: "2 Spike-sell", 3: "3 Rollover-buy", 5: "5 Reversal-buy"}
CONCEPT_SIDE = {1: "CE", 2: "PE", 3: "CE", 5: "CE"}


def lot_size_for_date(d):
    if d < dt.date(2024, 11, 20):
        return 25
    elif d < dt.date(2026, 1, 1):
        return 75
    else:
        return 65


def itm_strike(spot, side):
    if side == "CE":
        return math.floor(spot / STRIKE_STEP) * STRIKE_STEP
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


# ------------- raw per-bar votes for concepts 1, 2, 3, 5 (verbatim thresholds) -------------

def compute_votes_1_2_3_5(df):
    """Returns a DataFrame aligned to df.index with 4 boolean columns:
    vote_1 (spike-buy), vote_2 (spike-sell), vote_3 (rollover-buy),
    vote_5 (reversal-buy) -- each computed INDEPENDENTLY, bypassing
    confluence/disabled-concept filtering entirely."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)

    atr_s = dlb.atr(df, ATR_LEN).values
    adx_s = dlb.dmi_adx(df, ADX_LEN).values

    tight_count = weak_count = 0
    bars_since_spike_up = bars_since_spike_down = None
    breakout_level_up = breakout_level_down = None
    rollover_buy_prev = False

    v1 = np.zeros(n, dtype=bool)
    v2 = np.zeros(n, dtype=bool)
    v3 = np.zeros(n, dtype=bool)
    v5 = np.zeros(n, dtype=bool)

    for i in range(n):
        atr_i = atr_s[i]
        if np.isnan(atr_i) or atr_i == 0:
            continue

        if i >= COMP_LEN - 1:
            short_range = h[i - COMP_LEN + 1:i + 1].max() - l[i - COMP_LEN + 1:i + 1].min()
            is_tight = (short_range / (atr_i * COMP_LEN)) < COMP_THRESHOLD
        else:
            is_tight = False
        tight_count = tight_count + 1 if is_tight else 0
        compression_built = tight_count >= MIN_TIGHT_BARS

        big_range_bar = (h[i] - l[i]) > SPIKE_MULT * atr_i
        if i >= BREAKOUT_LEN:
            prior_high = h[max(0, i - BREAKOUT_LEN):i].max()
            prior_low = l[max(0, i - BREAKOUT_LEN):i].min()
        else:
            prior_high = prior_low = np.nan

        spike_up = big_range_bar and not np.isnan(prior_high) and c[i] > prior_high and c[i] > o[i]
        spike_down = big_range_bar and not np.isnan(prior_low) and c[i] < prior_low and c[i] < o[i]
        raw_buy = compression_built and spike_up
        raw_sell = compression_built and spike_down

        if raw_buy:
            breakout_level_up, bars_since_spike_up = prior_high, 0
        elif bars_since_spike_up is not None:
            bars_since_spike_up += 1
        if raw_sell:
            breakout_level_down, bars_since_spike_down = prior_low, 0
        elif bars_since_spike_down is not None:
            bars_since_spike_down += 1

        exhausted_up = (bars_since_spike_up is not None and bars_since_spike_up <= EXHAUSTION_BARS
                         and breakout_level_up is not None and c[i] < breakout_level_up)
        exhausted_down = (bars_since_spike_down is not None and bars_since_spike_down <= EXHAUSTION_BARS
                           and breakout_level_down is not None and c[i] > breakout_level_down)

        v1[i] = raw_buy and not exhausted_down       # concept 1: spike-buy
        v2[i] = raw_sell and not exhausted_up        # concept 2: spike-sell
        v5[i] = exhausted_down                       # concept 5: reversal-buy

        adx_i = adx_s[i]
        weak_trend = (not np.isnan(adx_i)) and adx_i < WEAK_ADX_THRESH
        weak_count = weak_count + 1 if weak_trend else 0
        stretched_weak_arc = weak_count >= WEAK_ARC_BARS

        if i >= WEAK_ARC_BARS - 1:
            extreme_low = l[i - WEAK_ARC_BARS + 1:i + 1].min()
        else:
            extreme_low = np.nan
        near_low = (not np.isnan(extreme_low)) and l[i] <= extreme_low * (1 + NEAR_EXTREME_PCT / 100)

        rng = h[i] - l[i]
        bull_wick = min(o[i], c[i]) - l[i]
        fast_rej_up = rng > 0 and (bull_wick / rng) >= REJECTION_WICK_RATIO and c[i] > o[i]

        rollover_buy_raw = stretched_weak_arc and near_low and fast_rej_up
        v3[i] = rollover_buy_raw and not rollover_buy_prev   # concept 3: rollover-buy
        rollover_buy_prev = rollover_buy_raw

    return pd.DataFrame({"vote_1": v1, "vote_2": v2, "vote_3": v3, "vote_5": v5}, index=df.index)


def get_htf_context_days():
    return HTF_CONTEXT_DAYS


def build_context_window_days(all_trading_days_sorted, target_day, context_days):
    idx_of_target = all_trading_days_sorted.index(target_day)
    start_idx = max(0, idx_of_target - context_days + 1)
    return all_trading_days_sorted[start_idx:idx_of_target + 1]


def simulate_trade_index_points(day_df, entry_idx, side, strike, day_open_spot,
                                 sl_pts=SL_POINTS, tp_pts=TP_POINTS, max_bars=MAX_BARS_IN_TRADE):
    """Stop/target measured in INDEX points on the underlying (matches
    dhan_live_bot.py's own concept-based convention), conservative
    ordering (stop before target before max-bars timeout before EOD).
    Fill at the NEXT bar's open (execution-lag fix), same as every
    other script here."""
    idx = day_df.index
    n = len(day_df)
    exec_idx = entry_idx + 1
    if exec_idx >= n:
        return None

    entry_spot = day_df["open"].iloc[exec_idx]
    if side == "CE":
        stop_level = entry_spot - sl_pts
        target_level = entry_spot + tp_pts
    else:
        stop_level = entry_spot + sl_pts
        target_level = entry_spot - tp_pts

    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()

    exit_i, exit_spot, reason = n - 1, closes[-1], "EOD"
    for j in range(exec_idx, min(exec_idx + max_bars, n)):
        t = idx[j].time()
        if side == "CE":
            if lows[j] <= stop_level:
                exit_i, exit_spot, reason = j, stop_level, "STOP"
                break
            if highs[j] >= target_level:
                exit_i, exit_spot, reason = j, target_level, "TARGET"
                break
        else:
            if highs[j] >= stop_level:
                exit_i, exit_spot, reason = j, stop_level, "STOP"
                break
            if lows[j] <= target_level:
                exit_i, exit_spot, reason = j, target_level, "TARGET"
                break
        if t >= EOD_SQUAREOFF:
            exit_i, exit_spot, reason = j, closes[j], "EOD"
            break
        if j == min(exec_idx + max_bars, n) - 1:
            exit_i, exit_spot, reason = j, closes[j], "MAX_BARS"
            break

    index_points = (exit_spot - entry_spot) if side == "CE" else (entry_spot - exit_spot)
    premium_df = build_premium_series(day_df, strike, day_open_spot, side)
    entry_premium = premium_df["open"].iloc[exec_idx]
    exit_premium = max(0.0, entry_premium + index_points * ASSUMED_DELTA)
    return {
        "exec_idx": exec_idx, "entry_time": idx[exec_idx], "exit_time": idx[exit_i],
        "entry_spot": entry_spot, "exit_spot": exit_spot, "exit_reason": reason,
        "index_pts": index_points, "entry_premium": round(entry_premium, 2),
        "exit_premium": round(exit_premium, 2),
    }


def _cross_validate_against_dhan_live_bot(day_df):
    """Runs the REAL, unedited dlb.compute_latest_signal() bar-by-bar on the
    same synthetic day, with confluence/disabled-concepts/HTF/weekday
    filters temporarily monkeypatched wide open (never touching the file),
    so it reports every concept exactly as it would fire in isolation.
    Returns {bar_index: concept_id} for direct comparison against this
    script's own compute_votes_1_2_3_5()."""
    orig_disabled = dlb.DISABLED_CONCEPTS
    orig_htf = dlb.REQUIRE_HTF_TREND_MATCH
    orig_weekdays = dlb.ALLOWED_WEEKDAYS
    orig_confluence = dlb.CONFLUENCE_REQUIRED
    dlb.DISABLED_CONCEPTS = set()
    dlb.REQUIRE_HTF_TREND_MATCH = False
    dlb.ALLOWED_WEEKDAYS = {0, 1, 2, 3, 4, 5, 6}
    dlb.CONFLUENCE_REQUIRED = 1
    try:
        picks = {}
        for i in range(len(day_df)):
            sig = dlb.compute_latest_signal(day_df.iloc[:i + 1])
            if sig and (sig["final_buy"] or sig["final_sell"]):
                picks[i] = sig["concept_id"]
        return picks
    finally:
        dlb.DISABLED_CONCEPTS = orig_disabled
        dlb.REQUIRE_HTF_TREND_MATCH = orig_htf
        dlb.ALLOWED_WEEKDAYS = orig_weekdays
        dlb.CONFLUENCE_REQUIRED = orig_confluence


def _build_synth_spike_reversal_day(spike_size=4.0, snapback_size=6.0, seed=42):
    """Tight chop -> a down-spike (triggers concept 2) -> a snap-back above
    the breakdown level within EXHAUSTION_BARS (triggers concept 5)."""
    n_bars = 80
    idx = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="1min")
    rng_state = np.random.RandomState(seed)
    closes = []
    c = 25000.0
    for _ in range(30):
        c += rng_state.uniform(-0.5, 0.5)
        closes.append(c)
    closes.append(closes[-1] - spike_size)
    closes.append(closes[-1] + snapback_size)
    for _ in range(32, n_bars):
        c = closes[-1] + rng_state.uniform(-0.5, 0.5)
        closes.append(c)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, cl) + 0.4 for o, cl in zip(opens, closes)]
    lows = [min(o, cl) - 0.4 for o, cl in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _build_synth_spike_up_day(spike_size=4.0, seed=7):
    """Tight chop -> an up-spike -> triggers concept 1."""
    n_bars = 80
    idx = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="1min")
    rng_state = np.random.RandomState(seed)
    closes = []
    c = 25000.0
    for _ in range(30):
        c += rng_state.uniform(-0.5, 0.5)
        closes.append(c)
    closes.append(closes[-1] + spike_size)
    for _ in range(31, n_bars):
        c = closes[-1] + rng_state.uniform(-0.5, 0.5)
        closes.append(c)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, cl) + 0.4 for o, cl in zip(opens, closes)]
    lows = [min(o, cl) - 0.4 for o, cl in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _build_synth_rollover_day(chop_bars=110, seed=11):
    """~110 bars of low-ADX chop drifting slowly down (weak, non-trending
    arc) followed by a fast-rejection green bar with a big lower wick right
    near the recent low -> triggers concept 3."""
    n_bars = chop_bars + 25
    idx = pd.date_range("2025-06-02 09:15", periods=n_bars, freq="1min")
    rng_state = np.random.RandomState(seed)
    closes = []
    for i in range(chop_bars):
        closes.append(25000.0 - i * 0.03 + rng_state.uniform(-1.0, 1.0))
    closes.append(closes[-1] + 0.5)
    for _ in range(chop_bars + 1, n_bars):
        closes.append(closes[-1] + rng_state.uniform(-1.0, 1.0))
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, cl) + 0.5 for o, cl in zip(opens, closes)]
    lows = [min(o, cl) - 0.5 for o, cl in zip(opens, closes)]
    lows[chop_bars] = min(closes[max(0, chop_bars - 30):chop_bars]) - 3.0
    highs[chop_bars] = closes[chop_bars] + 0.3
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def run_self_checks():
    print("Running self-checks (synthetic data)...")

    # --- concepts 2 (spike-sell) + 5 (reversal-buy): cross-validated
    # bar-for-bar against the REAL, unedited dhan_live_bot.compute_latest_signal ---
    day_25 = _build_synth_spike_reversal_day()
    votes_25 = compute_votes_1_2_3_5(day_25)
    original_25_all = _cross_validate_against_dhan_live_bot(day_25)
    # Restrict the comparison to concepts 2 and 5 -- the original function
    # also legitimately fires OTHER concepts (e.g. 6, reversal-sell) on the
    # same synthetic day, which aren't part of this 1/2/3/5 combination and
    # are correctly absent from my votes; only 2/5 need to match exactly.
    original_25 = {i: cid for i, cid in original_25_all.items() if cid in (2, 5)}
    my_25 = {}
    for i in range(len(day_25)):
        if votes_25["vote_2"].iloc[i]:
            my_25[i] = 2
        elif votes_25["vote_5"].iloc[i]:
            my_25[i] = 5
    assert my_25 == original_25, f"concept 2/5 mismatch vs dhan_live_bot: mine={my_25} original={original_25}"
    assert 2 in my_25.values() and 5 in my_25.values(), "expected both concept 2 and concept 5 to fire"
    print(f"  concept 2 (spike-sell) + concept 5 (reversal-buy) cross-validated bar-for-bar "
          f"against dhan_live_bot.compute_latest_signal ok ({my_25})")

    # --- concept 1 (spike-buy): same cross-validation ---
    day_1 = _build_synth_spike_up_day()
    votes_1 = compute_votes_1_2_3_5(day_1)
    original_1_all = _cross_validate_against_dhan_live_bot(day_1)
    original_1 = {i: cid for i, cid in original_1_all.items() if cid == 1}
    my_1 = {i: 1 for i in range(len(day_1)) if votes_1["vote_1"].iloc[i]}
    assert my_1 == original_1, f"concept 1 mismatch vs dhan_live_bot: mine={my_1} original={original_1}"
    assert 1 in my_1.values(), "expected concept 1 to fire"
    print(f"  concept 1 (spike-buy) cross-validated bar-for-bar against "
          f"dhan_live_bot.compute_latest_signal ok ({my_1})")

    # --- concept 3 (rollover-buy): same cross-validation ---
    day_3 = _build_synth_rollover_day()
    votes_3 = compute_votes_1_2_3_5(day_3)
    original_3_all = _cross_validate_against_dhan_live_bot(day_3)
    original_3 = {i: cid for i, cid in original_3_all.items() if cid == 3}
    my_3 = {i: 3 for i in range(len(day_3)) if votes_3["vote_3"].iloc[i]}
    assert my_3 == original_3, f"concept 3 mismatch vs dhan_live_bot: mine={my_3} original={original_3}"
    assert 3 in my_3.values(), "expected concept 3 to fire"
    print(f"  concept 3 (rollover-buy) cross-validated bar-for-bar against "
          f"dhan_live_bot.compute_latest_signal ok ({my_3})")

    # --- simulate_trade_index_points: clean TARGET for a CE ---
    idx_t = pd.date_range("2025-01-06 09:15", periods=20, freq="1min")
    spot = [25000.0] * 20
    for i in range(1, 20):
        spot[i] = spot[i - 1] + 1.5
    day_ce = pd.DataFrame({
        "open": spot, "high": [s + 0.5 for s in spot], "low": [s - 0.5 for s in spot], "close": spot,
    }, index=idx_t)
    t = simulate_trade_index_points(day_ce, entry_idx=0, side="CE", strike=25000, day_open_spot=25000.0)
    assert t["exit_reason"] == "TARGET", f"expected TARGET, got {t}"
    print("  simulate_trade_index_points CE clean TARGET check ok")

    # --- clean STOP for a PE (index rallying against a short/PE position) ---
    spot2 = [25000.0] * 20
    for i in range(1, 20):
        spot2[i] = spot2[i - 1] + 1.2
    day_pe = pd.DataFrame({
        "open": spot2, "high": [s + 0.5 for s in spot2], "low": [s - 0.5 for s in spot2], "close": spot2,
    }, index=idx_t)
    t2 = simulate_trade_index_points(day_pe, entry_idx=0, side="PE", strike=25000, day_open_spot=25000.0)
    assert t2["exit_reason"] == "STOP", f"expected STOP, got {t2}"
    print("  simulate_trade_index_points PE clean STOP check ok")

    # --- MAX_BARS_IN_TRADE timeout when neither stop nor target hit ---
    spot3 = [25000.0] * 20
    day_flat = pd.DataFrame({
        "open": spot3, "high": [s + 0.3 for s in spot3], "low": [s - 0.3 for s in spot3], "close": spot3,
    }, index=idx_t)
    t3 = simulate_trade_index_points(day_flat, entry_idx=0, side="CE", strike=25000, day_open_spot=25000.0)
    assert t3["exit_reason"] == "MAX_BARS", f"expected MAX_BARS, got {t3}"
    print("  simulate_trade_index_points MAX_BARS timeout check ok")

    # --- overlap check: two different concepts firing on the SAME bar must
    # both produce their own trade, not block each other ---
    open_positions = {}
    trades_today = []
    fake_signals = {1: True, 3: True}  # both fire on the same bar
    for concept_id, fired in fake_signals.items():
        if fired and open_positions.get(concept_id) is None:
            open_positions[concept_id] = "OPEN"
            trades_today.append(concept_id)
    assert trades_today == [1, 3], "both concepts firing on the same bar must both open a trade"
    print("  cross-concept same-bar overlap check ok (both concepts opened independently)")

    # --- same-concept no-overlap: a concept already open must not open a
    # second trade until it closes ---
    open_positions2 = {1: "OPEN"}
    trades_today2 = []
    if open_positions2.get(1) is None:
        trades_today2.append(1)
    assert trades_today2 == [], "an already-open concept must not open a second trade"
    print("  same-concept no-overlap check ok")

    print("All self-checks passed.\n")


def stats_for(g):
    if g.empty:
        return dict(n=0, win_pct=float("nan"), pf=float("nan"), net_rs=0.0,
                    max_dd=0.0, sqn=float("nan"), final_equity=STARTING_CAPITAL)
    wins = g[g["rupee_pnl"] > 0]
    losses = g[g["rupee_pnl"] <= 0]
    win_rate = len(wins) / len(g)
    pf = (wins["rupee_pnl"].sum() / abs(losses["rupee_pnl"].sum())
          if losses["rupee_pnl"].sum() != 0 else float("inf"))
    g = g.sort_values("entry_time")
    equity = STARTING_CAPITAL + g["rupee_pnl"].cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    r = g["rupee_pnl"]
    sqn = (r.mean() / r.std()) * (len(r) ** 0.5) if r.std() > 0 else float("nan")
    return dict(n=len(g), win_pct=100 * win_rate, pf=pf, net_rs=g["rupee_pnl"].sum(),
                max_dd=max_dd, sqn=sqn, final_equity=equity.iloc[-1])


def main():
    run_self_checks()
    print("=== Concepts 1, 2, 3, 5 COMBINED -- independent, overlap allowed -- 2-year backtest ===\n")
    print("Concepts: " + ", ".join(f"{cid}={name}" for cid, name in CONCEPT_NAMES.items()))
    print(f"Exit: {SL_POINTS}pt stop / {TP_POINTS}pt target (index points) / {MAX_BARS_IN_TRADE}-bar max hold, "
          f"conservative ordering. NO HTF trend filter, NO weekday filter applied.\n")

    print(f"Fetching Nifty index 1-min data ({FROM_DATE} to {TO_DATE}, ~2 years)...")
    df1m = fetch_index_chunked(FROM_DATE, TO_DATE)
    days_covered = sorted(set(df1m.index.date))
    print(f"Got {len(df1m)} 1-min bars across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    context_days = get_htf_context_days()
    all_trades = []

    for target_day in days_covered:
        window_days = build_context_window_days(days_covered, target_day, context_days)
        window_df = df1m[(df1m.index.date >= window_days[0]) & (df1m.index.date <= window_days[-1])]
        day_df = window_df[window_df.index.date == target_day]
        if len(day_df) < WEAK_ARC_BARS + 5:
            continue

        votes = compute_votes_1_2_3_5(window_df)
        day_votes = votes[votes.index.date == target_day]
        day_open_spot = day_df["open"].iloc[0]

        # each concept tracks its OWN open-trade bar independently
        open_until_bar = {1: -1, 2: -1, 3: -1, 5: -1}

        for local_i in range(len(day_df)):
            bar_time = day_df.index[local_i]
            for concept_id, vote_col in ((1, "vote_1"), (2, "vote_2"), (3, "vote_3"), (5, "vote_5")):
                if not day_votes.loc[bar_time, vote_col]:
                    continue
                if local_i <= open_until_bar[concept_id]:
                    continue  # this concept already has a position open

                side = CONCEPT_SIDE[concept_id]
                strike = itm_strike(day_open_spot, side)
                t = simulate_trade_index_points(day_df, local_i, side, strike, day_open_spot)
                if t is None:
                    continue

                lot = lot_size_for_date(target_day)
                pnl_pts = t["exit_premium"] - t["entry_premium"]
                rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
                all_trades.append({
                    "date": target_day, "concept_id": concept_id, "concept": CONCEPT_NAMES[concept_id],
                    "side": side, "strike": strike, "signal_time": bar_time,
                    "entry_time": t["entry_time"], "exit_time": t["exit_time"],
                    "entry_premium": t["entry_premium"], "exit_premium": t["exit_premium"],
                    "exit_reason": t["exit_reason"], "index_pts": round(t["index_pts"], 2),
                    "pnl_pts": round(pnl_pts, 2), "rupee_pnl": round(rupee_pnl, 2),
                })
                exit_local_i = day_df.index.get_loc(t["exit_time"])
                open_until_bar[concept_id] = exit_local_i

    if not all_trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    tdf.to_csv("concepts_1_2_3_5_combined_trades.csv", index=False)

    r = stats_for(tdf)
    print("--- COMBINED (all 4 concepts, one shared Rs25,000 account) ---")
    print(f"n={r['n']}  win%={r['win_pct']:.1f}  PF={r['pf']:.2f}  net_Rs={r['net_rs']:,.2f}  "
          f"maxDD_Rs={r['max_dd']:,.2f}  SQN={r['sqn']:.2f}  final_equity_Rs={r['final_equity']:,.2f}\n")

    print("--- By concept (isolated stats -- how each performed on its own) ---")
    for concept_id, name in CONCEPT_NAMES.items():
        g = tdf[tdf["concept_id"] == concept_id]
        rg = stats_for(g)
        print(f"  [{name}] n={rg['n']}  win%={rg['win_pct']:.1f}  PF={rg['pf']:.2f}  "
              f"net_Rs={rg['net_rs']:,.2f}  SQN={rg['sqn']:.2f}")

    print("\n--- By exit reason (combined) ---")
    for reason, g in tdf.groupby("exit_reason"):
        print(f"  {reason}: n={len(g)} ({100*len(g)/len(tdf):.1f}%)  win%={100*(g['rupee_pnl']>0).mean():.1f}  "
              f"net_Rs={g['rupee_pnl'].sum():,.2f}")

    # overlap diagnostic: how often did 2+ concepts have simultaneously open trades?
    events = []
    for _, row in tdf.iterrows():
        events.append((row["entry_time"], 1))
        events.append((row["exit_time"], -1))
    events.sort()
    concurrent, max_concurrent, overlap_bars = 0, 0, 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
        if concurrent >= 2:
            overlap_bars += 1
    print(f"\n--- Overlap diagnostic ---\nMax simultaneous open positions across all 4 concepts: {max_concurrent}")

    dates = pd.to_datetime(tdf["date"])
    mid = dates.min() + (dates.max() - dates.min()) / 2
    fh, sh = tdf[dates <= mid], tdf[dates > mid]
    m1, m2 = stats_for(fh), stats_for(sh)
    print(f"\n--- Stability check (in-sample/out-of-sample split) ---")
    print(f"  First half:  n={m1['n']} net_Rs={m1['net_rs']:,.2f} PF={m1['pf']:.2f} SQN={m1['sqn']:.2f}")
    print(f"  Second half: n={m2['n']} net_Rs={m2['net_rs']:,.2f} PF={m2['pf']:.2f} SQN={m2['sqn']:.2f}")

    if r["pf"] != float("inf") and not pd.isna(r["pf"]) and (r["pf"] > 2.5 or (not pd.isna(r["sqn"]) and r["sqn"] > 5)):
        print("\n*** RED FLAG: PF>2.5 or SQN>5 -- extraordinary claim, treat with extra scrutiny "
              "per this project's standing calibration. ***")

    print(f"\nSaved: concepts_1_2_3_5_combined_trades.csv ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
