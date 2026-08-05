"""
M-STRUCTURE (rounded-bottom support + flat-top resistance) BREAKOUT STRATEGY
=============================================================================
New concept, translated from your chart + description into concrete,
backtestable rules. Nothing here is imported from an already-locked file
except the shared proxy-premium model and conservative-ordering position
stepper (top3.build_premium_series / top3.step_long_premium_position --
reused verbatim, not reimplemented), and the OI regime classifier
(oi_trend_strategy_backtest.classify_oi_regime / fetch_futures_oi_chunked /
find_current_nifty_future -- also reused verbatim).

YOUR RULES, AS CODED (confirmed with you before building):
  - Direction: BOTH -- a long/CE trade on a breakout above the flat
    resistance ("M-top"), and a mirrored short/PE trade on a breakdown
    below the rounded support ("W-bottom").
  - Timeframe: 1-minute bars (your choice).
  - "Recent high/low" (the flat resistance / rounded support levels): the
    rolling max-high / min-low over the last STRUCT_LOOKBACK=30 bars
    (30 minutes) -- a modeling simplification for the fuzzy "M/W structure"
    read off a chart, documented here rather than hidden. Always computed
    from bars STRICTLY BEFORE the current one (shift(1)) -- no look-ahead.
  - "The rejected bottom will not breach by the next candles" / mirrored
    for the top: a structure-integrity filter. The extreme (low for longs,
    high for shorts) established in the OLDER part of the lookback window
    (bars i-30..i-5) must NOT have been broken by the most RECENT
    REJECT_HOLD_BARS=5 bars -- i.e. the rejection is still holding right
    now, not being actively re-tested/broken.
  - "Break the previous recent high ... and hold for a few minutes ...
    take trade": breakout bar's high must clear the resistance level, on
    above-average volume (volume > trailing 20-bar average -- the Nifty
    index feed carries real volume, confirmed earlier in this project).
    Then price must HOLD above the broken level for HOLD_CONFIRM_BARS=2
    more bars (your choice) before the entry is taken -- this rejects
    fakeout breakouts that immediately snap back below the level. Entry
    fills at the open of the bar AFTER the 2nd hold bar's close (signal
    confirms on a bar's close, fill is next bar's open -- the standing
    look-ahead-safety rule in this project).
  - Stop-loss / trail: exactly as you specified -- 5-point initial stop,
    4-point trail, using the same conservative intrabar ordering (this
    bar's low checked against the OLD stop before this bar's high is
    allowed to ratchet the stop up) already validated on every other
    strategy in this project.
  - OI ("look for the OI chart for one more confirmation"): SOFT-LOGGED
    only, not a hard filter (your choice) -- real OI is only available for
    the ~90 most recent calendar days (NIFTY futures roll monthly and
    Dhan drops expired contracts from its instrument list, confirmed in
    earlier OI investigation on this project). Filtering on it would
    shrink a 2-year backtest to ~90 days. Instead every trade is tagged
    with the OI regime (LONG_BUILDUP / SHORT_BUILDUP / SHORT_COVERING /
    LONG_UNWINDING / NEUTRAL / NO_DATA) at its entry time where real OI
    data exists, and the report breaks out P&L for OI-agreeing vs. not,
    purely as a reference diagnostic.

Two ways to run this (same convention as every other script in this
project):
  Self-checks + smoke test (safe any time, ~20-40s, no CSV writes):
      cd "C:\\dhan algo"
      python nifty_m_structure_backtest.py --smoke-test
  Full 2-year real backtest:
      cd "C:\\dhan algo"
      python nifty_m_structure_backtest.py
"""

import sys
import math
import datetime as dt
import pandas as pd
import numpy as np

import dhan_live_bot as dlb
import nifty_paper_trade_top3_live as top3
from oi_trend_strategy_backtest import (
    classify_oi_regime, fetch_futures_oi_chunked, find_current_nifty_future,
)
from full_institutional_analysis import full_metrics, monte_carlo_reshuffle, build_daily_pnl

# ============================== shared config ===============================
FROM_DATE = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

STARTING_CAPITAL = 25_000
BROKERAGE_PER_TRADE = 40.0

STRUCT_LOOKBACK = 30       # bars defining the rolling M-top/W-bottom structure level
REJECT_HOLD_BARS = 5       # recent bars checked for "the rejected extreme hasn't re-broken"
HOLD_CONFIRM_BARS = 2      # bars price must hold beyond the broken level before entry confirms
VOL_AVG_LOOKBACK = 20      # trailing bars for the volume-confirmation average
STRUCT_STOP_PTS, STRUCT_TRAIL_PTS = 5.0, 4.0   # exactly as specified
OI_CHANGE_LOOKBACK_MIN = 15  # matches oi_trend_strategy_backtest's own convention (1-min bars)

TRADES_CSV = "m_structure_trades.csv"


# ============================== data fetch ===================================
def fetch_index_chunked_with_volume(from_date, to_date, chunk_days=CHUNK_DAYS):
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
            "volume": data.get("volume", [0] * len(data["open"])), "timestamp": data["timestamp"],
        })
        df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
        chunks.append(df.set_index("time").sort_index())
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


# =============================================================================
# Structure level computation -- vectorized, all look-ahead safe (every level
# for bar i is computed strictly from bars < i via shift()).
# =============================================================================
def compute_structure_levels(day_df):
    high_s, low_s, vol_s = day_df["high"], day_df["low"], day_df["volume"]
    old_win = STRUCT_LOOKBACK - REJECT_HOLD_BARS
    out = pd.DataFrame(index=day_df.index)
    out["res_level"] = high_s.shift(1).rolling(STRUCT_LOOKBACK, min_periods=STRUCT_LOOKBACK).max()
    out["sup_level"] = low_s.shift(1).rolling(STRUCT_LOOKBACK, min_periods=STRUCT_LOOKBACK).min()
    out["res_level_old"] = high_s.shift(1 + REJECT_HOLD_BARS).rolling(old_win, min_periods=old_win).max()
    out["sup_level_old"] = low_s.shift(1 + REJECT_HOLD_BARS).rolling(old_win, min_periods=old_win).min()
    out["recent_max_high"] = high_s.shift(1).rolling(REJECT_HOLD_BARS, min_periods=REJECT_HOLD_BARS).max()
    out["recent_min_low"] = low_s.shift(1).rolling(REJECT_HOLD_BARS, min_periods=REJECT_HOLD_BARS).min()
    out["vol_avg"] = vol_s.shift(1).rolling(VOL_AVG_LOOKBACK, min_periods=VOL_AVG_LOOKBACK).mean()
    return out


def detect_m_structure_signals(day_df):
    """Returns a list of {'bar': i, 'side': 'CE'/'PE', 'event': ...} candidate
    signals for one day, already past the hold-confirmation check. `bar` is
    the ORIGINAL breakout/breakdown bar index (entry executes 2 bars later,
    at bar+HOLD_CONFIRM_BARS+1's open)."""
    lv = compute_structure_levels(day_df)
    highs = day_df["high"].to_numpy(); lows = day_df["low"].to_numpy(); vols = day_df["volume"].to_numpy()
    res = lv["res_level"].to_numpy(); sup = lv["sup_level"].to_numpy()
    res_old = lv["res_level_old"].to_numpy(); sup_old = lv["sup_level_old"].to_numpy()
    rec_max_h = lv["recent_max_high"].to_numpy(); rec_min_l = lv["recent_min_low"].to_numpy()
    vol_avg = lv["vol_avg"].to_numpy()
    n = len(day_df)
    out = []
    last_bar_ok = n - 1 - HOLD_CONFIRM_BARS - 1
    for i in range(STRUCT_LOOKBACK, last_bar_ok + 1):
        if any(np.isnan(x) for x in (res[i], sup[i], res_old[i], sup_old[i], rec_max_h[i], rec_min_l[i], vol_avg[i])):
            continue
        long_ok = (highs[i] > res[i] and vols[i] > vol_avg[i] and rec_min_l[i] >= sup_old[i])
        if long_ok:
            if all(lows[i + h] > res[i] for h in range(1, HOLD_CONFIRM_BARS + 1)):
                out.append({"bar": i, "side": "CE", "event": "MTOP_BREAKOUT", "level": res[i]})
                continue
        short_ok = (lows[i] < sup[i] and vols[i] > vol_avg[i] and rec_max_h[i] <= res_old[i])
        if short_ok:
            if all(highs[i + h] < sup[i] for h in range(1, HOLD_CONFIRM_BARS + 1)):
                out.append({"bar": i, "side": "PE", "event": "WBOTTOM_BREAKDOWN", "level": sup[i]})
    return out


def run_m_structure_standalone_for_day(day_df):
    trades = []
    sigs = detect_m_structure_signals(day_df)
    if not sigs:
        return trades
    n = len(day_df)
    day_open_spot = day_df["open"].iloc[0]
    strike_ce = top3.itm_call_strike(day_open_spot)
    strike_pe = top3.itm_put_strike(day_open_spot)
    premium_ce = top3.build_premium_series(day_df, strike_ce, day_open_spot, side="CE")
    premium_pe = top3.build_premium_series(day_df, strike_pe, day_open_spot, side="PE")
    lot = top3.lot_size_for_date(day_df.index[0].date())
    open_until_idx = -1
    for sig in sigs:
        j0 = sig["bar"]
        if j0 <= open_until_idx:
            continue
        exec_idx = j0 + HOLD_CONFIRM_BARS + 1
        if exec_idx >= n:
            continue
        strike = strike_ce if sig["side"] == "CE" else strike_pe
        premium_df = premium_ce if sig["side"] == "CE" else premium_pe
        entry_price = premium_df["open"].iloc[exec_idx]
        pos = {"entry_price": entry_price, "stop_level": entry_price - STRUCT_STOP_PTS,
               "peak": entry_price, "trailing_active": False, "last_checked_idx": exec_idx - 1}
        exit_info = top3.step_long_premium_position(pos, premium_df, STRUCT_TRAIL_PTS)
        if exit_info is None:
            exit_info = {"exit_price": premium_df["close"].iloc[-1], "reason": "EOD_FALLBACK",
                         "exit_time": premium_df.index[-1], "exit_idx": n - 1}
        pnl_pts = exit_info["exit_price"] - entry_price
        rupee_pnl = pnl_pts * lot - BROKERAGE_PER_TRADE
        trades.append({
            "strategy": "MStructure", "side": sig["side"], "strike": strike, "event": sig["event"],
            "breakout_level": sig["level"], "entry_time": premium_df.index[exec_idx],
            "exit_time": exit_info["exit_time"], "entry_price": entry_price, "exit_price": exit_info["exit_price"],
            "reason": exit_info["reason"], "pnl_pts": pnl_pts, "rupee_pnl": rupee_pnl,
            "date": day_df.index[0].date(),
        })
        open_until_idx = exit_info["exit_idx"]
    return trades


# =============================================================================
# OI soft-tagging (diagnostic only, never filters a trade)
# =============================================================================
def build_oi_regime_series(oi_df, index_price_df):
    """Join real futures OI onto the index price series (forward-filled),
    compute the price+OI quadrant regime over OI_CHANGE_LOOKBACK_MIN bars.
    Only covers whatever window real OI data exists for (~90 recent days)."""
    if oi_df.empty:
        return pd.DataFrame(columns=["oi_regime"])
    df = index_price_df[["close"]].join(oi_df[["oi"]], how="left")
    df["oi"] = df["oi"].ffill()
    df = df.dropna(subset=["oi"])
    oi_change = df["oi"].diff(OI_CHANGE_LOOKBACK_MIN)
    price_change = df["close"].diff(OI_CHANGE_LOOKBACK_MIN)
    df["oi_regime"] = [classify_oi_regime(pc, oc) for pc, oc in zip(price_change, oi_change)]
    return df[["oi_regime"]]


def tag_trades_with_oi(trades_df, oi_regime_df):
    if oi_regime_df.empty or trades_df.empty:
        trades_df["oi_regime"] = "NO_DATA"
        trades_df["oi_agrees"] = None
        return trades_df
    right = oi_regime_df.sort_index().reset_index()
    right.columns = ["entry_time"] + list(right.columns[1:])  # robust to whatever the index was named
    tagged = pd.merge_asof(
        trades_df.sort_values("entry_time"), right,
        on="entry_time", direction="backward", tolerance=pd.Timedelta(minutes=OI_CHANGE_LOOKBACK_MIN + 5),
    )
    tagged["oi_regime"] = tagged["oi_regime"].fillna("NO_DATA")

    def _agrees(row):
        if row["oi_regime"] == "NO_DATA":
            return None
        if row["side"] == "CE":
            return row["oi_regime"] == "LONG_BUILDUP"
        if row["side"] == "PE":
            return row["oi_regime"] == "SHORT_BUILDUP"
        return None

    tagged["oi_agrees"] = tagged.apply(_agrees, axis=1)
    return tagged.sort_values("entry_time").reset_index(drop=True)


# =============================================================================
# Self-checks -- hand-traced synthetic assertions, no network required
# =============================================================================
def run_self_checks():
    print("Running self-checks (synthetic data)...")
    top3.run_self_checks()  # re-verify the shared premium/position-management primitives first

    # --- structure levels: hand-traced rolling max/min with no look-ahead ---
    idx = pd.date_range("2025-01-06 09:15", periods=40, freq="1min")
    highs = [100 + (i % 5) for i in range(35)] + [100, 100, 100, 100, 108]  # spike at the very end
    lows = [95 - (i % 3) for i in range(35)] + [95, 95, 95, 95, 95]
    vols = [1000] * 39 + [5000]
    df = pd.DataFrame({
        "open": highs, "high": highs, "low": lows, "close": highs, "volume": vols,
    }, index=idx)
    lv = compute_structure_levels(df)
    # bar 39 (the spike) must NOT see its own high in res_level (shift(1) applied)
    assert lv["res_level"].iloc[39] == max(highs[9:39]), "res_level must exclude the current bar (look-ahead check)"
    print("  compute_structure_levels look-ahead safety (current bar excluded) ok")

    # --- breakout + hold-confirm: a real breakout that holds for 2 bars must fire;
    # an identical breakout that snaps back on hold-bar 2 must NOT fire ---
    n = STRUCT_LOOKBACK + 10
    idx2 = pd.date_range("2025-01-06 09:15", periods=n, freq="1min")
    base_h = [100.0] * STRUCT_LOOKBACK
    base_l = [95.0] * STRUCT_LOOKBACK
    base_v = [1000.0] * STRUCT_LOOKBACK
    # real breakout: bar STRUCT_LOOKBACK breaks 100 on high volume, next 2 bars hold above 100
    h1 = base_h + [105, 106, 107, 106, 106, 106, 106, 106, 106, 106]
    l1 = base_l + [99, 101, 102, 101, 101, 101, 101, 101, 101, 101]
    v1 = base_v + [3000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    df1 = pd.DataFrame({"open": h1, "high": h1, "low": l1, "close": h1, "volume": v1}, index=idx2)
    sigs1 = detect_m_structure_signals(df1)
    assert any(s["bar"] == STRUCT_LOOKBACK and s["side"] == "CE" for s in sigs1), (
        f"expected a confirmed CE breakout at bar {STRUCT_LOOKBACK}, got {sigs1}"
    )
    print("  real breakout + 2-bar hold confirmation fires ok")

    # fakeout: breaks 100 on volume, but snaps back below 100 on the 2nd hold bar
    h2 = base_h + [105, 106, 101, 106, 106, 106, 106, 106, 106, 106]
    l2 = base_l + [99, 101, 98, 101, 101, 101, 101, 101, 101, 101]  # bar +2's low (98) is back below 100
    v2 = v1
    df2 = pd.DataFrame({"open": h2, "high": h2, "low": l2, "close": h2, "volume": v2}, index=idx2)
    sigs2 = detect_m_structure_signals(df2)
    assert not any(s["bar"] == STRUCT_LOOKBACK for s in sigs2), (
        f"fakeout breakout (snaps back on hold bar 2) must NOT fire, got {sigs2}"
    )
    print("  fakeout breakout (fails 2-bar hold) correctly rejected ok")

    # --- rejection-hold filter: structure invalid if the recent bars already broke the old support ---
    # same breakout setup, but now the 5 most recent pre-breakout bars make a NEW LOWER low
    # than the older window -- support is being actively broken, not holding -- must NOT fire.
    base_l_broken = [95.0] * (STRUCT_LOOKBACK - REJECT_HOLD_BARS) + [90.0] * REJECT_HOLD_BARS
    df3 = pd.DataFrame({"open": h1, "high": h1, "low": base_l_broken + l1[STRUCT_LOOKBACK:], "close": h1, "volume": v1}, index=idx2)
    sigs3 = detect_m_structure_signals(df3)
    assert not any(s["bar"] == STRUCT_LOOKBACK and s["side"] == "CE" for s in sigs3), (
        f"support actively breaking (not holding) must block the long signal, got {sigs3}"
    )
    print("  rejection-hold filter blocks a long when support is actively breaking ok")

    # --- OI regime classification (reused verbatim) + trade tagging ---
    assert classify_oi_regime(10, 5000) == "LONG_BUILDUP"
    oi_idx = pd.date_range("2025-01-06 09:15", periods=20, freq="1min")
    oi_df = pd.DataFrame({"oi": [1000 + i * 50 for i in range(20)]}, index=oi_idx)
    price_df = pd.DataFrame({"close": [100 + i * 0.5 for i in range(20)]}, index=oi_idx)
    regime_df = build_oi_regime_series(oi_df, price_df)
    trades_df = pd.DataFrame([
        {"entry_time": oi_idx[19], "side": "CE", "rupee_pnl": 100},
        {"entry_time": pd.Timestamp("2025-02-01 09:15"), "side": "CE", "rupee_pnl": 50},  # outside OI window
    ])
    tagged = tag_trades_with_oi(trades_df, regime_df)
    assert tagged.iloc[0]["oi_regime"] == "LONG_BUILDUP" and tagged.iloc[0]["oi_agrees"] == True
    assert tagged.iloc[1]["oi_regime"] == "NO_DATA" and tagged.iloc[1]["oi_agrees"] is None
    print("  OI regime tagging (in-window agree + out-of-window NO_DATA) ok")

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


def run_full_pipeline(df1m, label="FULL 2-YEAR BACKTEST", fetch_oi=True):
    print(f"\n=== {label} ===")
    days_covered = sorted(set(df1m.index.date))

    all_trades = []
    for i, day in enumerate(days_covered):
        day_df = df1m[df1m.index.date == day]
        if len(day_df) >= STRUCT_LOOKBACK + REJECT_HOLD_BARS + HOLD_CONFIRM_BARS + 3:
            all_trades.extend(run_m_structure_standalone_for_day(day_df))
        if (i + 1) % 50 == 0:
            print(f"  ...scanned {i+1}/{len(days_covered)} days, {len(all_trades)} candidate trades so far")

    if not all_trades:
        print("No trades generated -- cannot proceed. This filter stack (structure + volume + "
              "rejection-hold + 2-bar confirm) may be too strict, or the levels/windows need widening.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)

    print(f"\n-- Entry-frequency diagnostics --")
    long_ct = (trades_df["side"] == "CE").sum(); short_ct = (trades_df["side"] == "PE").sum()
    traded_days = trades_df["date"].nunique()
    print(f"  {len(trades_df)} trades on {traded_days} of {len(days_covered)} days "
          f"({100*traded_days/len(days_covered):.1f}%)   CE breakouts={long_ct}  PE breakdowns={short_ct}")
    if traded_days / len(days_covered) > 0.95:
        print("  *** WARNING: firing on >95% of days -- this may have degenerated into something looser "
              "than the specific M/W structure it claims to test (same artifact pattern seen in the "
              "six-concepts screening pass on this project). Worth double-checking. ***")

    # --- OI soft-tagging ---
    if fetch_oi:
        print("\nFetching real NIFTY futures OI for soft-tagging (covers only the recent window it exists for)...")
        try:
            fut_id, fut_symbol, fut_expiry = find_current_nifty_future()
            oi_df = fetch_futures_oi_chunked(fut_id, FROM_DATE, TO_DATE)
        except Exception as e:
            print(f"  Could not fetch OI data ({e}) -- proceeding with all trades tagged NO_DATA.")
            oi_df = pd.DataFrame(columns=["oi"])
        regime_df = build_oi_regime_series(oi_df, df1m) if not oi_df.empty else pd.DataFrame(columns=["oi_regime"])
        trades_df = tag_trades_with_oi(trades_df, regime_df)
        n_with_oi = (trades_df["oi_regime"] != "NO_DATA").sum()
        print(f"  {n_with_oi} of {len(trades_df)} trades fall within the real-OI window "
              f"({100*n_with_oi/len(trades_df):.1f}%).")
        if n_with_oi > 0:
            agree = trades_df[trades_df["oi_agrees"] == True]
            disagree = trades_df[trades_df["oi_agrees"] == False]
            print(f"  Of those: {len(agree)} OI-agreed (net Rs{agree['rupee_pnl'].sum():,.2f}, "
                  f"win%={100*(agree['rupee_pnl']>0).mean():.1f})   "
                  f"{len(disagree)} OI-disagreed (net Rs{disagree['rupee_pnl'].sum():,.2f}, "
                  f"win%={100*(disagree['rupee_pnl']>0).mean() if len(disagree) else 0:.1f})")
    else:
        trades_df["oi_regime"] = "SKIPPED"
        trades_df["oi_agrees"] = None

    trades_df.to_csv(TRADES_CSV, index=False)
    print(f"\n{len(trades_df)} trades saved to {TRADES_CSV}")

    print(f"\n################ M-STRUCTURE -- HEADLINE RESULT ################")
    pf, sqn, n = compute_pf_sqn(trades_df)
    total_pnl = trades_df["rupee_pnl"].sum()
    win_rate = 100 * (trades_df["rupee_pnl"] > 0).mean()
    print(f"Trades: {n}   Win rate: {win_rate:.1f}%   PF: {pf:.2f}   SQN: {sqn:.2f}   Net P&L: Rs{total_pnl:,.2f}")
    if pf > 2.5 or sqn > 5:
        print("  *** RED FLAG per project rigor rule: PF>2.5 or SQN>5 -- treat as needing a bug hunt, "
              "NOT a real edge, before trusting this. (Calibration: real clean results here look like "
              "PF ~1.4-1.7, SQN ~2.5-3.) ***")

    long_trades = trades_df[trades_df["side"] == "CE"]
    short_trades = trades_df[trades_df["side"] == "PE"]
    for label, g in (("CE breakouts", long_trades), ("PE breakdowns", short_trades)):
        if len(g):
            pfx, sqnx, nx = compute_pf_sqn(g)
            print(f"  {label:<14} n={nx:<5} PF={pfx:5.2f} SQN={sqnx:5.2f} net=Rs{g['rupee_pnl'].sum():>10,.2f} "
                  f"win%={100*(g['rupee_pnl']>0).mean():.1f}")

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

    # capital adequacy check
    equity_curve = STARTING_CAPITAL + trades_df.sort_values("entry_time")["rupee_pnl"].cumsum()
    worst_dd = (equity_curve - equity_curve.cummax()).min()
    print(f"\nWorst point-in-time drawdown (trade-sequence, Rs{STARTING_CAPITAL:,} starting capital): "
          f"Rs{worst_dd:,.2f}")
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
    print(f"\n################ M-STRUCTURE -- FULL INSTITUTIONAL METRICS ################")
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
# Smoke test
# =============================================================================
def smoke_test():
    run_self_checks()
    print("=== SMOKE TEST: fetching ~35 real recent calendar days of Nifty data ===")
    to_date = dt.date.today().strftime("%Y-%m-%d")
    from_date = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    try:
        df1m = fetch_index_chunked_with_volume(from_date, to_date)
    except Exception as e:
        print(f"Could not fetch real data to smoke-test against: {e}")
        print("Self-checks above still passed (logic verified). This failure is specifically about "
              "reaching the Dhan API from wherever this was just run -- re-run on the machine with a "
              "working Dhan connection before trusting the smoke test fully.")
        return
    if df1m.empty:
        print("No data returned -- market may be fully closed with no recent data. Try again later.")
        return
    run_full_pipeline(df1m, label="SMOKE TEST (recent ~35 calendar days)", fetch_oi=False)
    print("\nSmoke test complete -- no crashes, all self-checks passed, full pipeline ran end-to-end "
          "on real data. (OI fetch skipped in smoke test to keep it fast -- runs in the full backtest.)")


def main():
    run_self_checks()
    print(f"=== M-STRUCTURE BREAKOUT STRATEGY -- 2-YEAR REAL BACKTEST ===")
    print("Fetching Nifty index 1-min data (2 years, with volume)...")
    df1m = fetch_index_chunked_with_volume(FROM_DATE, TO_DATE)
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
