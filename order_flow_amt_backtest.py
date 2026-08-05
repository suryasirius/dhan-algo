"""
Order Flow / Auction Market Theory (AMT) Scalping Model -- Nifty 1-min adaptation
===================================================================================
Standalone -- does not touch dhan_live_bot.py / backtest.py / FINAL_*.py or any
other concept script.

Good news from the diagnostic check: Dhan DOES return real, populated volume
for the Nifty index feed (confirmed: 0% zero-volume candles, mean ~907k).
So the volume-profile + LVN core of this model CAN be built. But the doc's
full trigger needs "footprint"/individual large-order print data, which we
do NOT have (we only have 1-min OHLCV bars, i.e. volume summed over the
minute, not broken into individual trade sizes). Per the doc's own section
12 ("Next Steps"), this is built as the suggested BASELINE: volume-profile +
LVN logic, with a volume-spike proxy standing in for "large order aggression"
since real order/print sizes aren't available. Read this list of adaptation
choices before trusting the results -- these are all judgment calls, not
things the source doc specifies for OHLCV-only data:

  1. "Aggression" proxy: a bar's volume >= AGGRESSION_VOL_MULT x its trailing
     20-bar average volume, at/near a validated LVN level. This is a stand-in
     for the doc's "large print/contract count" trigger, which needs
     tick/footprint data we don't have.
  2. Volume profile: each bar's total volume is bucketed into a price bin
     (width BIN_WIDTH_PTS) at that bar's typical price = (high+low+close)/3.
     This is an approximation -- true volume profiles use every individual
     trade price, not one bucket per 1-min bar.
  3. Market state (balance/imbalance): objective rule from the doc's own
     "Backtest rule" for Step 1 -- current price beyond IMBALANCE_STD_MULT
     standard deviations of the trailing volume-weighted average price (VWAP)
     = imbalance; inside that band = balance. Recomputed every bar from a
     rolling PROFILE_LOOKBACK_MIN-bar window (no lookahead -- window ends the
     bar BEFORE the current one).
  4. LVN (low-volume node): any price bin in the bottom LVN_PERCENTILE% of
     the trailing profile's volume distribution -- exactly as the doc's own
     "Backtest rule" for Step 2 states.
  5. Session mapping DROPPED: the doc splits Trend model -> New York session
     and Mean-reversion model -> London session. Nifty only trades in IST
     daytime hours, so there's no equivalent. Instead: Trend model fires
     whenever the OBJECTIVE state reads imbalance, Mean-reversion fires
     whenever it reads balance, at ANY time of day -- and we still do the
     usual 30-min session breakdown in the report, so we can see empirically
     if particular times of day work better (replacing geography with data).
  6. "Second drive only" for mean-reversion: a "drive" = price closes beyond
     the value-area edge (imbalance) then closes back inside it (reverts) --
     counted per day, per direction. Only the 2nd such completed drive is
     eligible to trigger a mean-reversion entry (matches the doc exactly).
  7. Target = the nearest point-of-control (POC, highest-volume bin) ahead of
     entry in the trade's favor, from the same trailing profile, IF it clears
     the doc's MIN_RR minimum (1:2.5). If no such POC clears that minimum,
     the trade is skipped entirely (source is explicit: "do not force large
     R:R when volatility doesn't support it" -- so we don't force a fallback
     target either, we just don't take that trade).
  8. Stop = trigger level (LVN price or value-area edge) +/- STOP_BUFFER_PTS,
     tight, matching "1-2 ticks beyond the level, not at it."
  9. Move to break-even after BREAKEVEN_AFTER_BARS confirming follow-through
     candle(s) -- doc says "as soon as a confirming candle appears," taken
     literally as 1 bar.
  10. Same-day only: any open position is closed at day-end (EOD), matching
      every other concept tested in this project (no overnight carry).
  11. Per-trade brokerage of Rs 40 (project's standing placeholder, per the
      doc's own explicit warning that commissions compound heavily at this
      frequency) is applied in the money simulation.
  12. Regime memory (found and fixed during testing, before this was ever
      run on real data): state is reclassified fresh every bar from a
      rolling trailing window, which means a normal pullback naturally
      drags the trailing average toward it -- so by the exact bar price
      reaches the pullback/LVN level, a pure bar-by-bar read would often
      already say "balance" (or even the opposite imbalance), and the
      Trend model's "pullback while still trending" entry would almost
      never fire. Fixed with a short memory: once IMBALANCE_UP/DOWN is
      read, that direction stays "active" through up to IMBALANCE_MEMORY_BARS
      of pullback bars before being considered over -- confirmed via a
      synthetic pullback scenario that it now fires as intended.

If this baseline shows any real edge, the NEXT step (per the doc's own
suggestion) would be layering in Cumulative Volume Delta (CVD) as a
secondary confirmation, which needs a slightly different data pull -- not
attempted yet, deliberately, until we know the baseline is worth building on.

A few tiny synthetic self-checks run automatically at the top of main() to
sanity-check the volume-profile/LVN/state functions before touching real
data (same rigor as previous concepts) -- if any fail, the script stops
before wasting time on the full backtest.
"""

import datetime as dt
import pandas as pd
import numpy as np

from dhan_live_bot import (
    dhan, UNDERLYING_SECURITY_ID, UNDERLYING_EXCHANGE_SEGMENT, UNDERLYING_INSTRUMENT_TYPE,
)

FROM_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year
TO_DATE = dt.date.today().strftime("%Y-%m-%d")
CHUNK_DAYS = 85

PROFILE_LOOKBACK_MIN = 60      # trailing bars used to build the rolling volume profile
BIN_WIDTH_PTS = 5              # price bin width for the volume profile
LVN_PERCENTILE = 10            # bottom 10% of volume bins = low-volume node
IMBALANCE_STD_MULT = 1.0       # beyond 1 std dev of trailing VWAP = imbalance
AGGRESSION_VOL_MULT = 1.5      # bar volume vs its trailing 20-bar average = "aggression"
AGGR_VOL_LOOKBACK = 20
LVN_TOUCH_TOLERANCE_PTS = 3    # how close price must get to an LVN bin to count as "at" it

STOP_BUFFER_PTS = 2            # tight stop beyond the trigger level
BREAKEVEN_AFTER_BARS = 1       # move stop to entry after this many confirming bars
MIN_RR = 2.5                   # minimum acceptable reward:risk (doc's stated minimum)
IMBALANCE_MEMORY_BARS = 10     # see note below -- how long an imbalance "regime" is remembered
                                # through a shallow pullback before it's considered over

BROKERAGE_PER_TRADE = 40.0
STARTING_CAPITAL = 10_000
LOT_SIZE = 65
ASSUMED_DELTA = 0.6

SESSIONS = [
    "09:15-09:45", "09:45-10:15", "10:15-10:45", "10:45-11:15",
    "11:15-11:45", "11:45-12:15", "12:15-12:45", "12:45-13:15",
    "13:15-13:45", "13:45-14:15", "14:15-14:45", "14:45-15:15", "15:15-15:30",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def fetch_historical(from_date, to_date):
    print(f"Requesting Nifty 1-min candles from {from_date} to {to_date}...")
    resp = dhan.intraday_minute_data(
        security_id=UNDERLYING_SECURITY_ID,
        exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
        instrument_type=UNDERLYING_INSTRUMENT_TYPE,
        from_date=from_date,
        to_date=to_date,
    )
    if isinstance(resp, dict) and resp.get("status") == "failure":
        print("\nDhan returned an error:")
        print(resp)
        raise SystemExit(1)
    data = resp.get("data", resp)
    df = pd.DataFrame({
        "open": data["open"], "high": data["high"],
        "low": data["low"], "close": data["close"],
        "volume": data["volume"], "timestamp": data["timestamp"],
    })
    df["time"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta(hours=5, minutes=30)
    return df.set_index("time").sort_index()


def fetch_historical_chunked(from_date, to_date, chunk_days=CHUNK_DAYS):
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        chunks.append(fetch_historical(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + dt.timedelta(days=1)
    full = pd.concat(chunks)
    return full[~full.index.duplicated(keep="first")].sort_index()


# ---------------- Core AMT building blocks ----------------

def build_volume_profile(typical_prices, volumes, bin_width=BIN_WIDTH_PTS):
    """Returns dict {bin_center_price: total_volume} from a trailing window."""
    profile = {}
    for p, v in zip(typical_prices, volumes):
        b = round(p / bin_width) * bin_width
        profile[b] = profile.get(b, 0) + v
    return profile


def get_poc(profile):
    if not profile:
        return None
    return max(profile.items(), key=lambda kv: kv[1])[0]


def get_lvn_levels(profile, percentile=LVN_PERCENTILE):
    if not profile:
        return []
    vols = sorted(profile.values())
    if not vols:
        return []
    idx = max(0, int(len(vols) * percentile / 100) - 1)
    threshold = vols[idx]
    return [price for price, v in profile.items() if v <= threshold]


def get_poc_ahead(profile, price, direction):
    """Highest-volume bin AHEAD of current price in the given direction ("up"/"down").
    Used for the Trend model's target: the single global POC can sit BEHIND price
    (in the range it just broke out of), which would never be a valid forward target
    for a continuation trade -- found via testing before this was ever run live."""
    if direction == "up":
        candidates = {p: v for p, v in profile.items() if p > price}
    else:
        candidates = {p: v for p, v in profile.items() if p < price}
    if not candidates:
        return None
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def classify_state(close, vwap, std, mult=IMBALANCE_STD_MULT):
    if std <= 0:
        return "BALANCE"
    if close > vwap + mult * std:
        return "IMBALANCE_UP"
    if close < vwap - mult * std:
        return "IMBALANCE_DOWN"
    return "BALANCE"


def nearest_lvn(price, lvn_levels, tolerance=LVN_TOUCH_TOLERANCE_PTS):
    candidates = [lvl for lvl in lvn_levels if abs(lvl - price) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl - price))


def run_self_checks():
    print("Running self-checks on core AMT functions (synthetic data)...")

    # Volume profile + POC: one heavy cluster should win as POC.
    prices = [100, 100, 100, 105, 110, 100, 100]
    vols = [1000, 1000, 1000, 50, 50, 1000, 1000]
    profile = build_volume_profile(prices, vols)
    assert get_poc(profile) == 100, f"POC test failed: {get_poc(profile)}"

    # LVN: the 105/110 bins have low volume relative to the 100 bin -> should be flagged.
    lvns = get_lvn_levels(profile, percentile=40)
    assert 105 in lvns or 110 in lvns, f"LVN test failed: {lvns}"
    assert 100 not in lvns, f"LVN test wrongly flagged the high-volume bin: {lvns}"

    # State classification.
    assert classify_state(120, 100, 10) == "IMBALANCE_UP"
    assert classify_state(80, 100, 10) == "IMBALANCE_DOWN"
    assert classify_state(105, 100, 10) == "BALANCE"

    # Nearest LVN with tolerance.
    assert nearest_lvn(107, [105, 200], tolerance=3) == 105
    assert nearest_lvn(150, [105, 200], tolerance=3) is None

    # POC-ahead: must ignore a heavier bin that's BEHIND price in the given direction.
    profile2 = {100: 2000, 150: 500}  # 100 is heavier overall, but behind price=120 going "up"
    assert get_poc_ahead(profile2, price=120, direction="up") == 150
    assert get_poc_ahead(profile2, price=120, direction="down") == 100
    assert get_poc_ahead(profile2, price=200, direction="up") is None  # nothing ahead

    print("All self-checks passed.\n")


# ---------------- Day simulation ----------------

def simulate_day_amt(day_df):
    opens = day_df["open"].tolist()
    highs = day_df["high"].tolist()
    lows = day_df["low"].tolist()
    closes = day_df["close"].tolist()
    vols = day_df["volume"].tolist()
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    idx = day_df.index
    n = len(day_df)

    trades = []
    position = None

    drive_count_up = 0
    drive_count_down = 0
    prev_state = "BALANCE"
    active_imbalance_dir = None   # "IMBALANCE_UP" / "IMBALANCE_DOWN" / None -- see IMBALANCE_MEMORY_BARS note
    bars_since_imbalance = 0

    for j in range(PROFILE_LOOKBACK_MIN, n):
        window_slice = slice(j - PROFILE_LOOKBACK_MIN, j)  # trailing, excludes bar j (no lookahead)
        w_typical = typical[window_slice]
        w_vols = vols[window_slice]

        total_vol = sum(w_vols)
        if total_vol <= 0:
            continue
        vwap = sum(p * v for p, v in zip(w_typical, w_vols)) / total_vol
        variance = sum(v * (p - vwap) ** 2 for p, v in zip(w_typical, w_vols)) / total_vol
        std = variance ** 0.5

        profile = build_volume_profile(w_typical, w_vols)
        lvn_levels = get_lvn_levels(profile)
        poc = get_poc(profile)

        avg_recent_vol_window = vols[max(0, j - AGGR_VOL_LOOKBACK):j]
        avg_recent_vol = sum(avg_recent_vol_window) / len(avg_recent_vol_window) if avg_recent_vol_window else 0
        is_aggression_bar = avg_recent_vol > 0 and vols[j] >= AGGRESSION_VOL_MULT * avg_recent_vol

        c = closes[j]
        state = classify_state(c, vwap, std)

        # --- update imbalance regime memory (see IMBALANCE_MEMORY_BARS note) ---
        if state in ("IMBALANCE_UP", "IMBALANCE_DOWN"):
            active_imbalance_dir = state
            bars_since_imbalance = 0
        else:
            bars_since_imbalance += 1
            if bars_since_imbalance > IMBALANCE_MEMORY_BARS:
                active_imbalance_dir = None

        # --- manage open position first ---
        if position is not None:
            exited, reason = False, None
            if position["side"] == "LONG":
                if position["breakeven_armed"] and (j - position["entry_bar"]) >= BREAKEVEN_AFTER_BARS \
                        and not position["at_breakeven"] and c > position["entry"]:
                    position["stop"] = max(position["stop"], position["entry"])
                    position["at_breakeven"] = True
                if c <= position["stop"]:
                    exited, reason = True, "STOP" if not position["at_breakeven"] else "BREAKEVEN_STOP"
                elif c >= position["target"]:
                    exited, reason = True, "TARGET"
            else:
                if position["breakeven_armed"] and (j - position["entry_bar"]) >= BREAKEVEN_AFTER_BARS \
                        and not position["at_breakeven"] and c < position["entry"]:
                    position["stop"] = min(position["stop"], position["entry"])
                    position["at_breakeven"] = True
                if c >= position["stop"]:
                    exited, reason = True, "STOP" if not position["at_breakeven"] else "BREAKEVEN_STOP"
                elif c <= position["target"]:
                    exited, reason = True, "TARGET"
            if exited:
                pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": idx[j], "side": position["side"],
                    "model": position["model"], "entry": position["entry"], "exit": c,
                    "points": pts, "reason": reason,
                })
                position = None

        # --- track "drives" for mean-reversion (2nd-drive-only rule) ---
        if prev_state == "IMBALANCE_UP" and state == "BALANCE":
            drive_count_up += 1
        if prev_state == "IMBALANCE_DOWN" and state == "BALANCE":
            drive_count_down += 1
        if state == "IMBALANCE_UP":
            drive_count_down = 0  # a fresh strong move resets the opposite counter
        if state == "IMBALANCE_DOWN":
            drive_count_up = 0

        # --- look for a new entry only if flat ---
        if position is None:
            o, h, l = opens[j], highs[j], lows[j]

            # TREND MODEL: fires while an imbalance regime is active (allowing for a shallow
            # pullback, per IMBALANCE_MEMORY_BARS), continuation direction, on a pullback to an
            # LVN with aggression. Also require the CURRENT bar not be showing the opposite
            # imbalance (that would mean the move has genuinely reversed, not just pulled back).
            if active_imbalance_dir == "IMBALANCE_UP" and state != "IMBALANCE_DOWN":
                lvl = nearest_lvn(l, lvn_levels)
                if lvl is not None and is_aggression_bar and c > o:
                    stop = lvl - STOP_BUFFER_PTS
                    risk = c - stop
                    target_poc = get_poc_ahead(profile, c, "up")
                    if risk > 0 and target_poc is not None:
                        reward = target_poc - c
                        if reward / risk >= MIN_RR:
                            position = {"side": "LONG", "model": "Trend", "entry": c, "entry_time": idx[j],
                                        "entry_bar": j, "stop": stop, "target": target_poc,
                                        "breakeven_armed": True, "at_breakeven": False}
            elif active_imbalance_dir == "IMBALANCE_DOWN" and state != "IMBALANCE_UP":
                lvl = nearest_lvn(h, lvn_levels)
                if lvl is not None and is_aggression_bar and c < o:
                    stop = lvl + STOP_BUFFER_PTS
                    risk = stop - c
                    target_poc = get_poc_ahead(profile, c, "down")
                    if risk > 0 and target_poc is not None:
                        reward = c - target_poc
                        if reward / risk >= MIN_RR:
                            position = {"side": "SHORT", "model": "Trend", "entry": c, "entry_time": idx[j],
                                        "entry_bar": j, "stop": stop, "target": target_poc,
                                        "breakeven_armed": True, "at_breakeven": False}

            # MEAN-REVERSION MODEL: fires in balance, only on the 2nd completed drive, fading back to POC.
            elif state == "BALANCE":
                if drive_count_up >= 2:
                    lvl = nearest_lvn(h, lvn_levels)
                    if lvl is not None and is_aggression_bar and c < o:
                        stop = lvl + STOP_BUFFER_PTS
                        risk = stop - c
                        if risk > 0 and poc is not None and poc < c:
                            reward = c - poc
                            if reward / risk >= MIN_RR:
                                position = {"side": "SHORT", "model": "MeanReversion", "entry": c, "entry_time": idx[j],
                                            "entry_bar": j, "stop": stop, "target": poc,
                                            "breakeven_armed": True, "at_breakeven": False}
                                drive_count_up = 0
                elif drive_count_down >= 2:
                    lvl = nearest_lvn(l, lvn_levels)
                    if lvl is not None and is_aggression_bar and c > o:
                        stop = lvl - STOP_BUFFER_PTS
                        risk = c - stop
                        if risk > 0 and poc is not None and poc > c:
                            reward = poc - c
                            if reward / risk >= MIN_RR:
                                position = {"side": "LONG", "model": "MeanReversion", "entry": c, "entry_time": idx[j],
                                            "entry_bar": j, "stop": stop, "target": poc,
                                            "breakeven_armed": True, "at_breakeven": False}
                                drive_count_down = 0

        prev_state = state

    if position is not None:
        c = closes[-1]
        pts = (c - position["entry"]) if position["side"] == "LONG" else (position["entry"] - c)
        trades.append({
            "entry_time": position["entry_time"], "exit_time": idx[-1], "side": position["side"],
            "model": position["model"], "entry": position["entry"], "exit": c,
            "points": pts, "reason": "EOD",
        })

    return trades


def classify_30min_session(ts):
    t = ts.time()
    for label in SESSIONS:
        start_s, end_s = label.split("-")
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        if dt.time(sh, sm) <= t < dt.time(eh, em):
            return label
    return "Outside usual session"


def report(trades_df, label):
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]
    print(f"\n===== {label} =====")
    print(f"Total trades: {len(trades_df)}   Win rate: {100*len(wins)/len(trades_df):.1f}%   "
          f"Net points: {trades_df['points'].sum():.2f}")
    if len(wins):
        print(f"Avg win: {wins['points'].mean():.2f}", end="  ")
    if len(losses):
        print(f"Avg loss: {losses['points'].mean():.2f}", end="")
    print()
    print(f"Exit reasons: {trades_df['reason'].value_counts().to_dict()}")
    print(f"By side: {trades_df.groupby('side')['points'].agg(['count','sum']).to_dict()}")


def main():
    run_self_checks()

    print("=== Order Flow / AMT Scalping Model (baseline: volume profile + LVN, no footprint data) ===\n")
    df = fetch_historical_chunked(FROM_DATE, TO_DATE)
    if df.empty:
        print("No data returned -- check your Data API subscription and date range.")
        return

    days_covered = sorted(set(df.index.date))
    print(f"Got {len(df)} candles across {len(days_covered)} trading day(s): "
          f"{days_covered[0]} to {days_covered[-1]}\n")

    all_trades = []
    for day in days_covered:
        day_df = df[df.index.date == day]
        if len(day_df) < PROFILE_LOOKBACK_MIN + 10:
            continue
        all_trades.extend(simulate_day_amt(day_df))

    if not all_trades:
        print("No trades were generated at all. The state/LVN/aggression/min-R:R filters "
              "stacked together may be too strict for this data -- let me know and I can loosen them.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    trades_df.to_csv("order_flow_amt_trades.csv", index=False)
    print(f"Saved {len(trades_df)} trades to order_flow_amt_trades.csv\n")

    report(trades_df, "ALL TRADES (Trend + Mean-Reversion combined)")

    for model_name in ["Trend", "MeanReversion"]:
        sub = trades_df[trades_df["model"] == model_name]
        if len(sub) > 0:
            report(sub, f"MODEL: {model_name}")
        else:
            print(f"\n===== MODEL: {model_name} =====\nNo trades.")

    trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
    print("\n--- By weekday ---")
    for wd in WEEKDAY_ORDER:
        g = trades_df[trades_df["weekday"] == wd]
        if len(g):
            print(f"{wd}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    trades_df["session"] = trades_df["entry_time"].apply(classify_30min_session)
    print("\n--- By 30-min session ---")
    for s in SESSIONS + ["Outside usual session"]:
        g = trades_df[trades_df["session"] == s]
        if len(g):
            print(f"{s}: {len(g)} trades, net {g['points'].sum():.2f} pts, "
                  f"win rate {100*(g['points']>0).mean():.1f}%")

    # --- Money simulation with brokerage ---
    rupee_pnl_gross = trades_df["points"] * ASSUMED_DELTA * LOT_SIZE
    rupee_pnl_net = rupee_pnl_gross - BROKERAGE_PER_TRADE
    trades_df["rupee_pnl_net"] = rupee_pnl_net
    trades_df["running_capital"] = STARTING_CAPITAL + rupee_pnl_net.cumsum()
    trades_df.to_csv("order_flow_amt_trades.csv", index=False)

    final_capital = trades_df["running_capital"].iloc[-1]
    max_dd = (trades_df["running_capital"] - trades_df["running_capital"].cummax()).min()
    print(f"\n--- Money simulation (Rs {STARTING_CAPITAL:,} start, Rs {BROKERAGE_PER_TRADE:.0f}/trade brokerage) ---")
    print(f"Ending capital: Rs {final_capital:,.2f}   Max drawdown: Rs {max_dd:,.2f}")
    if (trades_df["running_capital"] <= 0).any():
        wipeout_idx = trades_df[trades_df["running_capital"] <= 0].index[0]
        print(f"WARNING: capital wipes out at trade #{wipeout_idx+1} "
              f"({trades_df.loc[wipeout_idx,'entry_time']})")
    else:
        print("Never wipes out across the full period.")

    # --- Stability check: first half vs second half ---
    mid = len(trades_df) // 2
    h1, h2 = trades_df.iloc[:mid], trades_df.iloc[mid:]
    print(f"\n--- Stability check ---")
    print(f"First half:  {len(h1)} trades, net Rs {h1['rupee_pnl_net'].sum():,.2f}")
    print(f"Second half: {len(h2)} trades, net Rs {h2['rupee_pnl_net'].sum():,.2f}")


if __name__ == "__main__":
    main()
