"""Runs the 6-month offline backtest of the candle-analysis strategy against
real Nifty data, produces the locked-parameter headline result plus the
standard SL/target sweep, IS/OOS split, entry-frequency diagnostics, and a
report file. See candle_pa_6month_offline_backtest.py for the engine."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from candle_pa_6month_offline_backtest import (   # noqa: E402
    INPUT_FILE, SL_POINTS_DEFAULT, TP_POINTS_DEFAULT, CONCEPT_NAMES,
    attach_htf_trend, run_backtest, summarize, STARTING_CAPITAL,
    ASSUMED_DELTA, LOT_SIZE, LOTS_PER_TRADE, BROKERAGE_PER_ROUND_TRIP,
)

REPORT_FILE = Path(__file__).with_name("candle_pa_6month_report.txt")
TRADES_FILE = Path(__file__).with_name("candle_pa_6month_trades.csv")
SWEEP_FILE = Path(__file__).with_name("candle_pa_6month_sweep.csv")

SL_GRID = [4, 6, 8, 10, 12]
TP_GRID = [8, 10, 12, 16, 20, 24]


def load_data():
    frame = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    frame = frame.sort_values("datetime").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = frame.set_index("datetime")[["open", "high", "low", "close"]]
    df.index = df.index.tz_localize(None)
    df = attach_htf_trend(df)
    return df


def entry_frequency_diagnostics(trades_df, days_covered):
    if trades_df.empty:
        print("No trades -- cannot run diagnostics.")
        return
    trades_df = trades_df.copy()
    trades_df["date"] = trades_df["entry_time"].dt.date
    fire_days = trades_df["date"].nunique()
    print(f"  Days with >=1 signal: {fire_days} / {len(days_covered)} eligible days "
          f"({100*fire_days/len(days_covered):.1f}%)")
    print(f"  Direction split: LONG={int((trades_df['side']=='LONG').sum())}  "
          f"SHORT={int((trades_df['side']=='SHORT').sum())}")
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    clustering = trades_df["hour"].value_counts(normalize=True).sort_index()
    top_hour_pct = clustering.max() * 100
    print(f"  Entry-hour clustering: busiest hour holds {top_hour_pct:.1f}% of entries")
    print("  By concept:")
    for cid, g in trades_df.groupby("concept_id"):
        name = CONCEPT_NAMES.get(cid, str(cid))
        print(f"    {name:<32} trades={len(g)}")


def main():
    t0 = time.time()
    df = load_data()
    days_covered = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(days_covered)} trading days: "
          f"{days_covered[0]} to {days_covered[-1]}")

    # ---------- Headline: locked previous parameters (5 SL / 10 TP), no-lookahead fix ----------
    trades_fixed = run_backtest(df, SL_POINTS_DEFAULT, TP_POINTS_DEFAULT, fill_next_bar_open=True)
    trades_orig = run_backtest(df, SL_POINTS_DEFAULT, TP_POINTS_DEFAULT, fill_next_bar_open=False)
    s_fixed = summarize(trades_fixed, "5pt SL / 10pt TP, next-bar-open fill (rigor-fixed)")
    s_orig = summarize(trades_orig, "5pt SL / 10pt TP, same-bar-close fill (original, has look-ahead)")

    print(f"\nHeadline (previous 5/10, no-lookahead fill): trades={s_fixed['trades']} "
          f"win%={s_fixed['win_rate']:.1f} PF={s_fixed['pf']:.2f} SQN={s_fixed['sqn']:.2f} "
          f"net_pts={s_fixed['net_points']:.2f} net_Rs={s_fixed.get('net_rupee',0):,.2f} "
          f"max_dd_Rs={s_fixed.get('max_dd_rupee',0):,.2f}")
    print(f"Same-bar-close fill (original method) for comparison: trades={s_orig['trades']} "
          f"win%={s_orig['win_rate']:.1f} PF={s_orig['pf']:.2f} SQN={s_orig['sqn']:.2f} "
          f"net_pts={s_orig['net_points']:.2f} net_Rs={s_orig.get('net_rupee',0):,.2f}")

    print("\nEntry-frequency & clustering diagnostics (rigor-fixed run):")
    entry_frequency_diagnostics(trades_fixed, days_covered)

    # ---------- In-sample / out-of-sample split (first half vs second half by date) ----------
    if not trades_fixed.empty:
        trades_fixed_sorted = trades_fixed.sort_values("entry_time").reset_index(drop=True)
        mid_date = days_covered[len(days_covered) // 2]
        first_half = trades_fixed_sorted[trades_fixed_sorted["entry_time"].dt.date < mid_date]
        second_half = trades_fixed_sorted[trades_fixed_sorted["entry_time"].dt.date >= mid_date]
        s_first = summarize(first_half, "first half")
        s_second = summarize(second_half, "second half")
        print(f"\nIn-sample/out-of-sample split (cut at {mid_date}):")
        print(f"  First half:  trades={s_first['trades']} PF={s_first['pf']:.2f} "
              f"SQN={s_first['sqn']:.2f} net_Rs={s_first.get('net_rupee',0):,.2f}")
        print(f"  Second half: trades={s_second['trades']} PF={s_second['pf']:.2f} "
              f"SQN={s_second['sqn']:.2f} net_Rs={s_second.get('net_rupee',0):,.2f}")
    else:
        s_first = s_second = summarize(pd.DataFrame(), "n/a")

    # ---------- Standard SL x TP sweep ----------
    print(f"\nRunning SL x TP sweep ({len(SL_GRID)} x {len(TP_GRID)} = {len(SL_GRID)*len(TP_GRID)} combos)...")
    sweep_rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            tdf = run_backtest(df, sl, tp, fill_next_bar_open=True)
            s = summarize(tdf, f"sl{sl}_tp{tp}")
            sweep_rows.append(dict(sl=sl, tp=tp, **{k: v for k, v in s.items() if k != "label"}))
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_FILE, index=False)
    best = sweep_df.sort_values("sqn", ascending=False).iloc[0]
    print(f"Best combo by SQN: SL={best['sl']} TP={best['tp']} -> "
          f"PF={best['pf']:.2f} SQN={best['sqn']:.2f} trades={int(best['trades'])} "
          f"net_Rs={best.get('net_rupee',0):,.2f}")

    # grid-edge check
    edge = (best["sl"] in (min(SL_GRID), max(SL_GRID))) or (best["tp"] in (min(TP_GRID), max(TP_GRID)))
    print(f"Grid-edge check: best combo at grid boundary = {edge}")

    # ---------- Capital adequacy ----------
    if not trades_fixed.empty:
        rupee = (trades_fixed["points"] * ASSUMED_DELTA * LOT_SIZE * LOTS_PER_TRADE - BROKERAGE_PER_ROUND_TRIP)
        capital_curve = STARTING_CAPITAL + rupee.cumsum()
        wipeout = (capital_curve <= 0).any()
        print(f"\nCapital adequacy (Rs {STARTING_CAPITAL:,} start, {LOT_SIZE} lot size, "
              f"{ASSUMED_DELTA} assumed delta): wipeout at any point = {wipeout}")
        if wipeout:
            idx_w = int((capital_curve <= 0).idxmax())
            print(f"  Capital would hit <=0 around trade #{idx_w+1} of {len(trades_fixed)}")

    trades_fixed.to_csv(TRADES_FILE, index=False)

    # ---------- Report file ----------
    lines = [
        "Candle-Analysis (Price Action & Market Structure) Strategy -- 6-Month Offline Backtest",
        "",
        f"Data: {INPUT_FILE.name}, Nifty 1-min, {days_covered[0]} to {days_covered[-1]} "
        f"({len(days_covered)} trading days, ~6 months)",
        "Concepts active: 2 (Compression Spike, sell) and 4 (Rollover, sell) only -- matches the",
        "already-locked dhan_live_bot.py configuration; concepts 1,3,5,6,7,8,9 disabled per prior testing.",
        "Filters: Tuesday/Wednesday only, 5-min + 15-min EMA(9/21) trend must agree with signal direction.",
        "",
        "=== Headline: previous stop-loss / target (5pt SL / 10pt TP, 1:2 R:R) ===",
        f"Rigor-fixed (next-bar-open fill, no look-ahead): trades={s_fixed['trades']}, "
        f"win rate={s_fixed['win_rate']:.1f}%, PF={s_fixed['pf']:.2f}, SQN={s_fixed['sqn']:.2f}, "
        f"net points={s_fixed['net_points']:.2f}, net P&L (proxy)=Rs {s_fixed.get('net_rupee',0):,.2f}, "
        f"max drawdown (proxy)=Rs {s_fixed.get('max_dd_rupee',0):,.2f}",
        f"Original method (same-bar-close fill, has look-ahead -- reported for comparison only): "
        f"trades={s_orig['trades']}, PF={s_orig['pf']:.2f}, SQN={s_orig['sqn']:.2f}, "
        f"net points={s_orig['net_points']:.2f}, net P&L (proxy)=Rs {s_orig.get('net_rupee',0):,.2f}",
        "",
        "=== In-sample / out-of-sample split ===",
        f"First half:  trades={s_first['trades']}, PF={s_first['pf']:.2f}, SQN={s_first['sqn']:.2f}, "
        f"net P&L (proxy)=Rs {s_first.get('net_rupee',0):,.2f}",
        f"Second half: trades={s_second['trades']}, PF={s_second['pf']:.2f}, SQN={s_second['sqn']:.2f}, "
        f"net P&L (proxy)=Rs {s_second.get('net_rupee',0):,.2f}",
        "",
        f"=== SL x TP sweep ({len(SL_GRID)}x{len(TP_GRID)} grid) ===",
        f"Best by SQN: SL={best['sl']} / TP={best['tp']} -> PF={best['pf']:.2f}, SQN={best['sqn']:.2f}, "
        f"trades={int(best['trades'])}, net P&L (proxy)=Rs {best.get('net_rupee',0):,.2f}",
        f"Grid-edge check: best combo at a grid boundary = {edge}",
        "Full grid saved to candle_pa_6month_sweep.csv",
        "",
        "=== Money-simulation assumptions (proxy, no real option-chain data available) ===",
        f"Starting capital Rs {STARTING_CAPITAL:,}, Nifty lot size {LOT_SIZE}, assumed delta {ASSUMED_DELTA}, "
        f"{LOTS_PER_TRADE} lot/trade, brokerage Rs {BROKERAGE_PER_ROUND_TRIP:.0f}/round trip.",
        "Option premium move is approximated as underlying point move x assumed delta -- NOT real option prices.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {REPORT_FILE.name}, {TRADES_FILE.name}, {SWEEP_FILE.name}")
    print(f"Total run time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
