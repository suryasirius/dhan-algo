"""One-day Nifty ATM-option proxy backtest using the forming 15-minute candle direction."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from dhan_config import TRADE_QTY


INPUT_FILE = Path(__file__).with_name("dhan_historical_data_180days_v5.csv")
BACKTEST_DATE: str | None = None
SIGNAL_CANDLES = 5
STRIKE_STEP = 50
ATM_DELTA = 0.50
ASSUMED_ATM_PREMIUM = 100.0
BROKERAGE_PER_ROUND_TRIP = 40.0
TRADES_FILE = Path(__file__).with_name("htf_atm_option_1day_trades_v1.csv")
REPORT_FILE = Path(__file__).with_name("htf_atm_option_1day_report_v1.txt")


def nearest_atm_strike(spot_price: float) -> int:
    return int(math.floor((spot_price + STRIKE_STEP / 2) / STRIKE_STEP) * STRIKE_STEP)


def choose_backtest_day(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame["trade_date"] = frame["datetime"].dt.date.astype(str)
    chosen_date = BACKTEST_DATE or frame["trade_date"].max()
    day_frame = frame.loc[frame["trade_date"] == chosen_date].copy()
    if day_frame.empty:
        raise SystemExit(f"No Dhan candles found for {chosen_date}.")
    return day_frame, chosen_date


def simulate_trade(candles: pd.DataFrame, htf_start: pd.Timestamp, test_date: str) -> dict[str, object] | None:
    if len(candles) != 15:
        return None

    signal_candle = candles.iloc[SIGNAL_CANDLES - 1]
    entry_candle = candles.iloc[SIGNAL_CANDLES]
    exit_candle = candles.iloc[-1]
    htf_open = float(candles.iloc[0]["open"])
    signal_close = float(signal_candle["close"])
    signal_body = signal_close - htf_open
    if signal_body == 0:
        return None

    option_type = "CE" if signal_body > 0 else "PE"
    direction = 1 if option_type == "CE" else -1
    entry_spot = float(entry_candle["open"])
    exit_spot = float(exit_candle["close"])
    strike = nearest_atm_strike(entry_spot)
    option_entry = ASSUMED_ATM_PREMIUM
    option_exit = max(option_entry + direction * (exit_spot - entry_spot) * ATM_DELTA, 0.0)
    option_points = option_exit - option_entry
    gross_pnl = option_points * TRADE_QTY
    net_pnl = gross_pnl - BROKERAGE_PER_ROUND_TRIP
    spot_change = direction * (exit_spot - entry_spot)

    return {
        "date": test_date,
        "htf_start": htf_start,
        "signal_time": signal_candle["datetime"],
        "entry_time": entry_candle["datetime"],
        "exit_time": exit_candle["datetime"] + pd.Timedelta(minutes=1),
        "option_type": option_type,
        "atm_strike": strike,
        "signal_body_points": signal_body,
        "entry_spot": entry_spot,
        "exit_spot": exit_spot,
        "directional_spot_change": spot_change,
        "option_entry_proxy": option_entry,
        "option_exit_proxy": option_exit,
        "option_points_proxy": option_points,
        "quantity": TRADE_QTY,
        "gross_pnl_proxy": gross_pnl,
        "brokerage": BROKERAGE_PER_ROUND_TRIP,
        "net_pnl_proxy": net_pnl,
        "exit_reason": "15-minute candle close",
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file was not found: {INPUT_FILE.name}")

    frame = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    required_columns = {"datetime", "open", "high", "low", "close"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Input file is missing columns: {', '.join(missing_columns)}")

    frame = frame.sort_values("datetime").drop_duplicates(subset="timestamp").reset_index(drop=True)
    day_frame, test_date = choose_backtest_day(frame)
    day_frame["htf_start"] = day_frame["datetime"].dt.floor("15min")
    trades: list[dict[str, object]] = []

    for htf_start, candles in day_frame.groupby("htf_start", sort=True):
        trade = simulate_trade(candles.reset_index(drop=True), htf_start, test_date)
        if trade is not None:
            trades.append(trade)

    if not trades:
        raise SystemExit("No complete 15-minute candles with a non-zero five-minute direction were found.")

    trade_frame = pd.DataFrame(trades)
    trade_frame["cumulative_net_pnl_proxy"] = trade_frame["net_pnl_proxy"].cumsum()
    running_peak = trade_frame["cumulative_net_pnl_proxy"].cummax()
    trade_frame["drawdown_proxy"] = trade_frame["cumulative_net_pnl_proxy"] - running_peak
    trade_frame.to_csv(TRADES_FILE, index=False)

    wins = trade_frame["net_pnl_proxy"] > 0
    report_lines = [
        "Nifty Forming-15-Minute Candle ATM Option Backtest",
        "",
        f"Backtest date: {test_date}",
        f"Data: Dhan Nifty one-minute candles from {INPUT_FILE.name}",
        f"Completed 15-minute candles traded: {len(trade_frame)}",
        "",
        "Strategy:",
        f"- After {SIGNAL_CANDLES} completed one-minute candles, buy an ATM CE when the forming 15-minute candle is green; otherwise buy an ATM PE when it is red.",
        "- Enter at the open of the next one-minute candle and exit at the close of the 15-minute candle.",
        "- One independent trade per complete 15-minute candle.",
        "",
        "Results:",
        f"- Wins: {int(wins.sum())} / {len(trade_frame)} ({wins.mean() * 100:.2f}%)",
        f"- Gross proxy P&L: Rs {trade_frame['gross_pnl_proxy'].sum():,.2f}",
        f"- Brokerage assumed: Rs {trade_frame['brokerage'].sum():,.2f}",
        f"- Net proxy P&L: Rs {trade_frame['net_pnl_proxy'].sum():,.2f}",
        f"- Average proxy P&L per trade: Rs {trade_frame['net_pnl_proxy'].mean():,.2f}",
        f"- Maximum proxy drawdown: Rs {trade_frame['drawdown_proxy'].min():,.2f}",
        "",
        "Important limitation:",
        "- Dhan historical option premiums are unavailable in this project (DH-901), so this is not real option-premium P&L.",
        f"- The option result is a proxy using {ATM_DELTA:.2f} ATM delta and Rs {ASSUMED_ATM_PREMIUM:.2f} starting premium. It excludes theta, implied-volatility changes, bid-ask spread, and slippage.",
        "- This single-day result is only a workflow check, not evidence that the strategy is profitable.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("One-day ATM option proxy backtest complete")
    print(f"Date: {test_date}")
    print(f"Trades: {len(trade_frame)}")
    print(f"Wins: {int(wins.sum())} / {len(trade_frame)} ({wins.mean() * 100:.2f}%)")
    print(f"Net proxy P&L: Rs {trade_frame['net_pnl_proxy'].sum():,.2f}")
    print(f"Saved: {TRADES_FILE.name}")
    print(f"Saved: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
