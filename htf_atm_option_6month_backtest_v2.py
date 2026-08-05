"""Six-month ATM option proxy backtest for the forming 15-minute candle direction strategy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from htf_atm_option_1day_backtest_v1 import INPUT_FILE, simulate_trade


TRADES_FILE = Path(__file__).with_name("htf_atm_option_6month_trades_v2.csv")
DAILY_FILE = Path(__file__).with_name("htf_atm_option_6month_daily_v2.csv")
MONTHLY_FILE = Path(__file__).with_name("htf_atm_option_6month_monthly_v2.csv")
REPORT_FILE = Path(__file__).with_name("htf_atm_option_6month_report_v2.txt")


def calculate_profit_factor(values: pd.Series) -> float:
    gains = values.loc[values > 0].sum()
    losses = values.loc[values < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def build_daily_summary(trades: pd.DataFrame) -> pd.DataFrame:
    daily = trades.groupby("date", as_index=False).agg(
        trades=("net_pnl_proxy", "size"),
        wins=("net_pnl_proxy", lambda values: int((values > 0).sum())),
        gross_pnl_proxy=("gross_pnl_proxy", "sum"),
        brokerage=("brokerage", "sum"),
        net_pnl_proxy=("net_pnl_proxy", "sum"),
    )
    daily["win_rate"] = daily["wins"] / daily["trades"]
    daily["cumulative_net_pnl_proxy"] = daily["net_pnl_proxy"].cumsum()
    daily["drawdown_proxy"] = daily["cumulative_net_pnl_proxy"] - daily["cumulative_net_pnl_proxy"].cummax()
    return daily


def build_monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m")
    monthly = daily.assign(month=month).groupby("month", as_index=False).agg(
        trading_days=("date", "size"),
        trades=("trades", "sum"),
        wins=("wins", "sum"),
        gross_pnl_proxy=("gross_pnl_proxy", "sum"),
        brokerage=("brokerage", "sum"),
        net_pnl_proxy=("net_pnl_proxy", "sum"),
    )
    monthly["win_rate"] = monthly["wins"] / monthly["trades"]
    return monthly


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file was not found: {INPUT_FILE.name}")

    frame = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    required_columns = {"datetime", "timestamp", "open", "high", "low", "close"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Input file is missing columns: {', '.join(missing_columns)}")

    frame = frame.sort_values("datetime").drop_duplicates(subset="timestamp").reset_index(drop=True)
    frame["date"] = frame["datetime"].dt.date.astype(str)
    frame["htf_start"] = frame["datetime"].dt.floor("15min")
    trades: list[dict[str, object]] = []

    for test_date, day_frame in frame.groupby("date", sort=True):
        for htf_start, candles in day_frame.groupby("htf_start", sort=True):
            trade = simulate_trade(candles.reset_index(drop=True), htf_start, test_date)
            if trade is not None:
                trades.append(trade)

    if not trades:
        raise SystemExit("No qualifying trades were found.")

    trade_frame = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    trade_frame["cumulative_net_pnl_proxy"] = trade_frame["net_pnl_proxy"].cumsum()
    trade_frame["drawdown_proxy"] = trade_frame["cumulative_net_pnl_proxy"] - trade_frame["cumulative_net_pnl_proxy"].cummax()
    daily = build_daily_summary(trade_frame)
    monthly = build_monthly_summary(daily)
    trade_frame.to_csv(TRADES_FILE, index=False)
    daily.to_csv(DAILY_FILE, index=False)
    monthly.to_csv(MONTHLY_FILE, index=False)

    wins = trade_frame["net_pnl_proxy"] > 0
    net_pnl = trade_frame["net_pnl_proxy"].sum()
    profit_factor = calculate_profit_factor(trade_frame["net_pnl_proxy"])
    positive_days = daily["net_pnl_proxy"] > 0
    report_lines = [
        "Nifty Forming-15-Minute Candle ATM Option Backtest",
        "",
        f"Dhan index-candle date range: {trade_frame['date'].iloc[0]} to {trade_frame['date'].iloc[-1]}",
        f"Trading days: {len(daily)}",
        f"Trades: {len(trade_frame)}",
        "",
        "Fixed strategy rules:",
        "- After five completed one-minute candles, buy an ATM CE when the forming 15-minute candle is green; buy an ATM PE when it is red.",
        "- Enter at the next one-minute candle open and exit at the 15-minute candle close.",
        "- One independent trade per complete 15-minute candle.",
        "",
        "Six-month proxy results:",
        f"- Wins: {int(wins.sum())} / {len(trade_frame)} ({wins.mean() * 100:.2f}%)",
        f"- Profitable days: {int(positive_days.sum())} / {len(daily)} ({positive_days.mean() * 100:.2f}%)",
        f"- Gross proxy P&L: Rs {trade_frame['gross_pnl_proxy'].sum():,.2f}",
        f"- Brokerage assumed: Rs {trade_frame['brokerage'].sum():,.2f}",
        f"- Net proxy P&L: Rs {net_pnl:,.2f}",
        f"- Average proxy P&L per trade: Rs {trade_frame['net_pnl_proxy'].mean():,.2f}",
        f"- Profit factor: {profit_factor:.3f}",
        f"- Maximum trade-by-trade proxy drawdown: Rs {trade_frame['drawdown_proxy'].min():,.2f}",
        f"- Maximum day-by-day proxy drawdown: Rs {daily['drawdown_proxy'].min():,.2f}",
        "",
        "Important limitation:",
        "- Dhan historical option premiums are unavailable in this project (DH-901), so this is not real option-premium P&L.",
        "- The option result is a proxy and excludes theta, implied-volatility changes, bid-ask spread, and slippage.",
        "- Use this only to test whether the index-direction filter is promising; do not use it to place live trades.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("Six-month ATM option proxy backtest complete")
    print(f"Trading days: {len(daily)}")
    print(f"Trades: {len(trade_frame)}")
    print(f"Wins: {int(wins.sum())} / {len(trade_frame)} ({wins.mean() * 100:.2f}%)")
    print(f"Net proxy P&L: Rs {net_pnl:,.2f}")
    print(f"Profit factor: {profit_factor:.3f}")
    print(f"Saved: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
