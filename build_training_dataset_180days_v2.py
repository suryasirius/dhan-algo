"""Build a leak-free 15-minute-candle training dataset from six-month Dhan data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).with_name("dhan_historical_data_180days_v5.csv")
OUTPUT_FILE = Path(__file__).with_name("training_dataset_180days_v2.csv")
REQUIRED_COLUMNS = ("datetime", "timestamp", "open", "high", "low", "close")


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def calculate_atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["close"].shift()
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False).mean()


def load_complete_bars() -> tuple[pd.DataFrame, int]:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file was not found: {INPUT_FILE.name}")

    frame = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise SystemExit(f"Input file is missing columns: {', '.join(missing_columns)}")

    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.drop_duplicates(subset="timestamp").sort_values("datetime").reset_index(drop=True)
    frame["htf_start"] = frame["datetime"].dt.floor("15min")
    bar_sizes = frame.groupby("htf_start")["timestamp"].transform("size")
    incomplete_rows = int((bar_sizes != 15).sum())
    frame = frame.loc[bar_sizes == 15].copy().reset_index(drop=True)

    if frame.empty:
        raise SystemExit("No complete 15-minute bars were found in the input file.")

    return frame, incomplete_rows


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame["session_date"] = frame["datetime"].dt.date.astype(str)
    frame["body"] = frame["close"] - frame["open"]
    frame["range"] = frame["high"] - frame["low"]
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]
    frame["body_pct_of_range"] = frame["body"] / frame["range"].replace(0, np.nan)
    frame["ema9"] = frame["close"].ewm(span=9, adjust=False).mean()
    frame["ema21"] = frame["close"].ewm(span=21, adjust=False).mean()
    frame["ema_spread"] = frame["ema9"] - frame["ema21"]
    frame["rsi14"] = calculate_rsi(frame["close"])
    frame["atr14"] = calculate_atr(frame)
    frame["return_1m"] = frame["close"].pct_change()
    frame["return_5m"] = frame["close"].pct_change(5)

    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame["volume_average_20"] = frame["volume"].rolling(20, min_periods=5).mean()
        frame["volume_ratio_20"] = frame["volume"] / frame["volume_average_20"].replace(0, np.nan)

    bars = frame.groupby("htf_start", sort=False)
    frame["minute_in_15m"] = bars.cumcount() + 1
    frame["minutes_left"] = 15 - frame["minute_in_15m"]
    frame["htf_open"] = bars["open"].transform("first")
    frame["htf_high_so_far"] = bars["high"].cummax()
    frame["htf_low_so_far"] = bars["low"].cummin()
    frame["htf_close_so_far"] = frame["close"]
    frame["htf_body_so_far"] = frame["htf_close_so_far"] - frame["htf_open"]
    frame["htf_range_so_far"] = frame["htf_high_so_far"] - frame["htf_low_so_far"]
    frame["htf_upper_wick_so_far"] = frame["htf_high_so_far"] - frame[["htf_open", "htf_close_so_far"]].max(axis=1)
    frame["htf_lower_wick_so_far"] = frame[["htf_open", "htf_close_so_far"]].min(axis=1) - frame["htf_low_so_far"]

    frame["target_final_15m_close"] = bars["close"].transform("last")
    frame["target_final_15m_high"] = bars["high"].transform("max")
    frame["target_final_15m_low"] = bars["low"].transform("min")
    frame["target_bullish"] = (frame["target_final_15m_close"] > frame["htf_open"]).astype(int)
    frame["target_close_change"] = frame["target_final_15m_close"] - frame["close"]
    frame["target_high_change"] = frame["target_final_15m_high"] - frame["close"]
    frame["target_low_change"] = frame["target_final_15m_low"] - frame["close"]
    frame["target_new_high"] = (frame["target_final_15m_high"] > frame["htf_high_so_far"]).astype(int)
    frame["target_new_low"] = (frame["target_final_15m_low"] < frame["htf_low_so_far"]).astype(int)

    return frame.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    input_frame, incomplete_rows = load_complete_bars()
    dataset = build_dataset(input_frame)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE.name}")
    print(f"Rows: {len(dataset):,}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Removed rows from incomplete 15-minute bars: {incomplete_rows:,}")
    print(f"First timestamp: {dataset['datetime'].iloc[0]}")
    print(f"Last timestamp: {dataset['datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
