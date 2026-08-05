"""Download six months of clean one-minute Nifty candles from Dhan."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dhanhq import DhanContext, dhanhq

from dhan_config import (
    ACCESS_TOKEN,
    CLIENT_ID,
    UNDERLYING_EXCHANGE_SEGMENT,
    UNDERLYING_INSTRUMENT_TYPE,
    UNDERLYING_SECURITY_ID,
)


TOTAL_DAYS = 180
CHUNK_DAYS = 90
MIN_CANDLES_PER_SESSION = 300
PAUSE_BETWEEN_REQUESTS_SECONDS = 1
OUTPUT_FILE = Path(__file__).with_name("dhan_historical_data_180days_v5.csv")
REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close")


def describe_error(response: Any) -> str:
    if not isinstance(response, dict):
        return f"Unexpected response type: {type(response).__name__}: {str(response)[:500]}"

    remarks = response.get("remarks")
    if isinstance(remarks, dict) and remarks.get("error_message"):
        return f"{remarks.get('error_code', 'Dhan error')}: {remarks['error_message']}"

    return f"status={response.get('status')!r}, remarks={remarks!r}"


def response_to_frame(response: Any, chunk_start: dt.date, chunk_end: dt.date) -> pd.DataFrame:
    if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
        raise RuntimeError(describe_error(response))

    data = response["data"]
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise RuntimeError(f"Dhan response is missing: {', '.join(missing)}")

    candle_count = len(data["timestamp"])
    if candle_count == 0:
        return pd.DataFrame()

    columns: dict[str, Any] = {}
    for name, values in data.items():
        try:
            if len(values) == candle_count:
                columns[name] = values
        except TypeError:
            continue

    frame = pd.DataFrame(columns)
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    frame["source_chunk_start"] = chunk_start.isoformat()
    frame["source_chunk_end"] = chunk_end.isoformat()

    first_columns = ["datetime", "timestamp", "open", "high", "low", "close"]
    remaining_columns = [column for column in frame.columns if column not in first_columns]
    return frame[first_columns + remaining_columns]


def remove_incomplete_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    session_dates = frame["datetime"].dt.normalize()
    session_sizes = frame.groupby(session_dates).size()
    incomplete_sessions = session_sizes[session_sizes < MIN_CANDLES_PER_SESSION]

    if incomplete_sessions.empty:
        return frame

    print("\nDiscarding incomplete sessions:")
    for session_date, candle_count in incomplete_sessions.items():
        print(f"  {session_date.date()}: {candle_count} candles")

    valid_sessions = session_sizes[session_sizes >= MIN_CANDLES_PER_SESSION].index
    return frame.loc[session_dates.isin(valid_sessions)].copy()


def main() -> None:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=TOTAL_DAYS)
    client = dhanhq(DhanContext(CLIENT_ID, ACCESS_TOKEN))
    chunks: list[pd.DataFrame] = []
    current_start = start_date

    print(f"Downloading {TOTAL_DAYS} days: {start_date} to {end_date}")

    while current_start <= end_date:
        current_end = min(current_start + dt.timedelta(days=CHUNK_DAYS - 1), end_date)
        print(f"Requesting {current_start} to {current_end}...")
        response = client.intraday_minute_data(
            security_id=UNDERLYING_SECURITY_ID,
            exchange_segment=UNDERLYING_EXCHANGE_SEGMENT,
            instrument_type=UNDERLYING_INSTRUMENT_TYPE,
            from_date=current_start.isoformat(),
            to_date=current_end.isoformat(),
        )

        try:
            chunk = response_to_frame(response, current_start, current_end)
        except RuntimeError as error:
            raise SystemExit(f"Download stopped for {current_start} to {current_end}: {error}") from error

        print(f"  Received {len(chunk):,} candles")
        if not chunk.empty:
            chunks.append(chunk)

        current_start = current_end + dt.timedelta(days=1)
        if current_start <= end_date:
            time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)

    if not chunks:
        raise SystemExit("Dhan returned no candles; no file was created.")

    result = pd.concat(chunks, ignore_index=True)
    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    result = remove_incomplete_sessions(result)
    if result.empty:
        raise SystemExit("All downloaded sessions were incomplete; no file was created.")

    result.to_csv(OUTPUT_FILE, index=False)
    print("\nDownload complete")
    print(f"Saved: {OUTPUT_FILE.name}")
    print(f"Candles: {len(result):,}")
    print(f"First candle: {result['datetime'].iloc[0]}")
    print(f"Last candle: {result['datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
