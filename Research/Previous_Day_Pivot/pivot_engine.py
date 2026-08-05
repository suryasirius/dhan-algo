import pandas as pd
import numpy as np
import logging
from typing import Optional

class PivotEngine:
    def __init__(self, data: pd.DataFrame, logger: logging.Logger):
        self.data = data
        self.logger = logger
        self.daily_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None

    def calculate_daily_ohlc(self) -> pd.DataFrame:
        self.logger.info("Aggregating 1-minute data into daily OHLC format...")
        
        # Resample to daily frequency
        daily = self.data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Shift data to get Previous Day values
        daily['prev_open'] = daily['open'].shift(1)
        daily['prev_high'] = daily['high'].shift(1)
        daily['prev_low'] = daily['low'].shift(1)
        daily['prev_close'] = daily['close'].shift(1)
        
        self.daily_data = daily.dropna().copy()
        self.logger.info(f"Successfully calculated daily OHLC for {len(self.daily_data)} trading days.")
        return self.daily_data

    def calculate_pivots(self) -> None:
        if self.daily_data is None or self.daily_data.empty:
            self.logger.error("Daily data is empty. Cannot calculate pivots.")
            raise ValueError("Must run calculate_daily_ohlc() before calculate_pivots()")

        self.logger.info("Calculating Standard Pivot Points (Pivot, R1-R3, S1-S3)...")
        
        h = self.daily_data['prev_high']
        l = self.daily_data['prev_low']
        c = self.daily_data['prev_close']
        
        # Standard Pivot Point Calculations
        self.daily_data['pivot'] = (h + l + c) / 3.0
        
        p = self.daily_data['pivot']
        
        self.daily_data['r1'] = (2 * p) - l
        self.daily_data['s1'] = (2 * p) - h
        
        self.daily_data['r2'] = p + (h - l)
        self.daily_data['s2'] = p - (h - l)
        
        self.daily_data['r3'] = self.daily_data['r1'] + (h - l)
        self.daily_data['s3'] = self.daily_data['s1'] - (h - l)
        
        self.logger.info("Pivot calculations completed.")

    def merge_pivots_intraday(self) -> pd.DataFrame:
        if self.daily_data is None or 'pivot' not in self.daily_data.columns:
            self.logger.error("Pivot data missing. Cannot merge.")
            raise ValueError("Must run calculate_pivots() before merge_pivots_intraday()")

        self.logger.info("Merging daily Pivot levels back onto 1-minute intraday dataset...")
        
        # Extract date from datetime index to facilitate merging
        self.data['date_merge_key'] = self.data.index.date
        self.daily_data['date_merge_key'] = self.daily_data.index.date
        
        cols_to_merge = [
            'date_merge_key', 'prev_open', 'prev_high', 'prev_low', 'prev_close', 
            'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'
        ]
        
        merged = pd.merge(
            self.data.reset_index(),
            self.daily_data[cols_to_merge],
            on='date_merge_key',
            how='inner'
        )
        
        # Clean up merge keys and set index back to datetime
        merged.drop(columns=['date_merge_key'], inplace=True)
        merged.set_index('datetime', inplace=True)
        merged.sort_index(inplace=True)
        
        self.merged_data = merged
        self.logger.info(f"Merging complete. Final intraday dataset contains {len(self.merged_data)} rows.")
        return self.merged_data