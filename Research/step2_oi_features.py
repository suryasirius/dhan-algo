"""
Module: step2_oi_features.py
Description: Step 2 of the platform build. Calculates vectorised Open Interest 
             derivatives features, Z-scores, momentum, and institutional metrics.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Step2_OIFeatures")
logger.setLevel(logging.INFO)

class OIBuilder:
    def __init__(self, window: int = 20):
        self.window = window

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes institutional metrics and derivative features using vectorized operations."""
        if df.empty:
            logger.warning("Passed dataframe is empty. Returning empty frame.")
            return df

        df = df.copy()
        
        # Core Aggregations
        df['total_oi'] = df['call_oi'] + df['put_oi']
        df['oi_diff'] = df['put_oi'] - df['call_oi']
        df['pcr'] = df['put_oi'].div(df['call_oi'].replace(0, 1))
        
        # Derivatives Velocity & Momentum
        df['oi_delta'] = df['total_oi'].diff()
        df['oi_velocity'] = df['oi_delta'].diff()
        df['oi_momentum'] = df['total_oi'] - df['total_oi'].shift(self.window)
        
        # Z-Scores and Percentiles (Rolling window)
        rolling_mean = df['total_oi'].rolling(window=self.window).mean()
        rolling_std = df['total_oi'].rolling(window=self.window).std().replace(0, 1)
        df['z_score_oi'] = (df['total_oi'] - rolling_mean) / rolling_std
        
        # Institutional & Smart Money Metrics
        vol_mean = (df['call_vol'] + df['put_vol']).rolling(window=self.window).mean().replace(0, 1)
        current_vol = df['call_vol'] + df['put_vol']
        df['institutional_participation'] = (current_vol / vol_mean) * df['z_score_oi'].abs()
        df['smart_money_score'] = (df['oi_diff'] / df['total_oi'].replace(0, 1)) * df['institutional_participation']
        
        # Expansion Flags
        df['oi_expansion'] = np.where(df['z_score_oi'] > 1.5, 1, 0)
        
        logger.info("Open Interest features computed successfully.")
        return df.fillna(0)


if __name__ == "__main__":
    print("\n--- Testing Step 2: OI Feature Engineering ---")
    
    # Create sample testing data frame
    sample_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 101.5, 103.0],
        'close': [101.0, 102.0, 101.5, 103.0, 104.0],
        'call_oi': [1000000, 1050000, 1100000, 1150000, 1200000],
        'put_oi': [1200000, 1220000, 1250000, 1300000, 1350000],
        'call_vol': [20000, 22000, 25000, 27000, 30000],
        'put_vol': [30000, 32000, 35000, 38000, 42000]
    })
    
    builder = OIBuilder(window=3)
    processed_df = builder.compute_features(sample_data)
    
    print("Computed Columns:", list(processed_df.columns))
    print("\nLatest Row Z-Score OI:", processed_df['z_score_oi'].iloc[-1])
    print("Latest Row Smart Money Score:", processed_df['smart_money_score'].iloc[-1])
    print("------------------------------------------------\n")