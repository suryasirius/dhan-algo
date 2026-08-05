"""
Module: oi_feature_engineering.py
Description: Computes derivatives features, velocity, acceleration, Z-scores, and institutional metrics.
"""

import numpy as np
import pandas as pd

class OIFeatureEngineer:
    def __init__(self, window: int = 20):
        self.window = window

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['total_oi'] = df['call_oi'] + df['put_oi']
        df['oi_diff'] = df['put_oi'] - df['call_oi']
        df['oi_ratio'] = df['put_oi'].div(df['call_oi'].replace(0, 1))
        
        df['oi_delta'] = df['total_oi'].diff()
        df['oi_velocity'] = df['oi_delta'].diff()
        df['oi_acceleration'] = df['oi_velocity'].diff()
        df['oi_momentum'] = df['total_oi'] - df['total_oi'].shift(self.window)
        
        df['oi_trend'] = np.where(df['oi_delta'] > 0, 1, -1)
        df['oi_slope'] = df['total_oi'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        rolling_mean = df['total_oi'].rolling(window=self.window).mean()
        rolling_std = df['total_oi'].rolling(window=self.window).std().replace(0, 1)
        df['z_score_oi'] = (df['total_oi'] - rolling_mean) / rolling_std
        df['oi_percentile'] = df['total_oi'].rolling(window=self.window).rank(pct=True) * 100
        
        df['oi_spike'] = np.where(df['z_score_oi'].abs() > 2.0, 1, 0)
        df['oi_exhaustion'] = np.where((df['oi_percentile'] > 95) & (df['oi_delta'].abs() < rolling_std * 0.1), 1, 0)
        
        vol_mean = (df['call_vol'] + df['put_vol']).rolling(window=self.window).mean().replace(0, 1)
        current_vol = df['call_vol'] + df['put_vol']
        df['institutional_participation_score'] = (current_vol / vol_mean) * df['z_score_oi'].abs()
        df['smart_money_score'] = (df['oi_diff'] / df['total_oi']) * df['institutional_participation_score']
        
        return df.fillna(0)
