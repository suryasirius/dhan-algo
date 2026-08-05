import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("MarketDNA.FeatureEngine")

class FeatureEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.audit_records = []

    def extract_features(self) -> pd.DataFrame:
        logger.info("Aggregating 1-minute intraday candles into daily Market DNA bars...")
        
        # 1. Stage: Aggregation
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        if "oi" in self.df.columns:
            agg_rules["oi"] = "last"
            
        daily_df = self.df.groupby("trade_date").agg(agg_rules).reset_index()
        daily_df.rename(columns={"trade_date": "timestamp"}, inplace=True)
        
        # Check for incomplete sessions or missing intraday candles (e.g., fewer than expected minute candles per day)
        candle_counts = self.df.groupby("trade_date").size()
        expected_candles = candle_counts.median() * 0.7  # threshold for incomplete session
        incomplete_days = candle_counts[candle_counts < expected_candles].index
        
        for day in incomplete_days:
            self.audit_records.append({
                "Date": day,
                "Reason Removed": "Incomplete session (low candle count)",
                "Pipeline Stage": "Aggregation"
            })
            
        # Filter out incomplete days if any
        if len(incomplete_days) > 0:
            daily_df = daily_df[~daily_df["timestamp"].isin(incomplete_days)].reset_index(drop=True)

        # 2. Stage: Feature Calculation & Rolling Windows
        daily_df["returns"] = daily_df["close"].pct_change()
        daily_df["true_range"] = np.maximum(
            daily_df["high"] - daily_df["low"],
            np.maximum(
                abs(daily_df["high"] - daily_df["close"].shift(1)),
                abs(daily_df["low"] - daily_df["close"].shift(1))
            )
        )
        
        # Identify rows that will become NaN due to rolling window (e.g., 20-day volatility)
        daily_df["volatility_20d"] = daily_df["returns"].rolling(window=20).std()
        
        # Capture days dropped due to feature calculation (rolling window warmup)
        pre_dropna_count = len(daily_df)
        nan_mask = daily_df["volatility_20d"].isna() | daily_df["returns"].isna() | daily_df["true_range"].isna()
        dropped_feature_days = daily_df.loc[nan_mask, "timestamp"].tolist()
        
        for day in dropped_feature_days:
            self.audit_records.append({
                "Date": day,
                "Reason Removed": "Feature calculation warmup (rolling window NaN)",
                "Pipeline Stage": "Feature Calculation"
            })
            
        # Drop NaN rows
        daily_df = daily_df.dropna().reset_index(drop=True)
        
        return daily_df
