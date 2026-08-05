"""
WARNING - SYNTHETIC TARGETS, NOT REAL BACKTEST OUTPUT:
best_strategy / expected_win_rate / expected_profit_factor / expected_drawdown /
expected_confidence below are generated from hand-built linear formulas plus
np.random noise (see build_features()), NOT from any real historical backtest of
Opening Drive / Velez / Mean Reversion. Any model trained against these columns
(see model_trainer.py) learns to reproduce this formula, not real strategy
performance. Do NOT use this module's output for live or paper trading decisions
until it is replaced with labels derived from an actual backtest engine.
Confirmed as of this writing: step6_strategy_selector.py and step9_live_engine.py
(the live trading path) do NOT import this module - it is fully isolated from
live decisions. Keep it that way until the labels are real.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("StrategySelectorAI.Features")

class FeatureMatrixEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_features(self) -> pd.DataFrame:
        logger.info("Constructing quantitative feature matrix and synthetic backtest targets...")
        df = self.df.copy()
        
        # Clean up all column names
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Identify price and volume columns safely
        available_cols = list(df.columns)
        logger.info(f"Available columns in merged dataset: {available_cols}")
        
        c_close = next((c for c in ["close", "price", "ltp", "value"] if c in available_cols), available_cols[1] if len(available_cols) > 1 else available_cols[0])
        c_open = next((c for c in ["open", "opening"] if c in available_cols), c_close)
        c_high = next((c for c in ["high", "max"] if c in available_cols), c_close)
        c_low = next((c for c in ["low", "min"] if c in available_cols), c_close)
        c_volume = next((c for c in ["volume", "vol", "qty"] if c in available_cols), None)
        
        if "returns" not in df.columns:
            df["returns"] = df[c_close].pct_change().fillna(0)
        if "volatility_20d" not in df.columns:
            df["volatility_20d"] = df["returns"].rolling(20, min_periods=1).std().fillna(df["returns"].std())
            
        df["feat_volatility"] = df["volatility_20d"]
        df["feat_trend_strength"] = abs(df[c_close] - df[c_open]) / (df[c_high] - df[c_low] + 1e-8)
        df["feat_range_ratio"] = (df[c_high] - df[c_low]) / (df[c_close].shift(1) + 1e-8)
        
        if c_volume and c_volume in df.columns:
            vol_mean = df[c_volume].rolling(20, min_periods=1).mean()
            vol_std = df[c_volume].rolling(20, min_periods=1).std() + 1e-8
            df["feat_volume_zscore"] = (df[c_volume] - vol_mean) / vol_std
        else:
            df["feat_volume_zscore"] = 0.0
            
        df.fillna(0, inplace=True)
        
        # SYNTHETIC PLACEHOLDER TARGETS - see module warning at top of file.
        np.random.seed(42)
        n = len(df)

        od_score = 0.5 + 0.3 * df["feat_volume_zscore"] + 0.2 * np.random.normal(0, 0.1, n)
        velez_score = 0.5 + 0.4 * df["feat_trend_strength"] + 0.2 * np.random.normal(0, 0.1, n)
        mr_score = 0.5 + 0.3 * df["feat_volatility"] - 0.3 * df["feat_trend_strength"] + 0.2 * np.random.normal(0, 0.1, n)
        nt_score = 0.4 + 0.2 * abs(np.random.normal(0, 0.2, n))
        
        strategies_scores = pd.DataFrame({
            "Opening Drive": od_score,
            "Velez": velez_score,
            "Mean Reversion": mr_score,
            "No Trade": nt_score
        })
        
        df["best_strategy"] = strategies_scores.idxmax(axis=1)
        df["expected_win_rate"] = strategies_scores.max(axis=1).clip(0.35, 0.85)
        df["expected_profit_factor"] = (1.0 + df["expected_win_rate"] * 1.5).clip(1.0, 3.2)
        df["expected_drawdown"] = (0.05 + (1.0 - df["expected_win_rate"]) * 0.15).clip(0.01, 0.25)
        df["expected_risk"] = (df["expected_drawdown"] * 0.8).clip(0.01, 0.20)
        df["expected_confidence"] = df["expected_win_rate"].clip(0.50, 0.95)
        df["alternative_strategy"] = strategies_scores.apply(lambda x: x.nlargest(2).index[-1], axis=1)
        
        return df
