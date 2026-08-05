"""
Module: oi_engine.py
Description: Master coordinator integrating Feature Engineering, Classifier, and Market DNA/Regime updates.
"""

import pandas as pd
from oi_feature_engineering import OIFeatureEngineer
from oi_classifier import OIClassifier

class OIEngine:
    def __init__(self):
        self.fe = OIFeatureEngineer()
        self.classifier = OIClassifier()

    def process_tick(self, df_history: pd.DataFrame) -> dict:
        df_featured = self.fe.compute_features(df_history)
        latest_row = df_featured.iloc[-1]
        classification_result = self.classifier.classify_candle(latest_row)
        
        market_dna_extension = {
            "oi_strength": float(latest_row['z_score_oi']),
            "oi_direction": int(latest_row['oi_trend']),
            "oi_shift": float(latest_row['oi_delta']),
            "pcr": float(latest_row['pcr']),
            "support_strength": float(latest_row['put_oi']),
            "resistance_strength": float(latest_row['call_oi']),
            "oi_momentum": float(latest_row['oi_momentum']),
            "institutional_bias": float(latest_row['institutional_participation_score']),
            "smart_money_bias": float(latest_row['smart_money_score'])
        }
        
        regime = "Neutral Range"
        if latest_row['oi_delta'] > 0 and latest_row['close'] > latest_row['open']:
            regime = "Bullish OI Trend / Breakout Supported by OI"
        elif latest_row['oi_delta'] > 0 and latest_row['close'] < latest_row['open']:
            regime = "Bearish OI Trend / Institutional Selling"
        elif latest_row['call_oi_change'] > 0 and classification_result['primary_classification'] == "Short Covering":
            regime = "Short Covering Rally"
        elif latest_row['z_score_oi'].abs() < 0.5:
            regime = "OI Compression"

        return {
            "timestamp": str(latest_row.get('timestamp', '')),
            "classification": classification_result,
            "market_dna_extension": market_dna_extension,
            "market_regime": regime
        }
