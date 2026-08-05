"""
Module: oi_classifier.py
Description: Automatically classifies market candles into derivative states with confidence scores.
"""

import pandas as pd

class OIClassifier:
    def classify_candle(self, row: pd.Series) -> dict:
        price_change = row.get('close', 0) - row.get('open', 0)
        oi_change = row.get('oi_delta', 0)
        call_oi_chg = row.get('call_oi_change', 0)
        put_oi_chg = row.get('put_oi_change', 0)
        
        classification = "Neutral"
        confidence = 0.50
        
        if price_change > 0 and oi_change > 0:
            classification = "Long Build-up"
            confidence = min(0.60 + (abs(oi_change) / 100000.0), 0.95)
        elif price_change < 0 and oi_change > 0:
            classification = "Short Build-up"
            confidence = min(0.60 + (abs(oi_change) / 100000.0), 0.95)
        elif price_change < 0 and oi_change < 0:
            classification = "Long Unwinding"
            confidence = min(0.55 + (abs(oi_change) / 100000.0), 0.90)
        elif price_change > 0 and oi_change < 0:
            classification = "Short Covering"
            confidence = min(0.55 + (abs(oi_change) / 100000.0), 0.90)
            
        if put_oi_chg > call_oi_chg * 1.5:
            writing_zone = "Put Writing Zone (Bullish Support)"
        elif call_oi_chg > put_oi_chg * 1.5:
            writing_zone = "Call Writing Zone (Bearish Resistance)"
        else:
            writing_zone = "Balanced Option Flow"
            
        false_breakout_risk = 0.20
        if row.get('z_score_oi', 0) > 2.5 and price_change == 0:
            false_breakout_risk = 0.85
            classification = "OI Trap / Compression"

        return {
            "primary_classification": classification,
            "option_activity": writing_zone,
            "confidence_score": round(confidence * 100, 2),
            "false_breakout_risk": round(false_breakout_risk * 100, 2)
        }
