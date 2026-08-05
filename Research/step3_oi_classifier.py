"""
Module: step3_oi_classifier.py
Description: Step 3 of the platform build. Classifies market candles into derivative 
             states with dynamic confidence scoring.
"""

import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger("Step3_OIClassifier")
logger.setLevel(logging.INFO)

class OIClassifier:
    def __init__(self):
        # Mapping price change sign and OI change sign to market states
        self.classification_map = {
            ("positive", "positive"): ("Long Build-up", 0.85),
            ("negative", "positive"): ("Short Build-up", 0.85),
            ("negative", "negative"): ("Long Unwinding", 0.75),
            ("positive", "negative"): ("Short Covering", 0.75)
        }

    def classify_candle(self, row: pd.Series) -> Dict[str, Any]:
        """Classifies a single row/candle based on price and OI dynamics."""
        price_chg = row.get('close', 0) - row.get('open', 0)
        oi_chg = row.get('oi_delta', 0)
        z_score = row.get('z_score_oi', 0)
        
        p_state = "positive" if price_chg > 0 else ("negative" if price_chg < 0 else "neutral")
        o_state = "positive" if oi_chg > 0 else ("negative" if oi_chg < 0 else "neutral")
        
        classification, base_confidence = self.classification_map.get((p_state, o_state), ("Neutral Range", 0.50))
        
        # Override for high Z-score compressions or breakouts
        if abs(z_score) > 2.5 and p_state == "neutral":
            classification = "Fake Breakout / OI Compression"
            base_confidence = 0.90
        elif row.get('oi_expansion', 0) == 1 and p_state != "neutral":
            classification = f"Confirmed Breakout ({classification})"
            base_confidence = min(base_confidence + 0.12, 0.99)

        result = {
            "classification": classification,
            "confidence_score": round(base_confidence * 100, 2),
            "oi_strength": float(z_score),
            "smart_money_score": float(row.get('smart_money_score', 0.0))
        }
        
        logger.info(f"Classified candle as: {classification} with {result['confidence_score']}% confidence.")
        return result


if __name__ == "__main__":
    print("\n--- Testing Step 3: Market State Classification ---")
    
    # Sample row simulating a Long Build-up state
    sample_row = pd.Series({
        'open': 100.0,
        'close': 102.5,
        'oi_delta': 15000.0,
        'z_score_oi': 1.8,
        'oi_expansion': 1,
        'smart_money_score': 0.145
    })
    
    classifier = OIClassifier()
    classification_output = classifier.classify_candle(sample_row)
    
    for key, val in classification_output.items():
        print(f"{key:<20}: {val}")
    print("------------------------------------------------\n")