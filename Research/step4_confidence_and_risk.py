"""
Module: step4_confidence_and_risk.py
Description: Step 4 of the platform build. Evaluates multi-dimensional metrics 
             to output final trade confidence, risk tiering, and execution approval.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("Step4_ConfidenceRisk")
logger.setLevel(logging.INFO)

class ConfidenceAndRiskEngine:
    def __init__(self, min_confidence_threshold: float = 70.0):
        self.min_threshold = min_confidence_threshold

    def evaluate_trade(
        self,
        market_dna: Dict[str, Any],
        market_regime: Dict[str, Any],
        oi_intelligence: Dict[str, Any],
        technical_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates confidence score and risk rating based on multi-engine telemetry."""
        
        score = 50.0  # Baseline neutral score
        
        # 1. OI Intelligence Weighting (up to +25 points)
        oi_conf = oi_intelligence.get("confidence_score", 50.0)
        score += (oi_conf - 50.0) * 0.5
        
        # 2. Market Regime Alignment (up to +15 points)
        regime_type = market_regime.get("regime", "Neutral Range")
        if "Bullish" in regime_type or "Bearish" in regime_type or "Breakout" in regime_type:
            score += 15.0
        elif "Compression" in regime_type:
            score -= 10.0
            
        # 3. Technical Confirmation (up to +10 points)
        if technical_indicators.get("volume_expanding", False):
            score += 5.0
        if technical_indicators.get("price_near_vwap", True):
            score += 5.0

        confidence = max(0.0, min(100.0, round(score, 2)))
        
        # Risk Tier Assignment
        if confidence >= 82.0:
            risk_rating = "LOW"
            trade_quality = "A+"
        elif confidence >= 70.0:
            risk_rating = "MEDIUM"
            trade_quality = "A"
        else:
            risk_rating = "HIGH"
            trade_quality = "B"
            
        is_approved = confidence >= self.min_threshold

        result = {
            "confidence_score": confidence,
            "risk_rating": risk_rating,
            "trade_quality": trade_quality,
            "approved": is_approved
        }
        
        logger.info(f"Trade evaluated. Approved: {is_approved} (Confidence: {confidence}%, Risk: {risk_rating})")
        return result


if __name__ == "__main__":
    print("\n--- Testing Step 4: Confidence & Risk Engines ---")
    
    # Mock upstream telemetry dictionaries
    sample_dna = {"trend": "UP", "strength": "HIGH"}
    sample_regime = {"regime": "Bullish Trend"}
    sample_oi_intel = {"classification": "Confirmed Breakout (Long Build-up)", "confidence_score": 97.0}
    sample_technicals = {"volume_expanding": True, "price_near_vwap": True}
    
    engine = ConfidenceAndRiskEngine(min_confidence_threshold=70.0)
    evaluation = engine.evaluate_trade(sample_dna, sample_regime, sample_oi_intel, sample_technicals)
    
    for key, val in evaluation.items():
        print(f"{key:<20}: {val}")
    print("------------------------------------------------\n")