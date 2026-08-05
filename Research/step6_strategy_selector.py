"""
Module: step6_strategy_selector.py
Description: Step 6 of the platform build. Ranks Opening Drive, Velez, and
             ORB strategies using a hand-written scoring rubric based on
             Open Interest telemetry and Market Regime.

             HONEST LABEL (added after a project review): despite the class
             name below (kept as-is so step9_live_engine.py's import doesn't
             break), this is NOT a trained/machine-learning model. Every
             number here (+20, +15, +25, the 50.0 baseline, the 0.2 global
             modifier, etc.) is a manually chosen constant, not something
             learned from data. When this was actually tested -- backtesting
             the top 5 strategies with and without this selector's filtering
             -- it did not improve results. The two strategies now running
             live in step10_live_paper_best2.py deliberately do NOT use this
             module at all. Treat its output as a readable heuristic
             scorecard, not an AI recommendation.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("Step6_StrategySelector")
logger.setLevel(logging.INFO)

class StrategySelectorAI:
    """Class name kept unchanged (not renamed to e.g. StrategySelectorHeuristic)
    so step9_live_engine.py's `from step6_strategy_selector import
    StrategySelectorAI` keeps working without modification. See the module
    docstring above -- this is a rule-based scorecard, not a trained model."""
    def __init__(self):
        # Baseline probability scores for each strategy
        self.strategies = ["Opening Drive", "Velez", "ORB"]

    def evaluate_strategies(
        self, 
        market_regime: Dict[str, Any], 
        oi_intelligence: Dict[str, Any], 
        confidence_metrics: Dict[str, Any],
        time_of_day_minutes: int = 45 # Minutes from market open (e.g., 45 = 10:00 AM)
    ) -> Dict[str, Any]:
        """Ranks strategies and selects the optimal one based on live telemetry."""
        
        scores = {
            "Opening Drive": 50.0,
            "Velez": 50.0,
            "ORB": 50.0
        }
        
        oi_class = oi_intelligence.get("classification", "")
        oi_strength = oi_intelligence.get("oi_strength", 0.0)
        regime = market_regime.get("regime", "")
        
        # 1. Opening Drive Logic (Favors early morning, high momentum, breakouts)
        if time_of_day_minutes < 30:
            scores["Opening Drive"] += 20.0
        if "Breakout" in oi_class:
            scores["Opening Drive"] += 15.0
            
        # 2. ORB Logic (Favors range expansion, trend establishment post-morning volatility)
        if 30 <= time_of_day_minutes <= 120:
            scores["ORB"] += 15.0
        if "Build-up" in oi_class and abs(oi_strength) > 1.2:
            scores["ORB"] += 20.0
        if "Compression" in regime:
            scores["ORB"] -= 15.0 # ORB fails in tight compression
            
        # 3. Velez Logic (Favors trend continuation, strong institutional bias, pullbacks)
        if "Trend" in regime:
            scores["Velez"] += 25.0
        if oi_intelligence.get("smart_money_score", 0.0) > 0.1:
            scores["Velez"] += 15.0
            
        # Normalize scores to 0-100% and apply global confidence modifier
        global_modifier = (confidence_metrics.get("confidence_score", 50.0) - 50.0) * 0.2
        for strat in scores:
            scores[strat] = max(0.0, min(100.0, round(scores[strat] + global_modifier, 2)))

        # Rank strategies
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_strategy, best_score = ranked[0]
        
        # Generate a human-readable explanation of the rule-based score (not an AI-generated reason)
        reason = (
            f"Selected {best_strategy} with {best_score}% probability. "
            f"Market is showing '{oi_class}' with an OI Strength of {oi_strength:.2f}. "
            f"Regime is '{regime}'. This aligns perfectly with {best_strategy}'s edge."
        )

        result = {
            "selected_strategy": best_strategy,
            "probability_score": best_score,
            "rankings": {k: v for k, v in ranked},
            "explanation": reason
        }
        
        logger.info(f"Strategy Selector output: {best_strategy} ({best_score}%)")
        return result


if __name__ == "__main__":
    print("\n--- Testing Step 6: Strategy Selector AI ---")
    
    # Mock telemetry from upstream engines
    mock_regime = {"regime": "Bullish Trend"}
    mock_oi = {
        "classification": "Long Build-up", 
        "oi_strength": 1.8, 
        "smart_money_score": 0.15
    }
    mock_conf = {"confidence_score": 85.0}
    
    selector = StrategySelectorAI()
    
    # Test for mid-morning condition (e.g., 60 mins after open)
    decision = selector.evaluate_strategies(
        market_regime=mock_regime,
        oi_intelligence=mock_oi,
        confidence_metrics=mock_conf,
        time_of_day_minutes=60
    )
    
    print("Rankings:")
    for strat, score in decision["rankings"].items():
        print(f"  - {strat:<15}: {score}%")
        
    print(f"\nFinal Choice : {decision['selected_strategy']}")
    print(f"Explanation  : {decision['explanation']}")
    print("--------------------------------------------\n")