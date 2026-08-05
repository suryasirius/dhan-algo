"""
WARNING: reads best_strategy/expected_* columns straight from feature_matrix.py's
SYNTHETIC placeholder targets (formula + random noise, not a real backtest) - see
that module's docstring. The "Reason" string below is also boilerplate, not a
real explanation grounded in backtest data. Not wired into the live trading path
(step6_strategy_selector.py / step9_live_engine.py use an independent, real,
live-data-driven strategy selector instead).
"""
import pandas as pd
import logging

logger = logging.getLogger("StrategySelectorAI.Selector")

class StrategySelectorEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_recommendations(self) -> pd.DataFrame:
        logger.info("Generating daily strategy selections, confidence scores, and reasoning "
                     "(SYNTHETIC targets - see module warning).")
        
        recommendations = []
        for idx, row in self.df.iterrows():
            strat = row["best_strategy"]
            alt = row["alternative_strategy"]
            conf = row["expected_confidence"]
            
            reason = f"Selected {strat} due to optimal volatility and regime alignment matching backtest expectancy profiles."
            
            recommendations.append({
                "date": row["date"],
                "Best Strategy": strat,
                "Confidence": round(conf, 4),
                "Reason": reason,
                "Alternative Strategy": alt,
                "Expected Win Rate": round(row["expected_win_rate"], 4),
                "Expected Profit Factor": round(row["expected_profit_factor"], 4),
                "Expected Drawdown": round(row["expected_drawdown"], 4),
                "Expected Risk": round(row["expected_risk"], 4)
            })
            
        return pd.DataFrame(recommendations)
