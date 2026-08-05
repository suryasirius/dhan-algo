import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import config

class MonteCarloSimulator:
    def __init__(self, trades_df: pd.DataFrame, logger: logging.Logger):
        self.trades_df = trades_df
        self.logger = logger

    def run_simulations(self) -> Dict[str, Any]:
        """Executes Monte Carlo simulations to test statistical robustness and tail-risk drawdowns."""
        self.logger.info(f"Running {config.MONTE_CARLO_SIMULATIONS} Monte Carlo simulations...")
        
        if self.trades_df.empty or 'net_pnl' not in self.trades_df.columns:
            self.logger.warning("No trade returns available for Monte Carlo simulation.")
            return {}
            
        returns = self.trades_df['net_pnl'].values
        simulations_count = config.MONTE_CARLO_SIMULATIONS
        
        max_drawdowns = []
        final_equities = []
        
        for _ in range(simulations_count):
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            equity_curve = config.INITIAL_CAPITAL + np.cumsum(sampled_returns)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = equity_curve - peak
            
            max_drawdowns.append(abs(np.min(drawdown)))
            final_equities.append(equity_curve[-1])
            
        results = {
            "Median_Final_Equity": float(np.median(final_equities)),
            "Worst_Case_Drawdown": float(np.percentile(max_drawdowns, 99)),
            "Median_Max_Drawdown": float(np.median(max_drawdowns)),
            "Probability_of_Loss": float(np.mean(np.array(final_equities) < config.INITIAL_CAPITAL) * 100)
        }
        
        self.logger.info("Monte Carlo simulations completed successfully.")
        return results