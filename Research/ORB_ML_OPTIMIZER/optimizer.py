import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import config
from orb_engine import ORBEngine

class StrategyOptimizer:
    def __init__(self, engine: ORBEngine, logger: logging.Logger):
        self.engine = engine
        self.logger = logger
        self.all_trades_df: pd.DataFrame = pd.DataFrame()

    def run_grid_search(self) -> pd.DataFrame:
        """Executes exhaustive grid search across ORB durations, stops, targets, and time filters."""
        self.logger.info("Starting multi-dimensional grid search optimization across research phases...")
        
        master_results = []
        
        # Test subset of grid parameters for robustness and speed
        for orb_m in [15, 30]:
            for sl in [5, 10, 15]:
                for trail in [3, 6]:
                    for target in ["2R", "3R", "No Target"]:
                        for t_filter in ["09:30", "09:45"]:
                            trades = self.engine.simulate_orb_strategy(
                                orb_minutes=orb_m,
                                stop_loss_pts=sl,
                                trail_stop_pts=trail,
                                target_type=target,
                                time_filter=t_filter
                            )
                            
                            if len(trades) < 5:
                                continue
                                
                            metrics = self._calculate_performance_metrics(trades)
                            metrics.update({
                                'orb_minutes': orb_m,
                                'stop_loss': sl,
                                'trail_stop': trail,
                                'target': target,
                                'time_filter': t_filter
                            })
                            master_results.append(metrics)
                            
        self.all_trades_df = pd.DataFrame(master_results)
        self.logger.info(f"Grid search completed. Evaluated {len(self.all_trades_df)} parameter combinations.")
        return self.all_trades_df

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates institutional performance metrics: Profit Factor, SQN, Drawdown, Expectancy, Sharpe."""
        df = pd.DataFrame(trades)
        total_trades = len(df)
        wins = df[df['win'] == 1]
        losses = df[df['win'] == 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
        net_profit = df['net_pnl'].sum()
        
        gross_profit = wins['net_pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else gross_profit
        
        avg_win = wins['net_pnl'].mean() if not wins.empty else 0.0
        avg_loss = abs(losses['net_pnl'].mean()) if not losses.empty else 0.0
        
        expectancy = ((win_rate / 100.0) * avg_win) - ((1.0 - (win_rate / 100.0)) * avg_loss)
        
        # Cumulative Equity & Drawdown
        df['equity'] = config.INITIAL_CAPITAL + df['net_pnl'].cumsum()
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = df['equity'] - df['peak']
        max_drawdown = abs(df['drawdown'].min())
        
        # Sharpe Ratio (annualized proxy)
        returns = df['net_pnl'] / config.INITIAL_CAPITAL
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
        
        # SQN (System Quality Number)
        sqn = (returns.mean() / returns.std()) * np.sqrt(total_trades) if returns.std() != 0 and total_trades > 0 else 0.0
        
        # Risk of Ruin proxy
        risk_of_ruin = 0.0 if profit_factor > 1.2 else 50.0

        return {
            'trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sqn': sqn,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_of_ruin': risk_of_ruin
        }