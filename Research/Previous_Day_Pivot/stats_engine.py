import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
from typing import Dict, Any

class StatisticsCalculator:
    def __init__(self, metrics_df: pd.DataFrame, logger: logging.Logger):
        self.df = metrics_df
        self.logger = logger
        self.stats_result: Dict[str, Any] = {}

    def calculate_all(self) -> Dict[str, Any]:
        self.logger.info("Computing mathematical and statistical matrices...")
        
        total_days = len(self.df)
        if total_days == 0:
            self.logger.error("Empty metrics dataframe provided for statistics.")
            return {}

        # Pivot Metrics
        pivot_respect_pct = float((self.df['Num_Pivot_Touches'] == 0).mean() * 100)
        bullish_acc = float(self.df[self.df['Open_Above_Pivot']]['Pivot_Predicts'].mean() * 100) if not self.df[self.df['Open_Above_Pivot']].empty else 0.0
        bearish_acc = float(self.df[self.df['Open_Below_Pivot']]['Pivot_Predicts'].mean() * 100) if not self.df[self.df['Open_Below_Pivot']].empty else 0.0
        false_breakout_pct = float(self.df['Return_To_Pivot'].mean() * 100)
        breakout_pct = float(self.df['Breakout_Prob'].mean() * 100)
        reversal_pct = float(self.df['Reversal_Prob'].mean() * 100)

        # Movement Distributions
        moves = np.where(
            self.df['Intraday_Trend'] == 'Bullish', 
            self.df['Day_High'] - self.df['Day_Open'], 
            self.df['Day_Open'] - self.df['Day_Low']
        )
        
        avg_move = float(np.mean(moves))
        median_move = float(np.median(moves))
        mean_move = avg_move
        
        rounded_moves = np.round(moves, 1)
        mode_result = stats.mode(rounded_moves, keepdims=True)
        mode_move = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else avg_move
        
        max_move = float(np.max(moves))
        min_move = float(np.min(moves))
        std_dev = float(np.std(moves))
        variance = float(np.var(moves))

        # Day Category Probabilities
        trend_days_pct = float(self.df['Trending_Day'].mean() * 100)
        range_days_pct = float(self.df['Range_Day'].mean() * 100)
        inside_days_pct = float(self.df['Inside_Day'].mean() * 100)
        outside_days_pct = float(self.df['Outside_Day'].mean() * 100)
        gap_up_pct = float(self.df['Gap_Up'].mean() * 100)
        gap_down_pct = float(self.df['Gap_Down'].mean() * 100)

        # Comparative Predictors
        predictor_scores = {
            'Pivot': float(self.df['Pivot_Predicts'].mean()),
            'Previous High': float(self.df['Prev_H_Predicts'].mean()),
            'Previous Low': float(self.df['Prev_L_Predicts'].mean()),
            'Previous Close': float(self.df['Prev_C_Predicts'].mean())
        }
        best_predictor = max(predictor_scores, key=predictor_scores.get) # type: ignore

        overall_accuracy = float(self.df['Pivot_Predicts'].mean() * 100)
        
        # Market Condition Analysis
        gap_up_subset = self.df[self.df['Gap_Up']]
        gap_down_subset = self.df[self.df['Gap_Down']]
        
        gap_up_acc = gap_up_subset['Pivot_Predicts'].mean() if not gap_up_subset.empty else 0
        gap_down_acc = gap_down_subset['Pivot_Predicts'].mean() if not gap_down_subset.empty else 0
        
        best_condition = "Gap Up Environment" if gap_up_acc >= gap_down_acc else "Gap Down Environment"
        worst_condition = "Inside Day Range" if self.df['Inside_Day'].sum() > 0 else "Choppy Range Day"

        confidence_score = float(min(overall_accuracy + (pivot_respect_pct / 10.0), 100.0))

        self.stats_result = {
            "Total_Days_Analyzed": total_days,
            "Pivot_Respect_Percent": pivot_respect_pct,
            "Bullish_Accuracy_Percent": bullish_acc,
            "Bearish_Accuracy_Percent": bearish_acc,
            "False_Breakout_Percent": false_breakout_pct,
            "Breakout_Percent": breakout_pct,
            "Reversal_Percent": reversal_pct,
            "Average_Move": avg_move,
            "Median_Move": median_move,
            "Mean_Move": mean_move,
            "Mode_Move": mode_move,
            "Maximum_Move": max_move,
            "Minimum_Move": min_move,
            "Standard_Deviation": std_dev,
            "Variance": variance,
            "Trending_Days_Percent": trend_days_pct,
            "Range_Days_Percent": range_days_pct,
            "Inside_Days_Percent": inside_days_pct,
            "Outside_Days_Percent": outside_days_pct,
            "Gap_Up_Percent": gap_up_pct,
            "Gap_Down_Percent": gap_down_pct,
            "Best_Predictor": best_predictor,
            "Predictor_Scores": predictor_scores,
            "Overall_Trend_Prediction_Accuracy": overall_accuracy,
            "Best_Market_Condition": best_condition,
            "Worst_Market_Condition": worst_condition,
            "Confidence_Score": confidence_score
        }
        
        self.logger.info("Statistical metrics computation successfully finished.")
        return self.stats_result