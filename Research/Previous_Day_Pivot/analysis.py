import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import config

class MarketAnalyzer:
    def __init__(self, data: pd.DataFrame, logger: logging.Logger):
        self.data = data
        self.logger = logger
        self.daily_metrics: List[Dict[str, Any]] = []

    def calculate_trade_frictions(self, entry_price: float, exit_price: float) -> Dict[str, float]:
        """Calculates realistic costs for a complete round-trip trade based on config parameters."""
        qty = config.INITIAL_CAPITAL / entry_price
        turnover = (entry_price + exit_price) * qty
        
        brokerage = turnover * config.BROKERAGE_PCT
        exchange_charges = turnover * config.EXCHANGE_CHARGES_PCT
        gst = (brokerage + exchange_charges) * config.GST_PCT
        stt = exit_price * qty * config.STT_PCT
        stamp_duty = entry_price * qty * config.STAMP_DUTY_PCT
        slippage_cost = turnover * config.SLIPPAGE_PCT
        
        total_costs = brokerage + exchange_charges + gst + stt + stamp_duty + slippage_cost
        
        return {
            'total_frictions': total_costs,
            'slippage': slippage_cost,
            'taxes_and_fees': total_costs - slippage_cost
        }

    def analyze_day(self, date: pd.Timestamp, group: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Performs deep analysis on a single trading day's 1-minute data."""
        if len(group) < config.MIN_MINUTES_PER_DAY:
            return None

        # Daily OHLC
        day_open = float(group['open'].iloc[0])
        day_high = float(group['high'].max())
        day_low = float(group['low'].min())
        day_close = float(group['close'].iloc[-1])
        
        # Pivot Levels
        pivot = float(group['pivot'].iloc[0])
        r1 = float(group['r1'].iloc[0])
        s1 = float(group['s1'].iloc[0])
        
        # Previous Day Levels
        prev_h = float(group['prev_high'].iloc[0])
        prev_l = float(group['prev_low'].iloc[0])
        prev_c = float(group['prev_close'].iloc[0])

        # Core Metrics
        open_above_pivot = day_open > pivot
        open_below_pivot = day_open < pivot
        
        crosses_pivot = (group['high'] >= pivot) & (group['low'] <= pivot)
        num_pivot_touches = int(crosses_pivot.sum())
        
        time_above_pivot = int((group['close'] > pivot).sum())
        time_below_pivot = int((group['close'] < pivot).sum())
        
        max_move_above = max(day_high - pivot, 0.0) if day_high > pivot else 0.0
        max_move_below = max(pivot - day_low, 0.0) if day_low < pivot else 0.0
        
        distance_travelled = float(group['close'].diff().abs().sum())
        
        # Trend Definitions
        if day_close > day_open:
            intraday_trend = "Bullish"
        elif day_close < day_open:
            intraday_trend = "Bearish"
        else:
            intraday_trend = "Neutral"

        total_range = day_high - day_low
        trend_strength = abs(day_close - day_open) / total_range if total_range != 0 else 0.0
        
        gap_up = day_open > prev_c
        gap_down = day_open < prev_c
        
        # Market Conditions
        open_inside = prev_l <= day_open <= prev_h
        inside_day = (day_high <= prev_h) and (day_low >= prev_l)
        outside_day = (day_high >= prev_h) and (day_low <= prev_l)
        range_day = total_range < (prev_h - prev_l) * 0.8
        trending_day = trend_strength > 0.6
        
        # Probabilities & Behaviors
        return_to_pivot = (open_above_pivot and day_low <= pivot) or (open_below_pivot and day_high >= pivot)
        breakout = (day_close > r1) or (day_close < s1)
        reversal = (gap_up and day_close < day_open) or (gap_down and day_close > day_open)
        
        # Predictor Tests (Does opening above/below level X predict the daily trend?)
        pivot_predicts = (open_above_pivot and intraday_trend == "Bullish") or (open_below_pivot and intraday_trend == "Bearish")
        prev_h_predicts = (day_open > prev_h and intraday_trend == "Bullish")
        prev_l_predicts = (day_open < prev_l and intraday_trend == "Bearish")
        prev_c_predicts = (day_open > prev_c and intraday_trend == "Bullish") or (day_open < prev_c and intraday_trend == "Bearish")

        # Trading Frictions
        costs = self.calculate_trade_frictions(day_open, day_close)

        return {
            'Date': date,
            'Day_Open': day_open, 
            'Day_High': day_high, 
            'Day_Low': day_low, 
            'Day_Close': day_close,
            'Pivot': pivot, 
            'R1': r1, 
            'S1': s1,
            'Open_Above_Pivot': open_above_pivot,
            'Open_Below_Pivot': open_below_pivot,
            'Num_Pivot_Touches': num_pivot_touches,
            'Time_Above_Pivot': time_above_pivot,
            'Time_Below_Pivot': time_below_pivot,
            'Max_Move_Above_Pivot': max_move_above,
            'Max_Move_Below_Pivot': max_move_below,
            'Distance_Travelled': distance_travelled,
            'Intraday_Trend': intraday_trend,
            'Trend_Strength': trend_strength,
            'Reversal_Prob': reversal,
            'Breakout_Prob': breakout,
            'Return_To_Pivot': return_to_pivot,
            'Gap_Up': gap_up,
            'Gap_Down': gap_down,
            'Open_Inside_Prev': open_inside,
            'Inside_Day': inside_day,
            'Outside_Day': outside_day,
            'Range_Day': range_day,
            'Trending_Day': trending_day,
            'Pivot_Predicts': pivot_predicts,
            'Prev_H_Predicts': prev_h_predicts,
            'Prev_L_Predicts': prev_l_predicts,
            'Prev_C_Predicts': prev_c_predicts,
            'Total_Frictions': costs['total_frictions']
        }

    def run_analysis(self) -> pd.DataFrame:
        self.logger.info("Starting deeply granular intraday analysis on 1-minute timeframe...")
        
        # Extract unique dates manually handling the DatetimeIndex
        self.data['date_only'] = self.data.index.date
        unique_dates = self.data['date_only'].unique()
        
        total_days = len(unique_dates)
        processed = 0

        for d in unique_dates:
            group = self.data[self.data['date_only'] == d]
            metrics = self.analyze_day(d, group)
            
            if metrics:
                self.daily_metrics.append(metrics)
                
            processed += 1
            if processed % 20 == 0 or processed == total_days:
                self.logger.info(f"Analyzed {processed}/{total_days} trading days...")
                
        # Clean up the temporary column
        self.data.drop(columns=['date_only'], inplace=True)
                
        metrics_df = pd.DataFrame(self.daily_metrics)
        if not metrics_df.empty:
            metrics_df.set_index('Date', inplace=True)
            self.logger.info(f"Analysis complete. Total valid days processed: {len(metrics_df)}")
        else:
            self.logger.warning("Analysis produced no valid output days.")
            
        return metrics_df