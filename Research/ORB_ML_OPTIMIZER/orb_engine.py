import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import config

class ORBEngine:
    def __init__(self, data: pd.DataFrame, logger: logging.Logger):
        self.data = data
        self.logger = logger
        self.daily_features: pd.DataFrame = pd.DataFrame()

    def prepare_market_features(self) -> pd.DataFrame:
        """Calculates all foundational technical indicators, gaps, previous day context, and volume metrics."""
        self.logger.info("Engineering comprehensive market structure, gap, and volatility features...")
        
        # Resample 1-minute data to daily candles to get previous day metrics
        daily = self.data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Shift to get previous day reference values
        daily['prev_open'] = daily['open'].shift(1)
        daily['prev_high'] = daily['high'].shift(1)
        daily['prev_low'] = daily['low'].shift(1)
        daily['prev_close'] = daily['close'].shift(1)
        daily['prev_volume'] = daily['volume'].shift(1)
        daily['prev_range'] = daily['prev_high'] - daily['prev_low']
        
        # Standard Pivot Points
        daily['prev_pivot'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3.0
        
        # Day classifications
        daily['prev_trend'] = np.where(daily['prev_close'] > daily['prev_open'], 'Trend', 'Range')
        daily['prev_inside'] = (daily['prev_high'] <= daily['prev_high'].shift(1)) & (daily['prev_low'] >= daily['prev_low'].shift(1))
        daily['prev_outside'] = (daily['prev_high'] >= daily['prev_high'].shift(1)) & (daily['prev_low'] <= daily['prev_low'].shift(1))
        
        # Merge back to intraday dataframe
        self.data['date_key'] = self.data.index.date
        daily['date_key'] = daily.index.date
        
        cols_to_merge = [
            'date_key', 'prev_open', 'prev_high', 'prev_low', 'prev_close', 
            'prev_volume', 'prev_range', 'prev_pivot', 'prev_trend', 'prev_inside', 'prev_outside'
        ]
        
        merged = pd.merge(
            self.data.reset_index(),
            daily[cols_to_merge],
            on='date_key',
            how='inner'
        )
        merged.set_index('datetime', inplace=True)
        merged.sort_index(inplace=True)
        
        # Calculate VWAP intraday
        merged['typical_price'] = (merged['high'] + merged['low'] + merged['close']) / 3.0
        merged['tp_vol'] = merged['typical_price'] * merged['volume']
        
        # Group by date to calculate cumulative intraday VWAP and ATR proxy
        merged['date_group'] = merged.index.date
        merged['cum_tp_vol'] = merged.groupby('date_group')['tp_vol'].cumsum()
        merged['cum_vol'] = merged.groupby('date_group')['volume'].cumsum()
        merged['vwap'] = merged['cum_tp_vol'] / merged['cum_vol']
        
        # True Range and ATR proxy
        merged['prev_close_price'] = merged['close'].shift(1)
        merged['tr'] = np.maximum(
            merged['high'] - merged['low'],
            np.maximum(
                abs(merged['high'] - merged['prev_close_price']),
                abs(merged['low'] - merged['prev_close_price'])
            )
        )
        merged['atr'] = merged.groupby('date_group')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        
        self.data = merged
        self.logger.info("Feature engineering and market structure mapping complete.")
        return self.data

    def simulate_orb_strategy(
        self, 
        orb_minutes: int, 
        stop_loss_pts: float, 
        trail_stop_pts: float, 
        target_type: str,
        time_filter: str
    ) -> List[Dict[str, Any]]:
        """Simulates Opening Range Breakout across all trading sessions with strict transaction cost accounting."""
        trade_results = []
        
        # Group data by date_group to loop through each trading session cleanly
        grouped = self.data.groupby('date_group')
        
        for date, group in grouped:
            if len(group) < config.MIN_MINUTES_PER_DAY:
                continue
                
            # Match the timezone of the dataframe index to prevent comparison errors
            tz = group.index.tz
            market_open_time = pd.Timestamp(date).normalize() + pd.Timedelta(hours=9, minutes=15)
            if tz is not None:
                market_open_time = market_open_time.tz_localize(tz)
                
            orb_end_time = market_open_time + pd.Timedelta(minutes=orb_minutes)
            
            orb_data = group[(group.index >= market_open_time) & (group.index < orb_end_time)]
            trade_data = group[group.index >= orb_end_time]
            
            if orb_data.empty or trade_data.empty:
                continue
                
            orb_high = orb_data['high'].max()
            orb_low = orb_data['low'].min()
            orb_range = orb_high - orb_low
            
            if orb_range <= 0:
                continue
                
            # Check time filter for trade activation window
            filter_hour, filter_min = map(int, time_filter.split(':'))
            activation_time = pd.Timestamp(date).normalize() + pd.Timedelta(hours=filter_hour, minutes=filter_min)
            if tz is not None:
                activation_time = activation_time.tz_localize(tz)
            
            active_trade_data = trade_data[trade_data.index >= activation_time]
            if active_trade_data.empty:
                continue
                
            # Scan for breakout
            position = None  # 1 for Long, -1 for Short
            entry_price = 0.0
            entry_time = None
            stop_loss = 0.0
            target_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            
            for idx, row in active_trade_data.iterrows():
                c_price = row['close']
                h_price = row['high']
                l_price = row['low']
                
                if position is None:
                    # Long Breakout
                    if c_price > orb_high:
                        position = 1
                        entry_price = c_price * (1 + config.SLIPPAGE_PCT)
                        entry_time = idx
                        stop_loss = entry_price - stop_loss_pts
                        highest_since_entry = entry_price
                        
                        if target_type == "No Target":
                            target_price = float('inf')
                        else:
                            r_mult = float(target_type.replace('R', ''))
                            target_price = entry_price + (orb_range * r_mult)
                        break
                        
                    # Short Breakout
                    elif c_price < orb_low:
                        position = -1
                        entry_price = c_price * (1 - config.SLIPPAGE_PCT)
                        entry_time = idx
                        stop_loss = entry_price + stop_loss_pts
                        lowest_since_entry = entry_price
                        
                        if target_type == "No Target":
                            target_price = float('-inf')
                        else:
                            r_mult = float(target_type.replace('R', ''))
                            target_price = entry_price - (orb_range * r_mult)
                        break
                
                elif position == 1:
                    # Update trailing stop for long
                    if h_price > highest_since_entry:
                        highest_since_entry = h_price
                        if trail_stop_pts > 0:
                            potential_trail = highest_since_entry - trail_stop_pts
                            if potential_trail > stop_loss:
                                stop_loss = potential_trail
                                
                    # Check exit conditions
                    if l_price <= stop_loss:
                        exit_price = stop_loss * (1 - config.SLIPPAGE_PCT)
                        trade_results.append(self._compile_trade(date, position, entry_price, exit_price, entry_time, idx, orb_range, group))
                        position = None
                        break
                    elif h_price >= target_price:
                        exit_price = target_price
                        trade_results.append(self._compile_trade(date, position, entry_price, exit_price, entry_time, idx, orb_range, group))
                        position = None
                        break
                        
                elif position == -1:
                    # Update trailing stop for short
                    if l_price < lowest_since_entry:
                        lowest_since_entry = l_price
                        if trail_stop_pts > 0:
                            potential_trail = lowest_since_entry + trail_stop_pts
                            if potential_trail < stop_loss:
                                stop_loss = potential_trail
                                
                    # Check exit conditions
                    if h_price >= stop_loss:
                        exit_price = stop_loss * (1 + config.SLIPPAGE_PCT)
                        trade_results.append(self._compile_trade(date, position, entry_price, exit_price, entry_time, idx, orb_range, group))
                        position = None
                        break
                    elif l_price <= target_price:
                        exit_price = target_price
                        trade_results.append(self._compile_trade(date, position, entry_price, exit_price, entry_time, idx, orb_range, group))
                        position = None
                        break
            
            # Close open trade at market close if active
            if position is not None:
                final_row = group.iloc[-1]
                exit_price = final_row['close']
                trade_results.append(self._compile_trade(date, position, entry_price, exit_price, entry_time, group.index[-1], orb_range, group))

        return trade_results

    def _compile_trade(self, date, position, entry, exit, entry_time, exit_time, orb_range, day_group) -> Dict[str, Any]:
        """Calculates P&L, transaction frictions, R-multiple, and pre-trade contextual features."""
        qty = config.INITIAL_CAPITAL / entry
        gross_pnl = (exit - entry) * qty if position == 1 else (entry - exit) * qty
        
        turnover = (entry + exit) * qty
        brokerage = turnover * config.BROKERAGE_PCT
        exchange = turnover * config.EXCHANGE_CHARGES_PCT
        gst = (brokerage + exchange) * config.GST_PCT
        stt = exit * qty * config.STT_PCT
        stamp = entry * qty * config.STAMP_DUTY_PCT
        frictions = brokerage + exchange + gst + stt + stamp
        
        net_pnl = gross_pnl - frictions
        win = 1 if net_pnl > 0 else 0
        
        # Extract pre-trade features for ML feature matrix
        first_row = day_group.iloc[0]
        gap_val = first_row['open'] - first_row['prev_close']
        gap_pct = (gap_val / first_row['prev_close']) * 100
        
        return {
            'date': date,
            'position': position,
            'entry_price': entry,
            'exit_price': exit,
            'net_pnl': net_pnl,
            'win': win,
            'orb_range': orb_range,
            'gap_pct': gap_pct,
            'above_vwap': 1 if entry > first_row['vwap'] else 0,
            'above_pivot': 1 if entry > first_row['prev_pivot'] else 0,
            'day_of_week': pd.Timestamp(date).dayofweek,
            'atr': first_row['atr'],
            'volume': first_row['volume']
        }