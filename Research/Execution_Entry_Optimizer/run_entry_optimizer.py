r"""
Project: Execution_Entry_Optimizer
Folder: C:\dhan algo\Research\Execution_Entry_Optimizer
Description: Standalone Research Engine evaluating limit order entry offsets vs immediate market execution.
"""

import os
import sys
import math
import datetime
import numpy as np
import pandas as pd

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Define Paths
BASE_DIR = r"C:\dhan algo\Research\Execution_Entry_Optimizer"
DATA_PATH = r"C:\dhan algo\dhan_historical_data_180days.csv"
OUTPUT_CSV = os.path.join(BASE_DIR, "execution_optimizer_results.csv")
OUTPUT_XLSX = os.path.join(BASE_DIR, "execution_optimizer_summary.xlsx")
OUTPUT_HTML = os.path.join(BASE_DIR, "execution_dashboard.html")
OUTPUT_TXT = os.path.join(BASE_DIR, "Summary.txt")

os.makedirs(BASE_DIR, exist_ok=True)

# Configuration Parameters
INITIAL_CAPITAL = 100000.0
BROKERAGE_PER_TRADE = 40.0  # Rs 20 per leg
SLIPPAGE_POINTS = 0.5       # Slippage on market orders/stops
OFFSETS = list(range(0, 11)) # 0 (Market) to 10 points better

# ----------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ----------------------------------------------------------------------
def load_historical_data(path):
    if not os.path.exists(path):
        print(f"[*] Generating synthetic historical market data at {path}...")
        dates = pd.date_range(end=datetime.datetime.now(), periods=180*375, freq='1min')
        np.random.seed(42)
        price = 22000.0
        data = []
        for dt in dates:
            if dt.time() < datetime.time(9, 15) or dt.time() > datetime.time(15, 30):
                continue
            ret = np.random.normal(0.00001, 0.0008)
            price *= (1 + ret)
            high = price * (1 + abs(np.random.normal(0, 0.0004)))
            low = price * (1 - abs(np.random.normal(0, 0.0004)))
            close = np.random.uniform(low, high)
            volume = int(np.random.exponential(1000))
            data.append([dt, price, high, low, close, volume])
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # Normalize Column Names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Handle separate 'date' and 'time' or single timestamp
    if 'date' in df.columns and 'time' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    else:
        time_col = None
        for candidate in ['timestamp', 'datetime', 'date', 'time_stamp']:
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col:
            df.rename(columns={time_col: 'timestamp'}, inplace=True)
            
            # Auto-detect Unix Epoch timestamp vs Datetime string
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                first_val = df['timestamp'].iloc[0]
                if first_val > 1e11:  # Milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:  # Seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"[*] Data Loaded Successfully: {len(df):,} rows.")
    print(f"[*] Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

# ----------------------------------------------------------------------
# 2. ADAPTIVE SIGNAL GENERATION ENGINES
# ----------------------------------------------------------------------
def generate_signals(df):
    signals = []
    
    # Calculate indicators
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['date_only'] = df['timestamp'].dt.date
    df['time_only'] = df['timestamp'].dt.time
    
    grouped = df.groupby('date_only')
    
    for date, group in grouped:
        if len(group) < 10:
            continue
            
        # Strategy 1: Opening Range Breakout (ORB)
        orb_bars = group.iloc[:15]
        orb_high = orb_bars['high'].max()
        orb_low = orb_bars['low'].min()
        
        post_orb = group.iloc[15:]
        for global_idx, row in post_orb.iterrows():
            if row['close'] > orb_high:
                signals.append({
                    'strategy': 'ORB',
                    'sig_idx': global_idx,
                    'timestamp': row['timestamp'],
                    'signal_price': row['close'],
                    'direction': 'BUY',
                    'target': row['close'] + max((orb_high - orb_low), 10.0),
                    'stop_loss': orb_low,
                    'max_hold': 45
                })
                break
            elif row['close'] < orb_low:
                signals.append({
                    'strategy': 'ORB',
                    'sig_idx': global_idx,
                    'timestamp': row['timestamp'],
                    'signal_price': row['close'],
                    'direction': 'SELL',
                    'target': row['close'] - max((orb_high - orb_low), 10.0),
                    'stop_loss': orb_high,
                    'max_hold': 45
                })
                break

        # Strategy 2: Opening Drive (First 5 Bars)
        od_bars = group.iloc[:5]
        if len(od_bars) >= 5:
            first_bar = od_bars.iloc[0]
            last_od = od_bars.iloc[-1]
            od_global_idx = last_od.name
            
            move = last_od['close'] - first_bar['open']
            if move > 0:
                signals.append({
                    'strategy': 'Opening Drive',
                    'sig_idx': od_global_idx,
                    'timestamp': last_od['timestamp'],
                    'signal_price': last_od['close'],
                    'direction': 'BUY',
                    'target': last_od['close'] + 20.0,
                    'stop_loss': last_od['close'] - 10.0,
                    'max_hold': 30
                })
            elif move < 0:
                signals.append({
                    'strategy': 'Opening Drive',
                    'sig_idx': od_global_idx,
                    'timestamp': last_od['timestamp'],
                    'signal_price': last_od['close'],
                    'direction': 'SELL',
                    'target': last_od['close'] - 20.0,
                    'stop_loss': last_od['close'] + 10.0,
                    'max_hold': 30
                })

        # Strategy 3: Velez (20 EMA Rebound)
        velez_candidates = group.iloc[15:]
        for i in range(1, len(velez_candidates)):
            curr = velez_candidates.iloc[i]
            prev = velez_candidates.iloc[i-1]
            curr_global_idx = curr.name
            
            if prev['close'] <= prev['ema20'] and curr['close'] > curr['ema20']:
                signals.append({
                    'strategy': 'Velez',
                    'sig_idx': curr_global_idx,
                    'timestamp': curr['timestamp'],
                    'signal_price': curr['close'],
                    'direction': 'BUY',
                    'target': curr['close'] + 15.0,
                    'stop_loss': curr['low'] - 2.0,
                    'max_hold': 30
                })
            elif prev['close'] >= prev['ema20'] and curr['close'] < curr['ema20']:
                signals.append({
                    'strategy': 'Velez',
                    'sig_idx': curr_global_idx,
                    'timestamp': curr['timestamp'],
                    'signal_price': curr['close'],
                    'direction': 'SELL',
                    'target': curr['close'] - 15.0,
                    'stop_loss': curr['high'] + 2.0,
                    'max_hold': 30
                })

    return pd.DataFrame(signals)

# ----------------------------------------------------------------------
# 3. REPLAY ENGINE
# ----------------------------------------------------------------------
def simulate_variant(df, signals, strategy_name, offset):
    empty_result = {
        'Strategy': strategy_name,
        'Offset': f"+{offset}" if offset > 0 else "Market",
        'Numeric_Offset': offset,
        'Total Signals': 0,
        'Executed Trades': 0,
        'Missed Trades': 0,
        'Fill Rate %': 0.0,
        'Miss Rate %': 100.0,
        'Win Rate %': 0.0,
        'Loss Rate %': 0.0,
        'Net Profit': 0.0,
        'Gross Profit': 0.0,
        'Gross Loss': 0.0,
        'Profit Factor': 0.0,
        'Avg Win': 0.0,
        'Avg Loss': 0.0,
        'Avg Improvement (Pts)': 0.0,
        'Max Drawdown': 0.0,
        'Sharpe Ratio': 0.0,
        'Sortino Ratio': 0.0,
        'Calmar Ratio': 0.0,
        'SQN': 0.0,
        'Expectancy': 0.0,
        'Avg Hold Time (m)': 0.0,
        'Largest Winner': 0.0,
        'Largest Loser': 0.0,
        'Entry Quality Score': 0.0,
        'Execution Efficiency': 0.0
    }

    if signals.empty or 'strategy' not in signals.columns:
        return empty_result

    strat_signals = signals[signals['strategy'] == strategy_name].copy()
    if strat_signals.empty:
        return empty_result

    executed_trades = []
    missed_count = 0

    for _, sig in strat_signals.iterrows():
        sig_idx = sig['sig_idx']
        direction = sig['direction']
        sig_price = sig['signal_price']
        target = sig['target']
        sl = sig['stop_loss']
        max_hold = sig['max_hold']
        
        limit_price = sig_price if offset == 0 else ((sig_price - offset) if direction == 'BUY' else (sig_price + offset))

        future_bars = df.iloc[sig_idx + 1 : sig_idx + 1 + max_hold]
        if future_bars.empty:
            missed_count += 1
            continue
        
        filled = False
        fill_price = 0.0
        fill_idx = None
        
        if offset == 0:
            filled = True
            fill_price = sig_price + (SLIPPAGE_POINTS if direction == 'BUY' else -SLIPPAGE_POINTS)
            fill_idx = sig_idx + 1
            eval_bars = future_bars
        else:
            for idx, bar in future_bars.iterrows():
                if direction == 'BUY' and bar['low'] <= limit_price:
                    filled = True
                    fill_price = limit_price
                    fill_idx = idx
                    break
                elif direction == 'SELL' and bar['high'] >= limit_price:
                    filled = True
                    fill_price = limit_price
                    fill_idx = idx
                    break
            
            if filled:
                eval_bars = df.iloc[fill_idx : sig_idx + 1 + max_hold]
            else:
                missed_count += 1
                continue

        if eval_bars.empty:
            missed_count += 1
            continue

        exit_price = 0.0
        
        for idx, bar in eval_bars.iterrows():
            if direction == 'BUY':
                if bar['low'] <= sl:
                    exit_price = sl - SLIPPAGE_POINTS
                    break
                elif bar['high'] >= target:
                    exit_price = target
                    break
            else:
                if bar['high'] >= sl:
                    exit_price = sl + SLIPPAGE_POINTS
                    break
                elif bar['low'] <= target:
                    exit_price = target
                    break
                    
        if exit_price == 0.0:
            last_bar = eval_bars.iloc[-1]
            exit_price = last_bar['close']

        points = (exit_price - fill_price) if direction == 'BUY' else (fill_price - exit_price)
        gross_pnl = points * 50
        net_pnl = gross_pnl - BROKERAGE_PER_TRADE

        executed_trades.append({
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'holding_time': len(eval_bars),
            'improvement': (sig_price - fill_price) if direction == 'BUY' else (fill_price - sig_price)
        })

    total_signals = len(strat_signals)
    executed_count = len(executed_trades)
    fill_rate = (executed_count / total_signals * 100) if total_signals > 0 else 0
    miss_rate = 100 - fill_rate

    if executed_count == 0:
        empty_result.update({
            'Total Signals': total_signals,
            'Missed Trades': missed_count
        })
        return empty_result

    tdf = pd.DataFrame(executed_trades)
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] <= 0]
    
    win_rate = len(wins) / executed_count * 100
    loss_rate = 100 - win_rate
    net_profit = tdf['net_pnl'].sum()
    gross_profit = wins['net_pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses['net_pnl'].mean()) if len(losses) > 0 else 0.0
    
    tdf['cum_pnl'] = tdf['net_pnl'].cumsum()
    tdf['peak'] = tdf['cum_pnl'].cummax()
    tdf['drawdown'] = tdf['peak'] - tdf['cum_pnl']
    max_drawdown = tdf['drawdown'].max()
    
    returns = tdf['net_pnl']
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    downside_std = returns[returns < 0].std()
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    calmar = (net_profit / max_drawdown) if max_drawdown > 0 else net_profit
    sqn = (np.sqrt(executed_count) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    expectancy = returns.mean()

    avg_improvement = tdf['improvement'].mean()
    entry_quality_score = round(win_rate * profit_factor, 2)
    execution_efficiency = round((net_profit / (executed_count * 50)) if executed_count > 0 else 0, 2)

    return {
        'Strategy': strategy_name,
        'Offset': f"+{offset}" if offset > 0 else "Market",
        'Numeric_Offset': offset,
        'Total Signals': total_signals,
        'Executed Trades': executed_count,
        'Missed Trades': missed_count,
        'Fill Rate %': round(fill_rate, 2),
        'Miss Rate %': round(miss_rate, 2),
        'Win Rate %': round(win_rate, 2),
        'Loss Rate %': round(loss_rate, 2),
        'Net Profit': round(net_profit, 2),
        'Gross Profit': round(gross_profit, 2),
        'Gross Loss': round(gross_loss, 2),
        'Profit Factor': round(profit_factor, 2),
        'Avg Win': round(avg_win, 2),
        'Avg Loss': round(avg_loss, 2),
        'Avg Improvement (Pts)': round(avg_improvement, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2),
        'Calmar Ratio': round(calmar, 2),
        'SQN': round(sqn, 2),
        'Expectancy': round(expectancy, 2),
        'Avg Hold Time (m)': round(tdf['holding_time'].mean(), 1),
        'Largest Winner': round(tdf['net_pnl'].max(), 2),
        'Largest Loser': round(tdf['net_pnl'].min(), 2),
        'Entry Quality Score': entry_quality_score,
        'Execution Efficiency': execution_efficiency
    }

# ----------------------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    print("[*] Loading Historical Market Data...")
    df = load_historical_data(DATA_PATH)
    
    print("[*] Generating Strategy Replay Signals (ORB, Opening Drive, Velez)...")
    signals = generate_signals(df)
    print(f"[*] Total Signals Found: {len(signals)}")
    
    strategies = ['Opening Drive', 'Velez', 'ORB']
    all_results = []

    print("[*] Simulating Execution Variants across Offsets (+0 to +10)...")
    for strat in strategies:
        for offset in OFFSETS:
            res = simulate_variant(df, signals, strat, offset)
            all_results.append(res)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        results_df.to_excel(writer, sheet_name="All Variants", index=False)
        for strat in strategies:
            strat_df = results_df[results_df['Strategy'] == strat]
            strat_df.to_excel(writer, sheet_name=strat, index=False)

    master_bests = []
    for strat in strategies:
        strat_df = results_df[results_df['Strategy'] == strat]
        executed_df = strat_df[strat_df['Executed Trades'] > 0]
        
        if not executed_df.empty:
            best_idx = executed_df['Net Profit'].idxmax()
            best_row = executed_df.loc[best_idx]
            baseline_rows = strat_df[strat_df['Numeric_Offset'] == 0]
            baseline_profit = baseline_rows['Net Profit'].values[0] if not baseline_rows.empty else 0.0
            master_bests.append({
                'Strategy': strat,
                'Best Offset': best_row['Offset'],
                'Baseline Profit (Market)': baseline_profit,
                'Best Offset Profit': best_row['Net Profit'],
                'Profit Gain': best_row['Net Profit'] - baseline_profit,
                'Fill Rate %': best_row['Fill Rate %'],
                'Win Rate %': best_row['Win Rate %'],
                'Profit Factor': best_row['Profit Factor']
            })
        else:
            master_bests.append({
                'Strategy': strat,
                'Best Offset': 'N/A',
                'Baseline Profit (Market)': 0.0,
                'Best Offset Profit': 0.0,
                'Profit Gain': 0.0,
                'Fill Rate %': 0.0,
                'Win Rate %': 0.0,
                'Profit Factor': 0.0
            })

    master_df = pd.DataFrame(master_bests)
    
    # Use Numeric_Offset to resolve the comparison type issue
    global_best_numeric = results_df.groupby('Numeric_Offset')['Net Profit'].sum().idxmax()
    overall_market_profit = results_df[results_df['Numeric_Offset'] == 0]['Net Profit'].sum()
    best_offset_total_profit = results_df.groupby('Numeric_Offset')['Net Profit'].sum().max()
    
    use_limit_entries = best_offset_total_profit > overall_market_profit
    recommended_global_offset = f"+{global_best_numeric}" if (use_limit_entries and global_best_numeric > 0) else "Market (Offset 0)"

    with open(OUTPUT_TXT, "w") as f:
        f.write("====================================================================\n")
        f.write("           EXECUTION ENTRY OPTIMIZER - RESEARCH REPORT              \n")
        f.write("====================================================================\n\n")
        f.write(f"Global Recommendation: {'USE LIMIT ENTRIES' if use_limit_entries else 'STAY WITH MARKET ORDERS'}\n")
        f.write(f"Recommended Global Offset: {recommended_global_offset}\n\n")
        f.write("--- MASTER COMPARISON TABLE ---\n")
        f.write(master_df.to_string(index=False))
        f.write("\n\n====================================================================\n")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Execution Entry Optimizer Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px; }}
            h1, h2, h3 {{ color: #38bdf8; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #334155; padding: 10px; text-align: left; }}
            th {{ background-color: #0284c7; color: white; }}
            tr:nth-child(even) {{ background-color: #1e293b; }}
            .highlight {{ color: #4ade80; font-weight: bold; }}
            .badge {{ background: #2563eb; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Execution Entry Optimizer - Interactive Dashboard</h1>
        
        <div class="card">
            <h2>Global Optimal Entry Recommendation</h2>
            <p><strong>Primary Recommendation:</strong> <span class="badge">{"USE LIMIT ENTRIES" if use_limit_entries else "USE MARKET ORDERS"}</span></p>
            <p><strong>Optimal Global Offset:</strong> <span class="highlight">{recommended_global_offset}</span></p>
            <p><strong>Total Strategy Baseline Profit (Market):</strong> Rs {overall_market_profit:,.2f}</p>
            <p><strong>Total Strategy Optimized Profit:</strong> Rs {best_offset_total_profit:,.2f}</p>
            <p><strong>Net Optimization Alpha:</strong> <span class="highlight">Rs {best_offset_total_profit - overall_market_profit:,.2f}</span></p>
        </div>

        <div class="card">
            <h2>Strategy Performance Comparison</h2>
            {master_df.to_html(classes='table', index=False)}
        </div>

        <div class="card">
            <h2>All Tested Execution Variants</h2>
            {results_df.to_html(classes='table', index=False)}
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)

    print("\n====================================================================")
    print("                     RESEARCH OPTIMIZATION COMPLETE                 ")
    print("====================================================================")
    print(f"[*] CSV Exported:       {OUTPUT_CSV}")
    print(f"[*] Excel Report:       {OUTPUT_XLSX}")
    print(f"[*] Summary Text:       {OUTPUT_TXT}")
    print(f"[*] HTML Dashboard:     {OUTPUT_HTML}")
    print("====================================================================\n")

if __name__ == "__main__":
    main()