"""
WARNING: This module generates FULLY SYNTHETIC / RANDOM output (np.random-based
decisions, win rates, confidence, etc. - see run_engine() below) for demonstration
/ report-template purposes only. It does not read real market data, real backtest
results, or any other engine's real output. Do NOT use its output for real
trading decisions. Not wired into the live trading path (step9_live_engine.py
computes its own real, live-data-driven confidence/strategy decisions instead).
"""
import os
import pandas as pd
import numpy as np
import logging
import config

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DecisionEngine")

class MasterDecisionEngine:
    def __init__(self):
        config.init_directories()

    def run_engine(self):
        logger.info("Initializing Master Decision Engine...")
        
        # Load datasets safely or create synthetic aligned frames if paths differ
        if os.path.exists(config.MARKET_DNA_PATH):
            dna = pd.read_csv(config.MARKET_DNA_PATH)
        else:
            dna = pd.DataFrame({"timestamp": pd.date_range(end="2026-08-03", periods=100), "market_archetype": "Bullish Trend"})
            
        if os.path.exists(config.MARKET_REGIMES_PATH):
            regimes = pd.read_csv(config.MARKET_REGIMES_PATH)
        else:
            regimes = pd.DataFrame({"date": pd.date_range(end="2026-08-03", periods=100), "regime": "High Momentum"})

        # Build master decision dataframe
        dates = pd.date_range(end="2026-08-03", periods=50, freq='B')
        
        records = []
        np.random.seed(42)
        for d in dates:
            conf = np.random.uniform(20, 95)
            trade_flag = "TRADE" if conf >= 50 else "NO TRADE"
            strategy = np.random.choice(["Opening Drive", "Velez"]) if trade_flag == "TRADE" else "NONE"
            
            reasons = [
                "Market DNA trend alignment score supports momentum continuation.",
                "Volatility regime is stable within historical tolerances.",
                "Opening range breakout volume exceeds 20-day institutional average.",
                "VWAP distance confirms lack of overextension.",
                "Historical backtest expectancy for this setup exceeds 1.8 Profit Factor."
            ] if trade_flag == "TRADE" else [
                "Low volume node detected during opening auction.",
                "Conflicting signals between Market DNA archetype and Velez momentum vectors.",
                "High structural uncertainty in current regime classification.",
                "Poor historical expectancy during similar volatility contraction phases.",
                "Weak Market DNA directional conviction."
            ]

            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Market Type": np.random.choice(["Trending", "Mean Reverting", "Breakout", "Choppy"]),
                "Market DNA Summary": "Bullish Momentum Vector Active" if trade_flag=="TRADE" else "Consolidation / Chop Zone",
                "Selected Strategy": strategy,
                "Confidence %": round(conf, 2),
                "Expected Win Rate": round(np.random.uniform(65, 85) if trade_flag=="TRADE" else 35, 2),
                "Expected Profit Factor": round(np.random.uniform(2.1, 4.5) if trade_flag=="TRADE" else 0.8, 2),
                "Expected Drawdown": round(np.random.uniform(0.5, 2.0), 2),
                "Expected Reward": round(np.random.uniform(1.2, 3.0), 2),
                "Suggested Position Size": "1 Lot (Index F&O)",
                "Suggested Stop Loss": "45 Points",
                "Suggested Target": "110 Points",
                "Expected Holding Time": "35 Mins",
                "Risk Score": round(np.random.uniform(1.0, 4.5), 2),
                "Trade Quality Score (0-100)": round(conf * 1.05 if conf < 95 else 98, 2),
                "Final Recommendation": trade_flag,
                "Reason": " | ".join(reasons)
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(config.REPORTS_DIR, "AI_Trade_Decision_Master_Report.csv"), index=False)
        logger.info("Master decision report generated successfully.")
        return df

    def get_capital_risk_matrix(self):
        levels = [25000, 50000, 100000, 200000, 500000, 1000000]
        matrix = []
        for cap in levels:
            matrix.append({
                "Capital (?)": f"?{cap:,}",
                "Suggested Lots": max(1, cap // 25000),
                "Maximum Risk per Trade (?)": f"?{int(cap * 0.02):,}",
                "Maximum Daily Loss (?)": f"?{int(cap * 0.05):,}",
                "Max Consecutive Losses": 3
            })
        return pd.DataFrame(matrix)
