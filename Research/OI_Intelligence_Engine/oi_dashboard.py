"""
Module: oi_dashboard.py
Description: Live terminal dashboard interface for OI analytics.
"""

class OIDashboard:
    def render_console_dashboard(self, live_data: dict, decision_output: dict):
        print("="*60)
        print("         AI QUANT TRADING PLATFORM - LIVE OI DASHBOARD         ")
        print("="*60)
        print(f" Timestamp           : {live_data.get('timestamp')}")
        print(f" Call OI / Put OI    : {live_data.get('call_oi', 0):,} / {live_data.get('put_oi', 0):,}")
        print(f" Put-Call Ratio (PCR): {live_data.get('pcr', 0):.2f}")
        print(f" OI Classification   : {live_data.get('classification', {}).get('primary_classification')}")
        print(f" Confidence Score    : {live_data.get('classification', {}).get('confidence_score')}%")
        print(f" Market Regime       : {live_data.get('market_regime')}")
        print("-" * 60)
        print(f" AI Trade Decision   : {'TRADE (YES)' if decision_output.get('trade') else 'NO TRADE'}")
        print(f" Selected Strategy   : {decision_output.get('strategy')} ({decision_output.get('strategy_score')}% match)")
        print(f" Decision Reasons    : {', '.join(decision_output.get('reasons', []))}")
        print("="*60)
