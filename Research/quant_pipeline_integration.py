"""
Module: quant_pipeline_integration.py
Description: Integrates the OI_Intelligence_Engine into the AI Quant Platform pipeline.
             Acts as a non-invasive middleware layer between Market Regime and Strategy Selector.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Setup robust logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuantPipelineIntegrator")

# Add OI Engine path to system path for backward compatibility & imports
OI_ENGINE_PATH = r"C:\dhan algo\Research\OI_Intelligence_Engine"
if OI_ENGINE_PATH not in sys.path:
    sys.path.append(OI_ENGINE_PATH)

try:
    from oi_engine import OIEngine
    OI_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not load OI Engine modules: {e}. Running in fallback mode.")
    OI_ENGINE_AVAILABLE = False


class QuantPipelineConfig:
    """
    Configuration settings for the quant platform pipeline execution.
    Requirement 5: Configuration settings to enable or disable OI filtering.
    """
    def __init__(self, enable_oi_filtering: bool = True):
        self.enable_oi_filtering = enable_oi_filtering


class QuantPipelineOrchestrator:
    """
    Orchestrates the sequential pipeline flow:
    Historical Data -> Data Validation -> Market DNA -> Market Regime 
    -> [OI Intelligence Layer] -> Strategy Selector -> AI Decision Engine -> Risk Engine -> Trade
    """
    def __init__(self, config: Optional[QuantPipelineConfig] = None):
        self.config = config or QuantPipelineConfig(enable_oi_filtering=True)
        self.oi_engine = OIEngine() if (OI_ENGINE_AVAILABLE and self.config.enable_oi_filtering) else None
        
        if self.oi_engine:
            logger.info("OI Intelligence Engine successfully initialized and plugged into pipeline.")
        else:
            logger.info("OI Intelligence Engine is disabled or unavailable. Pipeline will bypass OI filtering.")

    def run_pipeline(
        self, 
        historical_df: pd.DataFrame, 
        existing_strategy_selector_func, 
        market_dna_data: Dict[str, Any],
        market_regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the pipeline steps sequentially while maintaining backward compatibility
        and handling missing OI data gracefully.
        
        Integration Points Documented:
        -----------------------------
        1. Historical Data & Validation: Handled prior to passing data into orchestrator.
        2. Market DNA & Regime: Received as input dictionary/context objects.
        3. OI Intelligence Integration Point: Computes advanced derivatives features, 
           Z-scores, and classification metrics if enabled and data is valid.
        4. Strategy Selector Integration Point: Appends OI intelligence context to the 
           payload without overriding Opening Drive, Velez, or ORB logic.
        """
        
        # --- STEP 1 & 2: Historical Data, Validation, Market DNA & Regime (Passed In) ---
        pipeline_context = {
            "market_dna": market_dna_data,
            "market_regime": market_regime_data,
            "oi_intelligence": None
        }

        # --- STEP 3: Open Interest Intelligence Layer ---
        # Requirement 3: If OI data is unavailable, system continues working normally.
        if self.config.enable_oi_filtering and self.oi_engine is not None:
            try:
                # Validate required columns for OI calculations
                required_cols = {'call_oi', 'put_oi', 'close', 'open'}
                if historical_df is not None and not historical_df.empty and required_cols.issubset(historical_df.columns):
                    oi_result = self.oi_engine.process_tick(historical_df)
                    pipeline_context["oi_intelligence"] = oi_result
                    logger.info(f"OI Intelligence processed successfully. Classification: {oi_result.get('classification', {}).get('primary_classification')}")
                else:
                    logger.warning("OI data columns missing or historical dataframe empty. Bypassing OI layer safely.")
            except Exception as e:
                logger.error(f"Error encountered during OI Intelligence execution: {e}. Falling back to standard flow.")
                pipeline_context["oi_intelligence"] = {"status": "error_fallback", "error": str(e)}
        else:
            logger.debug("OI filtering is disabled via configuration.")

        # --- STEP 4: Strategy Selector AI ---
        # Requirement 1: Non-invasive. Existing Opening Drive, Velez, or ORB logic remains untouched.
        # We pass the enhanced context bundle alongside standard inputs.
        try:
            logger.info("Passing context into Strategy Selector AI...")
            decision_output = existing_strategy_selector_func(pipeline_context)
        except TypeError:
            # Maintain backward compatibility if the existing strategy function expects raw parameters instead of context dict
            logger.warning("Existing strategy selector does not accept pipeline context dictionary. Falling back to legacy invocation.")
            decision_output = existing_strategy_selector_func(historical_df)

        # --- STEP 5 - 9: AI Decision Engine, Risk Engine, Trade Execution ---
        final_execution_payload = {
            "pipeline_context": pipeline_context,
            "strategy_decision": decision_output,
            "execution_status": "READY_FOR_RISK_ENGINE"
        }
        
        logger.info("Pipeline execution completed successfully up to Risk Engine boundary.")
        return final_execution_payload