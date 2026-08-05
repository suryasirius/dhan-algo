"""
Minimal regression test suite for the core, pure-function-friendly modules that
feed the live engine. Run with:  python -m pytest Research/tests -v
(from C:\\dhan algo, or `pytest tests/` from inside Research\\).

This intentionally covers the modules step9_live_engine.py actually depends on
(OIBuilder, OIClassifier, ConfidenceAndRiskEngine, StrategySelectorAI, and the
LiveEngine static scoring functions) - the project had zero automated tests
before this file.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from step2_oi_features import OIBuilder
from step3_oi_classifier import OIClassifier
from step4_confidence_and_risk import ConfidenceAndRiskEngine
from step6_strategy_selector import StrategySelectorAI
from step9_live_engine import LiveEngine


# ---------------------------------------------------------------------------
# step2: OIBuilder
# ---------------------------------------------------------------------------
def _sample_oi_df():
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 101.5, 103.0],
        'close': [101.0, 102.0, 101.5, 103.0, 104.0],
        'call_oi': [1_000_000, 1_050_000, 1_100_000, 1_150_000, 1_200_000],
        'put_oi': [1_200_000, 1_220_000, 1_250_000, 1_300_000, 1_350_000],
        'call_vol': [20_000, 22_000, 25_000, 27_000, 30_000],
        'put_vol': [30_000, 32_000, 35_000, 38_000, 42_000],
    })


def test_oibuilder_empty_df_returns_empty():
    result = OIBuilder(window=3).compute_features(pd.DataFrame())
    assert result.empty


def test_oibuilder_adds_expected_columns():
    df = OIBuilder(window=3).compute_features(_sample_oi_df())
    for col in ['total_oi', 'oi_diff', 'pcr', 'oi_delta', 'z_score_oi', 'smart_money_score']:
        assert col in df.columns
    assert df['total_oi'].iloc[-1] == pytest.approx(1_200_000 + 1_350_000)
    # PCR should be > 1 here since put_oi > call_oi throughout
    assert (df['pcr'] > 1).all()


def test_oibuilder_no_nans_in_output():
    df = OIBuilder(window=3).compute_features(_sample_oi_df())
    assert not df.isna().any().any()


# ---------------------------------------------------------------------------
# step3: OIClassifier
# ---------------------------------------------------------------------------
def test_oiclassifier_long_buildup():
    row = pd.Series({'open': 100.0, 'close': 102.5, 'oi_delta': 15000.0,
                      'z_score_oi': 1.0, 'oi_expansion': 0, 'smart_money_score': 0.1})
    result = OIClassifier().classify_candle(row)
    assert result["classification"] == "Long Build-up"
    assert 0 <= result["confidence_score"] <= 100


def test_oiclassifier_fake_breakout_on_extreme_zscore():
    row = pd.Series({'open': 100.0, 'close': 100.0, 'oi_delta': 0.0,
                      'z_score_oi': 3.0, 'oi_expansion': 0, 'smart_money_score': 0.0})
    result = OIClassifier().classify_candle(row)
    assert result["classification"] == "Fake Breakout / OI Compression"


def test_oiclassifier_confirmed_breakout_boosts_confidence():
    row = pd.Series({'open': 100.0, 'close': 102.5, 'oi_delta': 15000.0,
                      'z_score_oi': 1.8, 'oi_expansion': 1, 'smart_money_score': 0.15})
    result = OIClassifier().classify_candle(row)
    assert "Confirmed Breakout" in result["classification"]


# ---------------------------------------------------------------------------
# step4: ConfidenceAndRiskEngine
# ---------------------------------------------------------------------------
def test_confidence_engine_high_confidence_approved():
    engine = ConfidenceAndRiskEngine(min_confidence_threshold=70.0)
    result = engine.evaluate_trade(
        market_dna={"trend": "UP"},
        market_regime={"regime": "Bullish Trend"},
        oi_intelligence={"confidence_score": 95.0},
        technical_indicators={"volume_expanding": True, "price_near_vwap": True}
    )
    assert result["approved"] is True
    assert result["risk_rating"] in ("LOW", "MEDIUM")


def test_confidence_engine_low_confidence_rejected():
    engine = ConfidenceAndRiskEngine(min_confidence_threshold=70.0)
    result = engine.evaluate_trade(
        market_dna={},
        market_regime={"regime": "Neutral Range"},
        oi_intelligence={"confidence_score": 20.0},
        technical_indicators={"volume_expanding": False, "price_near_vwap": False}
    )
    assert result["approved"] is False
    assert result["risk_rating"] == "HIGH"


def test_confidence_engine_score_bounded_0_100():
    engine = ConfidenceAndRiskEngine()
    result = engine.evaluate_trade(
        market_dna={}, market_regime={"regime": "Breakout"},
        oi_intelligence={"confidence_score": 100.0},
        technical_indicators={"volume_expanding": True, "price_near_vwap": True}
    )
    assert 0.0 <= result["confidence_score"] <= 100.0


# ---------------------------------------------------------------------------
# step6: StrategySelectorAI
# ---------------------------------------------------------------------------
def test_strategy_selector_returns_valid_strategy():
    selector = StrategySelectorAI()
    result = selector.evaluate_strategies(
        market_regime={"regime": "Bullish Trend"},
        oi_intelligence={"classification": "Long Build-up", "oi_strength": 1.8, "smart_money_score": 0.15},
        confidence_metrics={"confidence_score": 85.0},
        time_of_day_minutes=60
    )
    assert result["selected_strategy"] in ("Opening Drive", "Velez", "ORB")
    assert set(result["rankings"].keys()) == {"Opening Drive", "Velez", "ORB"}
    for score in result["rankings"].values():
        assert 0.0 <= score <= 100.0


def test_strategy_selector_favors_opening_drive_early_with_breakout():
    selector = StrategySelectorAI()
    result = selector.evaluate_strategies(
        market_regime={"regime": "Neutral"},
        oi_intelligence={"classification": "Confirmed Breakout (Long Build-up)", "oi_strength": 0.5, "smart_money_score": 0.0},
        confidence_metrics={"confidence_score": 50.0},
        time_of_day_minutes=10  # early morning
    )
    assert result["rankings"]["Opening Drive"] >= 65.0  # 50 base + 20 (early) + 15 (breakout)


# ---------------------------------------------------------------------------
# step9: LiveEngine static scoring functions
# ---------------------------------------------------------------------------
def _sample_candles():
    return pd.DataFrame({
        'open': [100, 101, 102, 101.5, 103] * 5,
        'high': [101, 102, 103, 102.5, 104] * 5,
        'low': [99, 100, 101, 100.5, 102] * 5,
        'close': [100.5, 101.5, 102.5, 102, 103.5] * 5,
        'volume': [1000, 1200, 900, 1500, 1100] * 5,
    })


def test_compute_market_data_basic_shape():
    md = LiveEngine.compute_market_data(_sample_candles())
    for key in ["timestamp", "open", "high", "low", "close", "vwap", "atr", "pct_change"]:
        assert key in md
    assert md["low"] <= md["vwap"] <= md["high"] or md["vwap"] == md["high"] or md["vwap"] == md["low"]
    assert md["atr"] >= 0


def test_score_market_dna_bounded_and_labeled():
    md = LiveEngine.compute_market_data(_sample_candles())
    dna = LiveEngine.score_market_dna(md, {
        "classification": "Long Build-up", "confidence_score": 80,
        "oi_strength": 1.2, "smart_money_score": 0.08
    })
    assert 0.0 <= dna["overall_score"] <= 100.0
    assert dna["trend"] in ("Bullish", "Bearish", "Neutral")
    assert dna["market_regime"] != ""


def test_score_market_dna_waits_without_oi_data():
    md = LiveEngine.compute_market_data(_sample_candles())
    dna = LiveEngine.score_market_dna(md, None)
    assert dna["volume_profile"] == "Waiting for Live Data"
    assert dna["smart_money"] == "Waiting for Live Data"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
