"""Tests de las señales de confianza."""

from src.validation.confidence import ConfidenceThresholds, confidence_flags


def _record(**overrides):
    base = {
        "status": "ok",
        "rmse": 0.35,
        "radius": 23.5,
        "coverage_ratio": 0.52,
        "radius_filter_applied": True,
    }
    base.update(overrides)
    return base


def test_good_fit_needs_no_review():
    result = confidence_flags(_record())
    assert result["needs_review"] is False
    assert result["confidence"] == "alta"
    assert result["review_reasons"] == []


def test_high_rmse_is_flagged():
    result = confidence_flags(_record(rmse=0.90))
    assert result["needs_review"] is True
    assert result["confidence"] == "media"
    assert "RMSE alto" in result["review_reasons"][0]


def test_atypical_radius_is_flagged():
    """El caso que el RMSE solo no atrapa: ajuste limpio sobre un radio
    fuera de rango (se observó uno de 35.5 mm con RMSE 0.50)."""
    result = confidence_flags(_record(rmse=0.50, radius=35.5))
    assert result["needs_review"] is True
    assert any("radio atípico" in reason for reason in result["review_reasons"])


def test_two_independent_signals_lower_confidence_further():
    result = confidence_flags(_record(rmse=0.95, coverage_ratio=0.10))
    assert result["confidence"] == "baja"
    assert len(result["review_reasons"]) >= 2


def test_pipeline_error_is_always_low_confidence():
    result = confidence_flags({"status": "error", "error_type": "ValueError"})
    assert result["needs_review"] is True
    assert result["confidence"] == "baja"


def test_missing_fit_is_flagged():
    result = confidence_flags(_record(rmse=None, radius=None))
    assert result["needs_review"] is True
    assert result["confidence"] == "baja"


def test_no_plausible_candidate_is_flagged():
    result = confidence_flags(_record(radius_filter_applied=False))
    assert result["needs_review"] is True
    assert any("radio fisiológico" in reason for reason in result["review_reasons"])


def test_thresholds_are_configurable():
    strict = ConfidenceThresholds(rmse_max=0.20)
    assert confidence_flags(_record(rmse=0.35), strict)["needs_review"] is True
    assert confidence_flags(_record(rmse=0.35))["needs_review"] is False
