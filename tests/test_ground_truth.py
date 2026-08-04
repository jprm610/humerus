"""Tests de las métricas contra el ground truth del dataset."""

import numpy as np

from src.validation.ground_truth import axis_error, center_error, frame_diagnostics


def test_center_error_is_euclidean_distance():
    assert center_error([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == 5.0
    assert center_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_center_error_returns_none_when_ground_truth_missing():
    assert center_error(None, [1.0, 2.0, 3.0]) is None
    assert center_error([1.0, 2.0, 3.0], None) is None


def test_axis_error_zero_when_axis_matches_anatomical_reference():
    # HHC arriba, epicóndilos abajo: el eje anatómico es +Z.
    axis = {"direction": np.array([0.0, 0.0, 1.0]), "origin": np.zeros(3)}
    error = axis_error(axis, hhc=[0, 0, 300], le=[10, 0, 0], me=[-10, 0, 0])
    assert error == 0.0


def test_axis_error_ignores_axis_orientation():
    """El eje calculado puede apuntar proximal o distal; solo importa la recta."""
    reference = {"direction": np.array([0.0, 0.0, 1.0]), "origin": np.zeros(3)}
    flipped = {"direction": np.array([0.0, 0.0, -1.0]), "origin": np.zeros(3)}
    args = dict(hhc=[0, 0, 300], le=[10, 0, 0], me=[-10, 0, 0])
    assert axis_error(reference, **args) == axis_error(flipped, **args) == 0.0


def test_axis_error_measures_known_angle():
    axis = {"direction": np.array([1.0, 0.0, 1.0]), "origin": np.zeros(3)}
    error = axis_error(axis, hhc=[0, 0, 300], le=[10, 0, 0], me=[-10, 0, 0])
    assert error is not None
    np.testing.assert_allclose(error, 45.0, atol=1e-6)


def test_axis_error_none_without_epicondyles():
    axis = {"direction": np.array([0.0, 0.0, 1.0]), "origin": np.zeros(3)}
    assert axis_error(axis, hhc=[0, 0, 300], le=None, me=None) is None
    assert axis_error(None, hhc=[0, 0, 300], le=[1, 0, 0], me=[-1, 0, 0]) is None


def test_frame_diagnostics_detects_disagreement_with_fixed_frame():
    """
    El marco fijo asume medial = +X. Si el hueso dice que medial es +Y
    (según sus tuberosidades), el diagnóstico debe reportar 90 grados.
    """
    sphere = {"center": np.array([3.0, 4.0, 50.0]), "radius": 22.0}
    axis = {"direction": np.array([0.0, 0.0, 1.0]), "origin": np.zeros(3)}
    landmarks = {"GT": np.array([0.0, -10.0, 0.0]), "LT": np.array([0.0, 10.0, 0.0])}

    diagnostics = frame_diagnostics(sphere, axis, landmarks, side="R")
    np.testing.assert_allclose(diagnostics["frame_angle_deg"], 90.0, atol=1e-6)
    np.testing.assert_allclose(diagnostics["medial_offset_fixed"], 3.0, atol=1e-6)
    np.testing.assert_allclose(diagnostics["medial_offset_landmark"], 4.0, atol=1e-6)


def test_frame_diagnostics_returns_nulls_without_tuberosities():
    sphere = {"center": np.array([3.0, 4.0, 50.0]), "radius": 22.0}
    axis = {"direction": np.array([0.0, 0.0, 1.0]), "origin": np.zeros(3)}
    diagnostics = frame_diagnostics(sphere, axis, landmarks={}, side="R")
    assert diagnostics["frame_angle_deg"] is None
