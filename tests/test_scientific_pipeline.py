"""Tests del pipeline cientifico minimo."""

import struct
from pathlib import Path

import numpy as np
import pytest

from examples.demo_interactive_web import compute_from_seed, result_to_response, synthetic_humerus_points
from src.approximation.sphere import SphericalApproximator
from src.audit.trail import AuditTrail
from src.axis.longitudinal import AxisApproximator
from src.geometry.curvature import CurvatureCalculator, CurvatureData
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.optimization.best_fit import HumeralHeadBestFitSearch
from src.optimization.refinement import SphereOptimizer
from src.validation.risk_map import compute_risk_scores
from src.validation.viability import SeedValidator
from src.visualization.interactive_web import InteractiveWeb3D


def test_deterministic_seed_estimates_known_sphere():
    points, normals, seeds = synthetic_humerus_points()
    seed = seeds[10]
    audit = AuditTrail("seed_deterministic")

    sphere = SphericalApproximator(max_iterations=20, convergence_threshold=1e-6).approximate_from_seed(
        seed,
        points,
        normals,
        audit_trail=audit,
        initial_radius=22.0,
    )

    np.testing.assert_allclose(sphere["center"], np.array([12.0, 3.0, 80.0]), atol=1e-5)
    assert abs(sphere["radius"] - 22.0) < 1e-5
    assert sphere["error"] < 1e-5
    assert sphere["converged"]


def test_axis_estimation_on_synthetic_humerus():
    points, _, _ = synthetic_humerus_points()
    axis = AxisApproximator.compute_longitudinal_axis(points)

    assert axis["length"] > 100.0
    assert axis["validation"]["overall_valid"]
    assert abs(abs(axis["direction"][2]) - 1.0) < 1e-6
    assert axis["method"] == "diaphyseal_slice_axis"
    assert axis["axis_fit_point_count"] < axis["total_point_count"]


def test_shaft_pca_ignores_biased_irregular_ends():
    rng = np.random.default_rng(7)
    z = np.linspace(-120.0, 120.0, 90)
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    z_grid, angle_grid = np.meshgrid(z, angles)
    shaft = np.column_stack((
        8.0 * np.cos(angle_grid).ravel(),
        5.0 * np.sin(angle_grid).ravel(),
        z_grid.ravel(),
    ))

    proximal = rng.normal(loc=[45.0, 0.0, 165.0], scale=[16.0, 7.0, 9.0], size=(900, 3))
    distal = rng.normal(loc=[-35.0, 0.0, -160.0], scale=[13.0, 8.0, 8.0], size=(900, 3))
    points = np.vstack((shaft, proximal, distal))

    global_axis = AxisApproximator.compute_longitudinal_axis(points, method="pca")
    shaft_axis = AxisApproximator.compute_longitudinal_axis(points, method="shaft_pca", shaft_trim_fraction=0.28)
    true_axis = np.array([0.0, 0.0, 1.0])

    global_alignment = abs(np.dot(global_axis["direction"], true_axis))
    shaft_alignment = abs(np.dot(shaft_axis["direction"], true_axis))

    assert shaft_alignment > global_alignment
    assert shaft_alignment > 0.98


def test_diaphyseal_slice_axis_uses_only_shaft_on_incomplete_humerus():
    rng = np.random.default_rng(3)
    head_center = np.array([12.0, 3.0, 62.0])
    head_radius = 22.0

    theta = np.linspace(0.0, np.pi / 2.0, 32)
    phi = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    head = np.column_stack((
        head_center[0] + head_radius * np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
        head_center[1] + head_radius * np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
        head_center[2] + head_radius * np.cos(theta_grid).ravel(),
    ))

    z = np.linspace(-90.0, 40.0, 90)
    angles = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    z_grid, angle_grid = np.meshgrid(z, angles)
    shaft = np.column_stack((
        8.0 * np.cos(angle_grid).ravel(),
        6.0 * np.sin(angle_grid).ravel(),
        z_grid.ravel(),
    ))

    dense_tail = rng.normal(
        loc=[55.0, -25.0, -118.0],
        scale=[9.0, 9.0, 4.0],
        size=(2500, 3),
    )
    points = np.vstack((head, shaft, dense_tail))

    axis = AxisApproximator.compute_longitudinal_axis(points, method="diaphyseal_slice_axis")

    assert axis["axis_fit_strategy"] == "rough_pca_crop_slice_filter_ransac"
    assert not axis["is_complete_humerus"]
    assert axis["crop_mode"] == "head_only"
    assert axis["shaft_retained_slice_count"] >= 2
    assert axis["ransac_inlier_count"] >= 2
    assert axis["axis_fit_point_count"] < len(shaft)
    assert abs(np.dot(axis["direction"], np.array([0.0, 0.0, 1.0]))) > 0.99


def test_seed_validator_and_random_seed_selection():
    points, _, seeds = synthetic_humerus_points()
    validator = SeedValidator()

    assert validator.is_in_articulation_region(seeds[0], points, tolerance=0.1)
    selected = SphereOptimizer().select_random_seeds(seeds, n_seeds=5, random_seed=123)

    assert selected.shape == (5, 3)
    np.testing.assert_allclose(
        selected,
        SphereOptimizer().select_random_seeds(seeds, n_seeds=5, random_seed=123),
    )


def test_best_fit_search_recovers_synthetic_humeral_head():
    points, normals, _ = synthetic_humerus_points()
    result = HumeralHeadBestFitSearch(
        n_seeds=20,
        top_k=3,
        initial_radius=22.0,
        max_error=2.0,
        random_seed=5,
    ).search(points, normals)

    best = result["best"]

    assert result["head_side"] == "high_projection"
    assert result["candidate_count"] == 20
    assert result["valid_candidate_count"] == 20
    assert best["valid"]
    np.testing.assert_allclose(best["sphere"]["center"], [12.0, 3.0, 80.0], atol=1e-5)
    assert abs(best["sphere"]["radius"] - 22.0) < 1e-5
    assert best["sphere"]["error"] < 1e-5
    assert best["coverage_count"] > 1000
    assert best["score"] < 1.0


def _cap_with_nearby_bump(radius: float = 20.0, random_seed: int = 0) -> tuple:
    """
    Casquete esférico exacto (radio `radius`, centro en el origen) más un
    grupo de puntos contaminantes cerca del vértice, a un radio mayor,
    simulando un tubérculo vecino que se cuela en el vecindario local de
    la semilla. Los puntos contaminantes quedan siempre a pocos mm de la
    semilla (independiente de cómo se ajuste el radio de búsqueda), así
    que sesgan cada iteración del ajuste si no se descartan como outliers.
    """
    theta = np.linspace(0.0, np.pi / 3.0, 12)
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    cap = np.column_stack((
        radius * np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
        radius * np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
        radius * np.cos(theta_grid).ravel(),
    ))
    cap_normals = cap / np.linalg.norm(cap, axis=1)[:, None]

    rng = np.random.default_rng(random_seed)
    n_bump = 35
    jitter = rng.uniform(-2.5, 2.5, size=(n_bump, 2))
    bump_z = (radius + 6.0) + rng.uniform(-1.0, 1.0, size=n_bump)
    bump = np.column_stack((jitter[:, 0], jitter[:, 1], bump_z))
    bump_normals = bump / np.linalg.norm(bump, axis=1)[:, None]

    points = np.vstack((cap, bump))
    normals = np.vstack((cap_normals, bump_normals))
    seed = np.array([0.0, 0.0, radius])
    return points, normals, seed


def test_outlier_rejection_recovers_true_sphere_despite_biased_neighbors():
    points, normals, seed = _cap_with_nearby_bump()

    biased = SphericalApproximator(
        max_iterations=30, convergence_threshold=1e-6, outlier_margin=None
    ).approximate_from_seed(seed, points, normals, initial_radius=20.0)

    corrected = SphericalApproximator(
        max_iterations=30, convergence_threshold=1e-6, outlier_margin=2.0
    ).approximate_from_seed(seed, points, normals, initial_radius=20.0)

    # Sin rechazo de outliers, el vecindario contaminado sesga la esfera:
    # converge, pero a un radio y error claramente incorrectos.
    assert biased["converged"]
    assert abs(biased["radius"] - 20.0) > 3.0
    assert biased["error"] > 1.0

    # Con rechazo de outliers, el ajuste recupera la esfera verdadera.
    assert corrected["converged"]
    assert abs(corrected["radius"] - 20.0) < 0.05
    assert corrected["error"] < 0.05
    np.testing.assert_allclose(corrected["center"], np.zeros(3), atol=0.05)


def test_risk_scores_favor_head_region_over_shaft():
    points, _, _ = synthetic_humerus_points()
    head_indices = np.array([0, 100, 500])
    shaft_indices = np.array([2000, 4000, 6000])

    scores = compute_risk_scores(
        points,
        search_radius=22.0 * 1.15,
        initial_radius=22.0,
        max_error=2.0,
        radius_min=20.0,
        radius_max=40.0,
        min_neighbors=15,
        query_indices=np.concatenate([head_indices, shaft_indices]),
    )

    head_scores = scores[: len(head_indices)]
    shaft_scores = scores[len(head_indices):]
    assert np.all(head_scores < 0.75)
    assert np.all(shaft_scores == 1.0)


def test_best_fit_search_with_weighted_seeds_recovers_synthetic_head():
    points, normals, _ = synthetic_humerus_points()
    result = HumeralHeadBestFitSearch(
        n_seeds=20,
        top_k=3,
        initial_radius=22.0,
        max_error=2.0,
        random_seed=5,
        weight_seeds_by_risk=True,
    ).search(points, normals)

    best = result["best"]

    assert best["valid"]
    np.testing.assert_allclose(best["sphere"]["center"], [12.0, 3.0, 80.0], atol=1e-5)
    assert abs(best["sphere"]["radius"] - 22.0) < 1e-5
    assert best["sphere"]["error"] < 1e-5


def _head_and_wide_non_spherical_flange() -> tuple:
    """
    Cabeza semiesferica real en un extremo mas un disco plano, ancho y NO
    esferico en el otro extremo, simulando un codo (troclea/condilo/
    epicondilos) con mas dispersion radial que la cabeza real pero sin
    curvatura esferica compatible. Reproduce el caso encontrado con un
    humero real donde la esfera automatica termino en la punta equivocada.
    """
    center = np.array([12.0, 3.0, 80.0])
    radius = 22.0
    theta = np.linspace(0.0, np.pi / 2.0, 24)
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    head = np.column_stack((
        center[0] + radius * np.sin(theta_grid).ravel() * np.cos(phi_grid).ravel(),
        center[1] + radius * np.sin(theta_grid).ravel() * np.sin(phi_grid).ravel(),
        center[2] + radius * np.cos(theta_grid).ravel(),
    ))
    head_normals = (head - center) / np.linalg.norm(head - center, axis=1)[:, None]

    z = np.linspace(-217.6, 68.0, 160)
    a = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    z_grid, a_grid = np.meshgrid(z, a)
    shaft = np.column_stack((
        8.0 * np.cos(a_grid).ravel(),
        8.0 * np.sin(a_grid).ravel(),
        z_grid.ravel(),
    ))
    shaft_normals = np.column_stack((np.cos(a_grid).ravel(), np.sin(a_grid).ravel(), np.zeros(a_grid.size)))

    rng = np.random.default_rng(1)
    n_flange = 1200
    r_flange = rng.uniform(8.0, 45.0, n_flange)
    a_flange = rng.uniform(0.0, 2.0 * np.pi, n_flange)
    flange = np.column_stack((r_flange * np.cos(a_flange), r_flange * np.sin(a_flange), np.full(n_flange, -217.6)))
    flange_normals = np.tile(np.array([0.0, 0.0, -1.0]), (n_flange, 1))

    points = np.vstack((head, shaft, flange))
    normals = np.vstack((head_normals, shaft_normals, flange_normals))
    return points, normals


def test_head_side_selection_uses_risk_map_not_just_radial_spread():
    points, normals = _head_and_wide_non_spherical_flange()
    axis = AxisApproximator.compute_longitudinal_axis(points)
    searcher = HumeralHeadBestFitSearch(n_seeds=30, random_seed=5)

    origin = np.asarray(axis["origin"], dtype=float)
    direction = np.asarray(axis["direction"], dtype=float)
    direction = direction / np.linalg.norm(direction)
    projections = (points - origin) @ direction
    radial_distance = searcher._axis_radial_distance(points, origin, direction)

    length = float(np.max(projections) - np.min(projections))
    max_head_depth = min(90.0, max(35.0, searcher.proximal_fraction * float(axis["length"])))
    low_limit = float(np.min(projections))
    high_limit = float(np.max(projections))
    end_depth = min(max_head_depth, max(20.0, 0.25 * length))
    low_end = projections <= low_limit + end_depth
    high_end = projections >= high_limit - end_depth
    low_spread = searcher._end_radial_spread(radial_distance[low_end])
    high_spread = searcher._end_radial_spread(radial_distance[high_end])

    # El disco es mas ancho (mas dispersion radial) que la cabeza real: el
    # heuristico viejo (solo dispersion radial) elegiria el extremo
    # equivocado.
    old_heuristic_side = "high_projection" if high_spread >= low_spread else "low_projection"
    assert old_heuristic_side != "high_projection"

    # La cabeza real esta del lado de proyeccion alta en esta geometria.
    _, head_side = searcher._head_region_indices(points, axis)
    assert head_side == "high_projection"


def test_visualizer_has_selected_seed_trace():
    _, _, seeds = synthetic_humerus_points()
    viz = InteractiveWeb3D()
    viz.plot_selected_seed(seeds[3])

    assert len(viz.fig.data) == 1
    assert viz.fig.data[0].name == "Semilla Seleccionada"


def test_clicked_seed_response_contains_visual_traces():
    points, normals, seeds = synthetic_humerus_points()
    result = compute_from_seed(seeds[10], points, normals, initial_radius=22.0, max_error=2.0)
    response = result_to_response(result)

    np.testing.assert_allclose(response["center"], [12.0, 3.0, 80.0], atol=1e-5)
    assert abs(response["roc"] - 22.0) < 1e-5
    assert response["axis_length"] > 0
    assert 1.0 <= response["medial_offset"] <= 14.0
    assert 0.0 <= response["posterior_offset"] <= 10.0
    assert response["sphere_drawn"]
    assert response["valid"]
    assert response["morphology_reference_status"] == "en referencia"
    assert "z_score" in response["morphology_reference_values"]["roc"]
    assert response["axis_method"] == "diaphyseal_slice_axis"
    assert response["axis_completeness"] == "completo"
    assert response["axis_crop_mode"] == "head_and_tail"
    assert response["axis_ransac_inlier_count"] >= 2
    assert len(response["traces"]) == 4


def test_morphology_response_contains_offsets():
    points, normals, seeds = synthetic_humerus_points()
    result = compute_from_seed(seeds[10], points, normals, initial_radius=22.0, max_error=2.0)
    response = result_to_response(result)

    assert response["morphology"]["roc"] == response["roc"]
    assert response["total_offset"] >= response["medial_offset"]
    assert response["total_offset"] >= response["posterior_offset"]


def test_sphere_validation_uses_morphological_offsets():
    points, _, _ = synthetic_humerus_points()
    axis = AxisApproximator.compute_longitudinal_axis(points)
    audit = AuditTrail("morphology_valid")
    valid_sphere = {
        "center": np.array([12.0, 3.0, 80.0]),
        "radius": 22.0,
        "error": 0.1,
    }
    assert audit.is_valid_approximation(
        valid_sphere,
        axis=axis,
        surface_points=points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )

    audit = AuditTrail("morphology_invalid")
    invalid_sphere = {
        "center": np.array([35.0, 16.0, 80.0]),
        "radius": 22.0,
        "error": 0.1,
    }
    assert audit.is_valid_approximation(
        invalid_sphere,
        axis=axis,
        surface_points=points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )

    report = audit.get_report()
    validation_step = report["steps"][-1]["data"]
    assert not validation_step["morphology_reference_flags"]["all_in_reference"]

    strict_audit = AuditTrail("morphology_strict_invalid")
    assert not strict_audit.is_valid_approximation(
        invalid_sphere,
        axis=axis,
        surface_points=points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
        enforce_morphology_reference=True,
    )


def test_ascii_stl_loader_and_uniform_discretizer(tmp_path: Path):
    stl = tmp_path / "triangle.stl"
    stl.write_text(
        """solid tri
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid tri
""",
        encoding="utf-8",
    )

    mesh = STLLoader.load(str(stl))
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    np.testing.assert_allclose(mesh.normals[0], np.array([0.0, 0.0, 1.0]))

    points, normals = MeshDiscretizer().discretize_uniform(mesh.vertices, mesh.faces, n_samples=10, random_seed=1)
    assert points.shape == (10, 3)
    assert normals.shape == (10, 3)
    assert np.all(points[:, 0] >= 0.0)
    assert np.all(points[:, 1] >= 0.0)
    assert np.all(points[:, 0] + points[:, 1] <= 1.0 + 1e-12)


def test_binary_stl_loader(tmp_path: Path):
    stl = tmp_path / "triangle_binary.stl"
    normal = (0.0, 0.0, 1.0)
    vertices = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    with stl.open("wb") as fh:
        fh.write(b"binary test".ljust(80, b" "))
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<12fH", *(normal + vertices + (0,))))

    mesh = STLLoader.load(str(stl))
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_curvature_spherical_region_filter():
    curvature = CurvatureData(
        principal_k1=1.0 / 25.0,
        principal_k2=1.0 / 25.0,
        mean_curvature=1.0 / 25.0,
        gaussian_curvature=1.0 / (25.0 * 25.0),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    assert CurvatureCalculator.is_local_sphere(curvature, radius_estimate=25.0, tolerance=0.01)


def test_process_one_isolates_failures_instead_of_raising(tmp_path: Path):
    """
    Un STL ilegible no debe tumbar la corrida por lotes: el runner devuelve
    un registro con status='error' y el resto de los huesos sigue.
    """
    from src.batch.runner import process_one
    from src.dataset.catalog import BoneRecord

    broken = tmp_path / "roto.stl"
    broken.write_bytes(b"no soy un STL valido")

    record = BoneRecord(
        model_id="roto_001_M_50_R",
        cohort="test",
        stl_path=broken,
        side="R",
        pathology="NA",
        pathology_group="sano",
        landmarks={"HHC": np.array([1.0, 2.0, 3.0])},
    )

    row = process_one(record)

    assert row["status"] == "error"
    assert row["model_id"] == "roto_001_M_50_R"
    assert row["error_type"]
    # La metadata sigue presente para que la fila aparezca en el reporte.
    assert row["hhc_x"] == 1.0
    assert row["runtime_s"] >= 0.0



def _sphere_candidate(radius: float, score: float, error: float = 0.5) -> dict:
    return {
        "seed_index": 0,
        "seed": np.zeros(3),
        "sphere": {"center": np.array([0.0, 0.0, float(radius)]), "radius": float(radius),
                   "error": error, "converged": True},
        "valid": True,
        "score": score,
    }


def test_radius_filter_rejects_degenerate_candidate_with_better_score():
    """
    El ajuste sobre una vecindad casi plana converge a esferas enormes que
    pueden quedar con RMSE bajo y ganar el ranking. El filtro fisiológico
    debe descartarlas aunque tengan mejor score.
    """
    search = HumeralHeadBestFitSearch(head_radius_min=17.0, head_radius_max=40.0)
    degenerate = _sphere_candidate(radius=3200.0, score=0.01)
    plausible = _sphere_candidate(radius=23.0, score=0.50)

    rankable, applied = search._filter_by_radius([degenerate, plausible])

    assert applied is True
    assert rankable == [plausible]


def test_radius_filter_keeps_everything_when_no_candidate_is_plausible():
    """Sin candidatos dentro del rango se rankea igual: es preferible un
    ajuste implausible a no dar respuesta, pero queda marcado."""
    search = HumeralHeadBestFitSearch(head_radius_min=17.0, head_radius_max=40.0)
    candidates = [_sphere_candidate(radius=2.0, score=0.1),
                  _sphere_candidate(radius=900.0, score=0.2)]

    rankable, applied = search._filter_by_radius(candidates)

    assert applied is False
    assert rankable == candidates


def test_radius_filter_skips_candidates_without_sphere():
    """Los candidatos que fallaron (sphere=None) no deben romper el filtro."""
    search = HumeralHeadBestFitSearch()
    failed = {"seed_index": 1, "sphere": None, "valid": False, "score": float("inf")}
    plausible = _sphere_candidate(radius=24.0, score=0.4)

    rankable, applied = search._filter_by_radius([failed, plausible])

    assert applied is True
    assert rankable == [plausible]


def test_score_excludes_morphology_and_reference_penalties():
    """
    Ambos términos derivan del marco anatómico fijo, que sobre el dataset
    discrepa 36 grados de la anatomía real. Se siguen reportando como
    diagnóstico pero no deben mover el costo.
    """
    search = HumeralHeadBestFitSearch(max_error=2.0)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    sphere = {"center": np.array([0.0, 0.0, 25.0]), "radius": 25.0,
              "error": 0.5, "converged": True}

    sin_penalizacion = {"morphology_reference_values": {},
                        "morphology_reference_flags": {"all_in_reference": True}}
    con_penalizacion = {
        "morphology_reference_values": {"roc": {"z_score": 3.9}, "medial_offset": {"z_score": 3.5}},
        "morphology_reference_flags": {"all_in_reference": False},
    }

    limpio = search._score_candidate(sphere, sin_penalizacion, points, normals, np.arange(3))
    penalizado = search._score_candidate(sphere, con_penalizacion, points, normals, np.arange(3))

    assert limpio["score"] == pytest.approx(penalizado["score"])
    # ...pero se siguen reportando para poder diagnosticar.
    assert penalizado["score_components"]["morphology_penalty"] > 0
    assert penalizado["score_components"]["reference_penalty"] == 0.25
