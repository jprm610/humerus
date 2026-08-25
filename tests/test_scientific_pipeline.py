"""Tests del pipeline cientifico minimo."""

import struct
from pathlib import Path

import numpy as np

from examples.demo_interactive_web import (
    best_fit_to_response,
    compute_best_fit_search,
    compute_from_seed,
    result_to_response,
    synthetic_humerus_points,
)
from src.approximation.sphere import SphericalApproximator
from src.audit.trail import AuditTrail
from src.axis.longitudinal import AxisApproximator
from src.geometry.curvature import CurvatureCalculator, CurvatureData
from src.geometry.sphere import SphereGeometry
from src.mesh.cleaner import MeshCleaner
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.optimization.best_fit import HumeralHeadBestFitSearch
from src.optimization.refinement import SphereOptimizer
from src.optimization.sphere_ransac import SphereRansacConfig, SphereRansacFitter
from src.validation.sphere import ApproximationValidationConfig, SphereValidator
from src.validation.viability import SeedValidator
from src.visualization.interactive_web import InteractiveWeb3D


def synthetic_humerus_mesh():
    """Crea una malla sintética con casquete humeral y tallo cilíndrico."""
    center = np.array([12.0, 3.0, 80.0])
    radius = 22.0
    theta = np.linspace(0.05, np.pi / 2.0, 18)
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    vertices = []
    for t in theta:
        for p in phi:
            vertices.append(center + radius * np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t),
            ]))

    faces = []
    n_phi = len(phi)
    for i in range(len(theta) - 1):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            faces.append([a, c, b])
            faces.append([b, c, d])

    offset = len(vertices)
    z = np.linspace(-180.0, 68.0, 60)
    shaft_phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for zz in z:
        for p in shaft_phi:
            vertices.append([8.0 * np.cos(p), 8.0 * np.sin(p), zz])
    for i in range(len(z) - 1):
        for j in range(len(shaft_phi)):
            a = offset + i * len(shaft_phi) + j
            b = offset + i * len(shaft_phi) + (j + 1) % len(shaft_phi)
            c = offset + (i + 1) * len(shaft_phi) + j
            d = offset + (i + 1) * len(shaft_phi) + (j + 1) % len(shaft_phi)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


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


def test_mesh_cleaner_removes_degenerate_faces_and_builds_adjacency():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 4, 4],
    ])

    cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)

    assert cleaned.faces.shape == (2, 3)
    assert cleaned.cleaning_report["removed_degenerate_faces"] == 1
    assert cleaned.adjacency == [[1], [0]]
    assert cleaned.cleaning_report["component_count"] == 1


def test_sphere_from_four_points_recovers_known_sphere():
    center = np.array([4.0, -2.0, 7.0])
    radius = 13.0
    points = np.array([
        center + [radius, 0.0, 0.0],
        center + [0.0, radius, 0.0],
        center + [0.0, 0.0, radius],
        center + [-radius / np.sqrt(3.0), -radius / np.sqrt(3.0), -radius / np.sqrt(3.0)],
    ])

    sphere = SphereGeometry.from_four_points(points)

    assert sphere is not None
    fitted_center, fitted_radius = sphere
    np.testing.assert_allclose(fitted_center, center, atol=1e-8)
    assert abs(fitted_radius - radius) < 1e-8
    ransac_sphere = SphereRansacFitter.sphere_from_four_points(points)
    assert ransac_sphere is not None
    np.testing.assert_allclose(ransac_sphere[0], center, atol=1e-8)


def test_audit_delegates_sphere_validation_rules():
    points, _, _ = synthetic_humerus_points()
    axis = AxisApproximator.compute_longitudinal_axis(points)
    sphere = {
        "center": np.array([12.0, 3.0, 80.0]),
        "radius": 22.0,
        "error": 0.25,
    }

    expected = SphereValidator.validate_approximation(
        sphere,
        axis=axis,
        surface_points=points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
        config=ApproximationValidationConfig(max_error=2.0),
    )
    audit = AuditTrail("delegated_validation")
    actual_valid = audit.is_valid_approximation(
        sphere,
        max_error=2.0,
        axis=axis,
        surface_points=points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )
    validation_step = audit.get_report()["steps"][-1]["data"]

    assert actual_valid == expected["valid"]
    assert validation_step["morphology_reference_values"] == expected["morphology_reference_values"]
    assert validation_step["morphology_reference_flags"] == expected["morphology_reference_flags"]


def test_sphere_ransac_recovers_synthetic_humeral_head_mesh():
    vertices, faces = synthetic_humerus_mesh()
    cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)
    fitter = SphereRansacFitter(
        SphereRansacConfig(
            n_iterations=300,
            min_inlier_faces=30,
            min_inlier_area_ratio=0.001,
            random_seed=1,
        )
    )

    result = fitter.fit(cleaned)

    np.testing.assert_allclose(result["center"], [12.0, 3.0, 80.0], atol=0.15)
    assert abs(result["radius"] - 22.0) < 0.2
    assert result["rmse"] < 0.05
    assert result["mad"] < 0.05
    assert result["inlier_face_count"] > 500
    assert result["angular_coverage"] > 0.75
    assert result["articular_side_score"] >= 0.0
    assert result["valid"]


def test_ransac_score_penalizes_distal_local_sphere_with_bad_morphology():
    vertices, faces = synthetic_humerus_mesh()
    cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)
    axis = AxisApproximator.compute_longitudinal_axis(cleaned.face_centroids)
    fitter = SphereRansacFitter(SphereRansacConfig())
    common_support = {
        "mad": 0.02,
        "inlier_area_ratio": 0.25,
        "angular_coverage": 0.9,
        "normal_score": 0.98,
        "dominant_component_ratio": 1.0,
        "inlier_face_count": 1200,
        "inlier_area": float(0.25 * cleaned.face_areas.sum()),
        "connected_component_count": 1,
    }
    head_result = {
        **common_support,
        "center": np.array([12.0, 3.0, 80.0]),
        "radius": 22.0,
        "rmse": 0.02,
    }
    distal_result = {
        **common_support,
        "center": np.array([0.0, 0.0, -150.0]),
        "radius": 22.0,
        "rmse": 0.02,
    }

    head_score = fitter._score(cleaned, head_result, fitter._morphology(axis, head_result))
    distal_score = fitter._score(cleaned, distal_result, fitter._morphology(axis, distal_result))

    assert head_score["components"]["reference_miss_count"] == 0
    assert distal_score["components"]["reference_miss_count"] > 0
    assert distal_score["score"] > head_score["score"]


def test_ransac_compares_both_ends_and_rejects_elbow_sphere_on_sample_stl():
    mesh = STLLoader.load("data/sample_humeri/Right_humerus_bone_one-piece.stl")
    cleaned = MeshCleaner().clean(mesh.vertices, mesh.faces)
    axis = AxisApproximator.compute_longitudinal_axis(cleaned.face_centroids)
    result = SphereRansacFitter(
        SphereRansacConfig(
            n_iterations=500,
            min_inlier_faces=30,
            min_inlier_area_ratio=0.001,
            random_seed=42,
        )
    ).fit(cleaned, axis=axis)

    assert result["candidate_region_count"] == 2
    assert result["head_side"] == "low_projection"
    assert 17.0 <= result["morphology"]["roc"] <= 30.0
    assert 1.0 <= result["morphology"]["medial_offset"] <= 16.0
    assert 0.0 <= result["morphology"]["posterior_offset"] <= 10.0
    assert result["articular_side_score"] > 0.0
    assert result["candidate_region_summaries"][0]["score"] < result["candidate_region_summaries"][1]["score"]


def test_ransac_best_fit_response_serializes_articular_region():
    vertices, faces = synthetic_humerus_mesh()
    cleaned = MeshCleaner(keep_largest_component=False).clean(vertices, faces)
    points, normals = MeshDiscretizer().discretize_uniform(
        cleaned.vertices,
        cleaned.faces,
        n_samples=1200,
        random_seed=2,
    )

    result = compute_best_fit_search(
        points,
        normals,
        n_seeds=220,
        top_k=1,
        initial_radius=22.0,
        max_error=1.5,
        cleaned_mesh=cleaned,
    )
    response = best_fit_to_response(result)
    best = response["best"]

    assert response["automatic_method"] == "sphere_ransac"
    assert best["method"] == "sphere_ransac"
    assert len(best["articular_face_indices"]) > 500
    assert best["mad"] < 0.05
    assert best["inlier_area"] > 0.0
    assert best["angular_coverage"] > 0.75
    assert best["articular_side_score"] >= 0.0


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
