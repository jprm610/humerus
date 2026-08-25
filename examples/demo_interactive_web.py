"""Demo interactivo para seleccionar una semilla sobre el humero.

Uso basico:

    python3 examples/demo_interactive_web.py
    python3 examples/demo_interactive_web.py --use-seed-index --seed-index 8
    python3 examples/demo_interactive_web.py --seed 3.0 0.0 104.0
    python3 examples/demo_interactive_web.py --stl data/sample_humeri/model.stl

Por defecto abre una pagina local donde la semilla se escoge haciendo clic
sobre un punto real de la superficie discretizada. El calculo de esfera se
ejecuta en Python despues del clic.
"""

import argparse
import json
import sys
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import unquote

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.approximation.sphere import SphericalApproximator
from src.audit.trail import AuditTrail
from src.axis.longitudinal import AxisApproximator
from src.mesh.cleaner import CleanedMesh, MeshCleaner
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.optimization.best_fit import HumeralHeadBestFitSearch
from src.optimization.sphere_ransac import SphereRansacConfig, SphereRansacFitter
from src.visualization.interactive_web import InteractiveWeb3D


def synthetic_humerus_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crea una cabeza semi-esferica con diafisis cilindrica para pruebas."""
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
    head_normals = head - center
    head_normals = head_normals / np.linalg.norm(head_normals, axis=1)[:, None]

    z = np.linspace(-217.6, 68.0, 160)
    a = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    z_grid, a_grid = np.meshgrid(z, a)
    shaft_radius = 8.0
    shaft = np.column_stack((
        shaft_radius * np.cos(a_grid).ravel(),
        shaft_radius * np.sin(a_grid).ravel(),
        z_grid.ravel(),
    ))
    shaft_normals = np.column_stack((
        np.cos(a_grid).ravel(),
        np.sin(a_grid).ravel(),
        np.zeros(a_grid.size),
    ))

    points = np.vstack((head, shaft))
    normals = np.vstack((head_normals, shaft_normals))
    seed_candidates = head[::37]
    return points, normals, seed_candidates


def load_surface_from_stl(stl_path: str, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Carga y discretiza un STL real."""
    mesh = STLLoader.load(stl_path)
    discretizer = MeshDiscretizer()
    return discretizer.discretize_uniform(mesh.vertices, mesh.faces, samples, random_seed=42)


def load_cleaned_surface_from_stl(stl_path: str, samples: int) -> Tuple[CleanedMesh, np.ndarray, np.ndarray]:
    """Carga STL, limpia malla y discretiza su superficie limpia."""
    mesh = STLLoader.load(stl_path)
    cleaned = MeshCleaner().clean(mesh.vertices, mesh.faces)
    discretizer = MeshDiscretizer()
    points, normals = discretizer.discretize_uniform(
        cleaned.vertices,
        cleaned.faces,
        samples,
        random_seed=42,
    )
    return cleaned, points, normals


def load_demo_surface(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga superficie desde STL o, solo si se solicita, usa datos sinteticos."""
    if args.stl:
        surface_points, surface_normals = load_surface_from_stl(args.stl, args.samples)
        seed_candidates = surface_points
    elif args.synthetic_demo:
        surface_points, surface_normals, seed_candidates = synthetic_humerus_points()
    else:
        raise ValueError("Selecciona un STL en la interfaz web o usa --stl/--synthetic-demo")
    return surface_points, surface_normals, seed_candidates


def choose_seed(args: argparse.Namespace, candidates: np.ndarray) -> np.ndarray:
    """Escoge semilla por coordenadas explicitas o indice reproducible."""
    if args.seed is not None:
        return np.asarray(args.seed, dtype=float)

    if len(candidates) == 0:
        raise ValueError("No hay semillas candidatas; usa --seed x y z")

    index = int(args.seed_index) % len(candidates)
    return candidates[index]


def sphere_surface_trace(center: np.ndarray, radius: float, name: str) -> go.Surface:
    """Crea traza Plotly de la esfera ajustada como superficie."""
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 24)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return go.Surface(
        x=x,
        y=y,
        z=z,
        name=name,
        colorscale="Reds",
        showscale=False,
        opacity=0.45,
        hoverinfo="skip",
        meta={"resultTrace": True},
    )


def axis_traces(axis: Dict[str, Any]) -> list:
    """Crea trazas Plotly para el eje longitudinal."""
    origin = axis["origin"]
    end = axis["distal_point"]
    midpoint = (origin + end) / 2.0
    length = float(axis["length"])
    return [
        go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode="lines",
            name="Eje longitudinal PCA",
            line=dict(color="red", width=8),
            hoverinfo="skip",
            meta={"resultTrace": True},
        ),
        go.Scatter3d(
            x=[midpoint[0]],
            y=[midpoint[1]],
            z=[midpoint[2]],
            mode="text",
            name="Longitud eje PCA",
            text=[f"Eje PCA: {length:.2f} mm"],
            textposition="middle right",
            textfont=dict(color="darkred", size=13),
            hoverinfo="skip",
            showlegend=False,
            meta={"resultTrace": True},
        ),
    ]


def selected_seed_trace(seed: np.ndarray) -> go.Scatter3d:
    """Crea marcador destacado para la semilla clicada."""
    return go.Scatter3d(
        x=[seed[0]],
        y=[seed[1]],
        z=[seed[2]],
        mode="markers+text",
        name="Semilla seleccionada",
        marker=dict(size=11, color="gold", symbol="diamond"),
        text=["Semilla"],
        textposition="top center",
        hoverinfo="text",
        hovertext=[f"x={seed[0]:.4f}<br>y={seed[1]:.4f}<br>z={seed[2]:.4f}"],
        meta={"resultTrace": True},
    )


def best_fit_seed_trace(seed: np.ndarray, score: float) -> go.Scatter3d:
    """Crea marcador para la mejor semilla automática."""
    return go.Scatter3d(
        x=[seed[0]],
        y=[seed[1]],
        z=[seed[2]],
        mode="markers+text",
        name="Mejor semilla automática",
        marker=dict(size=10, color="limegreen", symbol="circle"),
        text=[f"Best fit {score:.3f}"],
        textposition="bottom center",
        hoverinfo="text",
        hovertext=[f"score={score:.4f}<br>x={seed[0]:.4f}<br>y={seed[1]:.4f}<br>z={seed[2]:.4f}"],
        meta={"resultTrace": True},
    )


def compute_from_seed(
    seed: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    initial_radius: float,
    max_error: float,
) -> Dict[str, Any]:
    """Calcula esfera, eje y auditoria para una semilla de superficie."""
    audit = AuditTrail("clicked_seed")
    audit.validate_seed(seed, surface_points, curvature_threshold=0.1)

    approximator = SphericalApproximator(max_iterations=30, convergence_threshold=1e-5)
    sphere = approximator.approximate_from_seed(
        seed,
        surface_points,
        surface_normals,
        audit_trail=audit,
        initial_radius=initial_radius,
    )
    axis = AxisApproximator.compute_longitudinal_axis(
        surface_points,
        method="diaphyseal_slice_axis",
    )
    audit.is_valid_approximation(
        sphere,
        max_error=max_error,
        axis=axis,
        surface_points=surface_points,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )

    return {
        "seed": seed,
        "sphere": sphere,
        "axis": axis,
        "sphere_center_inside_humerus": is_point_inside_surface_volume(sphere["center"], surface_points),
        "audit": audit.get_report(),
    }


def compute_best_fit_search(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    n_seeds: int,
    top_k: int,
    initial_radius: float,
    max_error: float,
    cleaned_mesh: CleanedMesh | None = None,
) -> Dict[str, Any]:
    """Ejecuta búsqueda automática de best-fit sphere."""
    if cleaned_mesh is not None:
        axis = AxisApproximator.compute_longitudinal_axis(
            cleaned_mesh.face_centroids,
            method="diaphyseal_slice_axis",
        )
        audit = AuditTrail("sphere_ransac_auto")
        fitter = SphereRansacFitter(
            SphereRansacConfig(
                n_iterations=n_seeds,
                distance_tolerance=max(1.0, min(2.5, max_error)),
                random_seed=42,
            )
        )
        result = fitter.fit(cleaned_mesh, axis=axis, audit_trail=audit)
        return ransac_fit_to_best_fit(result, cleaned_mesh, audit)

    axis = AxisApproximator.compute_longitudinal_axis(
        surface_points,
        method="diaphyseal_slice_axis",
    )
    searcher = HumeralHeadBestFitSearch(
        n_seeds=n_seeds,
        top_k=top_k,
        initial_radius=initial_radius,
        max_error=max_error,
        random_seed=42,
    )
    return searcher.search(surface_points, surface_normals, axis=axis)


def ransac_fit_to_best_fit(
    result: Dict[str, Any],
    cleaned_mesh: CleanedMesh,
    audit: AuditTrail,
) -> Dict[str, Any]:
    """Adapta el resultado RANSAC al contrato JSON usado por la demo."""
    sphere = {
        "center": result["center"],
        "radius": float(result["radius"]),
        "error": float(result["rmse"]),
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
    }
    articular_faces = np.asarray(result["articular_face_indices"], dtype=int)
    articular_points = cleaned_mesh.face_centroids[articular_faces] if len(articular_faces) else np.empty((0, 3))
    candidate = {
        "seed_index": -1,
        "seed": np.asarray(result["center"], dtype=float),
        "sphere": sphere,
        "valid": bool(result["valid"]),
        "score": float(result["score"]),
        "audit": audit.get_report(),
        "coverage_count": int(result["inlier_face_count"]),
        "coverage_ratio": float(result["inlier_area_ratio"]),
        "coverage_tolerance": float(SphereRansacConfig().distance_tolerance),
        "score_components": result["score_components"],
        "morphology": result.get("morphology", {}),
        "morphology_z_scores": result["morphology_z_scores"],
        "morphology_reference_flags": result["morphology_reference_flags"],
        "morphology_reference_values": {},
        "mad": float(result["mad"]),
        "radial_p95": float(result.get("radial_p95", 0.0)),
        "inlier_area": float(result["inlier_area"]),
        "angular_coverage": float(result["angular_coverage"]),
        "angular_compactness": float(result.get("angular_compactness", 0.0)),
        "connected_component_count": int(result["connected_component_count"]),
        "dominant_component_ratio": float(result["dominant_component_ratio"]),
        "normal_score": float(result["normal_score"]),
        "articular_side_score": float(result.get("articular_side_score", 0.0)),
        "articular_face_indices": articular_faces,
        "articular_points": articular_points,
        "method": "sphere_ransac",
        "reasons": result["reasons"],
    }
    return {
        "axis": result["axis"],
        "head_region_count": int(result["candidate_face_count"]),
        "head_side": result["head_side"],
        "candidate_count": int(result.get("candidate_region_count", 1)),
        "valid_candidate_count": int(result.get("valid_region_count", int(bool(result["valid"])))),
        "candidate_region_summaries": result.get("candidate_region_summaries", []),
        "automatic_method": "sphere_ransac",
        "cleaning_report": cleaned_mesh.cleaning_report,
        "best": candidate,
        "top_candidates": [candidate],
        "all_candidates": [candidate],
    }


def best_fit_to_response(best_fit: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte ranking automático a JSON compacto."""
    best = best_fit.get("best")
    top_candidates = [candidate_to_response(candidate) for candidate in best_fit.get("top_candidates", [])]
    payload = {
        "automatic_method": best_fit.get("automatic_method", "seed_population"),
        "cleaning_report": best_fit.get("cleaning_report", {}),
        "head_region_count": int(best_fit.get("head_region_count", 0)),
        "candidate_count": int(best_fit.get("candidate_count", 0)),
        "valid_candidate_count": int(best_fit.get("valid_candidate_count", 0)),
        "candidate_region_summaries": best_fit.get("candidate_region_summaries", []),
        "top_candidates": top_candidates,
        "best": candidate_to_response(best) if best else None,
    }
    return payload


def candidate_to_response(candidate: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Serializa un candidato evitando arrays numpy."""
    if candidate is None:
        return None
    sphere = candidate.get("sphere")
    payload = {
        "seed_index": int(candidate.get("seed_index", -1)),
        "seed": np.asarray(candidate.get("seed", np.zeros(3))).tolist(),
        "score": float(candidate.get("score", float("inf"))),
        "valid": bool(candidate.get("valid", False)),
        "coverage_count": int(candidate.get("coverage_count", 0)),
        "coverage_ratio": float(candidate.get("coverage_ratio", 0.0)),
        "coverage_tolerance": float(candidate.get("coverage_tolerance", 0.0)),
        "score_components": candidate.get("score_components", {}),
        "morphology": candidate.get("morphology", {}),
        "morphology_z_scores": candidate.get("morphology_z_scores", {}),
        "morphology_reference_flags": candidate.get("morphology_reference_flags", {}),
        "morphology_reference_values": candidate.get("morphology_reference_values", {}),
        "method": candidate.get("method", "seed_population"),
        "mad": float(candidate.get("mad", candidate.get("score_components", {}).get("mad_norm", 0.0))),
        "radial_p95": float(candidate.get("radial_p95", 0.0)),
        "inlier_area": float(candidate.get("inlier_area", 0.0)),
        "angular_coverage": float(candidate.get("angular_coverage", 0.0)),
        "angular_compactness": float(candidate.get("angular_compactness", 0.0)),
        "connected_component_count": int(candidate.get("connected_component_count", 0)),
        "dominant_component_ratio": float(candidate.get("dominant_component_ratio", 0.0)),
        "normal_score": float(candidate.get("normal_score", 0.0)),
        "articular_side_score": float(candidate.get("articular_side_score", 0.0)),
        "articular_face_indices": np.asarray(candidate.get("articular_face_indices", []), dtype=int).tolist(),
        "reasons": candidate.get("reasons", []),
    }
    if sphere is not None:
        morphology = candidate.get("morphology", {})
        payload.update({
            "center": np.asarray(sphere["center"]).tolist(),
            "radius": float(sphere["radius"]),
            "roc": float(morphology.get("roc", abs(float(sphere["radius"])))),
            "medial_offset": float(morphology.get("medial_offset", 0.0)),
            "posterior_offset": float(morphology.get("posterior_offset", 0.0)),
            "total_offset": float(morphology.get("total_offset", 0.0)),
            "error": float(sphere["error"]),
            "iterations": int(sphere.get("iterations", 0)),
            "converged": bool(sphere.get("converged", False)),
        })
    else:
        payload["error_message"] = candidate.get("error_message", "candidate failed")
    return payload


def best_fit_traces(best_fit: Dict[str, Any]) -> list:
    """Crea trazas Plotly para el mejor candidato automático."""
    best = best_fit.get("best")
    if not best or best.get("sphere") is None:
        return []
    sphere = best["sphere"]
    traces = [
        sphere_surface_trace(
            sphere["center"],
            sphere["radius"],
            f"Best-fit automática score={best['score']:.3f}",
        ),
        best_fit_seed_trace(best["seed"], best["score"]),
    ]
    articular_points = np.asarray(best.get("articular_points", np.empty((0, 3))), dtype=float)
    if articular_points.ndim == 2 and len(articular_points) > 0:
        traces.append(
            go.Scatter3d(
                x=articular_points[:, 0],
                y=articular_points[:, 1],
                z=articular_points[:, 2],
                mode="markers",
                name="Región articular RANSAC",
                marker=dict(size=3, color="orange", opacity=0.92),
                hoverinfo="skip",
                meta={"resultTrace": True},
            )
        )
    traces.extend(axis_traces(best_fit["axis"]))
    return traces


def is_point_inside_surface_volume(point: np.ndarray, surface_points: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Estima si un punto está dentro del volumen del húmero usando la envolvente convexa.

    Esto no reemplaza una prueba volumétrica exacta de malla cerrada, pero rechaza
    centros claramente fuera del hueso y evita dibujar esferas absurdas.
    """
    points = np.asarray(surface_points, dtype=float)
    point = np.asarray(point, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or point.shape != (3,):
        return False
    if len(points) < 4:
        return False

    try:
        hull = ConvexHull(points)
        equations = hull.equations
        return bool(np.all(equations[:, :3] @ point + equations[:, 3] <= tolerance))
    except QhullError:
        mins = points.min(axis=0) - tolerance
        maxs = points.max(axis=0) + tolerance
        return bool(np.all(point >= mins) and np.all(point <= maxs))


def surface_points_trace(surface_points: np.ndarray, name: str = "Superficie clicable") -> go.Scatter3d:
    """Crea traza clicable para puntos de superficie."""
    indices = np.arange(len(surface_points))
    return go.Scatter3d(
        x=surface_points[:, 0],
        y=surface_points[:, 1],
        z=surface_points[:, 2],
        mode="markers",
        name=name,
        customdata=indices,
        marker=dict(size=2, color="steelblue", opacity=0.72),
        hovertemplate=(
            "Punto %{customdata}<br>"
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<extra>clic para usar semilla</extra>"
        ),
        meta={"surfaceTrace": True},
    )


def make_selection_figure(surface_points: np.ndarray | None = None) -> go.Figure:
    """Crea figura inicial clicable con puntos de superficie."""
    fig = go.Figure()
    if surface_points is not None:
        fig.add_trace(surface_points_trace(surface_points))
    fig.update_layout(
        title="Carga un STL y selecciona una semilla en la cabeza del humero",
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        width=1200,
        height=820,
        showlegend=True,
        hovermode="closest",
    )
    return fig


def build_selection_html(
    fig: go.Figure,
    initial_best_fit: Dict[str, Any] | None = None,
    initial_filename: str | None = None,
    initial_points: int = 0,
) -> str:
    """Construye HTML con manejador de clic y llamada a Python."""
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    initial_result = None
    if initial_best_fit is not None:
        initial_result = {
            "filename": initial_filename or "superficie precargada",
            "points": int(initial_points),
            "best_fit": best_fit_to_response(initial_best_fit),
        }
    initial_result_json = json.dumps(initial_result, cls=PlotlyJSONEncoder)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Seleccion de semilla humeral</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f8; color: #1f2933; }}
    #bar {{ display: flex; align-items: center; gap: 12px; padding: 12px 18px; background: white; border-bottom: 1px solid #d9dee3; font-size: 14px; }}
    #status {{ flex: 1; }}
    #plot {{ width: 100vw; height: calc(100vh - 58px); }}
    input[type=file] {{ max-width: 340px; }}
    button {{ border: 1px solid #b9c2cc; background: #f8fafc; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    button:disabled {{ opacity: 0.55; cursor: default; }}
    code {{ background: #eef1f4; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="bar">
    <input id="stlFile" type="file" accept=".stl,model/stl,application/sla">
    <button id="uploadButton">Cargar STL</button>
    <div id="status">
      Selecciona un archivo STL. Luego haz clic sobre un punto real de la cabeza humeral para usarlo como semilla.
    </div>
  </div>
  <div id="plot"></div>
  <script>
    const fig = {fig_json};
    const initialResult = {initial_result_json};
    const plot = document.getElementById('plot');
    const status = document.getElementById('status');
    const fileInput = document.getElementById('stlFile');
    const uploadButton = document.getElementById('uploadButton');
    let surfaceLoaded = fig.data.length > 0;
    let resultTraceCount = 0;
    let surfaceTraceCount = fig.data.length;

    Plotly.newPlot(plot, fig.data, fig.layout, {{responsive: true}});

    function traceHasMeta(trace, key) {{
      return trace && trace.meta && trace.meta[key] === true;
    }}

    function resultTraceIndices() {{
      return plot.data
        .map((trace, index) => traceHasMeta(trace, 'resultTrace') ? index : -1)
        .filter(index => index >= 0);
    }}

    function surfaceTraceIndices() {{
      return plot.data
        .map((trace, index) => traceHasMeta(trace, 'surfaceTrace') ? index : -1)
        .filter(index => index >= 0);
    }}

    function clearResultTraces() {{
      const indices = resultTraceIndices().sort((a, b) => b - a);
      if (indices.length > 0) {{
        Plotly.deleteTraces(plot, indices);
        resultTraceCount = 0;
      }}
    }}

    function keepSurfaceSelectable() {{
      const indices = surfaceTraceIndices();
      if (indices.length > 0 && indices[indices.length - 1] !== plot.data.length - 1) {{
        Plotly.moveTraces(plot, indices, plot.data.length - indices.length);
      }}
    }}

    function bestFitSummary(result) {{
      if (!result.best_fit || !result.best_fit.best) {{
        return 'Best-fit: <code>sin candidatos</code>.';
      }}
      const best = result.best_fit.best;
      const method = result.best_fit.automatic_method || best.method || 'seed_population';
      const extra = method === 'sphere_ransac'
        ? `MAD <code>${{best.mad.toFixed(3)}} mm</code> ` +
          `P95 <code>${{best.radial_p95.toFixed(3)}} mm</code> ` +
          `área <code>${{best.inlier_area.toFixed(1)}} mm²</code> ` +
          `angular <code>${{(100 * best.angular_coverage).toFixed(1)}}%</code> ` +
          `compacta <code>${{(100 * best.angular_compactness).toFixed(1)}}%</code> ` +
          `lado <code>${{(100 * best.articular_side_score).toFixed(1)}}%</code> ` +
          `conectividad <code>${{(100 * best.dominant_component_ratio).toFixed(1)}}%</code> `
        : '';
      return (
        `Best-fit automático <code>${{method}}</code>: ` +
        `score <code>${{best.score.toFixed(3)}}</code> ` +
        `ROC <code>${{(best.roc ?? best.radius).toFixed(3)}} mm</code> ` +
        `MO <code>${{(best.medial_offset ?? 0).toFixed(3)}} mm</code> ` +
        `PO <code>${{(best.posterior_offset ?? 0).toFixed(3)}} mm</code> ` +
        `RMSE <code>${{best.error.toFixed(3)}} mm</code> ` +
        extra +
        `cobertura <code>${{best.coverage_count}} (${{(100 * best.coverage_ratio).toFixed(1)}}%)</code> ` +
        `candidatos válidos <code>${{result.best_fit.valid_candidate_count}}/${{result.best_fit.candidate_count}}</code>.`
      );
    }}

    if (initialResult) {{
      status.innerHTML =
        `STL cargado: <code>${{initialResult.filename}}</code> ` +
        `Puntos: <code>${{initialResult.points}}</code>. ` +
        `${{bestFitSummary(initialResult)}} ` +
        `Haz clic en la cabeza humeral para comparar con una semilla manual.`;
    }}

    uploadButton.addEventListener('click', async function() {{
      const file = fileInput.files[0];
      if (!file) {{
        status.textContent = 'Selecciona primero un archivo STL.';
        return;
      }}

      uploadButton.disabled = true;
      status.innerHTML = `Cargando <code>${{file.name}}</code> y discretizando superficie...`;

      try {{
        const response = await fetch('/upload-stl', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/octet-stream',
            'X-Filename': encodeURIComponent(file.name)
          }},
          body: await file.arrayBuffer()
        }});
        const result = await response.json();
        if (!response.ok) {{
          throw new Error(result.error || 'No se pudo cargar STL');
        }}

        clearResultTraces();
        if (plot.data.length > 0) {{
          Plotly.deleteTraces(plot, Array.from({{length: plot.data.length}}, (_, i) => i));
        }}
        Plotly.addTraces(plot, result.traces);
        surfaceTraceCount = result.traces.length;
        resultTraceCount = 0;
        surfaceLoaded = true;
        status.innerHTML =
          `STL cargado: <code>${{result.filename}}</code> ` +
          `Puntos: <code>${{result.points}}</code>. ` +
          `${{bestFitSummary(result)}} ` +
          `Haz clic en la cabeza humeral para comparar con una semilla manual.`;
      }} catch (error) {{
        status.textContent = `Error cargando STL: ${{error.message}}`;
      }} finally {{
        uploadButton.disabled = false;
      }}
    }});

    plot.on('plotly_click', async function(event) {{
      if (!surfaceLoaded) {{
        status.textContent = 'Carga primero un STL para seleccionar una semilla real.';
        return;
      }}
      const point = event.points[0];
      const pointIndex = point.customdata;
      if (pointIndex === undefined || pointIndex === null) {{
        return;
      }}
      status.innerHTML = `Calculando esfera desde punto <code>${{pointIndex}}</code>...`;

      try {{
        const response = await fetch('/approximate', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{point_index: pointIndex}})
        }});
        const result = await response.json();
        if (!response.ok) {{
          throw new Error(result.error || 'Error desconocido');
        }}

        clearResultTraces();
        Plotly.addTraces(plot, result.traces);
        resultTraceCount = resultTraceIndices().length;
        keepSurfaceSelectable();

        status.innerHTML =
          `Semilla: <code>[${{result.seed.map(v => v.toFixed(3)).join(', ')}}]</code> ` +
          `Centro: <code>[${{result.center.map(v => v.toFixed(3)).join(', ')}}]</code> ` +
          `ROC: <code>${{result.roc.toFixed(3)}} mm</code> ` +
          `Eje tallo: <code>${{result.axis_length.toFixed(3)}} mm</code> ` +
          `Modelo: <code>${{result.axis_completeness}}</code> ` +
          `Crop: <code>${{result.axis_crop_mode}}</code> ` +
          `RANSAC: <code>${{result.axis_ransac_inlier_count}}/${{result.axis_fit_centerline_point_count}}</code> ` +
          `Offset medial: <code>${{result.medial_offset.toFixed(3)}} mm</code> ` +
          `Offset posterior: <code>${{result.posterior_offset.toFixed(3)}} mm</code> ` +
          `Referencia: <code>${{result.morphology_reference_status}}</code> ` +
          `Z ref: <code>ROC ${{result.reference_z_scores.roc.toFixed(2)}}, MO ${{result.reference_z_scores.medial_offset.toFixed(2)}}, PO ${{result.reference_z_scores.posterior_offset.toFixed(2)}}</code> ` +
          `RMSE: <code>${{result.error.toFixed(4)}} mm</code> ` +
          `Valida: <code>${{result.valid}}</code> ` +
          `Dibujo esfera: <code>${{result.sphere_drawn ? 'si' : 'omitido'}}</code>`;
      }} catch (error) {{
        status.textContent = `No se pudo aproximar la esfera: ${{error.message}}`;
      }}
    }});
  </script>
</body>
</html>"""


def result_to_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte resultado Python a respuesta JSON para el navegador."""
    seed = result["seed"]
    sphere = result["sphere"]
    axis = result["axis"]
    axis_length = float(axis["length"])
    morphology = AuditTrail.compute_morphological_metrics(
        sphere,
        axis,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )
    validation_data = next(
        (
            step["data"]
            for step in reversed(result["audit"].get("steps", []))
            if step.get("step_name") == "validate_approximation"
        ),
        {},
    )
    reference_flags = validation_data.get("morphology_reference_flags", {})
    reference_values = validation_data.get("morphology_reference_values", {})
    reference_reasons = validation_data.get("morphology_reference_reasons", [])
    reference_status = (
        "en referencia"
        if reference_flags.get("all_in_reference", False)
        else "fuera de referencia"
    )
    reference_z_scores = {
        key: float(value.get("z_score", float("nan")))
        for key, value in reference_values.items()
    }
    sphere_center_inside_humerus = bool(result.get("sphere_center_inside_humerus", True))
    sphere_drawn = bool(sphere_center_inside_humerus)
    traces = [
        selected_seed_trace(seed),
        *axis_traces(axis),
    ]
    if sphere_drawn:
        traces.insert(
            1,
            sphere_surface_trace(
                sphere["center"],
                sphere["radius"],
                f"Esfera RMSE={sphere['error']:.3f}mm",
            ),
        )

    visual_reason = "drawn"
    if not sphere_drawn:
        if not sphere_center_inside_humerus:
            visual_reason = "sphere center is outside the approximated humerus volume"
        else:
            visual_reason = "sphere was not drawn"
    return {
        "seed": seed.tolist(),
        "center": sphere["center"].tolist(),
        "radius": float(sphere["radius"]),
        "roc": float(morphology["roc"]),
        "medial_offset": float(morphology["medial_offset"]),
        "posterior_offset": float(morphology["posterior_offset"]),
        "total_offset": float(morphology["total_offset"]),
        "morphology": morphology,
        "error": float(sphere["error"]),
        "iterations": int(sphere["iterations"]),
        "converged": bool(sphere["converged"]),
        "axis_length": axis_length,
        "axis_method": axis.get("method", ""),
        "axis_fit_point_count": int(axis.get("axis_fit_point_count", 0)),
        "axis_fit_strategy": axis.get("axis_fit_strategy", ""),
        "axis_projected_length": float(axis.get("projected_length", axis_length)),
        "axis_is_complete_humerus": bool(axis.get("is_complete_humerus", False)),
        "axis_completeness": "completo" if axis.get("is_complete_humerus", False) else "incompleto",
        "axis_crop_mode": axis.get("crop_mode", ""),
        "axis_roi_point_count": int(axis.get("roi_point_count", 0)),
        "axis_slice_valid_count": int(axis.get("slice_valid_count", 0)),
        "axis_spike_filtered_slice_count": int(axis.get("spike_filtered_slice_count", 0)),
        "axis_fit_centerline_point_count": int(axis.get("axis_fit_centerline_point_count", 0)),
        "axis_ransac_inlier_count": int(axis.get("ransac_inlier_count", 0)),
        "axis_ransac_outlier_count": int(axis.get("ransac_outlier_count", 0)),
        "morphology_reference_flags": reference_flags,
        "morphology_reference_values": reference_values,
        "morphology_reference_statistics": validation_data.get("morphology_reference_statistics", {}),
        "morphology_reference_ranges": validation_data.get("morphology_reference_ranges", {}),
        "morphology_reference_reasons": reference_reasons,
        "morphology_reference_status": reference_status,
        "reference_z_scores": reference_z_scores,
        "sphere_center_inside_humerus": sphere_center_inside_humerus,
        "sphere_drawn": sphere_drawn,
        "sphere_visual_reason": visual_reason,
        "valid": bool(result["audit"]["final_valid"]),
        "traces": json.loads(json.dumps(traces, cls=PlotlyJSONEncoder)),
    }


def surface_to_response(
    filename: str,
    surface_points: np.ndarray,
    best_fit: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convierte una superficie cargada en respuesta JSON para Plotly."""
    traces = [surface_points_trace(surface_points, name=f"Superficie STL: {filename}")]
    if best_fit is not None:
        traces.extend(best_fit_traces(best_fit))
    return {
        "filename": filename,
        "points": int(len(surface_points)),
        "best_fit": best_fit_to_response(best_fit) if best_fit is not None else None,
        "traces": json.loads(json.dumps(traces, cls=PlotlyJSONEncoder)),
    }


def load_surface_from_upload(filename: str, data: bytes, samples: int) -> Tuple[CleanedMesh, np.ndarray, np.ndarray]:
    """Carga un STL recibido desde el navegador."""
    suffix = Path(filename).suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return load_cleaned_surface_from_stl(tmp.name, samples)


def run_selection_server(args: argparse.Namespace) -> None:
    """Sirve una pagina local para seleccionar semilla con clic."""
    state: Dict[str, Any] = {
        "surface_points": None,
        "surface_normals": None,
        "cleaned_mesh": None,
        "filename": None,
        "best_fit": None,
    }
    if args.stl or args.synthetic_demo:
        if args.stl:
            cleaned_mesh, surface_points, surface_normals = load_cleaned_surface_from_stl(args.stl, args.samples)
        else:
            cleaned_mesh = None
            surface_points, surface_normals, _ = load_demo_surface(args)
        state["surface_points"] = surface_points
        state["surface_normals"] = surface_normals
        state["cleaned_mesh"] = cleaned_mesh
        state["filename"] = Path(args.stl).name if args.stl else "synthetic-demo"
        state["best_fit"] = compute_best_fit_search(
            surface_points,
            surface_normals,
            args.best_fit_seeds,
            args.best_fit_top,
            args.initial_radius,
            args.max_error,
            cleaned_mesh=cleaned_mesh,
        )

    fig = make_selection_figure(state["surface_points"])
    if state["best_fit"] is not None:
        for trace in best_fit_traces(state["best_fit"]):
            fig.add_trace(trace)
    html = build_selection_html(
        fig,
        initial_best_fit=state["best_fit"],
        initial_filename=state["filename"],
        initial_points=len(state["surface_points"]) if state["surface_points"] is not None else 0,
    ).encode("utf-8")

    class SelectionHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *handler_args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            if self.path == "/upload-stl":
                self._handle_upload_stl()
                return
            if self.path == "/approximate":
                self._handle_approximate()
                return
            self.send_error(404)

        def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
            response = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def _handle_upload_stl(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Archivo vacio")
                filename = unquote(self.headers.get("X-Filename", "uploaded.stl"))
                filename = Path(filename).name
                data = self.rfile.read(length)
                cleaned_mesh, surface_points, surface_normals = load_surface_from_upload(filename, data, args.samples)
                best_fit = compute_best_fit_search(
                    surface_points,
                    surface_normals,
                    args.best_fit_seeds,
                    args.best_fit_top,
                    args.initial_radius,
                    args.max_error,
                    cleaned_mesh=cleaned_mesh,
                )
                state["surface_points"] = surface_points
                state["surface_normals"] = surface_normals
                state["cleaned_mesh"] = cleaned_mesh
                state["filename"] = filename
                state["best_fit"] = best_fit
                self._send_json(surface_to_response(filename, surface_points, best_fit=best_fit))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status_code=400)

        def _handle_approximate(self) -> None:
            try:
                surface_points = state["surface_points"]
                surface_normals = state["surface_normals"]
                if surface_points is None or surface_normals is None:
                    raise ValueError("Carga primero un STL")

                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                point_index = int(payload["point_index"])
                if point_index < 0 or point_index >= len(surface_points):
                    raise ValueError("Indice de punto fuera de rango")

                seed = surface_points[point_index]
                result = compute_from_seed(
                    seed,
                    surface_points,
                    surface_normals,
                    args.initial_radius,
                    args.max_error,
                )
                self._send_json(result_to_response(result))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status_code=400)

    server = ThreadingHTTPServer((args.host, args.port), SelectionHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"App local de seleccion STL: {url}")
    print("Carga un STL en el navegador y haz clic en la cabeza del humero para calcular la esfera.")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


def run_deterministic_demo(args: argparse.Namespace) -> None:
    """Ejecuta aproximacion de eje y esfera desde una semilla elegida."""
    surface_points, surface_normals, seed_candidates = load_demo_surface(args)

    seed = choose_seed(args, seed_candidates)
    result = compute_from_seed(
        seed,
        surface_points,
        surface_normals,
        args.initial_radius,
        args.max_error,
    )
    sphere = result["sphere"]
    axis = result["axis"]

    viz = InteractiveWeb3D(title="Semilla determinada: eje y esfera articular")
    viz.plot_points(surface_points, name="Superficie discretizada", color="steelblue", size=2)
    viz.plot_selected_seed(seed)
    viz.plot_sphere(sphere["center"], sphere["radius"], name=f"Esfera ajustada RMSE={sphere['error']:.3f}mm")
    viz.plot_axis(axis["origin"], axis["direction"], axis["length"], name="Eje longitudinal PCA")

    print("\nResultado de la semilla seleccionada")
    print(f"  Semilla: {seed.tolist()}")
    print(f"  Centro esfera: {sphere['center'].round(4).tolist()}")
    print(f"  Radio: {sphere['radius']:.4f} mm")
    print(f"  RMSE: {sphere['error']:.4f} mm")
    print(f"  Iteraciones: {sphere['iterations']} converged={sphere['converged']}")
    print(f"  Eje origen: {axis['origin'].round(4).tolist()}")
    print(f"  Eje direccion: {axis['direction'].round(4).tolist()}")
    print(f"  Eje longitud: {axis['length']:.4f} mm")
    print(f"  Modelo eje: {'completo' if axis.get('is_complete_humerus') else 'incompleto'}")
    print(f"  Crop eje: {axis.get('crop_mode', 'n/a')}")
    print(
        "  RANSAC eje: "
        f"{axis.get('ransac_inlier_count', 0)}/{axis.get('axis_fit_centerline_point_count', 0)} "
        "centros inlier"
    )
    morphology = AuditTrail.compute_morphological_metrics(
        sphere,
        axis,
        medial_direction=np.array([1.0, 0.0, 0.0]),
        posterior_direction=np.array([0.0, 1.0, 0.0]),
    )
    print(f"  Offset medial: {morphology['medial_offset']:.4f} mm")
    print(f"  Offset posterior: {morphology['posterior_offset']:.4f} mm")
    print(f"  Auditoria final: {result['audit']['final_valid']}")

    if args.output:
        viz.save(args.output)
    if not args.no_browser:
        viz.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="App local para cargar STL, elegir semilla y aproximar esfera.")
    parser.add_argument("--stl", help="Precarga un STL real. Tambien puedes cargarlo desde la interfaz web.")
    parser.add_argument("--samples", type=int, default=8000, help="Puntos a muestrear del STL.")
    parser.add_argument("--seed", nargs=3, type=float, help="Semilla explicita: x y z.")
    parser.add_argument("--seed-index", type=int, default=10, help="Indice reproducible entre candidatas.")
    parser.add_argument("--use-seed-index", action="store_true", help="No abre selector; usa --seed-index directamente.")
    parser.add_argument("--synthetic-demo", action="store_true", help="Usa el humero sintetico solo para demos/tests.")
    parser.add_argument("--initial-radius", type=float, default=22.0, help="Radio inicial de esfera en mm.")
    parser.add_argument("--max-error", type=float, default=2.0, help="RMSE maximo aceptado por auditoria.")
    parser.add_argument("--best-fit-seeds", type=int, default=1000, help="Semillas/iteraciones para busqueda best-fit automatica.")
    parser.add_argument("--best-fit-top", type=int, default=5, help="Cantidad de candidatos best-fit a reportar.")
    parser.add_argument("--output", help="Guarda HTML en esta ruta.")
    parser.add_argument("--no-browser", action="store_true", help="No abre navegador; util para tests.")
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor de seleccion.")
    parser.add_argument("--port", type=int, default=8765, help="Puerto del servidor de seleccion.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.seed is not None or parsed_args.use_seed_index or parsed_args.output:
        run_deterministic_demo(parsed_args)
    else:
        run_selection_server(parsed_args)
