"""Mapa de riesgo de falsos positivos para búsquedas de esfera local.

Ajusta una esfera local en cada punto consultado y puntúa qué tan probable
es que `HumeralHeadBestFitSearch` lo acepte como candidato válido. Esta
lógica vivía solo en `examples/demo_head_risk_map.py`; se movió aquí para
que la búsqueda automática (`HumeralHeadBestFitSearch`) pueda reusarla al
elegir semillas, en vez de duplicarla.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.differential import DifferentialAnalyzer


def fit_local_spheres(
    points: np.ndarray,
    search_radius: float,
    initial_radius: float,
    min_neighbors: int = 15,
    query_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajusta una esfera local en cada punto consultado usando sus vecinos.

    Mismo mecanismo de vecindario (radio fijo, sin filtro de normal) que
    `SphericalApproximator._get_local_points`, reusando
    `DifferentialAnalyzer.fit_sphere_to_neighbors` para el ajuste.

    Parameters
    ----------
    points : np.ndarray
        Todos los puntos de la superficie (shape: (N, 3)). Se usan como
        universo de vecinos aunque solo se consulte un subconjunto.
    search_radius : float
        Radio de vecindario local (mm)
    initial_radius : float
        Radio inicial para la optimización del ajuste
    min_neighbors : int
        Vecinos mínimos requeridos para intentar el ajuste
    query_indices : np.ndarray, optional
        Índices de `points` a evaluar. Si es None, se evalúan todos.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (rmse, fitted_radius) alineados con `query_indices`; NaN donde no
        hay suficientes vecinos o el ajuste falla.
    """
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    query_indices = np.arange(len(points)) if query_indices is None else np.asarray(query_indices, dtype=int)

    rmse = np.full(len(query_indices), np.nan)
    fitted_radius = np.full(len(query_indices), np.nan)

    for out_pos, point_index in enumerate(query_indices):
        point = points[point_index]
        neighbor_idx = tree.query_ball_point(point, search_radius)
        neighbor_idx = [j for j in neighbor_idx if j != point_index]
        if len(neighbor_idx) < min_neighbors:
            continue
        try:
            _, radius, error = DifferentialAnalyzer.fit_sphere_to_neighbors(
                point, points[neighbor_idx], initial_radius=initial_radius
            )
            rmse[out_pos] = error
            fitted_radius[out_pos] = radius
        except (ValueError, np.linalg.LinAlgError):
            continue

    return rmse, fitted_radius


def risk_score_from_fit(
    rmse: np.ndarray,
    fitted_radius: np.ndarray,
    max_error: float,
    radius_min: float,
    radius_max: float,
) -> np.ndarray:
    """
    Score de riesgo por punto en [0, 1]: 0 = candidato válido (buen ajuste
    Y radio dentro de [radius_min, radius_max]), 1 = el buscador lo
    descartaría (RMSE en o sobre max_error, o radio fuera de rango).
    """
    finite = np.isfinite(rmse) & np.isfinite(fitted_radius)
    rmse_ratio = np.where(finite, rmse / max_error, np.inf)

    radius_half = 0.5 * (radius_max - radius_min)
    outside_by = np.maximum(radius_min - fitted_radius, fitted_radius - radius_max)
    outside_by = np.maximum(outside_by, 0.0)
    radius_ratio = np.where(finite, outside_by / radius_half, np.inf)

    score = np.maximum(rmse_ratio, radius_ratio)
    return np.clip(score, 0.0, 1.0)


def compute_risk_scores(
    points: np.ndarray,
    search_radius: float,
    initial_radius: float,
    max_error: float,
    radius_min: float = 20.0,
    radius_max: float = 40.0,
    min_neighbors: int = 15,
    query_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Atajo: ajusta esferas locales y devuelve solo el score de riesgo."""
    rmse, fitted_radius = fit_local_spheres(points, search_radius, initial_radius, min_neighbors, query_indices)
    return risk_score_from_fit(rmse, fitted_radius, max_error, radius_min, radius_max)
