"""Primitivas geométricas para esferas 3D."""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


class SphereGeometry:
    """Operaciones matemáticas puras sobre esferas."""

    @staticmethod
    def from_four_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Calcula la esfera definida por cuatro puntos no coplanares."""
        points = np.asarray(points, dtype=float)
        if points.shape != (4, 3):
            raise ValueError("points debe tener shape (4, 3)")
        p0 = points[0]
        matrix = 2.0 * (points[1:] - p0)
        rhs = np.sum(points[1:] ** 2, axis=1) - np.sum(p0 ** 2)
        if abs(np.linalg.det(matrix)) < 1e-9:
            return None
        center = np.linalg.solve(matrix, rhs)
        radius = float(np.linalg.norm(center - p0))
        if not np.all(np.isfinite(center)) or not np.isfinite(radius):
            return None
        return center, radius

    @staticmethod
    def algebraic_initial_fit(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Ajuste algebraico rápido usado como semilla de optimización."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("Se requieren al menos 4 puntos 3D")
        a = np.column_stack((points, np.ones(len(points))))
        b = -np.sum(points * points, axis=1)
        coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
        center = -0.5 * coeffs[:3]
        radius_sq = np.dot(center, center) - coeffs[3]
        radius = float(np.sqrt(max(radius_sq, 1e-12)))
        return center, radius

    @staticmethod
    def radial_residuals(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """Residuo geométrico firmado distancia-a-esfera."""
        points = np.asarray(points, dtype=float)
        center = np.asarray(center, dtype=float)
        return np.linalg.norm(points - center, axis=1) - float(radius)

    @staticmethod
    def radial_unit_vectors(points: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectores radiales unitarios y máscara de puntos válidos."""
        points = np.asarray(points, dtype=float)
        center = np.asarray(center, dtype=float)
        radial = points - center
        norms = np.linalg.norm(radial, axis=1)
        valid = norms > 1e-12
        unit = np.zeros_like(radial, dtype=float)
        unit[valid] = radial[valid] / norms[valid, None]
        return unit, valid

    @staticmethod
    def normal_alignment(points: np.ndarray, normals: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Concordancia absoluta entre normal de superficie y dirección radial."""
        normals = np.asarray(normals, dtype=float)
        radial_unit, valid = SphereGeometry.radial_unit_vectors(points, center)
        alignment = np.zeros(len(radial_unit), dtype=float)
        normal_norm = np.linalg.norm(normals, axis=1)
        normal_valid = normal_norm > 1e-12
        both = valid & normal_valid
        alignment[both] = np.abs(np.sum(
            (normals[both] / normal_norm[both, None]) * radial_unit[both],
            axis=1,
        ))
        return alignment

    @staticmethod
    def angular_coverage(points: np.ndarray, center: np.ndarray) -> float:
        """Estimación acotada de cobertura angular de puntos sobre una esfera."""
        radial_unit, valid = SphereGeometry.radial_unit_vectors(points, center)
        dirs = radial_unit[valid]
        if len(dirs) < 2:
            return 0.0
        centroid = dirs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm <= 1e-12:
            return 1.0
        center_dir = centroid / centroid_norm
        angles = np.arccos(np.clip(dirs @ center_dir, -1.0, 1.0))
        return float(np.clip(np.percentile(angles, 95) / (0.5 * np.pi), 0.0, 1.0))

    @staticmethod
    def robust_fit(
        points: np.ndarray,
        initial_center: np.ndarray,
        initial_radius: float,
        weights: Optional[np.ndarray] = None,
        radius_bounds: Tuple[float, float] = (0.0, np.inf),
        f_scale: float = 1.0,
        max_nfev: int = 300,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Refina esfera minimizando distancia geométrica radial con pérdida Huber."""
        points = np.asarray(points, dtype=float)
        if len(points) < 4:
            raise ValueError("Se requieren al menos 4 puntos para ajustar esfera")
        if weights is None:
            weights = np.ones(len(points), dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        low, high = radius_bounds
        bounded_radius = float(np.clip(initial_radius, low, high))
        initial = np.array([
            float(initial_center[0]),
            float(initial_center[1]),
            float(initial_center[2]),
            bounded_radius,
        ])

        def residual(params: np.ndarray) -> np.ndarray:
            return SphereGeometry.radial_residuals(points, params[:3], params[3]) * weights

        optimized = least_squares(
            residual,
            initial,
            loss="huber",
            f_scale=max(float(f_scale), 1e-6),
            bounds=([-np.inf, -np.inf, -np.inf, low], [np.inf, np.inf, np.inf, high]),
            max_nfev=max_nfev,
        )
        center = optimized.x[:3]
        radius = float(optimized.x[3])
        raw = SphereGeometry.radial_residuals(points, center, radius)
        rmse = float(np.sqrt(np.mean(raw ** 2)))
        median = float(np.median(raw))
        mad = float(np.median(np.abs(raw - median)))
        return center, radius, rmse, mad
