"""Aproximación de esfera en superficie articular."""

import numpy as np
from typing import Dict, Optional, Any
from ..audit.trail import AuditTrail
from ..geometry.sphere import SphereGeometry


class SphericalApproximator:
    """
    Aproxima esfera a superficie articular del húmero.
    
    Algoritmo:
    1. Comenzar desde semilla
    2. Iterar: calcular curvatura local → ajustar esfera
    3. Validar convergencia
    4. Auditar cada paso
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        verbose: bool = False
    ):
        """
        Inicializa aproximador de esfera.
        
        Parameters
        ----------
        max_iterations : int
            Máximo número de iteraciones
        convergence_threshold : float
            Umbral de convergencia (cambio relativo)
        verbose : bool
            Si imprimir progreso
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
    
    def approximate_from_seed(
        self,
        seed_point: np.ndarray,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        audit_trail: Optional[AuditTrail] = None,
        initial_radius: float = 30.0
    ) -> Dict[str, Any]:
        """
        Aproxima esfera comenzando desde semilla.
        
        Parameters
        ----------
        seed_point : np.ndarray
            Punto inicial de semilla (shape: (3,))
        surface_points : np.ndarray
            Puntos de la superficie (shape: (N, 3))
        surface_normals : np.ndarray
            Normales en puntos (shape: (N, 3))
        audit_trail : AuditTrail, optional
            Sistema de auditoría
        initial_radius : float
            Radio inicial estimado (mm)
        
        Returns
        -------
        Dict[str, float]
            {
                'center': np.ndarray (3,),
                'radius': float,
                'error': float,
                'iterations': int,
                'converged': bool
            }
        
        Notes
        -----
        El algoritmo:
        1. Comienza con centro = semilla, radio = inicial
        2. En cada iteración:
           - Encontrar puntos cercanos (región local)
           - Ajustar esfera a puntos cercanos
           - Verificar convergencia
        3. Registrar en auditoría
        """
        seed_point = np.asarray(seed_point, dtype=float)
        surface_points = np.asarray(surface_points, dtype=float)
        surface_normals = np.asarray(surface_normals, dtype=float)

        if seed_point.shape != (3,):
            raise ValueError("seed_point debe tener shape (3,)")
        if surface_points.ndim != 2 or surface_points.shape[1] != 3:
            raise ValueError("surface_points debe tener shape (N, 3)")
        if len(surface_points) < 4:
            raise ValueError("Se requieren al menos 4 puntos para ajustar una esfera")

        center = seed_point.copy()
        radius = float(initial_radius)
        converged = False
        error = float("inf")

        if audit_trail:
            audit_trail.log_step("sphere_initialization", {
                "seed": seed_point.tolist(),
                "initial_radius": radius,
                "surface_points": int(len(surface_points)),
            })

        for iteration in range(1, self.max_iterations + 1):
            local_indices = self._get_local_points(seed_point, surface_points, surface_normals, max(radius * 1.15, 10.0))
            if len(local_indices) < 4:
                local_indices = self._get_local_points(seed_point, surface_points, surface_normals, max(radius * 1.5, 15.0))
            if len(local_indices) < 4:
                raise ValueError("No hay suficientes puntos locales para ajustar la esfera")

            local_points = surface_points[local_indices]
            fitted = self._fit_sphere(local_points, center, radius)
            center_new = fitted["center"]
            radius_new = fitted["radius"]
            error = fitted["error"]

            if audit_trail:
                audit_trail.log_step("sphere_iteration", {
                    "iteration": iteration,
                    "local_points": int(len(local_points)),
                    "center": center_new.tolist(),
                    "radius": float(radius_new),
                    "error": float(error),
                })

            converged = self._check_convergence(center, center_new, radius, radius_new)
            center = center_new
            radius = radius_new
            if converged:
                break

        result = {
            "center": center,
            "radius": float(radius),
            "error": float(error),
            "iterations": iteration,
            "converged": converged,
            "seed": seed_point,
            "local_point_count": int(len(local_indices)),
        }

        if audit_trail:
            audit_trail.log_step("sphere_result", {
                "center": center.tolist(),
                "radius": float(radius),
                "error": float(error),
                "iterations": iteration,
                "converged": converged,
            })

        return result
    
    def _get_local_points(
        self,
        center: np.ndarray,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        search_radius: float = 15.0
    ) -> np.ndarray:
        """
        Obtiene puntos locales alrededor del centro.
        
        Parameters
        ----------
        center : np.ndarray
            Centro de búsqueda (shape: (3,))
        surface_points : np.ndarray
            Todos los puntos (shape: (N, 3))
        surface_normals : np.ndarray
            Normales correspondientes
        search_radius : float
            Radio de búsqueda (mm)
        
        Returns
        -------
        np.ndarray
            Índices de puntos dentro del radio
        """
        distances = np.linalg.norm(surface_points - center, axis=1)
        mask = distances <= search_radius
        if len(surface_normals) == len(surface_points):
            nearest_idx = int(np.argmin(distances))
            reference = surface_normals[nearest_idx]
            reference_norm = np.linalg.norm(reference)
            normal_norms = np.linalg.norm(surface_normals, axis=1)
            valid_normals = normal_norms > 1e-12
            if reference_norm > 1e-12 and np.any(valid_normals):
                unit_normals = np.zeros_like(surface_normals, dtype=float)
                unit_normals[valid_normals] = surface_normals[valid_normals] / normal_norms[valid_normals, None]
                reference = reference / reference_norm
                mask &= (unit_normals @ reference) >= 0.25
        return np.where(mask)[0]
    
    def _fit_sphere(
        self,
        points: np.ndarray,
        initial_center: np.ndarray,
        initial_radius: float
    ) -> Dict[str, Any]:
        """
        Ajusta esfera a conjunto de puntos.
        
        Minimiza: Σ ||center - p||² - R²
        
        Parameters
        ----------
        points : np.ndarray
            Puntos para ajustar (shape: (M, 3))
        initial_center : np.ndarray
            Centro inicial (shape: (3,))
        initial_radius : float
            Radio inicial
        
        Returns
        -------
        Dict[str, float]
            {'center': np.ndarray, 'radius': float, 'error': float}
        """
        points = np.asarray(points, dtype=float)
        if len(points) < 4:
            raise ValueError("Se requieren al menos 4 puntos")

        try:
            center, radius = SphereGeometry.algebraic_initial_fit(points)
        except (ValueError, np.linalg.LinAlgError):
            center = np.asarray(initial_center, dtype=float)
            radius = float(initial_radius)

        if not np.all(np.isfinite(center)) or not np.isfinite(radius):
            center = np.asarray(initial_center, dtype=float)
            radius = float(initial_radius)

        center, radius, rmse, _ = SphereGeometry.robust_fit(
            points,
            center,
            radius,
            radius_bounds=(1e-6, np.inf),
            f_scale=1.0,
            max_nfev=200,
        )

        return {"center": center, "radius": radius, "error": rmse}
    
    def _check_convergence(
        self,
        center_old: np.ndarray,
        center_new: np.ndarray,
        radius_old: float,
        radius_new: float
    ) -> bool:
        """
        Verifica si algoritmo ha convergido.
        
        Parameters
        ----------
        center_old : np.ndarray
            Centro anterior
        center_new : np.ndarray
            Centro nuevo
        radius_old : float
            Radio anterior
        radius_new : float
            Radio nuevo
        
        Returns
        -------
        bool
            True si ha convergido
        """
        center_change = np.linalg.norm(center_new - center_old)
        radius_change = abs(radius_new - radius_old) / max(radius_old, 1e-6)
        
        return (center_change < self.convergence_threshold and
                radius_change < self.convergence_threshold)
