"""Validación de viabilidad de semillas."""

import numpy as np
from typing import Optional

from ..geometry.sphere import SphereGeometry


class SeedValidator:
    """
    Valida que puntos semilla sean viables para aproximación.
    
    Implementar en esta clase:
    - Validación de punto dentro de región articular
    - Validación de curvatura compatible con esfera
    - Validación de distancia a otra geometría del húmero
    """
    
    def __init__(self, curvature_data: Optional[np.ndarray] = None):
        """
        Inicializa validador de semillas.
        
        Parameters
        ----------
        curvature_data : np.ndarray, optional
            Datos de curvatura principal para región articular
        """
        self.curvature_data = curvature_data
    
    def is_in_articulation_region(
        self, 
        point: np.ndarray, 
        articulation_points: np.ndarray,
        tolerance: float = 5.0
    ) -> bool:
        """
        Valida si punto está en región articular.
        
        Parameters
        ----------
        point : np.ndarray
            Punto a validar (shape: (3,))
        articulation_points : np.ndarray
            Puntos de la región articular (shape: (N, 3))
        tolerance : float
            Tolerancia en mm
        
        Returns
        -------
        bool
            True si punto está en región
        """
        if articulation_points is None or len(articulation_points) == 0:
            return False
        distances = np.linalg.norm(np.asarray(articulation_points) - point, axis=1)
        return bool(distances.min() <= tolerance)
    
    def has_spherical_curvature(
        self,
        point: np.ndarray,
        neighbors: np.ndarray,
        radius_estimate: float = 30.0,
        tolerance: float = 0.1
    ) -> bool:
        """
        Valida si punto tiene curvatura compatible con esfera.
        
        Una superficie es esférica si κ₁ ≈ κ₂ ≈ 1/R
        
        Parameters
        ----------
        point : np.ndarray
            Punto a validar (shape: (3,))
        neighbors : np.ndarray
            Puntos vecinos (shape: (M, 3))
        radius_estimate : float
            Radio estimado de la esfera (mm)
        tolerance : float
            Tolerancia relativa
        
        Returns
        -------
        bool
            True si curvatura es compatible
        """
        if neighbors is None or len(neighbors) < 4:
            return False

        points = np.vstack([point, neighbors])
        try:
            center, radius = SphereGeometry.algebraic_initial_fit(points)
        except (ValueError, np.linalg.LinAlgError):
            return False
        relative_error = abs(radius - radius_estimate) / max(radius_estimate, 1e-12)
        residuals = SphereGeometry.radial_residuals(points, center, radius)
        rmse = np.sqrt(np.mean(residuals ** 2))
        return bool(relative_error <= tolerance and rmse <= max(2.0, tolerance * radius_estimate))
    
    def is_away_from_other_surfaces(
        self,
        point: np.ndarray,
        other_surface_points: np.ndarray,
        min_distance: float = 10.0
    ) -> bool:
        """
        Valida que punto está alejado de otras superficies del húmero.
        
        Previene divergencias a superficies no deseadas.
        
        Parameters
        ----------
        point : np.ndarray
            Punto a validar (shape: (3,))
        other_surface_points : np.ndarray
            Puntos de otras superficies (shape: (K, 3))
        min_distance : float
            Distancia mínima requerida (mm)
        
        Returns
        -------
        bool
            True si punto está suficientemente alejado
        """
        if other_surface_points is None or len(other_surface_points) == 0:
            return True
        distances = np.linalg.norm(np.asarray(other_surface_points) - point, axis=1)
        return bool(distances.min() >= min_distance)
