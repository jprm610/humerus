"""Análisis diferencial de superficies."""

import numpy as np
from typing import Tuple
from .sphere import SphereGeometry


class DifferentialAnalyzer:
    """
    Análisis diferencial de superficies en mallas 3D.
    
    Métodos:
    - Ajuste de superficies cuadráticas locales
    - Cálculo de operadores Weingarten
    - Análisis de formas fundamentales
    """
    
    @staticmethod
    def fit_quadric_surface(
        point: np.ndarray,
        neighbors: np.ndarray,
        normal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Ajusta superficie cuadrática local (WLOP).
        
        z = a*x² + b*y² + c*xy + d*x + e*y + f
        
        Parameters
        ----------
        point : np.ndarray
            Punto central (shape: (3,))
        neighbors : np.ndarray
            Puntos vecinos (shape: (M, 3))
        normal : np.ndarray
            Vector normal unitario (shape: (3,))
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            (Hessian, gradient, fit_error)
        """
        point = np.asarray(point, dtype=float)
        neighbors = np.asarray(neighbors, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            raise ValueError("La normal no puede ser cero")
        normal = normal / normal_norm
        if len(neighbors) < 6:
            raise ValueError("Se requieren al menos 6 vecinos para ajustar una cuádrica")

        tangent_x = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(tangent_x) < 1e-8:
            tangent_x = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        tangent_x = tangent_x / np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x)

        local = neighbors - point
        x = local @ tangent_x
        y = local @ tangent_y
        z = local @ normal

        design = np.column_stack((x * x, y * y, x * y, x, y, np.ones_like(x)))
        coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
        a, b, c, d, e, _ = coeffs
        predicted = design @ coeffs
        fit_error = float(np.sqrt(np.mean((predicted - z) ** 2)))

        hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=float)
        gradient = np.array([d, e], dtype=float)
        return hessian, gradient, fit_error
    
    @staticmethod
    def compute_shape_operator(
        first_fundamental: np.ndarray,
        second_fundamental: np.ndarray
    ) -> np.ndarray:
        """
        Calcula operador de forma (Weingarten).
        
        S = (I⁻¹) * II
        
        Eigenvalores de S son curvaturas principales.
        
        Parameters
        ----------
        first_fundamental : np.ndarray
            Primera forma fundamental (shape: (2, 2))
        second_fundamental : np.ndarray
            Segunda forma fundamental (shape: (2, 2))
        
        Returns
        -------
        np.ndarray
            Operador de forma (shape: (2, 2))
        """
        return np.linalg.solve(first_fundamental, second_fundamental)
    
    @staticmethod
    def compute_principal_curvatures(
        Hessian: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calcula curvaturas principales desde Hessiana.
        
        Parameters
        ----------
        Hessian : np.ndarray
            Matriz Hessiana (shape: (2, 2))
        
        Returns
        -------
        Tuple[float, float, np.ndarray, np.ndarray]
            (k1, k2, e1, e2)
            k1, k2: curvaturas principales
            e1, e2: direcciones principales
        """
        # Eigenvalores y eigenvectores
        eigenvalues, eigenvectors = np.linalg.eigh(Hessian)
        
        # Ordenar: mayor a menor
        sorted_idx = np.argsort(-eigenvalues)
        k1 = eigenvalues[sorted_idx[0]]
        k2 = eigenvalues[sorted_idx[1]]
        e1 = eigenvectors[:, sorted_idx[0]]
        e2 = eigenvectors[:, sorted_idx[1]]
        
        return k1, k2, e1, e2
    
    @staticmethod
    def fit_sphere_to_neighbors(
        point: np.ndarray,
        neighbors: np.ndarray,
        initial_radius: float = 30.0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Ajusta esfera localmente a conjunto de puntos.
        
        Minimiza ||center - point||² - R² para todos los puntos.
        
        Parameters
        ----------
        point : np.ndarray
            Punto central (shape: (3,))
        neighbors : np.ndarray
            Puntos para ajustar (shape: (M, 3))
        initial_radius : float
            Radio inicial para optimización
        
        Returns
        -------
        Tuple[np.ndarray, float, float]
            (center, radius, error)
            center: centro de esfera ajustada (shape: (3,))
            radius: radio ajustado
            error: error cuadrático medio
        
        Notes
        -----
        Usar método de mínimos cuadrados (Levenberg-Marquardt).
        """
        points = np.vstack([point, neighbors])
        if len(points) < 4:
            raise ValueError("Se requieren al menos 4 puntos para ajustar esfera")

        center, radius = SphereGeometry.algebraic_initial_fit(points)
        center, radius, error, _ = SphereGeometry.robust_fit(
            points,
            center,
            radius,
            radius_bounds=(1e-6, np.inf),
            f_scale=1.0,
            max_nfev=200,
        )
        return center, radius, error
    
    @staticmethod
    def estimate_curvature_bounds(
        neighbors: np.ndarray,
        normal: np.ndarray,
        window_size: float = 2.0
    ) -> Tuple[float, float]:
        """
        Estima límites de curvatura usando análisis local.
        
        Parameters
        ----------
        neighbors : np.ndarray
            Puntos vecinos (shape: (M, 3))
        normal : np.ndarray
            Normal en punto central
        window_size : float
            Tamaño de ventana local (mm)
        
        Returns
        -------
        Tuple[float, float]
            (min_curvature, max_curvature)
        """
        if len(neighbors) < 6:
            return 0.0, 0.0
        point = neighbors.mean(axis=0)
        hessian, _, _ = DifferentialAnalyzer.fit_quadric_surface(point, neighbors, normal)
        k1, k2, _, _ = DifferentialAnalyzer.compute_principal_curvatures(hessian)
        return float(min(k1, k2)), float(max(k1, k2))
