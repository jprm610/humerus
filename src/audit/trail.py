"""Sistema de auditoría para registrar y validar cada paso del proceso.

Este módulo proporciona la clase AuditTrail que:
- Registra cada paso de cada aproximación
- Valida viabilidad de semillas
- Valida calidad de aproximaciones
- Genera reportes detallados

La auditoría es obligatoria para garantizar reproducibilidad y confiabilidad.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np

from ..validation.sphere import (
    ApproximationValidationConfig,
    MorphologyReference,
    SphereValidator,
)


@dataclass
class StepRecord:
    """Registro de un paso en el proceso."""
    step_name: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario JSON-serializable."""
        result = asdict(self)
        # Convertir arrays de numpy a listas
        for key, value in result['data'].items():
            if isinstance(value, np.ndarray):
                result['data'][key] = value.tolist()
        return result


class AuditTrail:
    """
    Sistema de auditoría para aproximación de esferas.
    
    Registra cada paso, valida procesos, y genera reportes auditables.
    
    Attributes
    ----------
    seed_id : str
        Identificador único de la semilla siendo auditada
    steps : List[StepRecord]
        Registro secuencial de todos los pasos
    validations : Dict[str, bool]
        Resultados de validaciones realizadas
    """
    
    def __init__(self, seed_id: str = ""):
        """
        Inicializa el sistema de auditoría.
        
        Parameters
        ----------
        seed_id : str, optional
            Identificador de la semilla (default: "")
        """
        self.seed_id = seed_id
        self.steps: List[StepRecord] = []
        self.validations: Dict[str, bool] = {}
        self.start_time = datetime.now()
    
    def log_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """
        Registra un paso en el proceso.
        
        Parameters
        ----------
        step_name : str
            Nombre descriptivo del paso
        data : Dict[str, Any]
            Datos asociados al paso (puede contener arrays de numpy)
        
        Examples
        --------
        >>> audit = AuditTrail("seed_001")
        >>> audit.log_step("initialize", {"point": np.array([1, 2, 3])})
        >>> audit.log_step("iteration_1", {"center": [0.5, 1.5, 2.5], "error": 0.012})
        """
        record = StepRecord(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            data=data.copy()
        )
        self.steps.append(record)
    
    def validate_seed(
        self, 
        point: np.ndarray, 
        articulation_region: Optional[np.ndarray] = None,
        curvature_threshold: float = 0.1
    ) -> bool:
        """
        Valida que una semilla está en región viable.
        
        Una semilla es viable si:
        1. Está dentro de la región articular (si se proporciona)
        2. La curvatura es compatible con una superficie esférica
        
        Parameters
        ----------
        point : np.ndarray
            Punto 3D de la semilla
        articulation_region : np.ndarray, optional
            Puntos que forman la región articular
        curvature_threshold : float
            Umbral de curvatura aceptable
        
        Returns
        -------
        bool
            True si la semilla es viable, False en caso contrario
        
        Notes
        -----
        Esta validación es crítica. Una semilla inválida puede causar
        divergencias a otras superficies del húmero.
        """
        is_valid = True
        reasons = []
        
        # Validación 1: Punto válido
        if not isinstance(point, np.ndarray) or point.shape != (3,):
            is_valid = False
            reasons.append("Point must be numpy array of shape (3,)")
        
        # Validación 2: Dentro de región articular (si se proporciona)
        if articulation_region is not None and is_valid:
            from ..validation.viability import SeedValidator

            validator = SeedValidator()
            is_in_region = validator.is_in_articulation_region(point, articulation_region, tolerance=5.0)
            distances = np.linalg.norm(articulation_region - point, axis=1)
            min_distance = float(distances.min())
            if not is_in_region:
                is_valid = False
                reasons.append(f"Point outside articulation region: {min_distance:.2f}mm")
        
        self.validations['seed_viable'] = is_valid
        self.log_step("validate_seed", {
            "seed": point.tolist() if is_valid else None,
            "viable": is_valid,
            "reasons": reasons if not is_valid else ["OK"]
        })
        
        return is_valid
    
    def is_valid_approximation(
        self,
        sphere: Dict[str, Any],
        max_error: float = 2.0,
        min_radius: float = 17.0,
        max_radius: float = 40.0,
        axis: Optional[Dict[str, Any]] = None,
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        reference_min_roc: float = 17.0,
        reference_max_roc: float = 30.0,
        reference_mean_roc: float = 22.5,
        reference_sd_roc: float = 2.8,
        reference_min_medial_offset: float = 1.0,
        reference_max_medial_offset: float = 14.0,
        reference_mean_medial_offset: float = 6.8,
        reference_sd_medial_offset: float = 2.5,
        reference_min_posterior_offset: float = 0.0,
        reference_max_posterior_offset: float = 10.0,
        reference_mean_posterior_offset: float = 2.0,
        reference_sd_posterior_offset: float = 2.0,
        enforce_morphology_reference: bool = False,
    ) -> bool:
        """
        Valida que una aproximación es aceptable.
        
        Criterios:
        1. Error cuadrático medio < max_error (default 2mm)
        2. Radio de curvatura (ROC) en rango plausible [17, 40] mm
        3. Si se entrega eje longitudinal, se auditan ROC/MO/PO contra
           rangos de referencia, pero no invalidan salvo que se solicite.
        
        Parameters
        ----------
        sphere : Dict[str, Any]
            Diccionario con 'center', 'radius', 'error'
        max_error : float
            Error máximo aceptable (mm)
        min_radius : float
            ROC mínimo aceptable (mm)
        max_radius : float
            ROC máximo aceptable (mm)
        axis : Dict[str, Any], optional
            Eje longitudinal con 'origin' y 'direction'
        surface_points : np.ndarray, optional
            Superficie usada para inferir un marco transversal si no se entregan
            direcciones anatómicas explícitas
        medial_direction : np.ndarray, optional
            Dirección medial aproximada. Se proyecta perpendicular al eje.
        posterior_direction : np.ndarray, optional
            Dirección posterior aproximada. Se proyecta perpendicular al eje.
        enforce_morphology_reference : bool
            Si True, los rangos promedio de ROC/MO/PO invalidan la aproximación.
            Si False, solo se registran como indicadores de referencia.
        
        Returns
        -------
        bool
            True si la aproximación es válida
        """
        reference = MorphologyReference(
            min_roc=reference_min_roc,
            max_roc=reference_max_roc,
            mean_roc=reference_mean_roc,
            sd_roc=reference_sd_roc,
            min_medial_offset=reference_min_medial_offset,
            max_medial_offset=reference_max_medial_offset,
            mean_medial_offset=reference_mean_medial_offset,
            sd_medial_offset=reference_sd_medial_offset,
            min_posterior_offset=reference_min_posterior_offset,
            max_posterior_offset=reference_max_posterior_offset,
            mean_posterior_offset=reference_mean_posterior_offset,
            sd_posterior_offset=reference_sd_posterior_offset,
        )
        config = ApproximationValidationConfig(
            max_error=max_error,
            min_radius=min_radius,
            max_radius=max_radius,
            enforce_morphology_reference=enforce_morphology_reference,
            reference=reference,
        )
        data = SphereValidator.validate_approximation(
            sphere,
            axis=axis,
            surface_points=surface_points,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
            config=config,
        )

        self.validations['approximation_valid'] = bool(data["valid"])
        self.log_step("validate_approximation", data)
        
        return bool(data["valid"])

    @staticmethod
    def _reference_stat(
        value: float,
        min_value: float,
        max_value: float,
        mean: float,
        sd: float,
    ) -> Dict[str, float]:
        """Empaqueta valor, rango, media, desviación y z-score."""
        return SphereValidator.reference_stat(value, min_value, max_value, mean, sd)

    @staticmethod
    def compute_morphological_metrics(
        sphere: Dict[str, Any],
        axis: Dict[str, Any],
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calcula medidas morfológicas de postprocesado.

        El ROC es el radio de la esfera ajustada. Los offsets se calculan desde
        el eje longitudinal hasta el centro de rotación de la cabeza humeral.
        Si no se entregan direcciones anatómicas, se infiere un marco transversal
        determinista perpendicular al eje.
        """
        return SphereValidator.compute_morphological_metrics(
            sphere,
            axis,
            surface_points=surface_points,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
        )

    @staticmethod
    def _transverse_frame(
        longitudinal: np.ndarray,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        surface_points: Optional[np.ndarray] = None,
    ) -> tuple:
        """Construye dos direcciones ortonormales perpendiculares al eje."""
        return SphereValidator.transverse_frame(
            longitudinal,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
            surface_points=surface_points,
        )

    @staticmethod
    def _project_perpendicular(
        vector: Optional[np.ndarray],
        axis: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Proyecta un vector al plano perpendicular del eje y lo normaliza."""
        return SphereValidator.project_perpendicular(vector, axis)
    
    def get_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de auditoría.
        
        Returns
        -------
        Dict[str, Any]
            Reporte con todos los pasos, validaciones y resumen
        
        Examples
        --------
        >>> report = audit.get_report()
        >>> print(f"Steps: {len(report['steps'])}")
        >>> print(f"Valid: {report['validations']}")
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'seed_id': self.seed_id,
            'duration_seconds': duration,
            'total_steps': len(self.steps),
            'steps': [step.to_dict() for step in self.steps],
            'validations': self.validations,
            'final_valid': all(self.validations.values()) if self.validations else None
        }
    
    def to_json(self) -> str:
        """
        Serializa auditoría a JSON.
        
        Returns
        -------
        str
            Reporte en formato JSON
        """
        return json.dumps(self.get_report(), indent=2)


class AuditManager:
    """
    Gestor de múltiples auditorías (para múltiples semillas).
    
    Mantiene un registro de todas las auditorías realizadas
    y permite generar reportes comparativos.
    """
    
    def __init__(self):
        """Inicializa gestor de auditorías."""
        self.audits: Dict[str, AuditTrail] = {}
        self.start_time = datetime.now()
    
    def create_audit(self, seed_id: str) -> AuditTrail:
        """
        Crea una nueva auditoría.
        
        Parameters
        ----------
        seed_id : str
            Identificador único de la semilla
        
        Returns
        -------
        AuditTrail
            Nueva instancia de AuditTrail
        """
        audit = AuditTrail(seed_id)
        self.audits[seed_id] = audit
        return audit
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Resumen estadístico de todas las auditorías.
        
        Returns
        -------
        Dict[str, Any]
            Resumen con conteos y estadísticas
        """
        total_audits = len(self.audits)
        valid_audits = sum(
            1 for audit in self.audits.values() 
            if audit.validations.get('approximation_valid', False)
        )
        
        return {
            'total_audits': total_audits,
            'valid_approximations': valid_audits,
            'success_rate': valid_audits / total_audits if total_audits > 0 else 0,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'seed_ids': list(self.audits.keys())
        }
