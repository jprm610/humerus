"""Fuente única de parámetros del pipeline.

Estos mismos números estaban duplicados en `best_fit.py`, `risk_map.py`,
los tres demos de `examples/` y el script de `entrega_profesor/`. Los
valores por defecto son exactamente los que se venían usando, para que las
corridas de validación sean comparables con los resultados anteriores.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineParams:
    """Parámetros de una corrida del pipeline, de extremo a extremo."""

    # Discretización de la malla
    samples: int = 4000
    discretize_seed: int = 42

    # Ajuste de esfera / mapa de riesgo
    initial_radius: float = 22.5
    max_error: float = 2.0
    radius_min: float = 20.0
    radius_max: float = 40.0
    min_neighbors: int = 15
    search_radius_factor: float = 1.15

    # Búsqueda automática de la cabeza humeral
    n_seeds: int = 80
    top_k: int = 5
    random_seed: int = 11
    weight_seeds_by_risk: bool = True

    # Rango fisiológico del radio de la cabeza humeral, usado para descartar
    # candidatos antes de rankearlos. Mismo rango que ya evaluaba
    # `AuditTrail.is_valid_approximation`, donde nunca llegaba a activarse
    # porque se aplicaba solo al ganador. Medido sobre los 229: los ajustes
    # correctos caen entre 18.9 y 32.0 mm, así que [17, 40] no los toca y sí
    # elimina los candidatos degenerados (hay radios de hasta 320.000 mm).
    head_radius_min: float = 17.0
    head_radius_max: float = 40.0

    # Umbral de éxito para el reporte (mm de error contra el HHC medido)
    success_threshold_mm: float = 5.0

    @property
    def search_radius(self) -> float:
        """Radio de vecindario local del mapa de riesgo."""
        return self.initial_radius * self.search_radius_factor
