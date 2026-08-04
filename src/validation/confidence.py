"""Señales de confianza para marcar ajustes que conviene revisar a mano.

El sistema no tiene forma de saber si acertó: sin ground truth, lo único que
conoce de un ajuste es su RMSE y cuánta superficie cubre. Sobre los 229
húmeros del dataset la validación existente aceptó 229/229, incluido un
error de 97 mm, porque sus umbrales (RMSE máximo 2.0 mm, radio 17–40 mm)
están fuera del rango que producen los datos reales.

Este módulo no decide nada del ajuste: solo marca los casos dudosos para
que una persona los mire. Los umbrales se calibran empíricamente contra el
HHC medido (ver `scripts/calibrate_confidence.py`).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConfidenceThresholds:
    """
    Umbrales de revisión, calibrados sobre los 229 húmeros del dataset.

    Sobre esa corrida, `rmse > 0.75` o radio fuera de [19, 34] marca los 6
    ajustes que quedaron fuera de tolerancia sin ninguna falsa alarma. Ese
    resultado **es dentro de muestra**: los umbrales se eligieron mirando
    esas mismas 6 fallas, así que no debe leerse como "detecta el 100%".
    Con tan pocos positivos, la separación perfecta es frágil — hay que
    recalibrar contra ground truth nuevo antes de confiar en la tasa.

    El sesgo es deliberado hacia marcar de más: el costo de una falsa
    alarma es que alguien mire un ajuste bueno; el de una falla no
    detectada es un centro de rotación equivocado.
    """

    rmse_max: float = 0.75
    coverage_min: float = 0.20
    radius_min: float = 19.0
    radius_max: float = 34.0


def confidence_flags(
    record: Dict[str, Any],
    thresholds: Optional[ConfidenceThresholds] = None,
) -> Dict[str, Any]:
    """
    Evalúa un registro de resultado y devuelve los motivos de revisión.

    Parameters
    ----------
    record : dict
        Fila producida por `src.batch.runner.process_one`.

    Returns
    -------
    dict
        `needs_review` (bool), `review_reasons` (lista legible) y
        `confidence` ('alta' | 'media' | 'baja').
    """
    thresholds = thresholds or ConfidenceThresholds()
    reasons: List[str] = []

    if record.get("status") != "ok":
        return {"needs_review": True, "review_reasons": ["el pipeline falló"],
                "confidence": "baja"}

    rmse = record.get("rmse")
    coverage = record.get("coverage_ratio")
    radius = record.get("radius")

    if rmse is None or radius is None:
        return {"needs_review": True, "review_reasons": ["no se obtuvo un ajuste"],
                "confidence": "baja"}

    if rmse > thresholds.rmse_max:
        reasons.append(f"RMSE alto ({rmse:.2f} > {thresholds.rmse_max:.2f} mm)")
    if coverage is not None and coverage < thresholds.coverage_min:
        reasons.append(f"cobertura baja ({coverage:.2f} < {thresholds.coverage_min:.2f})")
    if not (thresholds.radius_min <= radius <= thresholds.radius_max):
        reasons.append(f"radio atípico ({radius:.1f} mm fuera de "
                       f"[{thresholds.radius_min:.0f}, {thresholds.radius_max:.0f}])")
    if not record.get("radius_filter_applied", True):
        reasons.append("ningún candidato tenía radio fisiológico")

    # Dos señales independientes en desacuerdo con el ajuste pesan más que una.
    if len(reasons) >= 2:
        confidence = "baja"
    elif reasons:
        confidence = "media"
    else:
        confidence = "alta"

    return {
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "confidence": confidence,
    }
