"""Métricas del resultado del pipeline contra el ground truth del dataset.

Hasta ahora la validación era cualitativa (mirar un HTML y juzgar si la
esfera "cae bien"). Los CSVs del dataset traen `HHC`, el centro de la
cabeza humeral medido, lo que convierte esa validación en un error en
milímetros.
"""

from typing import Any, Dict, Optional

import numpy as np


def center_error(fitted_center, hhc) -> Optional[float]:
    """Distancia euclídea en mm entre el centro ajustado y el HHC medido."""
    if fitted_center is None or hhc is None:
        return None
    fitted = np.asarray(fitted_center, dtype=float)
    truth = np.asarray(hhc, dtype=float)
    if fitted.shape != (3,) or truth.shape != (3,):
        return None
    return float(np.linalg.norm(fitted - truth))


def axis_error(axis: Optional[Dict[str, Any]], hhc, le, me) -> Optional[float]:
    """
    Ángulo en grados entre el eje longitudinal calculado y el eje anatómico
    de referencia: el vector que va del punto medio entre epicóndilos hasta
    el centro de la cabeza humeral.

    Devuelve None si faltan los epicóndilos (solo 87 de 229 modelos los
    traen) o si el eje no es utilizable.
    """
    if axis is None or hhc is None or le is None or me is None:
        return None

    direction = np.asarray(axis.get("direction"), dtype=float)
    if direction.shape != (3,):
        return None
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None
    direction = direction / norm

    elbow_center = 0.5 * (np.asarray(le, dtype=float) + np.asarray(me, dtype=float))
    reference = np.asarray(hhc, dtype=float) - elbow_center
    reference_norm = np.linalg.norm(reference)
    if reference_norm <= 1e-12:
        return None
    reference = reference / reference_norm

    # El eje calculado no tiene sentido garantizado (puede apuntar proximal
    # o distal), así que se toma el ángulo agudo entre las dos rectas.
    cosine = abs(float(np.dot(direction, reference)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def frame_diagnostics(
    sphere: Optional[Dict[str, Any]],
    axis: Optional[Dict[str, Any]],
    landmarks: Optional[Dict[str, np.ndarray]],
    side: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mide (sin corregirlo) el efecto del marco anatómico fijo.

    `HumeralHeadBestFitSearch` asume direcciones medial/posterior fijas
    `[1,0,0]` y `[0,1,0]` para calcular los offsets morfológicos que
    alimentan el `reference_penalty` (0.25 del score). Esa suposición no se
    sostiene en un dataset con hombros izquierdos y derechos y un sistema
    de coordenadas distinto por escaneo de CT.

    Esta función compara el offset medial/posterior calculado con el marco
    fijo contra uno derivado de landmarks reales del propio hueso: el eje
    que va de la tuberosidad menor (LT, anterior/medial) a la mayor (GT,
    lateral/posterior). El desacuerdo angular entre ambos marcos es la
    evidencia para decidir si el fix vale la pena.
    """
    result: Dict[str, Any] = {
        "frame_angle_deg": None,
        "medial_offset_fixed": None,
        "medial_offset_landmark": None,
        "posterior_offset_fixed": None,
        "posterior_offset_landmark": None,
    }
    if sphere is None or axis is None or not landmarks:
        return result
    if "GT" not in landmarks or "LT" not in landmarks:
        return result

    center = np.asarray(sphere.get("center"), dtype=float)
    origin = np.asarray(axis.get("origin"), dtype=float)
    longitudinal = np.asarray(axis.get("direction"), dtype=float)
    if center.shape != (3,) or origin.shape != (3,) or longitudinal.shape != (3,):
        return result
    norm = np.linalg.norm(longitudinal)
    if norm <= 1e-12:
        return result
    longitudinal = longitudinal / norm

    def _perpendicular(vector: np.ndarray) -> Optional[np.ndarray]:
        projected = vector - np.dot(vector, longitudinal) * longitudinal
        magnitude = np.linalg.norm(projected)
        return None if magnitude <= 1e-9 else projected / magnitude

    # Marco fijo, tal como lo usa best_fit.py hoy.
    fixed_medial = _perpendicular(np.array([1.0, 0.0, 0.0]))
    # Marco derivado del hueso: GT -> LT apunta hacia medial en un hombro
    # derecho; en uno izquierdo la lateralidad invierte el sentido.
    landmark_medial = _perpendicular(landmarks["LT"] - landmarks["GT"])
    if fixed_medial is None or landmark_medial is None:
        return result
    if side == "L":
        landmark_medial = -landmark_medial

    offset_vector = (center - origin) - np.dot(center - origin, longitudinal) * longitudinal

    fixed_posterior = np.cross(longitudinal, fixed_medial)
    landmark_posterior = np.cross(longitudinal, landmark_medial)

    cosine = abs(float(np.dot(fixed_medial, landmark_medial)))
    result["frame_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    result["medial_offset_fixed"] = abs(float(np.dot(offset_vector, fixed_medial)))
    result["medial_offset_landmark"] = abs(float(np.dot(offset_vector, landmark_medial)))
    result["posterior_offset_fixed"] = abs(float(np.dot(offset_vector, fixed_posterior)))
    result["posterior_offset_landmark"] = abs(float(np.dot(offset_vector, landmark_posterior)))
    return result
