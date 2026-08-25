"""Reglas compartidas de validación de esferas humerales."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MorphologyReference:
    """Rangos, medias y desviaciones de referencia morfológica."""

    min_roc: float = 17.0
    max_roc: float = 30.0
    mean_roc: float = 22.5
    sd_roc: float = 2.8
    min_medial_offset: float = 1.0
    max_medial_offset: float = 14.0
    mean_medial_offset: float = 6.8
    sd_medial_offset: float = 2.5
    min_posterior_offset: float = 0.0
    max_posterior_offset: float = 10.0
    mean_posterior_offset: float = 2.0
    sd_posterior_offset: float = 2.0


@dataclass
class ApproximationValidationConfig:
    """Configuración para validación clásica de una esfera ajustada."""

    max_error: float = 2.0
    min_radius: float = 17.0
    max_radius: float = 40.0
    enforce_morphology_reference: bool = False
    reference: MorphologyReference = field(default_factory=MorphologyReference)


@dataclass
class SurfaceSupportValidationConfig:
    """Configuración para validar soporte superficial de una esfera RANSAC."""

    min_radius: float = 17.0
    max_radius: float = 40.0
    min_inlier_faces: int = 20
    min_inlier_area_ratio: float = 0.005
    min_dominant_component_ratio: float = 0.85


class SphereValidator:
    """Validador puro: decide validez y produce datos auditables."""

    @staticmethod
    def validate_approximation(
        sphere: Dict[str, Any],
        axis: Optional[Dict[str, Any]] = None,
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        config: Optional[ApproximationValidationConfig] = None,
    ) -> Dict[str, Any]:
        """Valida RMSE, ROC plausible y referencia morfológica opcional."""
        config = config or ApproximationValidationConfig()
        is_valid = True
        reasons = []
        error = float(sphere.get("error", float("inf")))
        radius = abs(float(sphere.get("radius", 0.0)))

        if error > config.max_error:
            is_valid = False
            reasons.append(f"Error {error:.3f}mm exceeds max {config.max_error}mm")
        if not (config.min_radius <= radius <= config.max_radius):
            is_valid = False
            reasons.append(f"ROC {radius:.2f}mm outside range [{config.min_radius}, {config.max_radius}]")

        morphology = None
        reference_flags = None
        reference_values = None
        reference_reasons = []
        if axis is not None:
            try:
                summary = SphereValidator.morphology_summary(
                    sphere,
                    axis,
                    surface_points=surface_points,
                    medial_direction=medial_direction,
                    posterior_direction=posterior_direction,
                    reference=config.reference,
                )
                morphology = summary["morphology"]
                reference_flags = summary["morphology_reference_flags"]
                reference_values = summary["morphology_reference_values"]
                reference_reasons = summary["morphology_reference_reasons"]

                if config.enforce_morphology_reference and not reference_flags["all_in_reference"]:
                    is_valid = False
                    reasons.extend(reference_reasons)
            except ValueError as exc:
                is_valid = False
                reasons.append(str(exc))

        data = {
            "valid": is_valid,
            "error": error,
            "roc": radius,
            "roc_plausibility_range": [float(config.min_radius), float(config.max_radius)],
            "morphology_reference_ranges": SphereValidator.reference_ranges(config.reference),
            "morphology_reference_statistics": SphereValidator.reference_statistics(config.reference),
            "enforce_morphology_reference": bool(config.enforce_morphology_reference),
            "reasons": reasons if not is_valid else ["OK"],
        }
        if morphology is not None:
            data["morphology"] = morphology
        if reference_flags is not None:
            data["morphology_reference_flags"] = reference_flags
            data["morphology_reference_values"] = reference_values
            data["morphology_reference_reasons"] = reference_reasons if reference_reasons else ["OK"]
        return data

    @staticmethod
    def validate_surface_support(
        result: Dict[str, Any],
        total_area: float,
        config: SurfaceSupportValidationConfig,
    ) -> Dict[str, Any]:
        """Valida criterios duros de una esfera con soporte triangular."""
        valid = True
        reasons = []
        radius = abs(float(result.get("radius", 0.0)))
        inlier_faces = int(result.get("inlier_face_count", 0))
        inlier_area = float(result.get("inlier_area", 0.0))
        dominant = float(result.get("dominant_component_ratio", 0.0))
        center = np.asarray(result.get("center", np.zeros(3)), dtype=float)

        if not (config.min_radius <= radius <= config.max_radius):
            valid = False
            reasons.append("radius outside plausible range")
        if inlier_faces < config.min_inlier_faces:
            valid = False
            reasons.append("too few inlier faces")
        if inlier_area < config.min_inlier_area_ratio * float(total_area):
            valid = False
            reasons.append("inlier area too small")
        if int(result.get("connected_component_count", 0)) < 1 or dominant < config.min_dominant_component_ratio:
            valid = False
            reasons.append("no dominant connected component")
        if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius):
            valid = False
            reasons.append("non-finite sphere parameters")

        return {"valid": valid, "reasons": reasons if reasons else ["OK"]}

    @staticmethod
    def compute_morphological_metrics(
        sphere: Dict[str, Any],
        axis: Dict[str, Any],
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calcula ROC y offsets transversal-medial/posterior."""
        center = np.asarray(sphere.get("center"), dtype=float)
        origin = np.asarray(axis.get("origin"), dtype=float)
        longitudinal = np.asarray(axis.get("direction"), dtype=float)

        if center.shape != (3,):
            raise ValueError("sphere['center'] debe tener shape (3,)")
        if origin.shape != (3,) or longitudinal.shape != (3,):
            raise ValueError("axis debe incluir 'origin' y 'direction' con shape (3,)")

        norm = np.linalg.norm(longitudinal)
        if norm <= 1e-12:
            raise ValueError("axis['direction'] no puede ser vector cero")
        longitudinal = longitudinal / norm

        medial_unit, posterior_unit = SphereValidator.transverse_frame(
            longitudinal,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
            surface_points=surface_points,
        )
        vec = center - origin
        axial_position = float(np.dot(vec, longitudinal))
        closest_axis_point = origin + axial_position * longitudinal
        offset_vector = center - closest_axis_point
        signed_medial = float(np.dot(offset_vector, medial_unit))
        signed_posterior = float(np.dot(offset_vector, posterior_unit))

        return {
            "roc": abs(float(sphere.get("radius", 0.0))),
            "axis_point": closest_axis_point.tolist(),
            "offset_vector": offset_vector.tolist(),
            "total_offset": float(np.linalg.norm(offset_vector)),
            "medial_offset": abs(signed_medial),
            "posterior_offset": abs(signed_posterior),
            "signed_medial_offset": signed_medial,
            "signed_posterior_offset": signed_posterior,
            "axial_position": axial_position,
            "medial_direction": medial_unit.tolist(),
            "posterior_direction": posterior_unit.tolist(),
            "longitudinal_direction": longitudinal.tolist(),
        }

    @staticmethod
    def morphology_summary(
        sphere: Dict[str, Any],
        axis: Dict[str, Any],
        surface_points: Optional[np.ndarray] = None,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        reference: Optional[MorphologyReference] = None,
    ) -> Dict[str, Any]:
        """Calcula métricas, flags, z-scores y razones morfológicas."""
        reference = reference or MorphologyReference()
        morphology = SphereValidator.compute_morphological_metrics(
            sphere,
            axis,
            surface_points=surface_points,
            medial_direction=medial_direction,
            posterior_direction=posterior_direction,
        )
        values = {
            "roc": morphology["roc"],
            "medial_offset": morphology["medial_offset"],
            "posterior_offset": morphology["posterior_offset"],
        }
        ranges = SphereValidator.reference_ranges(reference)
        reference_values = {
            "roc": SphereValidator.reference_stat(
                values["roc"], reference.min_roc, reference.max_roc, reference.mean_roc, reference.sd_roc
            ),
            "medial_offset": SphereValidator.reference_stat(
                values["medial_offset"],
                reference.min_medial_offset,
                reference.max_medial_offset,
                reference.mean_medial_offset,
                reference.sd_medial_offset,
            ),
            "posterior_offset": SphereValidator.reference_stat(
                values["posterior_offset"],
                reference.min_posterior_offset,
                reference.max_posterior_offset,
                reference.mean_posterior_offset,
                reference.sd_posterior_offset,
            ),
        }
        flags = {
            "roc_in_reference": bool(ranges["roc"][0] <= values["roc"] <= ranges["roc"][1]),
            "medial_offset_in_reference": bool(
                ranges["medial_offset"][0] <= values["medial_offset"] <= ranges["medial_offset"][1]
            ),
            "posterior_offset_in_reference": bool(
                ranges["posterior_offset"][0] <= values["posterior_offset"] <= ranges["posterior_offset"][1]
            ),
        }
        flags["all_in_reference"] = all(flags.values())
        reasons = []
        labels = {
            "roc": "ROC",
            "medial_offset": "Medial offset",
            "posterior_offset": "Posterior offset",
        }
        for key, flag_key in (
            ("roc", "roc_in_reference"),
            ("medial_offset", "medial_offset_in_reference"),
            ("posterior_offset", "posterior_offset_in_reference"),
        ):
            if not flags[flag_key]:
                low, high = ranges[key]
                reasons.append(f"{labels[key]} {values[key]:.2f}mm outside reference [{low}, {high}]")
        return {
            "morphology": morphology,
            "morphology_reference_flags": flags,
            "morphology_reference_values": reference_values,
            "morphology_reference_reasons": reasons,
            "z_scores": {key: value["z_score"] for key, value in reference_values.items()},
        }

    @staticmethod
    def reference_stat(value: float, min_value: float, max_value: float, mean: float, sd: float) -> Dict[str, float]:
        """Empaqueta valor, rango, media, desviación y z-score."""
        sd = float(sd)
        z_score = (float(value) - float(mean)) / sd if sd > 0 else float("nan")
        return {
            "value": float(value),
            "min": float(min_value),
            "max": float(max_value),
            "mean": float(mean),
            "sd": sd,
            "z_score": float(z_score),
        }

    @staticmethod
    def reference_ranges(reference: MorphologyReference) -> Dict[str, list]:
        """Rangos de referencia morfológica."""
        return {
            "roc": [float(reference.min_roc), float(reference.max_roc)],
            "medial_offset": [float(reference.min_medial_offset), float(reference.max_medial_offset)],
            "posterior_offset": [float(reference.min_posterior_offset), float(reference.max_posterior_offset)],
        }

    @staticmethod
    def reference_statistics(reference: MorphologyReference) -> Dict[str, Dict[str, float]]:
        """Media y desviación de referencia morfológica."""
        return {
            "roc": {"mean": float(reference.mean_roc), "sd": float(reference.sd_roc)},
            "medial_offset": {
                "mean": float(reference.mean_medial_offset),
                "sd": float(reference.sd_medial_offset),
            },
            "posterior_offset": {
                "mean": float(reference.mean_posterior_offset),
                "sd": float(reference.sd_posterior_offset),
            },
        }

    @staticmethod
    def transverse_frame(
        longitudinal: np.ndarray,
        medial_direction: Optional[np.ndarray] = None,
        posterior_direction: Optional[np.ndarray] = None,
        surface_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construye dos direcciones ortonormales perpendiculares al eje."""
        medial = SphereValidator.project_perpendicular(medial_direction, longitudinal)
        if medial is None and surface_points is not None:
            points = np.asarray(surface_points, dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 3:
                centered = points - points.mean(axis=0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                for candidate in vh[1:]:
                    medial = SphereValidator.project_perpendicular(candidate, longitudinal)
                    if medial is not None:
                        break

        if medial is None:
            for candidate in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
                medial = SphereValidator.project_perpendicular(candidate, longitudinal)
                if medial is not None:
                    break

        posterior = SphereValidator.project_perpendicular(posterior_direction, longitudinal)
        if posterior is not None:
            posterior = posterior - np.dot(posterior, medial) * medial
            posterior_norm = np.linalg.norm(posterior)
            posterior = posterior / posterior_norm if posterior_norm > 1e-12 else None

        if posterior is None:
            posterior = np.cross(longitudinal, medial)
            posterior = posterior / np.linalg.norm(posterior)

        return medial, posterior

    @staticmethod
    def project_perpendicular(vector: Optional[np.ndarray], axis: np.ndarray) -> Optional[np.ndarray]:
        """Proyecta un vector al plano perpendicular del eje y lo normaliza."""
        if vector is None:
            return None
        projected = np.asarray(vector, dtype=float)
        if projected.shape != (3,):
            return None
        projected = projected - np.dot(projected, axis) * axis
        norm = np.linalg.norm(projected)
        if norm <= 1e-12:
            return None
        return projected / norm
