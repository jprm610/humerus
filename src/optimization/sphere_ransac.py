"""RANSAC y refinamiento robusto de esfera sobre superficie triangular."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..audit.trail import AuditTrail
from ..axis.longitudinal import AxisApproximator
from ..geometry.sphere import SphereGeometry
from ..mesh.cleaner import CleanedMesh, MeshCleaner
from ..validation.sphere import SphereValidator, SurfaceSupportValidationConfig


@dataclass
class SphereRansacConfig:
    """Parámetros de búsqueda RANSAC esférica."""

    n_iterations: int = 1000
    distance_tolerance: float = 1.5
    normal_angle_max: float = 25.0
    radius_range: Tuple[float, float] = (17.0, 40.0)
    random_seed: int = 42
    min_inlier_faces: int = 20
    min_inlier_area_ratio: float = 0.005
    refinement_iterations: int = 10
    neighbor_angle_max: float = 45.0
    proximal_fraction: float = 0.35
    morphology_weight: float = 2.0
    reference_miss_weight: float = 8.0
    morphology_z_cap: float = 6.0
    residual_p95_weight: float = 2.5
    angular_compactness_weight: float = 4.0
    min_angular_compactness: float = 0.45
    core_distance_tolerance: float = 1.1
    articular_side_min: float = 0.0
    articular_side_weight: float = 3.0


class SphereRansacFitter:
    """Detecta la superficie articular y ajusta una esfera robusta."""

    def __init__(self, config: Optional[SphereRansacConfig] = None):
        self.config = config or SphereRansacConfig()

    def fit(
        self,
        mesh: CleanedMesh,
        axis: Optional[Dict[str, Any]] = None,
        audit_trail: Optional[AuditTrail] = None,
    ) -> Dict[str, Any]:
        """Ejecuta RANSAC, segmentación conectada y refinamiento geométrico."""
        axis = axis or AxisApproximator.compute_longitudinal_axis(mesh.face_centroids)
        candidate_regions = self._candidate_end_regions(mesh, axis)
        if not candidate_regions:
            raise ValueError("No hay suficientes caras proximales candidatas para RANSAC")

        if audit_trail:
            audit_trail.log_step("sphere_ransac_start", {
                "candidate_regions": [
                    {"head_side": region["head_side"], "candidate_faces": int(len(region["faces"]))}
                    for region in candidate_regions
                ],
                "n_iterations": int(self.config.n_iterations),
                "distance_tolerance": float(self.config.distance_tolerance),
                "normal_angle_max": float(self.config.normal_angle_max),
            })

        rng = np.random.default_rng(self.config.random_seed)
        region_results = []
        attempts = max(1, int(self.config.n_iterations))
        for region in candidate_regions:
            best = self._ransac_region(mesh, axis, region["faces"], rng, attempts)
            if best is None:
                continue
            refined = self._segment_and_refine(mesh, best["center"], best["radius"], best["articular_face_indices"], axis)
            morphology = self._morphology(axis, refined)
            score_parts = self._score(mesh, refined, morphology)
            region_results.append({
                "head_side": region["head_side"],
                "candidate_face_count": int(len(region["faces"])),
                "raw_score": float(best["raw_score"]),
                "local_score": float(best["local_score"]),
                "initial_morphology_penalty": float(best["morphology_penalty"]),
                "initial_reference_miss_count": int(best["reference_miss_count"]),
                "refined": refined,
                "morphology": morphology,
                "score_parts": score_parts,
            })

        if not region_results:
            raise ValueError("RANSAC no encontró una esfera compatible con los extremos del húmero")

        valid_results = [entry for entry in region_results if entry["score_parts"]["valid"]]
        best_region = min(valid_results or region_results, key=lambda entry: entry["score_parts"]["score"])
        refined = best_region["refined"]
        morphology = best_region["morphology"]
        score_parts = best_region["score_parts"]
        result = {
            **refined,
            "axis": axis,
            "head_side": best_region["head_side"],
            "candidate_face_count": int(best_region["candidate_face_count"]),
            "candidate_region_count": int(len(region_results)),
            "valid_region_count": int(sum(1 for entry in region_results if entry["score_parts"]["valid"])),
            "candidate_region_summaries": [
                self._region_summary(entry)
                for entry in sorted(region_results, key=lambda item: item["score_parts"]["score"])
            ],
            "morphology": morphology.get("metrics"),
            "morphology_z_scores": morphology.get("z_scores", {}),
            "morphology_reference_flags": morphology.get("flags", {}),
            "score": score_parts["score"],
            "score_components": score_parts["components"],
            "valid": bool(score_parts["valid"]),
            "reasons": score_parts["reasons"],
        }

        if audit_trail:
            audit_trail.log_step("sphere_ransac_result", self._json_result(result))
        return result

    @staticmethod
    def sphere_from_four_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Calcula la esfera que pasa por cuatro puntos no coplanares."""
        return SphereGeometry.from_four_points(points)

    def _candidate_end_regions(self, mesh: CleanedMesh, axis: Dict[str, Any]) -> list:
        """Selecciona ambos extremos del húmero para no confundir cabeza y codo."""
        origin = np.asarray(axis["origin"], dtype=float)
        distal = np.asarray(axis["distal_point"], dtype=float)
        direction = distal - origin
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = np.asarray(axis["direction"], dtype=float)
            norm = np.linalg.norm(direction)
        direction = direction / norm

        centroids = mesh.face_centroids
        projections = (centroids - origin) @ direction
        radial = self._axis_radial_distance(centroids, origin, direction)
        length = float(projections.max() - projections.min())
        max_depth = min(90.0, max(35.0, self.config.proximal_fraction * float(axis["length"])))
        end_depth = min(max_depth, max(20.0, 0.25 * length))
        low_limit = float(projections.min())
        high_limit = float(projections.max())
        low_spread = self._spread(radial[projections <= low_limit + end_depth])
        high_spread = self._spread(radial[projections >= high_limit - end_depth])

        regions = [
            {
                "head_side": "low_projection",
                "faces": np.where(projections <= low_limit + max_depth)[0],
                "spread": low_spread,
            },
            {
                "head_side": "high_projection",
                "faces": np.where(projections >= high_limit - max_depth)[0],
                "spread": high_spread,
            },
        ]
        filtered = [region for region in regions if len(region["faces"]) >= 4]
        if len(filtered) == 2 and np.array_equal(filtered[0]["faces"], filtered[1]["faces"]):
            return [filtered[0]]
        return filtered

    def _proximal_candidate_faces(self, mesh: CleanedMesh, axis: Dict[str, Any]) -> tuple:
        """Compatibilidad: retorna el extremo más ancho, usado solo por código legado."""
        regions = self._candidate_end_regions(mesh, axis)
        if not regions:
            return np.asarray([], dtype=int), "none"
        best = max(regions, key=lambda region: region.get("spread", 0.0))
        return best["faces"], best["head_side"]

    def _ransac_region(
        self,
        mesh: CleanedMesh,
        axis: Dict[str, Any],
        candidate_faces: np.ndarray,
        rng: np.random.Generator,
        attempts: int,
    ) -> Optional[Dict[str, Any]]:
        """Busca la mejor esfera inicial dentro de un extremo candidato."""
        best = None
        for _ in range(attempts):
            sample = self._sample_four_faces(mesh, candidate_faces, rng)
            if sample is None:
                continue
            sphere = self.sphere_from_four_points(mesh.face_centroids[sample])
            if sphere is None:
                continue
            center, radius = sphere
            if not (self.config.radius_range[0] <= radius <= self.config.radius_range[1]):
                continue
            evaluated = self._evaluate_sphere(mesh, center, radius, candidate_faces, axis=axis)
            component = self._largest_component(mesh.adjacency, evaluated["compatible_faces"])
            if len(component) < self.config.min_inlier_faces:
                continue
            inlier_area = float(mesh.face_areas[component].sum())
            if inlier_area < self.config.min_inlier_area_ratio * float(mesh.face_areas.sum()):
                continue
            summary = self._candidate_summary(mesh, center, radius, component, evaluated, axis)
            if best is None or summary["raw_score"] < best["raw_score"]:
                best = summary
        return best

    def _sample_four_faces(
        self,
        mesh: CleanedMesh,
        candidate_faces: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """Muestrea cuatro caras separadas para evitar parches diminutos."""
        if len(candidate_faces) < 4:
            return None
        areas = mesh.face_areas[candidate_faces].astype(float)
        probabilities = areas / areas.sum() if areas.sum() > 0 else None
        first = int(rng.choice(candidate_faces, p=probabilities))
        selected = [first]
        points = mesh.face_centroids
        for _ in range(3):
            remaining = np.setdiff1d(candidate_faces, np.asarray(selected), assume_unique=False)
            if len(remaining) == 0:
                return None
            distances = np.min(
                np.linalg.norm(points[remaining, None, :] - points[np.asarray(selected)][None, :, :], axis=2),
                axis=1,
            )
            spread_weights = distances ** 2
            area_weights = mesh.face_areas[remaining]
            weights = spread_weights * np.maximum(area_weights, 1e-12)
            if weights.sum() <= 0:
                choice = int(rng.choice(remaining))
            else:
                choice = int(rng.choice(remaining, p=weights / weights.sum()))
            selected.append(choice)
        return np.asarray(selected, dtype=int)

    def _evaluate_sphere(
        self,
        mesh: CleanedMesh,
        center: np.ndarray,
        radius: float,
        face_indices: np.ndarray,
        axis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evalúa residuo radial y concordancia normal para caras candidatas."""
        points = mesh.face_centroids[face_indices]
        normals = mesh.face_normals[face_indices]
        residuals = np.abs(SphereGeometry.radial_residuals(points, center, radius))
        radial_unit, valid = SphereGeometry.radial_unit_vectors(points, center)
        normal_alignment = SphereGeometry.normal_alignment(points, normals, center)
        side_alignment = self._articular_side_alignment(points, center, axis)
        threshold = np.cos(np.deg2rad(self.config.normal_angle_max))
        compatible = (
            (residuals <= self.config.distance_tolerance)
            & (normal_alignment >= threshold)
            & (side_alignment >= self.config.articular_side_min)
            & valid
        )
        return {
            "residuals": residuals,
            "normal_alignment": normal_alignment,
            "side_alignment": side_alignment,
            "compatible_faces": face_indices[compatible],
        }

    def _segment_and_refine(
        self,
        mesh: CleanedMesh,
        center: np.ndarray,
        radius: float,
        initial_faces: np.ndarray,
        axis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alterna expansión conectada y refit robusto hasta estabilizar."""
        current_faces = np.asarray(initial_faces, dtype=int)
        converged = False
        iterations = 0
        for iteration in range(1, self.config.refinement_iterations + 1):
            iterations = iteration
            expanded = self._expand_region(mesh, center, radius, current_faces, axis)
            refined_center, refined_radius, rmse, mad = self._robust_refit(mesh, expanded, center, radius)
            changed = set(expanded.tolist()) != set(current_faces.tolist())
            moved = np.linalg.norm(refined_center - center) + abs(refined_radius - radius)
            current_faces = expanded
            center = refined_center
            radius = refined_radius
            if not changed and moved < 1e-4:
                converged = True
                break

        _, _, rmse, mad = self._robust_refit(mesh, current_faces, center, radius)
        current_faces, center, radius, rmse, mad = self._trim_to_core_region(
            mesh,
            current_faces,
            center,
            radius,
            rmse,
            mad,
            axis,
        )

        angular = self._angular_coverage(mesh.face_centroids[current_faces], center)
        angular_compactness = self._angular_compactness(mesh.face_centroids[current_faces], center)
        residuals = np.abs(SphereGeometry.radial_residuals(mesh.face_centroids[current_faces], center, radius))
        radial_p95 = float(np.percentile(residuals, 95)) if len(residuals) else float("inf")
        inlier_area = float(mesh.face_areas[current_faces].sum())
        labels, sizes = MeshCleaner.connected_components(self._restricted_adjacency(mesh.adjacency, current_faces))
        dominant_ratio = float(sizes.max() / max(1, sizes.sum())) if len(sizes) else 0.0
        normal_score = self._normal_score(mesh, center, current_faces)
        side_score = self._articular_side_score(mesh.face_centroids[current_faces], center, axis)
        return {
            "center": center,
            "radius": float(radius),
            "error": float(rmse),
            "rmse": float(rmse),
            "mad": float(mad),
            "inlier_area": inlier_area,
            "inlier_area_ratio": float(inlier_area / max(float(mesh.face_areas.sum()), 1e-12)),
            "angular_coverage": float(angular),
            "angular_compactness": float(angular_compactness),
            "radial_p95": radial_p95,
            "normal_score": float(normal_score),
            "articular_side_score": float(side_score),
            "dominant_component_ratio": dominant_ratio,
            "connected_component_count": int(len(sizes)),
            "articular_face_indices": np.asarray(current_faces, dtype=int),
            "inlier_face_count": int(len(current_faces)),
            "iterations": int(iterations),
            "converged": bool(converged),
        }

    def _trim_to_core_region(
        self,
        mesh: CleanedMesh,
        face_indices: np.ndarray,
        center: np.ndarray,
        radius: float,
        rmse: float,
        mad: float,
        axis: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Conserva el núcleo conectado con residuo bajo tras el refit."""
        if len(face_indices) < self.config.min_inlier_faces:
            return face_indices, center, radius, rmse, mad
        residuals = np.abs(SphereGeometry.radial_residuals(mesh.face_centroids[face_indices], center, radius))
        side_alignment = self._articular_side_alignment(mesh.face_centroids[face_indices], center, axis)
        tolerance = min(float(self.config.distance_tolerance), float(self.config.core_distance_tolerance))
        core_faces = face_indices[
            (residuals <= tolerance)
            & (side_alignment >= self.config.articular_side_min)
        ]
        core_component = self._largest_component(mesh.adjacency, core_faces)
        if len(core_component) < self.config.min_inlier_faces:
            return face_indices, center, radius, rmse, mad
        if mesh.face_areas[core_component].sum() < self.config.min_inlier_area_ratio * float(mesh.face_areas.sum()):
            return face_indices, center, radius, rmse, mad
        refined_center, refined_radius, refined_rmse, refined_mad = self._robust_refit(
            mesh,
            core_component,
            center,
            radius,
        )
        return core_component, refined_center, refined_radius, refined_rmse, refined_mad

    def _expand_region(
        self,
        mesh: CleanedMesh,
        center: np.ndarray,
        radius: float,
        seed_faces: np.ndarray,
        axis: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Expande una región compatible hacia caras vecinas."""
        selected = set(int(face) for face in seed_faces)
        queue = list(selected)
        normal_threshold = np.cos(np.deg2rad(self.config.normal_angle_max))
        neighbor_threshold = np.cos(np.deg2rad(self.config.neighbor_angle_max))
        while queue:
            face = queue.pop()
            for neighbor in mesh.adjacency[face]:
                if neighbor in selected:
                    continue
                point = mesh.face_centroids[neighbor]
                radial = point - center
                radial_norm = np.linalg.norm(radial)
                if radial_norm <= 1e-12:
                    continue
                residual = abs(radial_norm - radius)
                radial_unit = radial / radial_norm
                side_ok = self._articular_side_alignment(
                    mesh.face_centroids[np.asarray([neighbor])],
                    center,
                    axis,
                )[0] >= self.config.articular_side_min
                normal_ok = abs(float(np.dot(mesh.face_normals[neighbor], radial_unit))) >= normal_threshold
                smooth_ok = abs(float(np.dot(mesh.face_normals[neighbor], mesh.face_normals[face]))) >= neighbor_threshold
                if residual <= self.config.distance_tolerance and side_ok and normal_ok and smooth_ok:
                    selected.add(int(neighbor))
                    queue.append(int(neighbor))
        return np.asarray(sorted(selected), dtype=int)

    def _robust_refit(
        self,
        mesh: CleanedMesh,
        face_indices: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> tuple:
        """Refina centro/radio minimizando distancia geométrica radial."""
        return SphereGeometry.robust_fit(
            mesh.face_centroids[face_indices],
            center,
            radius,
            weights=np.sqrt(np.maximum(mesh.face_areas[face_indices], 1e-12)),
            radius_bounds=self.config.radius_range,
            f_scale=self.config.distance_tolerance,
            max_nfev=300,
        )

    def _candidate_summary(
        self,
        mesh: CleanedMesh,
        center: np.ndarray,
        radius: float,
        face_indices: np.ndarray,
        evaluated: Dict[str, Any],
        axis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resumen barato usado durante RANSAC para escoger candidato inicial."""
        residuals = np.abs(SphereGeometry.radial_residuals(mesh.face_centroids[face_indices], center, radius))
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        area = float(mesh.face_areas[face_indices].sum())
        angular = self._angular_coverage(mesh.face_centroids[face_indices], center)
        local_score = (mad + 1e-6) / ((area / max(mesh.face_areas.sum(), 1e-12)) + 1e-3) / (angular + 0.05)
        morphology = self._morphology(axis, {"center": center, "radius": radius, "rmse": mad})
        morphology_penalty, reference_miss_count = self._morphology_penalty(morphology)
        raw_score = (
            local_score
            + self.config.morphology_weight * morphology_penalty
            + self.config.reference_miss_weight * reference_miss_count
        )
        return {
            "center": center,
            "radius": float(radius),
            "articular_face_indices": np.asarray(face_indices, dtype=int),
            "raw_score": float(raw_score),
            "local_score": float(local_score),
            "morphology_penalty": float(morphology_penalty),
            "reference_miss_count": int(reference_miss_count),
            "compatible_face_count": int(len(evaluated["compatible_faces"])),
        }

    def _score(
        self,
        mesh: CleanedMesh,
        result: Dict[str, Any],
        morphology: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calcula score final y validación dura."""
        mad_norm = float(result["mad"]) / max(self.config.distance_tolerance, 1e-6)
        area_ratio = float(result["inlier_area_ratio"])
        angular = float(result["angular_coverage"])
        normal_score = float(result["normal_score"])
        connectivity = float(result["dominant_component_ratio"])
        radial_p95_norm = float(result.get("radial_p95", result["rmse"])) / max(self.config.distance_tolerance, 1e-6)
        angular_compactness = float(result.get("angular_compactness", 1.0))
        articular_side_score = float(result.get("articular_side_score", 1.0))
        compactness_penalty = max(0.0, self.config.min_angular_compactness - angular_compactness)
        articular_side_penalty = max(0.0, self.config.articular_side_min - articular_side_score)
        support = max(area_ratio, 1e-4) * max(angular, 0.03) * max(normal_score, 0.03) * max(connectivity, 0.03)
        local_fit_score = mad_norm / support
        morphology_penalty, reference_miss_count = self._morphology_penalty(morphology)
        reference_miss_penalty = self.config.reference_miss_weight * reference_miss_count
        score = (
            local_fit_score
            + self.config.morphology_weight * morphology_penalty
            + reference_miss_penalty
            + self.config.residual_p95_weight * radial_p95_norm
            + self.config.angular_compactness_weight * compactness_penalty
            + self.config.articular_side_weight * articular_side_penalty
        )

        support_validation = SphereValidator.validate_surface_support(
            result,
            total_area=float(mesh.face_areas.sum()),
            config=SurfaceSupportValidationConfig(
                min_radius=self.config.radius_range[0],
                max_radius=self.config.radius_range[1],
                min_inlier_faces=self.config.min_inlier_faces,
                min_inlier_area_ratio=self.config.min_inlier_area_ratio,
            ),
        )

        return {
            "score": float(score),
            "components": {
                "mad_norm": mad_norm,
                "area_ratio": area_ratio,
                "angular_coverage": angular,
                "normal_score": normal_score,
                "connectivity": connectivity,
                "radial_p95_norm": radial_p95_norm,
                "angular_compactness": angular_compactness,
                "compactness_penalty": compactness_penalty,
                "articular_side_score": articular_side_score,
                "articular_side_penalty": articular_side_penalty,
                "local_fit_score": local_fit_score,
                "morphology_penalty": morphology_penalty,
                "morphology_weight": float(self.config.morphology_weight),
                "reference_miss_count": int(reference_miss_count),
                "reference_miss_penalty": float(reference_miss_penalty),
                "support": support,
            },
            "valid": bool(support_validation["valid"]),
            "reasons": support_validation["reasons"],
        }

    def _morphology_penalty(self, morphology: Dict[str, Any]) -> Tuple[float, int]:
        """Penaliza candidatos con ROC/MO/PO lejos de la morfología humeral esperada."""
        z_scores = morphology.get("z_scores", {})
        finite_z = [
            min(abs(float(value)), float(self.config.morphology_z_cap))
            for value in z_scores.values()
            if np.isfinite(float(value))
        ]
        z_penalty = float(np.mean([value ** 2 for value in finite_z])) if finite_z else 0.0
        flags = morphology.get("flags", {})
        reference_keys = (
            "roc_in_reference",
            "medial_offset_in_reference",
            "posterior_offset_in_reference",
        )
        reference_miss_count = sum(1 for key in reference_keys if not bool(flags.get(key, True)))
        return z_penalty, int(reference_miss_count)

    @staticmethod
    def _region_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resumen JSON-friendly de un extremo evaluado por RANSAC."""
        refined = entry["refined"]
        morphology = entry["morphology"]
        score_parts = entry["score_parts"]
        return {
            "head_side": entry["head_side"],
            "candidate_face_count": int(entry["candidate_face_count"]),
            "score": float(score_parts["score"]),
            "valid": bool(score_parts["valid"]),
            "raw_score": float(entry["raw_score"]),
            "local_score": float(entry["local_score"]),
            "initial_morphology_penalty": float(entry["initial_morphology_penalty"]),
            "initial_reference_miss_count": int(entry["initial_reference_miss_count"]),
            "radius": float(refined["radius"]),
            "center": np.asarray(refined["center"], dtype=float).tolist(),
            "rmse": float(refined["rmse"]),
            "mad": float(refined["mad"]),
            "inlier_area": float(refined["inlier_area"]),
            "angular_coverage": float(refined["angular_coverage"]),
            "angular_compactness": float(refined["angular_compactness"]),
            "radial_p95": float(refined["radial_p95"]),
            "articular_side_score": float(refined["articular_side_score"]),
            "inlier_face_count": int(refined["inlier_face_count"]),
            "morphology": morphology.get("metrics", {}),
            "morphology_z_scores": morphology.get("z_scores", {}),
            "morphology_reference_flags": morphology.get("flags", {}),
            "score_components": score_parts["components"],
            "reasons": score_parts["reasons"],
        }

    @staticmethod
    def _morphology(axis: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas morfológicas y z-scores de referencia."""
        sphere = {"center": result["center"], "radius": result["radius"], "error": result["rmse"]}
        summary = SphereValidator.morphology_summary(
            sphere,
            axis,
            medial_direction=np.array([1.0, 0.0, 0.0]),
            posterior_direction=np.array([0.0, 1.0, 0.0]),
        )
        return {
            "metrics": summary["morphology"],
            "z_scores": summary["z_scores"],
            "flags": summary["morphology_reference_flags"],
        }

    @staticmethod
    def _largest_component(adjacency: list, face_indices: np.ndarray) -> np.ndarray:
        """Retorna el componente conectado más grande dentro de face_indices."""
        face_set = set(int(face) for face in face_indices)
        visited = set()
        best = []
        for start in face_set:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in face_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(component) > len(best):
                best = component
        return np.asarray(sorted(best), dtype=int)

    @staticmethod
    def _restricted_adjacency(adjacency: list, face_indices: np.ndarray) -> list:
        """Crea adyacencia compacta restringida a un conjunto de caras."""
        index = {int(face): pos for pos, face in enumerate(face_indices)}
        restricted = [[] for _ in range(len(face_indices))]
        for face, pos in index.items():
            restricted[pos] = [index[n] for n in adjacency[face] if n in index]
        return restricted

    @staticmethod
    def _axis_radial_distance(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        vecs = points - origin
        axial = np.outer(vecs @ direction, direction)
        return np.linalg.norm(vecs - axial, axis=1)

    @staticmethod
    def _spread(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        return float(np.percentile(values, 90))

    @staticmethod
    def _angular_coverage(points: np.ndarray, center: np.ndarray) -> float:
        """Estimación acotada de cobertura angular de un conjunto sobre la esfera."""
        return SphereGeometry.angular_coverage(points, center)

    @staticmethod
    def _angular_compactness(points: np.ndarray, center: np.ndarray) -> float:
        """Mide si los puntos forman un casquete compacto en vez de rodear la esfera."""
        radial_unit, valid = SphereGeometry.radial_unit_vectors(points, center)
        dirs = radial_unit[valid]
        if len(dirs) == 0:
            return 0.0
        return float(np.linalg.norm(dirs.mean(axis=0)))

    @staticmethod
    def _articular_side_alignment(
        points: np.ndarray,
        center: np.ndarray,
        axis: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Alineación con el hemisferio externo de la cabeza respecto al eje."""
        points = np.asarray(points, dtype=float)
        if axis is None or len(points) == 0:
            return np.ones(len(points), dtype=float)
        origin = np.asarray(axis.get("origin"), dtype=float)
        direction = np.asarray(axis.get("direction"), dtype=float)
        center = np.asarray(center, dtype=float)
        if origin.shape != (3,) or direction.shape != (3,) or center.shape != (3,):
            return np.ones(len(points), dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-12:
            return np.ones(len(points), dtype=float)
        direction = direction / direction_norm
        axial_position = float(np.dot(center - origin, direction))
        axis_point = origin + axial_position * direction
        offset = center - axis_point
        offset_norm = np.linalg.norm(offset)
        if offset_norm <= 1e-6:
            return np.ones(len(points), dtype=float)
        offset_unit = offset / offset_norm
        radial_unit, valid = SphereGeometry.radial_unit_vectors(points, center)
        alignment = np.zeros(len(points), dtype=float)
        alignment[valid] = radial_unit[valid] @ offset_unit
        return alignment

    @staticmethod
    def _articular_side_score(
        points: np.ndarray,
        center: np.ndarray,
        axis: Optional[Dict[str, Any]],
    ) -> float:
        """Promedio acotado de cuánto soporte cae en el hemisferio articular."""
        alignment = SphereRansacFitter._articular_side_alignment(points, center, axis)
        if len(alignment) == 0:
            return 0.0
        return float(np.mean(np.clip(alignment, -1.0, 1.0)))

    @staticmethod
    def _normal_score(mesh: CleanedMesh, center: np.ndarray, face_indices: np.ndarray) -> float:
        points = mesh.face_centroids[face_indices]
        alignment = SphereGeometry.normal_alignment(points, mesh.face_normals[face_indices], center)
        if len(alignment) == 0:
            return 0.0
        return float(np.mean(np.clip(alignment, 0.0, 1.0)))

    @staticmethod
    def _json_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte resultado a tipos serializables para auditoría."""
        payload = {}
        for key, value in result.items():
            if key == "axis":
                continue
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            elif isinstance(value, dict):
                payload[key] = SphereRansacFitter._json_result(value)
            elif isinstance(value, (np.floating, np.integer)):
                payload[key] = value.item()
            else:
                payload[key] = value
        return payload
