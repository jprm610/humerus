"""Limpieza topológica ligera de mallas triangulares."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class CleanedMesh:
    """Malla limpia con métricas de triángulo y conectividad."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_centroids: np.ndarray
    adjacency: List[List[int]]
    cleaning_report: Dict[str, Any]


class MeshCleaner:
    """Prepara una malla STL para algoritmos basados en conectividad."""

    def __init__(
        self,
        vertex_precision: int = 6,
        min_area: float = 1e-8,
        keep_largest_component: bool = True,
    ):
        self.vertex_precision = int(vertex_precision)
        self.min_area = float(min_area)
        self.keep_largest_component = bool(keep_largest_component)

    def clean(self, vertices: np.ndarray, faces: np.ndarray) -> CleanedMesh:
        """Elimina degenerados, deduplica vértices y construye adyacencia."""
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices debe tener shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces debe tener shape (M, 3)")

        original_vertex_count = len(vertices)
        original_face_count = len(faces)
        finite_vertex_mask = np.all(np.isfinite(vertices), axis=1)
        valid_face_mask = np.all(finite_vertex_mask[faces], axis=1)
        vertices = vertices[finite_vertex_mask]
        if np.count_nonzero(finite_vertex_mask) != original_vertex_count:
            remap = -np.ones(original_vertex_count, dtype=int)
            remap[np.where(finite_vertex_mask)[0]] = np.arange(len(vertices))
            faces = remap[faces[valid_face_mask]]
        else:
            faces = faces[valid_face_mask]

        vertices, faces = self._deduplicate_vertices(vertices, faces)
        areas, normals, centroids = self._face_geometry(vertices, faces)
        nondegenerate = areas > self.min_area
        faces = faces[nondegenerate]
        areas = areas[nondegenerate]
        normals = normals[nondegenerate]
        centroids = centroids[nondegenerate]

        if len(faces) == 0:
            raise ValueError("La malla no conserva triángulos válidos después de limpieza")

        adjacency = self.build_adjacency(faces)
        component_labels, component_sizes = self.connected_components(adjacency)
        kept_component_count = int(len(component_sizes))
        removed_small_component_faces = 0

        if self.keep_largest_component and len(component_sizes) > 1:
            largest = int(np.argmax(component_sizes))
            keep = component_labels == largest
            removed_small_component_faces = int(np.count_nonzero(~keep))
            faces = faces[keep]
            areas = areas[keep]
            normals = normals[keep]
            centroids = centroids[keep]
            vertices, faces = self._compact_vertices(vertices, faces)
            areas, normals, centroids = self._face_geometry(vertices, faces)
            adjacency = self.build_adjacency(faces)
            component_labels, component_sizes = self.connected_components(adjacency)

        report = {
            "original_vertex_count": int(original_vertex_count),
            "original_face_count": int(original_face_count),
            "clean_vertex_count": int(len(vertices)),
            "clean_face_count": int(len(faces)),
            "removed_nonfinite_faces": int(original_face_count - np.count_nonzero(valid_face_mask)),
            "removed_degenerate_faces": int(np.count_nonzero(~nondegenerate)),
            "component_count_before_filter": kept_component_count,
            "component_count": int(len(component_sizes)),
            "removed_small_component_faces": removed_small_component_faces,
            "total_area": float(np.sum(areas)),
        }

        return CleanedMesh(
            vertices=vertices,
            faces=faces,
            face_normals=normals,
            face_areas=areas,
            face_centroids=centroids,
            adjacency=adjacency,
            cleaning_report=report,
        )

    def vertex_normals(self, mesh: CleanedMesh) -> np.ndarray:
        """Calcula normales por vértice ponderadas por área de cara."""
        normals = np.zeros_like(mesh.vertices, dtype=float)
        weighted = mesh.face_normals * mesh.face_areas[:, None]
        for face, normal in zip(mesh.faces, weighted):
            normals[face] += normal
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        return normals

    @staticmethod
    def build_adjacency(faces: np.ndarray) -> List[List[int]]:
        """Construye adyacencia de triángulos por aristas compartidas."""
        edge_to_faces: Dict[tuple, List[int]] = {}
        for face_index, face in enumerate(np.asarray(faces, dtype=int)):
            edges = (
                tuple(sorted((int(face[0]), int(face[1])))),
                tuple(sorted((int(face[1]), int(face[2])))),
                tuple(sorted((int(face[2]), int(face[0])))),
            )
            for edge in edges:
                edge_to_faces.setdefault(edge, []).append(face_index)

        adjacency = [set() for _ in range(len(faces))]
        for owners in edge_to_faces.values():
            if len(owners) < 2:
                continue
            for owner in owners:
                adjacency[owner].update(other for other in owners if other != owner)
        return [sorted(neighbors) for neighbors in adjacency]

    @staticmethod
    def connected_components(adjacency: List[List[int]]) -> tuple:
        """Etiqueta componentes conectados de una lista de adyacencia."""
        labels = -np.ones(len(adjacency), dtype=int)
        sizes = []
        label = 0
        for start in range(len(adjacency)):
            if labels[start] >= 0:
                continue
            stack = [start]
            labels[start] = label
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                for neighbor in adjacency[current]:
                    if labels[neighbor] < 0:
                        labels[neighbor] = label
                        stack.append(neighbor)
            sizes.append(size)
            label += 1
        return labels, np.asarray(sizes, dtype=int)

    def _deduplicate_vertices(self, vertices: np.ndarray, faces: np.ndarray) -> tuple:
        """Deduplica vértices por coordenadas redondeadas."""
        rounded = np.round(vertices, decimals=self.vertex_precision)
        unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        deduped_vertices = np.zeros((len(unique), 3), dtype=float)
        counts = np.bincount(inverse)
        np.add.at(deduped_vertices, inverse, vertices)
        deduped_vertices /= counts[:, None]
        return deduped_vertices, inverse[faces]

    @staticmethod
    def _compact_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple:
        """Descarta vértices no usados y remapea caras."""
        used = np.unique(faces)
        remap = -np.ones(len(vertices), dtype=int)
        remap[used] = np.arange(len(used))
        return vertices[used], remap[faces]

    @staticmethod
    def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple:
        """Calcula área, normal y centroide por triángulo."""
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        areas = 0.5 * lengths
        normals = np.zeros_like(cross, dtype=float)
        valid = lengths > 1e-12
        normals[valid] = cross[valid] / lengths[valid, None]
        centroids = triangles.mean(axis=1)
        return areas, normals, centroids
