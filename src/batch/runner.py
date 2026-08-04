"""Corrida del pipeline sobre muchos húmeros, con aislamiento de fallas.

El pipeline solo tenía `try/except` por semilla (`best_fit.py:125`), así
que un STL problemático abortaba la corrida entera. Acá cada hueso se
procesa aislado: si falla, se registra con `status='error'` y la corrida
sigue.

Los resultados se escriben incrementalmente a JSONL, una línea por hueso,
para que una corrida interrumpida no pierda nada y para poder reanudar.
"""

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from ..axis.longitudinal import AxisApproximator
from ..config import PipelineParams
from ..dataset.catalog import BoneRecord
from ..mesh.discretizer import MeshDiscretizer
from ..mesh.loader import STLLoader
from ..optimization.best_fit import HumeralHeadBestFitSearch
from ..validation.confidence import confidence_flags
from ..validation.ground_truth import axis_error, center_error, frame_diagnostics


def _base_record(record: BoneRecord) -> Dict[str, Any]:
    """Columnas de metadata, presentes tanto en éxito como en error."""
    hhc = record.hhc
    return {
        "model_id": record.model_id,
        "cohort": record.cohort,
        "pathology": record.pathology,
        "pathology_group": record.pathology_group,
        "side": record.side,
        "sex": record.sex,
        "age": record.age,
        "bmi": record.bmi,
        "cadaver": record.cadaver,
        "whole_humerus": record.whole_humerus,
        "notes": record.notes,
        "hhc_x": None if hhc is None else float(hhc[0]),
        "hhc_y": None if hhc is None else float(hhc[1]),
        "hhc_z": None if hhc is None else float(hhc[2]),
    }


def _candidate_rows(record: BoneRecord, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Aplana los candidatos de un húmero, con su error contra el HHC medido.

    Ajustar 80 esferas por hueso es la parte cara de la corrida (~30 s). Con
    los candidatos guardados, probar una función de costo distinta es
    re-rankear un CSV: milisegundos en vez de 10 minutos de recómputo.
    """
    hhc = record.hhc
    rows = []
    for rank, candidate in enumerate(result.get("all_candidates", [])):
        sphere = candidate.get("sphere")
        if sphere is None:
            continue
        center = np.asarray(sphere["center"], dtype=float)
        components = candidate.get("score_components", {})
        rows.append({
            "model_id": record.model_id,
            "rank": rank,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "radius": float(sphere["radius"]),
            "rmse": float(sphere["error"]),
            "converged": bool(sphere.get("converged", False)),
            "score": float(candidate.get("score", float("inf"))),
            "coverage_ratio": float(candidate.get("coverage_ratio", 0.0)),
            "rmse_norm": float(components.get("rmse_norm", float("nan"))),
            "coverage_penalty": float(components.get("coverage_penalty", float("nan"))),
            "convergence_penalty": float(components.get("convergence_penalty", float("nan"))),
            "morphology_penalty": float(components.get("morphology_penalty", float("nan"))),
            "reference_penalty": float(components.get("reference_penalty", float("nan"))),
            "center_error_mm": center_error(center, hhc),
        })
    return rows


def process_one(
    record: BoneRecord,
    params: Optional[PipelineParams] = None,
    export_candidates: bool = False,
) -> Dict[str, Any]:
    """
    Corre el pipeline completo sobre un húmero y lo compara con su HHC.

    Nunca propaga excepciones: cualquier falla se devuelve como un registro
    con `status='error'`, para que un hueso malo no tumbe la corrida.

    Con `export_candidates`, el registro incluye `_candidates`: la lista de
    los 80 candidatos con su error real, para re-rankear offline.
    """
    params = params or PipelineParams()
    row = _base_record(record)
    started = time.time()

    try:
        mesh = STLLoader.load(str(record.stl_path))
        extent = np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(axis=0)
        points, normals = MeshDiscretizer().discretize_uniform(
            mesh.vertices, mesh.faces, params.samples, random_seed=params.discretize_seed,
        )

        axis = AxisApproximator.compute_longitudinal_axis(points)

        result = HumeralHeadBestFitSearch(
            n_seeds=params.n_seeds,
            top_k=params.top_k,
            initial_radius=params.initial_radius,
            max_error=params.max_error,
            random_seed=params.random_seed,
            weight_seeds_by_risk=params.weight_seeds_by_risk,
            risk_radius_min=params.radius_min,
            risk_radius_max=params.radius_max,
            risk_min_neighbors=params.min_neighbors,
            head_radius_min=params.head_radius_min,
            head_radius_max=params.head_radius_max,
        ).search(points, normals, axis=axis)

        best = result.get("best")
        sphere = best.get("sphere") if best else None
        components = (best or {}).get("score_components", {})

        row.update({
            "status": "ok",
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "bbox_max_extent": float(np.max(extent)),
            "axis_length": float(axis.get("length", float("nan"))),
            "axis_valid": bool(axis.get("validation", {}).get("overall_valid", False)),
            "head_side": result.get("head_side"),
            "candidate_count": int(result.get("candidate_count", 0)),
            "valid_candidate_count": int(result.get("valid_candidate_count", 0)),
            "plausible_candidate_count": int(result.get("plausible_candidate_count", 0)),
            "radius_filter_applied": bool(result.get("radius_filter_applied", False)),
            "center_x": None if sphere is None else float(sphere["center"][0]),
            "center_y": None if sphere is None else float(sphere["center"][1]),
            "center_z": None if sphere is None else float(sphere["center"][2]),
            "radius": None if sphere is None else float(sphere["radius"]),
            "rmse": None if sphere is None else float(sphere["error"]),
            "converged": None if sphere is None else bool(sphere.get("converged", False)),
            "score": None if best is None else float(best.get("score", float("nan"))),
            "valid": None if best is None else bool(best.get("valid", False)),
            "coverage_ratio": None if best is None else float(best.get("coverage_ratio", 0.0)),
            "reference_penalty": float(components.get("reference_penalty", 0.0)) if components else None,
            "morphology_penalty": float(components.get("morphology_penalty", 0.0)) if components else None,
            "center_error_mm": center_error(None if sphere is None else sphere["center"], record.hhc),
            "axis_error_deg": axis_error(
                axis, record.hhc, record.landmarks.get("LE"), record.landmarks.get("ME"),
            ),
        })
        row.update(frame_diagnostics(sphere, axis, record.landmarks, record.side))
        if export_candidates:
            row["_candidates"] = _candidate_rows(record, result)

    except Exception as exc:  # aislamiento a nivel hueso, a propósito
        row.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:300],
            "traceback": traceback.format_exc()[-1500:],
        })

    row["runtime_s"] = round(time.time() - started, 2)
    flags = confidence_flags(row)
    row["needs_review"] = flags["needs_review"]
    row["confidence"] = flags["confidence"]
    row["review_reasons"] = "; ".join(flags["review_reasons"])
    return row


def _worker(args) -> Dict[str, Any]:
    """Entrada picklable para el pool (nada de closures ni lambdas)."""
    record, params, export_candidates = args
    return process_one(record, params, export_candidates)


def load_done_ids(jsonl_path) -> Set[str]:
    """`model_id` ya presentes en el JSONL, para reanudar sin reprocesar."""
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    done: Set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["model_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # línea truncada por una interrupción: se reprocesa
    return done


def run_batch(
    records: Sequence[BoneRecord],
    params: Optional[PipelineParams] = None,
    workers: int = 10,
    out_jsonl=None,
    resume: bool = False,
    on_result: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    candidates_jsonl=None,
) -> List[Dict[str, Any]]:
    """
    Procesa `records` en paralelo, escribiendo cada resultado apenas llega.

    Devuelve solo los resultados calculados en esta corrida (los saltados
    por `resume` ya están en el JSONL).

    Con `candidates_jsonl`, los 80 candidatos de cada húmero se escriben
    aparte para poder re-rankear offline.
    """
    params = params or PipelineParams()
    pending = list(records)

    if resume and out_jsonl is not None:
        done = load_done_ids(out_jsonl)
        if done:
            pending = [record for record in pending if record.model_id not in done]

    if not pending:
        return []

    export_candidates = candidates_jsonl is not None
    handle = _open_append(out_jsonl)
    candidate_handle = _open_append(candidates_jsonl)

    payload = [(record, params, export_candidates) for record in pending]
    results: List[Dict[str, Any]] = []
    total = len(payload)

    def consume(index: int, row: Dict[str, Any]) -> None:
        candidates = row.pop("_candidates", None)
        results.append(row)
        _emit(handle, row)
        if candidates:
            for candidate in candidates:
                _emit(candidate_handle, candidate)
        if on_result:
            on_result(index, total, row)

    try:
        workers = max(1, min(int(workers), total))
        if workers == 1:
            iterator = (process_one(record, params, export_candidates)
                        for record, _, _ in payload)
            for index, row in enumerate(iterator, start=1):
                consume(index, row)
        else:
            with mp.Pool(processes=workers) as pool:
                for index, row in enumerate(pool.imap_unordered(_worker, payload), start=1):
                    consume(index, row)
    finally:
        for open_handle in (handle, candidate_handle):
            if open_handle is not None:
                open_handle.close()

    return results


def _open_append(path):
    if path is None:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return open(target, "a", encoding="utf-8")


def _emit(handle, row: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()  # una interrupción no debe perder lo ya calculado
