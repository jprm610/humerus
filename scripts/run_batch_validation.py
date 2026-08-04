"""Corre el pipeline sobre el dataset y mide el error contra el HHC medido.

No interactivo: sin navegador, sin servidor, sin ventanas. Escribe una
línea JSON por hueso a medida que termina, así una interrupción no pierde
trabajo y `--resume` continúa donde quedó.

Uso:
    # muestra estratificada por patología (n=60, censo de GHOA severa)
    python scripts/run_batch_validation.py --stratified --workers 10

    # dataset completo, reutilizando lo ya calculado
    python scripts/run_batch_validation.py --all --workers 10 --resume

    # prueba rápida
    python scripts/run_batch_validation.py --limit 5 --workers 2
"""

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.batch.runner import load_done_ids, run_batch          # noqa: E402
from src.config import PipelineParams                          # noqa: E402
from src.dataset.catalog import load_catalog, stratified_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(ROOT / "data" / "database"),
                        help="Directorio del dataset extraído")
    parser.add_argument("--out", default=str(ROOT / "results"),
                        help="Directorio de salida")
    parser.add_argument("--jsonl", default=None,
                        help="Ruta del JSONL de resultados (default: <out>/raw_results.jsonl)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="Saltar los model_id ya presentes en el JSONL")
    parser.add_argument("--export-candidates", action="store_true",
                        help="Guardar los 80 candidatos por húmero en <out>/candidatos.jsonl, "
                             "para probar funciones de costo sin volver a ajustar esferas")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stratified", action="store_true",
                           help="Muestra estratificada por patología (n=60)")
    selection.add_argument("--all", dest="run_all", action="store_true",
                           help="Los 229 húmeros")
    parser.add_argument("--cohort", default=None,
                        help="Filtrar por cohorte (hill_sachs, paired_shoulder, single_shoulder)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Correr solo estos model_id (para reproducir o depurar casos puntuales)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N (después de filtrar)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl) if args.jsonl else out_dir / "raw_results.jsonl"

    catalog = load_catalog(args.data_root)
    print(f"Catálogo: {len(catalog)} húmeros con STL y metadata")

    records = catalog
    if args.models:
        wanted = set(args.models)
        records = [record for record in records if record.model_id in wanted]
        missing = wanted - {record.model_id for record in records}
        if missing:
            raise SystemExit(f"model_id no encontrados en el catálogo: {sorted(missing)}")
    if args.cohort:
        records = [record for record in records if record.cohort == args.cohort]
    if args.stratified:
        records = stratified_sample(records)
    if args.limit:
        records = records[:args.limit]

    strata = collections.Counter(record.pathology_group for record in records)
    print(f"Seleccionados: {len(records)}")
    for group, count in strata.most_common():
        print(f"    {group:16s} {count:3d}")

    if args.resume:
        already = load_done_ids(jsonl) & {record.model_id for record in records}
        if already:
            print(f"Reanudando: {len(already)} ya calculados, faltan {len(records) - len(already)}")

    params = PipelineParams()
    started = time.time()

    def report(index: int, total: int, row: dict) -> None:
        if row["status"] != "ok":
            detail = f"ERROR {row.get('error_type')}: {row.get('error_msg', '')[:60]}"
        elif row.get("center_error_mm") is None:
            detail = "sin candidato válido"
        else:
            detail = f"err={row['center_error_mm']:6.2f}mm  r={row['radius']:.2f}"
        print(f"[{index:3d}/{total:3d}] {row['model_id']:34s} {detail}  ({row['runtime_s']:.0f}s)",
              flush=True)

    candidates_jsonl = (out_dir / "candidatos.jsonl") if args.export_candidates else None
    results = run_batch(
        records, params=params, workers=args.workers,
        out_jsonl=jsonl, resume=args.resume, on_result=report,
        candidates_jsonl=candidates_jsonl,
    )

    elapsed = time.time() - started
    ok = [row for row in results if row["status"] == "ok"]
    errors = [row for row in results if row["status"] != "ok"]
    measured = [row["center_error_mm"] for row in ok if row.get("center_error_mm") is not None]

    print(f"\n=== Resumen ({elapsed / 60:.1f} min) ===")
    print(f"Procesados en esta corrida: {len(results)}  |  ok: {len(ok)}  errores: {len(errors)}")
    if measured:
        measured_sorted = sorted(measured)
        median = measured_sorted[len(measured_sorted) // 2]
        within = sum(1 for value in measured if value <= params.success_threshold_mm)
        print(f"Error vs HHC: mediana={median:.2f}mm  min={measured_sorted[0]:.2f}  "
              f"max={measured_sorted[-1]:.2f}")
        print(f"Dentro de {params.success_threshold_mm:.0f}mm: {within}/{len(measured)} "
              f"({100 * within / len(measured):.0f}%)")
    for row in errors:
        print(f"  ERROR {row['model_id']}: {row.get('error_type')} {row.get('error_msg', '')[:80]}")
    print(f"\nResultados: {jsonl}")
    if candidates_jsonl is not None:
        print(f"Candidatos: {candidates_jsonl}")


if __name__ == "__main__":
    main()
