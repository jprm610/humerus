"""Compara dos corridas húmero por húmero.

Los agregados esconden intercambios: un cambio que arregla 40 huesos y rompe
20 se ve igual en la mediana que uno que arregla 20 y no rompe nada. Este
script informa la migración caso por caso.

Uso:
    python scripts/compare_runs.py --base results/baseline_229.jsonl \
                                   --nuevo results/raw_results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PipelineParams  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    frame = pd.DataFrame(rows).drop_duplicates(subset="model_id", keep="last")
    return frame.set_index("model_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(ROOT / "results" / "baseline_229.jsonl"))
    parser.add_argument("--nuevo", default=str(ROOT / "results" / "raw_results.jsonl"))
    parser.add_argument("--umbral", type=float, default=PipelineParams().success_threshold_mm)
    parser.add_argument("--out", default=str(ROOT / "results" / "comparacion.csv"))
    args = parser.parse_args()

    base, nuevo = load(Path(args.base)), load(Path(args.nuevo))
    shared = base.index.intersection(nuevo.index)
    if len(shared) == 0:
        raise SystemExit("Las dos corridas no comparten ningún model_id")

    t = args.umbral
    d = pd.DataFrame({
        "pathology_group": nuevo.loc[shared, "pathology_group"],
        "err_base": base.loc[shared, "center_error_mm"],
        "err_nuevo": nuevo.loc[shared, "center_error_mm"],
        "radio_base": base.loc[shared, "radius"],
        "radio_nuevo": nuevo.loc[shared, "radius"],
    })
    d["delta"] = d.err_nuevo - d.err_base
    d["fallaba"] = d.err_base > t
    d["falla"] = d.err_nuevo > t

    arreglados = d[d.fallaba & ~d.falla]
    rotos = d[~d.fallaba & d.falla]
    siguen_mal = d[d.fallaba & d.falla]
    siguen_bien = d[~d.fallaba & ~d.falla]

    print(f"Comparando {len(shared)} húmeros  (umbral {t:.0f} mm)\n")
    print(f"{'':22s} {'BASE':>10s} {'NUEVO':>10s}")
    print(f"{'error mediano':22s} {d.err_base.median():9.2f}  {d.err_nuevo.median():9.2f}")
    print(f"{'percentil 90':22s} {d.err_base.quantile(.9):9.2f}  {d.err_nuevo.quantile(.9):9.2f}")
    print(f"{'error máximo':22s} {d.err_base.max():9.2f}  {d.err_nuevo.max():9.2f}")
    print(f"{'fuera de tolerancia':22s} {int(d.fallaba.sum()):6d}     {int(d.falla.sum()):6d}"
          f"   ({100 * d.fallaba.mean():.0f}% -> {100 * d.falla.mean():.0f}%)")

    print(f"\nMIGRACIÓN CASO POR CASO")
    print(f"  arreglados (fallaba -> pasa) : {len(arreglados):3d}")
    print(f"  rotos      (pasaba -> falla) : {len(rotos):3d}")
    print(f"  siguen fallando              : {len(siguen_mal):3d}")
    print(f"  siguen pasando               : {len(siguen_bien):3d}")
    print(f"  NETO                         : {len(arreglados) - len(rotos):+3d}")

    mejoran = int((d.delta < -0.01).sum())
    empeoran = int((d.delta > 0.01).sum())
    print(f"\n  bajan el error: {mejoran}   suben el error: {empeoran}   sin cambio: {len(d) - mejoran - empeoran}")

    if len(rotos):
        print(f"\nCASOS ROTOS (los que hay que mirar):")
        print(rotos.sort_values("delta", ascending=False)[
            ["pathology_group", "err_base", "err_nuevo", "radio_base", "radio_nuevo"]
        ].round(2).to_string())

    print(f"\nPOR PATOLOGÍA (tasa fuera de tolerancia)")
    by = d.groupby("pathology_group").agg(n=("falla", "size"), base=("fallaba", "mean"),
                                          nuevo=("falla", "mean"))
    by["cambio"] = by.nuevo - by.base
    print((by * [1, 100, 100, 100]).round(0).astype(int).to_string())

    d.round(3).to_csv(args.out, encoding="utf-8-sig")
    print(f"\nDetalle: {args.out}")


if __name__ == "__main__":
    main()
