"""Catálogo del dataset de húmeros: metadata clínica + ground truth anatómico.

El dataset trae tres cohortes (`hill_sachs`, `paired_shoulder`,
`single_shoulder`), cada una con su CSV de metadata y su carpeta de STLs.
Los CSVs incluyen `HHCx/y/z`, el centro de la cabeza humeral medido, que es
el ground truth contra el cual se valida la esfera que ajusta el pipeline.

Los tres CSVs no comparten esquema: `hill_sachs` tiene 44 columnas y no trae
epicóndilos (`LE`/`ME`), los otros dos tienen 50. Además los encabezados
vienen con comillas inconsistentes (`"Notes"` vs `Notes`), así que los
nombres de columna se normalizan al leer.
"""

import csv
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Landmarks anatómicos con prefijo `<nombre>x/y/z` en los CSVs.
LANDMARK_KEYS = (
    "GC",    # glenoid center
    "TS",    # trigonum spinae
    "IA",    # inferior angle
    "PLA",   # posterolateral acromion
    "AC",    # acromioclavicular
    "LT",    # lesser tuberosity
    "GT",    # greater tuberosity
    "SCT",   # surgical neck top
    "SCB",   # surgical neck bottom
    "Crest",  # bicipital crest
    "HHC",   # humeral head center  <- ground truth
    "LE",    # lateral epicondyle
    "ME",    # medial epicondyle
)

#: Grupos de patología usados para estratificar y para desglosar el informe.
PATHOLOGY_GROUPS = (
    "sano",
    "Hill-Sachs",
    "GHOA leve",
    "GHOA moderada",
    "GHOA severa",
    "cuff tear",
    "otro",
)

#: Cuotas de la muestra estratificada (n=60). `GHOA severa` es censo: solo
#: hay 11 en todo el dataset y es el único modo de falla observado.
STRATA_QUOTAS: Dict[str, int] = {
    "GHOA severa": 11,
    "GHOA moderada": 9,
    "GHOA leve": 10,
    "Hill-Sachs": 10,
    "cuff tear": 8,
    "sano": 12,
    "otro": 0,
}

_COHORT_FILES = {
    "hill_sachs": "hill_sachs_shoulder_model_data.csv",
    "paired_shoulder": "paired_shoulder_model_data.csv",
    "single_shoulder": "single_shoulder_model_data.csv",
}

_COHORT_STL_DIRS = {
    "hill_sachs": "hill_sachs_STLs",
    "paired_shoulder": "paired_shoulder_STLs",
    "single_shoulder": "single_shoulder_STLs",
}


@dataclass
class BoneRecord:
    """Un húmero del dataset con su metadata y ground truth."""

    model_id: str
    cohort: str
    stl_path: Path
    sex: Optional[str] = None            # 'M' / 'F'
    age: Optional[float] = None
    side: Optional[str] = None           # 'R' / 'L'
    height: Optional[float] = None
    weight: Optional[float] = None
    bmi: Optional[float] = None
    pathology: str = "NA"                # texto crudo del CSV
    pathology_group: str = "otro"        # grupo normalizado
    cadaver: Optional[bool] = None
    whole_humerus: Optional[bool] = None
    notes: str = ""
    landmarks: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def hhc(self) -> Optional[np.ndarray]:
        """Centro de la cabeza humeral (ground truth), o None si falta."""
        return self.landmarks.get("HHC")

    @property
    def has_epicondyles(self) -> bool:
        return "LE" in self.landmarks and "ME" in self.landmarks


def normalize_pathology(raw: Optional[str]) -> str:
    """
    Agrupa el texto libre de `Pathology` en uno de `PATHOLOGY_GROUPS`.

    El grado de artrosis glenohumeral domina sobre `cuff tear` en las
    cadenas compuestas (p.ej. "cuff tear, mild GHOA" -> "GHOA leve"),
    porque la deformación de la cabeza es lo que afecta al ajuste de la
    esfera; un desgarro del manguito rotador no cambia la geometría ósea.

    Nota: el dataset trae "moderate CHOA" (3 casos) — un typo de "GHOA".
    Sin contemplarlo, esos casos caerían en "otro" y se perderían del
    estrato de artrosis moderada.
    """
    if raw is None:
        return "otro"
    text = raw.strip().strip('"').lower()
    if text in ("", "na", "n/a", "none"):
        return "sano"
    if "hill-sachs" in text or "hill sachs" in text:
        return "Hill-Sachs"
    # 'choa' cubre el typo presente en el dataset.
    for grade, group in (("severe", "GHOA severa"),
                         ("moderate", "GHOA moderada"),
                         ("mild", "GHOA leve")):
        if f"{grade} ghoa" in text or f"{grade} choa" in text:
            return group
    if "cuff tear" in text:
        return "cuff tear"
    return "otro"


def _clean_key(key: Optional[str]) -> str:
    return (key or "").strip().strip('"').strip()


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip().strip('"')
    if text in ("", "NA", "N/A", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Optional[str]) -> Optional[bool]:
    number = _to_float(value)
    return None if number is None else bool(int(number))


def _read_landmarks(row: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Extrae los landmarks presentes. Los ausentes vienen como `0,0,0`
    (no como celda vacía), así que el origen exacto se trata como faltante:
    ningún landmark anatómico real cae en el origen del sistema del CT.
    """
    landmarks: Dict[str, np.ndarray] = {}
    for name in LANDMARK_KEYS:
        coords = [_to_float(row.get(f"{name}{axis}")) for axis in ("x", "y", "z")]
        if any(c is None for c in coords):
            continue
        point = np.array(coords, dtype=float)
        if np.allclose(point, 0.0):
            continue
        landmarks[name] = point
    return landmarks


def _parse_cohort_csv(csv_path: Path, cohort: str, stl_dir: Path) -> List[BoneRecord]:
    records: List[BoneRecord] = []
    missing_stl: List[str] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for raw_row in csv.DictReader(handle):
            row = {_clean_key(k): v for k, v in raw_row.items() if k is not None}
            model_id = (row.get("Model ID") or "").strip().strip('"')
            if not model_id:
                continue

            stl_path = stl_dir / f"{model_id}_Humerus.stl"
            if not stl_path.exists():
                missing_stl.append(model_id)
                continue

            sex_flag = _to_float(row.get("M/F 1/0"))
            side_flag = _to_float(row.get("R/L 1/0"))
            pathology = (row.get("Pathology") or "NA").strip().strip('"')

            records.append(BoneRecord(
                model_id=model_id,
                cohort=cohort,
                stl_path=stl_path,
                sex=None if sex_flag is None else ("M" if sex_flag else "F"),
                age=_to_float(row.get("Age (years)")),
                side=None if side_flag is None else ("R" if side_flag else "L"),
                height=_to_float(row.get("Height (mm)")),
                weight=_to_float(row.get("Weight (kg)")),
                bmi=_to_float(row.get("BMI (kg/m2)")),
                pathology=pathology,
                pathology_group=normalize_pathology(pathology),
                cadaver=_to_bool(row.get("Cadaver 1/0")),
                whole_humerus=_to_bool(row.get("Whole Humerus 1/0")),
                notes=(row.get("Notes") or "").strip().strip('"'),
                landmarks=_read_landmarks(row),
            ))

    if missing_stl:
        warnings.warn(
            f"{cohort}: {len(missing_stl)} fila(s) de CSV sin STL correspondiente: "
            f"{', '.join(missing_stl)}",
            stacklevel=2,
        )
    return records


def load_catalog(root) -> List[BoneRecord]:
    """
    Carga las tres cohortes desde `root` (el directorio `data/database`).

    Devuelve solo los modelos que tienen STL en disco, ordenados por
    `model_id`. Se espera exactamente una fila de CSV sin STL
    (`paired_shoulder_005_F_40_R`), reportada como warning.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    records: List[BoneRecord] = []
    for cohort, csv_name in _COHORT_FILES.items():
        csv_path = root / csv_name
        stl_dir = root / _COHORT_STL_DIRS[cohort]
        if not csv_path.exists():
            warnings.warn(f"Falta el CSV de la cohorte {cohort}: {csv_path}", stacklevel=2)
            continue
        records.extend(_parse_cohort_csv(csv_path, cohort, stl_dir))

    records.sort(key=lambda record: record.model_id)
    return records


def _balanced_pick(bucket: List[BoneRecord], quota: int, rng: random.Random) -> List[BoneRecord]:
    """
    Elige `quota` registros de `bucket` repartiendo entre las combinaciones
    de lado (R/L) y completitud del húmero, para no confundir el efecto de
    la patología con el de un hueso truncado por el CT.
    """
    if quota >= len(bucket):
        return list(bucket)

    groups: Dict[tuple, List[BoneRecord]] = {}
    for record in bucket:
        groups.setdefault((record.side, record.whole_humerus), []).append(record)
    for group in groups.values():
        rng.shuffle(group)

    # Round-robin sobre los grupos: reparte lo más parejo posible y es
    # determinista dado el mismo seed.
    order = sorted(groups, key=lambda key: (str(key[0]), str(key[1])))
    picked: List[BoneRecord] = []
    while len(picked) < quota:
        progressed = False
        for key in order:
            if not groups[key]:
                continue
            picked.append(groups[key].pop())
            progressed = True
            if len(picked) == quota:
                break
        if not progressed:
            break
    return picked


def stratified_sample(
    records: Sequence[BoneRecord],
    quotas: Optional[Dict[str, int]] = None,
    seed: int = 11,
) -> List[BoneRecord]:
    """
    Muestra estratificada por grupo de patología, determinista dado `seed`.

    Si un estrato tiene menos elementos que su cuota se toma completo
    (censo), sin redistribuir el faltante a otros estratos: el objetivo es
    cubrir los estratos clínicamente relevantes, no llegar a un n exacto.
    """
    quotas = STRATA_QUOTAS if quotas is None else quotas
    rng = random.Random(seed)

    buckets: Dict[str, List[BoneRecord]] = {group: [] for group in PATHOLOGY_GROUPS}
    for record in records:
        buckets.setdefault(record.pathology_group, []).append(record)

    sample: List[BoneRecord] = []
    for group, quota in quotas.items():
        if quota <= 0:
            continue
        bucket = sorted(buckets.get(group, []), key=lambda record: record.model_id)
        sample.extend(_balanced_pick(bucket, quota, rng))

    sample.sort(key=lambda record: record.model_id)
    return sample
