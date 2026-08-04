"""Tests del catálogo del dataset y del muestreo estratificado."""

from pathlib import Path

import numpy as np
import pytest

from src.dataset.catalog import (
    BoneRecord,
    STRATA_QUOTAS,
    load_catalog,
    normalize_pathology,
    stratified_sample,
)

# Encabezado con las comillas inconsistentes que trae el dataset real:
# 'Pathology' y 'Notes' aparecen entrecomillados en unos CSVs y no en otros.
HEADER_WITH_EPICONDYLES = (
    'Model ID,M/F 1/0,Age (years),R/L 1/0,Height (mm),Weight (kg),BMI (kg/m2),'
    '"Pathology",Cadaver 1/0,Whole Humerus 1/0,"Notes",'
    'GTx,GTy,GTz,LTx,LTy,LTz,HHCx,HHCy,HHCz,LEx,LEy,LEz,MEx,MEy,MEz'
)

# La cohorte hill_sachs no trae epicóndilos: esquema distinto, mismo parser.
HEADER_WITHOUT_EPICONDYLES = (
    'Model ID,M/F 1/0,Age (years),R/L 1/0,Height (mm),Weight (kg),BMI (kg/m2),'
    'Pathology,Cadaver 1/0,Whole Humerus 1/0,Notes,'
    'GTx,GTy,GTz,LTx,LTy,LTz,HHCx,HHCy,HHCz'
)


def _write_dataset(root: Path, cohort_rows: dict) -> None:
    """Arma un dataset mínimo en disco: CSV por cohorte + STLs vacíos."""
    csv_names = {
        "hill_sachs": "hill_sachs_shoulder_model_data.csv",
        "paired_shoulder": "paired_shoulder_model_data.csv",
        "single_shoulder": "single_shoulder_model_data.csv",
    }
    stl_dirs = {
        "hill_sachs": "hill_sachs_STLs",
        "paired_shoulder": "paired_shoulder_STLs",
        "single_shoulder": "single_shoulder_STLs",
    }
    root.mkdir(parents=True, exist_ok=True)
    for cohort, (header, rows, stl_ids) in cohort_rows.items():
        (root / csv_names[cohort]).write_text(
            "\n".join([header] + rows) + "\n", encoding="utf-8",
        )
        stl_dir = root / stl_dirs[cohort]
        stl_dir.mkdir(exist_ok=True)
        for model_id in stl_ids:
            (stl_dir / f"{model_id}_Humerus.stl").write_bytes(b"")


@pytest.mark.parametrize("raw,expected", [
    ("NA", "sano"),
    ("", "sano"),
    (None, "otro"),
    ("Hill-Sachs", "Hill-Sachs"),
    ("mild GHOA", "GHOA leve"),
    ("moderate GHOA", "GHOA moderada"),
    ("severe GHOA", "GHOA severa"),
    ("cuff tear", "cuff tear"),
    # El grado de artrosis domina sobre el desgarro del manguito: la
    # deformación ósea es lo que afecta al ajuste de la esfera.
    ("cuff tear, mild GHOA", "GHOA leve"),
    ("cuff tear, moderate GHOA, prior lateral acromial fracture", "GHOA moderada"),
    ("cuff tear/calcific tendonitis", "cuff tear"),
    ("cuff tear/prior cuff repair", "cuff tear"),
    # Typo presente en el dataset real (3 casos): 'CHOA' por 'GHOA'.
    ("moderate CHOA", "GHOA moderada"),
    ("anterior glenoid rim defect", "otro"),
    ("prior proximal humerus fracture", "otro"),
])
def test_normalize_pathology(raw, expected):
    assert normalize_pathology(raw) == expected


def test_load_catalog_parses_both_schemas(tmp_path: Path):
    root = tmp_path / "database"
    _write_dataset(root, {
        "paired_shoulder": (
            HEADER_WITH_EPICONDYLES,
            ['paired_001_F_56_R,0,56,1,1626,63.6,24,"NA",1,1,possible labrum,'
             '1,2,3,4,5,6,10,11,12,20,21,22,30,31,32'],
            ["paired_001_F_56_R"],
        ),
        "hill_sachs": (
            HEADER_WITHOUT_EPICONDYLES,
            ['hill_001_M_23_L,1,23,0,1829,89.1,26.6,Hill-Sachs,0,0,superior angle clipped,'
             '1,2,3,4,5,6,7,8,9'],
            ["hill_001_M_23_L"],
        ),
    })

    catalog = load_catalog(root)
    assert len(catalog) == 2

    paired = next(record for record in catalog if record.cohort == "paired_shoulder")
    assert paired.sex == "F" and paired.side == "R"
    assert paired.age == 56 and paired.bmi == 24
    assert paired.cadaver is True and paired.whole_humerus is True
    assert paired.pathology_group == "sano"
    assert paired.notes == "possible labrum"
    np.testing.assert_allclose(paired.hhc, [10, 11, 12])
    assert paired.has_epicondyles

    hill = next(record for record in catalog if record.cohort == "hill_sachs")
    assert hill.sex == "M" and hill.side == "L"
    assert hill.pathology_group == "Hill-Sachs"
    assert hill.whole_humerus is False
    # Sin columnas LE/ME el parser no debe inventar landmarks.
    assert not hill.has_epicondyles
    np.testing.assert_allclose(hill.hhc, [7, 8, 9])


def test_zero_landmark_is_treated_as_missing(tmp_path: Path):
    """Los landmarks ausentes vienen como 0,0,0, no como celda vacía."""
    root = tmp_path / "database"
    _write_dataset(root, {
        "single_shoulder": (
            HEADER_WITH_EPICONDYLES,
            ['single_001_M_56_L,1,56,0,1651,89.5,32.8,"NA",1,0,"very short humerus",'
             '1,2,3,4,5,6,10,11,12,0,0,0,0,0,0'],
            ["single_001_M_56_L"],
        ),
    })

    record = load_catalog(root)[0]
    assert record.hhc is not None
    assert not record.has_epicondyles
    assert "LE" not in record.landmarks and "ME" not in record.landmarks


def test_csv_row_without_stl_is_skipped_with_warning(tmp_path: Path):
    root = tmp_path / "database"
    _write_dataset(root, {
        "paired_shoulder": (
            HEADER_WITH_EPICONDYLES,
            [
                'paired_001_F_56_R,0,56,1,1626,63.6,24,"NA",1,1,x,1,2,3,4,5,6,10,11,12,20,21,22,30,31,32',
                'paired_005_F_40_R,0,40,1,1600,60,23,"NA",1,1,x,1,2,3,4,5,6,10,11,12,20,21,22,30,31,32',
            ],
            ["paired_001_F_56_R"],  # el 005 no tiene STL, igual que en el dataset real
        ),
    })

    with pytest.warns(UserWarning, match="paired_005_F_40_R"):
        catalog = load_catalog(root)
    assert [record.model_id for record in catalog] == ["paired_001_F_56_R"]


def _record(model_id: str, group: str, side: str = "R", whole: bool = True) -> BoneRecord:
    return BoneRecord(
        model_id=model_id, cohort="test", stl_path=Path(f"{model_id}.stl"),
        side=side, whole_humerus=whole, pathology=group, pathology_group=group,
    )


def test_stratified_sample_respects_quotas_and_is_deterministic():
    records = (
        [_record(f"severa_{i:02d}", "GHOA severa") for i in range(11)]
        + [_record(f"sano_{i:02d}", "sano", side="L" if i % 2 else "R") for i in range(40)]
        + [_record(f"leve_{i:02d}", "GHOA leve") for i in range(30)]
    )

    sample = stratified_sample(records)
    groups = [record.pathology_group for record in sample]

    # Estrato más chico que su cuota -> censo completo, sin redistribuir.
    assert groups.count("GHOA severa") == 11
    assert groups.count("sano") == STRATA_QUOTAS["sano"]
    assert groups.count("GHOA leve") == STRATA_QUOTAS["GHOA leve"]

    assert [r.model_id for r in stratified_sample(records)] == [r.model_id for r in sample]
    assert [r.model_id for r in stratified_sample(records, seed=99)] != [r.model_id for r in sample]


def test_stratified_sample_balances_side():
    records = (
        [_record(f"r_{i:02d}", "sano", side="R") for i in range(20)]
        + [_record(f"l_{i:02d}", "sano", side="L") for i in range(20)]
    )
    sample = stratified_sample(records, quotas={"sano": 12})
    sides = [record.side for record in sample]
    assert sides.count("R") == sides.count("L") == 6
