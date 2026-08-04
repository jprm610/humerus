"""Catálogo del dataset de húmeros con ground truth anatómico."""

from .catalog import (
    BoneRecord,
    PATHOLOGY_GROUPS,
    STRATA_QUOTAS,
    load_catalog,
    normalize_pathology,
    stratified_sample,
)

__all__ = [
    "BoneRecord",
    "PATHOLOGY_GROUPS",
    "STRATA_QUOTAS",
    "load_catalog",
    "normalize_pathology",
    "stratified_sample",
]
