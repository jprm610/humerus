"""Ejecución por lotes del pipeline sobre el dataset."""

from .runner import load_done_ids, process_one, run_batch

__all__ = ["load_done_ids", "process_one", "run_batch"]
