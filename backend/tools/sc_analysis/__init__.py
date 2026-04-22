from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parents[1]
SCGPT_SOURCE_DIR = BACKEND_DIR / "scgpt_source"

for path in (BACKEND_DIR, SCGPT_SOURCE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from .core import run_analysis_to_dir
from .skill import SingleCellAnalysisParams, run_single_cell_skill

__all__ = [
    "SingleCellAnalysisParams",
    "run_analysis_to_dir",
    "run_single_cell_skill",
]
