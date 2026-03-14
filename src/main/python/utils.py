"""Compatibility wrapper for notebook imports."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memo.ml.data import create_dataloaders, get_data, prepare_data, set_seed
from memo.ml.preprocessing import min_max_normalize
from memo.ml.visualization import plot_heatmap

__all__ = [
    "create_dataloaders",
    "get_data",
    "min_max_normalize",
    "plot_heatmap",
    "prepare_data",
    "set_seed",
]
