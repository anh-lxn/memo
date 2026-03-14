"""Compatibility wrapper for notebook imports."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memo.ml.evaluation import ModelEvaluator
from memo.ml.inference import ModelPredictor
from memo.ml.models import MembraneModel
from memo.ml.training import Trainer

__all__ = ["MembraneModel", "Trainer", "ModelEvaluator", "ModelPredictor"]

