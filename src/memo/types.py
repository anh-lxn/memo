from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(slots=True)
class SensorFrame:
    """Single acquisition sample from the membrane sensors."""

    sensors: np.ndarray
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabeledSample:
    """Recorded sample with the current CSV schema."""

    sensors: np.ndarray
    x: float
    y: float
    force: float
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    """Model output for position and optional force."""

    x: float | None = None
    y: float | None = None
    force: float | None = None
    raw_output: np.ndarray | None = None


@dataclass(slots=True)
class NormalizationStats:
    """Min/max statistics for the eight sensor channels."""

    mins: np.ndarray
    maxs: np.ndarray
