from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from memo.types import SensorFrame


SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]


class SensorReader(ABC):
    """Abstract interface for membrane sensor acquisition backends."""

    @abstractmethod
    def read(self) -> SensorFrame:
        """Return the next sensor frame."""


@dataclass
class MockReader(SensorReader):
    """Generates synthetic sensor values for local development."""

    sensor_count: int = 8
    seed: int | None = None
    baseline: float = 0.0
    noise_std: float = 1.0
    source: str = "mock"

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def read(self) -> SensorFrame:
        sensors = self._rng.normal(
            loc=self.baseline,
            scale=self.noise_std,
            size=self.sensor_count,
        ).astype(float)
        return SensorFrame(
            sensors=sensors,
            timestamp=datetime.utcnow(),
            source=self.source,
            metadata={"reader": type(self).__name__},
        )


@dataclass
class CsvReplayReader(SensorReader):
    """Replays frames from an existing dataset without changing CSV schema."""

    csv_path: str
    loop: bool = False
    source: str = "csv_replay"

    def __post_init__(self):
        self._data = pd.read_csv(self.csv_path)
        missing_columns = [column for column in SENSOR_COLUMNS if column not in self._data.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"CSV is missing required sensor columns: {missing}")
        self._index = 0

    def read(self) -> SensorFrame:
        if self._index >= len(self._data):
            if not self.loop:
                raise StopIteration("Reached end of CSV replay dataset.")
            self._index = 0

        row = self._data.iloc[self._index]
        self._index += 1

        sensors = row[SENSOR_COLUMNS].to_numpy(dtype=float)
        metadata = {
            "reader": type(self).__name__,
            "row_index": self._index - 1,
        }
        for label in ("X", "Y", "F"):
            if label in row.index:
                metadata[label.lower()] = float(row[label])

        return SensorFrame(
            sensors=sensors,
            timestamp=datetime.utcnow(),
            source=self.source,
            metadata=metadata,
        )
