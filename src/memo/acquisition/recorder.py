from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from memo.types import LabeledSample


CSV_COLUMNS = ["timestamp", "X", "Y", "F"] + [f"Sensor R{i}" for i in range(1, 9)]
RAW_CSV_COLUMNS = ["date", "time", "X", "Y", "F"] + [f"Sensor R{i}" for i in range(1, 9)]


class CsvSampleRecorder:
    """Appends labeled membrane samples to a CSV file."""

    def __init__(self, csv_path, overwrite: bool = False):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self._write_header()
        else:
            self._ensure_header()
        self.sample_count = self._count_rows()

    def _write_header(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    def _ensure_header(self):
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        self._write_header()

    def _count_rows(self) -> int:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)

    def append_sample(self, sample: LabeledSample) -> dict[str, float | str]:
        timestamp = sample.timestamp or datetime.utcnow()
        row = {
            "timestamp": timestamp.replace(microsecond=0).isoformat(),
            "X": float(sample.x),
            "Y": float(sample.y),
            "F": float(sample.force),
        }
        for index, value in enumerate(sample.sensors, start=1):
            row[f"Sensor R{index}"] = float(value)

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        self.sample_count += 1
        return row


    def remove_point(self, point: tuple[float, float]) -> int:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return 0

        target_x = float(point[0])
        target_y = float(point[1])
        kept_rows: list[dict[str, str]] = []
        removed_count = 0

        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    row_x = float(row["X"])
                    row_y = float(row["Y"])
                except (KeyError, TypeError, ValueError):
                    kept_rows.append(row)
                    continue

                if row_x == target_x and row_y == target_y:
                    removed_count += 1
                else:
                    kept_rows.append(row)

        self._write_header()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            for row in kept_rows:
                writer.writerow(row)

        self.sample_count = self._count_rows()
        return removed_count

    def read_saved_points(self) -> list[tuple[float, float]]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []

        points: list[tuple[float, float]] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    points.append((float(row["X"]), float(row["Y"])))
                except (KeyError, TypeError, ValueError):
                    continue
        return points


class RawSampleFileWriter:
    """Writes each recorded sample to its own CSV file inside data/raw."""

    file_pattern = re.compile(r"^(?P<index>\d{4})_.*N\.csv$")

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _next_index(self) -> int:
        max_index = 0
        for csv_path in self.output_dir.glob("*.csv"):
            match = self.file_pattern.match(csv_path.name)
            if match:
                max_index = max(max_index, int(match.group("index")))
        return max_index + 1

    def write_sample(self, sample: LabeledSample) -> Path:
        timestamp = sample.timestamp or datetime.utcnow()
        force_value = int(round(float(sample.force)))
        file_path = self.output_dir / f"{self._next_index():04d}_{force_value}N.csv"

        row = {
            "date": timestamp.date().isoformat(),
            "time": timestamp.replace(microsecond=0).time().isoformat(),
            "X": float(sample.x),
            "Y": float(sample.y),
            "F": float(sample.force),
        }
        for index, value in enumerate(sample.sensors, start=1):
            row[f"Sensor R{index}"] = float(value)

        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

        return file_path
