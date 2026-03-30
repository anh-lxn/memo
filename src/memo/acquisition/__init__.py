"""Sensor acquisition and simple desktop UI."""

from .readers import CsvReplayReader, MockReader, SensorReader
from .recorder import CSV_COLUMNS, CsvSampleRecorder
from .ui import run_app
from .widgets import CalibrationStatusPanel, LiveSensorPlot, XYGridPlot

__all__ = [
    "CSV_COLUMNS",
    "CalibrationStatusPanel",
    "CsvReplayReader",
    "CsvSampleRecorder",
    "LiveSensorPlot",
    "MockReader",
    "SensorReader",
    "XYGridPlot",
    "run_app",
]
