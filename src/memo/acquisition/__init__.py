"""Sensor acquisition and simple desktop UI."""

from .readers import Ads1115Reader, CsvReplayReader, MockReader, SensorReader, SerialSensorReader, UnavailableSensorReader
from .recorder import CSV_COLUMNS, CsvSampleRecorder
from .ui import run_app
from .widgets import CalibrationStatusPanel, LiveSensorPlot, XYGridPlot

__all__ = [
    "Ads1115Reader",
    "CSV_COLUMNS",
    "CalibrationStatusPanel",
    "CsvReplayReader",
    "CsvSampleRecorder",
    "LiveSensorPlot",
    "MockReader",
    "SensorReader",
    "SerialSensorReader",
    "UnavailableSensorReader",
    "XYGridPlot",
    "run_app",
]
