"""Visualization windows for the MeMo project."""

from .plots import CalibrationStatusPanel, LiveSensorPlot, XYGridPlot

__all__ = [
    "CalibrationStatusPanel",
    "LiveSensorPlot",
    "XYGridPlot",
]

try:  # Optional on machines without working torch / heatmap deps.
    from .heatmap_window import HeatmapWindow, XYHeatmapPlot, run_heatmap_window

    __all__.extend(
        [
            "HeatmapWindow",
            "XYHeatmapPlot",
            "run_heatmap_window",
        ]
    )
except Exception:  # pragma: no cover - depends on local runtime
    pass

try:  # Optional on machines without working torch / model deps.
    from .xy_model_heatmap import XYModelHeatmapWindow, run_xy_model_heatmap

    __all__.extend(
        [
            "XYModelHeatmapWindow",
            "run_xy_model_heatmap",
        ]
    )
except Exception:  # pragma: no cover - depends on local runtime
    pass
