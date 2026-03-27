from __future__ import annotations

from collections import deque
import math
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget, QVBoxLayout


class LiveSensorPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axis = self.figure.add_subplot(111)
        self._build_layout()
        self.update_values(np.zeros(8, dtype=float))

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_values(self, values):
        values = np.asarray(values, dtype=float)
        self.axis.clear()
        self.axis.bar(np.arange(1, len(values) + 1), values, color="#1f77b4")
        self.axis.set_title("Live Plot der 8 Sensorwerte")
        self.axis.set_xlabel("Sensor")
        self.axis.set_ylabel("Wert")
        self.axis.set_xticks(np.arange(1, len(values) + 1))
        self.axis.set_ylim(0, 4)
        self.axis.set_yticks(np.arange(0, 4.5, 0.5))
        self.canvas.draw_idle()


class LiveForcePlot(QWidget):
    def __init__(self, history_seconds: float = 5.0, parent=None):
        super().__init__(parent)
        self.history_seconds = history_seconds
        self.timestamps: deque[datetime] = deque()
        self.values: deque[float] = deque()
        self.start_time: datetime | None = None
        self.current_time: datetime | None = None
        self.stream_active = False
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axis = self.figure.add_subplot(111)
        self._build_layout()
        self._redraw()

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def append_value(self, timestamp: datetime, value: float):
        if self.start_time is None:
            self.start_time = timestamp
        self.current_time = timestamp
        self.timestamps.append(timestamp)
        self.values.append(float(value))
        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.values.popleft()
        self._redraw()

    def advance_time(self, timestamp: datetime):
        if self.start_time is None:
            self.start_time = timestamp
        self.current_time = timestamp
        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.values.popleft()
        self._redraw()

    def set_stream_active(self, active: bool):
        self.stream_active = bool(active)
        self._redraw()

    def _redraw(self):
        self.axis.clear()
        self.axis.set_title("Live Plot Kraft ueber Zeit")
        self.axis.set_xlabel("Zeit seit Start [s]")
        self.axis.set_ylabel("Kraft [N]")
        self.axis.set_ylim(0.0, 25.0)
        self.axis.set_yticks(np.arange(0.0, 25.0 + 0.001, 2.5))

        reference_time = self.current_time.timestamp() if self.current_time is not None else None
        start_time = self.start_time.timestamp() if self.start_time is not None else None

        if self.timestamps and reference_time is not None and start_time is not None:
            x_values = np.array([timestamp.timestamp() - start_time for timestamp in self.timestamps], dtype=float)
            y_values = np.array(self.values, dtype=float)
            trace_color = "#0f8b6d" if self.stream_active else "#b42318"
            self.axis.plot(x_values, y_values, color=trace_color, linewidth=2.0)
            self.axis.scatter(x_values[-1:], y_values[-1:], color=trace_color, s=28, zorder=3)
            elapsed = max(0.0, reference_time - start_time)
            half_window = self.history_seconds / 2.0
            window_start = max(0.0, elapsed - half_window)
            window_end = window_start + self.history_seconds
            self.axis.set_xlim(window_start, window_end)
        else:
            self.axis.set_xlim(0.0, self.history_seconds)

        self.axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.canvas.draw_idle()


class XYGridPlot(QWidget):
    def __init__(
        self,
        x_limits,
        y_limits,
        grid_spacing: float,
        corner_marker_size: float | None = None,
        membrane_size: float = 450.0 * math.sqrt(2.0),
        sensor_size: tuple[float, float] = (18.0, 33.0),
        sensor_definitions: list[dict[str, float | str]] | None = None,
        on_point_selected=None,
        parent=None,
    ):
        super().__init__(parent)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.grid_spacing = grid_spacing
        self.corner_marker_size = corner_marker_size or grid_spacing
        self.membrane_size = membrane_size
        self.sensor_size = sensor_size
        self.sensor_definitions = sensor_definitions or [
            {"name": "R1", "x": -30.0, "y": 200.0, "rotation": 0.0},
            {"name": "R2", "x": 45.0, "y": 182.0, "rotation": 90.0},
            {"name": "R3", "x": -200.0, "y": 30.0, "rotation": 90.0},
            {"name": "R4", "x": -182.0, "y": -45.0, "rotation": 0.0},
            {"name": "R5", "x": 182.0, "y": 45.0, "rotation": 0.0},
            {"name": "R6", "x": 200.0, "y": -30.0, "rotation": 90.0},
            {"name": "R7", "x": -45.0, "y": -182.0, "rotation": 90.0},
            {"name": "R8", "x": 30.0, "y": -200.0, "rotation": 0.0},
        ]
        self.on_point_selected = on_point_selected
        self.figure = Figure(figsize=(5, 4)) 
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axis = self.figure.add_subplot(111)
        self.test_points = self._generate_test_points()
        self.saved_points: set[tuple[float, float]] = set()
        self.current_index = 0 if self.test_points else None
        self.manual_selected_point: tuple[float, float] | None = None
        self._build_layout()
        self._draw_grid()
        self.canvas.mpl_connect("button_press_event", self._handle_click)

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def _membrane_vertices(self) -> list[tuple[float, float]]:
        half_size = self.membrane_size / 2.0
        return [
            (0.0, half_size),
            (half_size, 0.0),
            (0.0, -half_size),
            (-half_size, 0.0),
        ]

    def _eyelet_positions(self) -> list[tuple[float, float]]:
        ring_scale = 0.9
        return [(corner_x * ring_scale, corner_y * ring_scale) for corner_x, corner_y in self._membrane_vertices()]

    def _point_within_eyelet_bounds(self, point: tuple[float, float]) -> bool:
        x_value, y_value = float(point[0]), float(point[1])
        eyelet_positions = self._eyelet_positions()
        diamond_half_extent = max(abs(x) + abs(y) for x, y in eyelet_positions)
        return abs(x_value) + abs(y_value) <= diamond_half_extent + 1e-9

    def _draw_corner_markers(self):
        profile_size = 30.0
        profile_offset = 20.0
        corners = [
            ("top", 0.0, (self.membrane_size / 2.0) + profile_offset, 1.0),
            ("bottom", 0.0, -(self.membrane_size / 2.0) - profile_offset, 1.0),
            ("left", -(self.membrane_size / 2.0) - profile_offset, 0.0, 0.3),
            ("right", (self.membrane_size / 2.0) + profile_offset, 0.0, 0.3),
        ]
        for _, center_x, center_y, alpha in corners:
            marker = Rectangle(
                (center_x - (profile_size / 2.0), center_y - (profile_size / 2.0)),
                profile_size,
                profile_size,
                facecolor="black",
                edgecolor="black",
                alpha=alpha,
                linewidth=1.0,
                clip_on=False,
                zorder=2.5,
            )
            self.axis.add_patch(marker)

    def _draw_membrane(self):
        membrane = Polygon(
            self._membrane_vertices(),
            closed=True,
            facecolor="#4a90e2",
            edgecolor="#1f4f82",
            alpha=0.18,
            linewidth=1.8,
            zorder=0,
        )
        self.axis.add_patch(membrane)
        return membrane

    def _draw_membrane_grid(self, membrane_patch):
        half_size = self.membrane_size / 2.0
        sample_x_positions = sorted({point[0] for point in self.test_points})
        sample_y_positions = sorted({point[1] for point in self.test_points})

        for x_value in sample_x_positions:
            line = self.axis.plot(
                [x_value, x_value],
                [-half_size, half_size],
                color="#1f4f82",
                linewidth=0.9,
                alpha=0.7,
                zorder=0.8,
            )[0]
            line.set_clip_path(membrane_patch)

        for y_value in sample_y_positions:
            line = self.axis.plot(
                [-half_size, half_size],
                [y_value, y_value],
                color="#1f4f82",
                linewidth=0.9,
                alpha=0.7,
                zorder=0.8,
            )[0]
            line.set_clip_path(membrane_patch)

        self._draw_sample_axes(membrane_patch)

    def _draw_sample_axes(self, membrane_patch):
        if not self.test_points:
            return

        vertical_points = [point for point in self.test_points if math.isclose(point[0], 0.0, abs_tol=1e-9)]

        if vertical_points:
            min_y = min(point[1] for point in vertical_points)
            max_y = max(point[1] for point in vertical_points)
            line = self.axis.plot(
                [0.0, 0.0],
                [min_y, max_y],
                color="black",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
                alpha=0.8,
                zorder=1.2,
            )[0]
            line.set_clip_path(membrane_patch)

    def _draw_eyelets(self):
        ring_radius = 7.0
        for corner_x, corner_y in self._eyelet_positions():
            eyelet = Circle(
                (corner_x, corner_y),
                ring_radius,
                facecolor="white",
                edgecolor="#203548",
                linewidth=1.8,
                zorder=3.5,
            )
            self.axis.add_patch(eyelet)

    def _draw_sensors(self):
        sensor_width, sensor_height = self.sensor_size
        half_width = sensor_width / 2.0
        half_height = sensor_height / 2.0
        for sensor_data in self.sensor_definitions:
            center_x = float(sensor_data["x"])
            center_y = float(sensor_data["y"])
            rotation = float(sensor_data.get("rotation", 0.0))
            name = str(sensor_data.get("name", "R?"))

            sensor = Rectangle(
                (center_x - half_width, center_y - half_height),
                sensor_width,
                sensor_height,
                facecolor="#f5a623",
                edgecolor="#7a4d00",
                alpha=0.75,
                linewidth=1.0,
                zorder=2,
            )
            sensor.set_transform(
                Affine2D().rotate_deg_around(center_x, center_y, rotation) + self.axis.transData
            )
            self.axis.add_patch(sensor)
            self.axis.text(
                center_x,
                center_y,
                name,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                zorder=3,
            )

    def _grid_ticks(self, limits):
        start = math.ceil(limits[0] / self.grid_spacing) * self.grid_spacing
        stop = math.floor(limits[1] / self.grid_spacing) * self.grid_spacing
        if start > stop:
            return np.array([0.0])
        ticks = np.arange(start, stop + (self.grid_spacing * 0.5), self.grid_spacing)
        if limits[0] <= 0 <= limits[1] and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        return ticks

    def _generate_test_points(self):
        half_size = self.membrane_size / 2.0
        x_values = self._grid_ticks((-half_size, half_size))
        y_values = self._grid_ticks((-half_size, half_size))
        points = []
        for y_value in sorted(y_values, reverse=True):
            for x_value in sorted(x_values):
                point = (float(x_value), float(y_value))
                if abs(point[0]) + abs(point[1]) <= half_size + 1e-9 and self._point_within_eyelet_bounds(point):
                    points.append(point)
        return points

    def _draw_test_points(self):
        if not self.test_points:
            return
        saved = np.array([point for point in self.test_points if point in self.saved_points], dtype=float)
        pending = np.array([point for point in self.test_points if point not in self.saved_points], dtype=float)

        if len(pending) > 0:
            self.axis.scatter(pending[:, 0], pending[:, 1], color="red", s=34, zorder=3)
        if len(saved) > 0:
            self.axis.scatter(saved[:, 0], saved[:, 1], color="green", s=34, zorder=4)

        selected_point = self.get_selected_point()
        if selected_point is not None:
            selected_is_saved = selected_point in self.saved_points
            self.axis.scatter(
                [selected_point[0]],
                [selected_point[1]],
                color="green" if selected_is_saved else "red",
                edgecolors="black",
                linewidths=1.2,
                s=90,
                zorder=5,
            )
        else:
            active_point = self.get_active_point()
            if active_point is not None:
                self.axis.scatter(
                    [active_point[0]],
                    [active_point[1]],
                    color="red",
                    edgecolors="black",
                    linewidths=1.2,
                    s=90,
                    zorder=5,
                )

    def _add_legend(self):
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markeredgecolor="none", markeredgewidth=0, markersize=7, label="Offenes Sample"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markeredgecolor="none", markeredgewidth=0, markersize=7, label="Gespeichertes Sample"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markeredgecolor="black", markeredgewidth=1.2, markersize=9, label="Aktuelles Sample"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#4a90e2", markeredgecolor="#1f4f82", markeredgewidth=1.0, alpha=0.35, markersize=10, label="Membran"),
            Patch(facecolor="#f5a623", edgecolor="#7a4d00", alpha=0.75, label="Sensor (R1-R8)"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markeredgecolor="black", markeredgewidth=1.0, markersize=10, label="Aluprofil Top Level"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markeredgecolor="black", markeredgewidth=1.0, alpha=0.3, markersize=10, label="Aluprofil Low Level"),
        ]
        self.axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95, borderaxespad=0.0)

    def _configure_axes(self):
        tick_step = self.grid_spacing
        x_ticks = np.arange(
            math.ceil(self.x_limits[0] / tick_step) * tick_step,
            self.x_limits[1] + (tick_step * 0.5),
            tick_step,
        )
        y_ticks = np.arange(
            math.ceil(self.y_limits[0] / tick_step) * tick_step,
            self.y_limits[1] + (tick_step * 0.5),
            tick_step,
        )
        self.axis.set_xticks(x_ticks)
        self.axis.set_yticks(y_ticks)
        self.axis.tick_params(axis="both", labelsize=8)
        self.axis.set_xlabel("X [mm]")
        self.axis.set_ylabel("Y [mm]")
        self.axis.spines["left"].set_visible(True)
        self.axis.spines["bottom"].set_visible(True)
        self.axis.spines["left"].set_color("#54606e")
        self.axis.spines["bottom"].set_color("#54606e")
        self.axis.spines["left"].set_linewidth(1.0)
        self.axis.spines["bottom"].set_linewidth(1.0)
        self.axis.spines["top"].set_visible(False)
        self.axis.spines["right"].set_visible(False)

    def _draw_grid(self):
        self.axis.clear()
        self.figure.subplots_adjust(right=0.78)
        self.axis.set_title("XY Draufsicht")
        self.axis.set_xlim(*self.x_limits)
        self.axis.set_ylim(*self.y_limits)
        self._configure_axes()

        membrane_patch = self._draw_membrane()
        self._draw_membrane_grid(membrane_patch)
        self._draw_sensors()
        self._draw_test_points()
        self.axis.set_aspect("equal", adjustable="box")
        self._draw_corner_markers()
        self._draw_eyelets()
        self._add_legend()
        self.canvas.draw_idle()

    def get_current_point(self) -> tuple[float, float] | None:
        if self.current_index is None or self.current_index >= len(self.test_points):
            return None
        return self.test_points[self.current_index]

    def get_selected_point(self) -> tuple[float, float] | None:
        return self.manual_selected_point

    def get_active_point(self) -> tuple[float, float] | None:
        if self.manual_selected_point is not None and self.manual_selected_point not in self.saved_points:
            return self.manual_selected_point
        return self.get_current_point()

    def is_point_saved(self, point: tuple[float, float] | None) -> bool:
        if point is None:
            return False
        normalized_point = (float(point[0]), float(point[1]))
        return normalized_point in self.saved_points

    def mark_point_saved(self, point: tuple[float, float]):
        normalized_point = (float(point[0]), float(point[1]))
        self.saved_points.add(normalized_point)
        if self.manual_selected_point == normalized_point:
            self.manual_selected_point = None
        self._advance_to_next_point()
        self._draw_grid()
        self._notify_point_selected()

    def reset_samples(self):
        self.saved_points.clear()
        self.current_index = 0 if self.test_points else None
        self.manual_selected_point = None
        self._draw_grid()
        self._notify_point_selected()

    def reset_saved_point(self, point: tuple[float, float]):
        normalized_point = (float(point[0]), float(point[1]))
        if normalized_point in self.saved_points:
            self.saved_points.remove(normalized_point)
        self.manual_selected_point = normalized_point
        self._advance_to_next_point()
        self._draw_grid()
        self._notify_point_selected()

    def load_saved_points(self, points: list[tuple[float, float]]):
        self.saved_points = {
            (float(point[0]), float(point[1]))
            for point in points
            if (float(point[0]), float(point[1])) in self.test_points
        }
        self.manual_selected_point = None
        self._advance_to_next_point()
        self._draw_grid()
        self._notify_point_selected()

    def _advance_to_next_point(self):
        for index, point in enumerate(self.test_points):
            if point not in self.saved_points:
                self.current_index = index
                return
        self.current_index = None

    def has_remaining_points(self) -> bool:
        return self.get_current_point() is not None

    def total_sample_count(self) -> int:
        return len(self.test_points)

    def remaining_sample_count(self) -> int:
        return len(self.test_points) - len(self.saved_points)

    def _all_points(self):
        return list(self.test_points)

    def _pending_points(self):
        return [point for point in self.test_points if point not in self.saved_points]

    def _handle_click(self, event):
        if event.inaxes != self.axis or event.xdata is None or event.ydata is None:
            return
        points = self._all_points()
        if not points:
            return

        click = np.array([float(event.xdata), float(event.ydata)])
        distances = [np.linalg.norm(np.array(point) - click) for point in points]
        nearest_point = points[int(np.argmin(distances))]
        self.manual_selected_point = nearest_point
        self._draw_grid()
        self._notify_point_selected()

    def _notify_point_selected(self):
        if self.on_point_selected is None:
            return
        selected_point = self.get_selected_point()
        if selected_point is not None:
            self.on_point_selected(selected_point, self.is_point_saved(selected_point))
            return
        active_point = self.get_active_point()
        self.on_point_selected(active_point, False)


class CalibrationStatusPanel(QWidget):
    def __init__(self, baseline_values, tolerance: float, parent=None):
        super().__init__(parent)
        self.baseline_values = np.asarray(baseline_values, dtype=float)
        self.tolerance = float(tolerance)
        self.live_values = np.full(len(self.baseline_values), np.nan, dtype=float)
        self.calibration_active = False
        self._rows: list[tuple[QLabel, QLabel, QLabel, QLabel]] = []
        self._build_layout()
        self._refresh_display()

    def _build_layout(self):
        layout = QVBoxLayout(self)
        title = QLabel("Kalibrierung / Basisspannung")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.summary_label = QLabel("Kalibrierung nicht gestartet")
        layout.addWidget(title)
        layout.addWidget(self.summary_label)

        grid = QGridLayout()
        grid.addWidget(QLabel("Sensor"), 0, 0)
        grid.addWidget(QLabel("Basis"), 0, 1)
        grid.addWidget(QLabel("Live"), 0, 2)
        grid.addWidget(QLabel("Status"), 0, 3)

        for index, baseline in enumerate(self.baseline_values, start=1):
            sensor_label = QLabel(f"R{index}")
            baseline_label = QLabel(f"{baseline:.3f}")
            live_label = QLabel("-")
            status_label = QLabel("Nicht kalibriert")
            status_label.setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 4px;")
            grid.addWidget(sensor_label, index, 0)
            grid.addWidget(baseline_label, index, 1)
            grid.addWidget(live_label, index, 2)
            grid.addWidget(status_label, index, 3)
            self._rows.append((sensor_label, baseline_label, live_label, status_label))

        layout.addLayout(grid)
        layout.addStretch(1)

    def start_calibration(self):
        self.calibration_active = True
        self._refresh_display()

    def set_live_values(self, values):
        self.live_values = np.asarray(values, dtype=float)
        self._refresh_display()

    def _refresh_display(self):
        if not self.calibration_active:
            self.summary_label.setText("Kalibrierung nicht gestartet")
        matched_count = 0

        for index, (_, _, live_label, status_label) in enumerate(self._rows):
            live_value = self.live_values[index] if index < len(self.live_values) else float("nan")
            baseline_value = self.baseline_values[index]
            if np.isnan(live_value):
                live_label.setText("-")
                status_label.setText("Keine Live-Daten")
                status_label.setStyleSheet("background-color: #e2e3e5; color: #41464b; padding: 4px;")
                continue

            live_label.setText(f"{live_value:.3f}")
            is_calibrated = self.calibration_active and abs(live_value - baseline_value) <= self.tolerance
            if is_calibrated:
                matched_count += 1
                status_label.setText("Kalibriert")
                status_label.setStyleSheet("background-color: #d4edda; color: #155724; padding: 4px;")
            elif self.calibration_active:
                status_label.setText("Noch nicht kalibriert")
                status_label.setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 4px;")
            else:
                status_label.setText("Bereit")
                status_label.setStyleSheet("background-color: #e2e3e5; color: #41464b; padding: 4px;")

        if self.calibration_active:
            self.summary_label.setText(
                f"Kalibrierung aktiv: {matched_count}/{len(self.baseline_values)} Sensoren im Toleranzbereich (+/- {self.tolerance:.3f})"
            )

    def all_calibrated(self) -> bool:
        if not self.calibration_active or len(self.live_values) != len(self.baseline_values):
            return False
        return bool(np.all(np.abs(self.live_values - self.baseline_values) <= self.tolerance))
