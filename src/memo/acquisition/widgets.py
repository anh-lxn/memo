from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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


class XYGridPlot(QWidget):
    def __init__(
        self,
        x_limits,
        y_limits,
        grid_spacing: float,
        corner_marker_size: float | None = None,
        membrane_size: float = 350.0,
        max_abs_coordinate: float | None = None,
        sensor_size: tuple[float, float] = (18.0, 10.0),
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
        self.max_abs_coordinate = max_abs_coordinate
        self.sensor_size = sensor_size
        self.sensor_definitions = sensor_definitions or [
            {"name": "R1", "x": -110.0, "y": 110.0, "rotation": 0.0},
            {"name": "R2", "x": 0.0, "y": 120.0, "rotation": 0.0},
            {"name": "R3", "x": 110.0, "y": 110.0, "rotation": 0.0},
            {"name": "R4", "x": -120.0, "y": 0.0, "rotation": 90.0},
            {"name": "R5", "x": 120.0, "y": 0.0, "rotation": 90.0},
            {"name": "R6", "x": -110.0, "y": -110.0, "rotation": 0.0},
            {"name": "R7", "x": 0.0, "y": -120.0, "rotation": 0.0},
            {"name": "R8", "x": 110.0, "y": -110.0, "rotation": 0.0},
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

    @staticmethod
    def _rotate_point_ccw(point: tuple[float, float]) -> tuple[float, float]:
        x_value, y_value = float(point[0]), float(point[1])
        return (-y_value, x_value)

    def _draw_corner_markers(self):
        half_size = self.corner_marker_size / 2.0
        corners = [
            (self.x_limits[0], self.y_limits[1], 1.0),
            (self.x_limits[1], self.y_limits[0], 1.0),
            (self.x_limits[1], self.y_limits[1], 0.3),
            (self.x_limits[0], self.y_limits[0], 0.3),
        ]
        for center_x, center_y, alpha in corners:
            rotated_x, rotated_y = self._rotate_point_ccw((center_x, center_y))
            marker = Rectangle(
                (rotated_x - half_size, rotated_y - half_size),
                self.corner_marker_size,
                self.corner_marker_size,
                facecolor="black",
                edgecolor="black",
                alpha=alpha,
                linewidth=1.0,
                clip_on=False,
                zorder=2,
            )
            self.axis.add_patch(marker)

    def _draw_membrane(self):
        half_size = self.membrane_size / 2.0
        membrane = Rectangle(
            (-half_size, -half_size),
            self.membrane_size,
            self.membrane_size,
            facecolor="#4a90e2",
            edgecolor="#1f4f82",
            alpha=0.18,
            linewidth=1.5,
            zorder=0,
        )
        self.axis.add_patch(membrane)

    def _draw_sensors(self):
        sensor_width, sensor_height = self.sensor_size
        half_width = sensor_width / 2.0
        half_height = sensor_height / 2.0
        for sensor_data in self.sensor_definitions:
            center_x, center_y = self._rotate_point_ccw((float(sensor_data["x"]), float(sensor_data["y"])))
            rotation = float(sensor_data.get("rotation", 0.0)) + 90.0
            name = str(sensor_data.get("name", "R?"))

            sensor = Rectangle(
                (center_x - half_width, center_y - half_height),
                sensor_width,
                sensor_height,
                facecolor="#f5a623",
                edgecolor="#7a4d00",
                alpha=0.75,
                linewidth=1.0,
                zorder=1,
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
                fontsize=8,
                color="black",
                zorder=2,
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
                if self.max_abs_coordinate is not None and max(abs(point[0]), abs(point[1])) > self.max_abs_coordinate:
                    continue
                points.append(point)
        return points

    def _draw_test_points(self):
        if not self.test_points:
            return
        saved = np.array(
            [self._rotate_point_ccw(point) for point in self.test_points if point in self.saved_points],
            dtype=float,
        )
        pending = np.array(
            [self._rotate_point_ccw(point) for point in self.test_points if point not in self.saved_points],
            dtype=float,
        )

        if len(pending) > 0:
            self.axis.scatter(pending[:, 0], pending[:, 1], color="red", s=34, zorder=3)
        if len(saved) > 0:
            self.axis.scatter(saved[:, 0], saved[:, 1], color="green", s=34, zorder=4)

        selected_point = self.get_selected_point()
        if selected_point is not None:
            selected_is_saved = selected_point in self.saved_points
            rotated_selected = self._rotate_point_ccw(selected_point)
            self.axis.scatter(
                [rotated_selected[0]],
                [rotated_selected[1]],
                color="green" if selected_is_saved else "red",
                edgecolors="black",
                linewidths=1.2,
                s=90,
                zorder=5,
            )
        else:
            active_point = self.get_active_point()
            if active_point is not None:
                rotated_active = self._rotate_point_ccw(active_point)
                self.axis.scatter(
                    [rotated_active[0]],
                    [rotated_active[1]],
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

    def _draw_grid(self):
        self.axis.clear()
        self.figure.subplots_adjust(right=0.78)
        self.axis.set_title("XY Draufsicht")
        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")
        self.axis.set_xlim(*self.x_limits)
        self.axis.set_ylim(*self.y_limits)
        self.axis.set_xticks(self._grid_ticks(self.x_limits))
        self.axis.set_yticks(self._grid_ticks(self.y_limits))
        self.axis.tick_params(axis="x", pad=12)
        self.axis.tick_params(axis="y", pad=12)
        self._draw_membrane()
        self._draw_sensors()
        self._draw_test_points()
        self.axis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        if self.x_limits[0] <= 0 <= self.x_limits[1]:
            self.axis.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=1)
        if self.y_limits[0] <= 0 <= self.y_limits[1]:
            self.axis.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=1)
        self.axis.set_aspect("equal", adjustable="box")
        self._draw_corner_markers()
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
        distances = [np.linalg.norm(np.array(self._rotate_point_ccw(point)) - click) for point in points]
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
