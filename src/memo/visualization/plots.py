"""Compatibility exports for acquisition UI widgets."""

from collections import deque
import math
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget, QVBoxLayout

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


class LiveSensorPlot(QWidget):
    def __init__(self, baseline_value: float = 2.5, threshold: float = 0.1, parent=None):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")
        self.baseline_value = float(baseline_value)
        self.threshold = float(threshold)
        self._latest_values = np.zeros(8, dtype=float)
        self._visible_sensor_indices = list(range(8))
        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("left", "Verstärkte Spannung in V")
        self.plot_widget.setLabel("bottom", "Sensor")
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("#344054"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#344054"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("#98a2b3"))
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("#98a2b3"))
        self.plot_widget.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.plot_widget.setYRange(0.0, 4.0, padding=0.02)
        self.plot_widget.setXRange(0.5, 8.5, padding=0.0)
        self.plot_widget.getPlotItem().getViewBox().setLimits(xMin=0.5, xMax=8.5, yMin=0.0)
        ticks = [(index, str(index)) for index in range(1, 9)]
        self.plot_widget.getAxis("bottom").setTicks([ticks])
        self._bar_item = pg.BarGraphItem(
            x=np.arange(1, 9, dtype=float),
            height=np.zeros(8, dtype=float),
            width=0.7,
            brush=pg.mkBrush("#1f77b4"),
            pen=pg.mkPen("#1b5f91", width=1.0),
        )
        self._threshold_band = pg.LinearRegionItem(
            values=(self.baseline_value - self.threshold, self.baseline_value + self.threshold),
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(245, 158, 11, 70),
            pen=pg.mkPen(None),
        )
        self._baseline_line = pg.InfiniteLine(
            pos=self.baseline_value,
            angle=0,
            movable=False,
            pen=pg.mkPen("#f59e0b", width=1.5, style=Qt.DashLine),
        )
        self._baseline_line.setOpacity(0.7)
        self.plot_widget.addItem(self._threshold_band)
        self.plot_widget.addItem(self._bar_item)
        self.plot_widget.addItem(self._baseline_line)
        self._title_label = QLabel("Live Plot der 8 Sensorwerte")
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("color: #1f2a37; font-size: 14pt; font-weight: 600;")
        self._sensor_filter_panel = QWidget()
        self._sensor_filter_panel.setObjectName("sensorFilterPanel")
        self._sensor_checkboxes: list[QCheckBox] = []
        self._build_layout()
        self.update_values(np.zeros(8, dtype=float))

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._title_label)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        content_layout.addWidget(self.plot_widget, stretch=1)

        filter_layout = QVBoxLayout(self._sensor_filter_panel)
        filter_layout.setContentsMargins(10, 10, 10, 10)
        filter_layout.setSpacing(0)
        for index in range(8):
            checkbox = QCheckBox(f"R{index + 1}")
            checkbox.setChecked(True)
            checkbox.setMinimumHeight(36)
            checkbox.toggled.connect(self._update_visible_sensors)
            self._sensor_checkboxes.append(checkbox)
            filter_layout.addWidget(checkbox)
            if index < 7:
                filter_layout.addStretch(1)
        self._sensor_filter_panel.setStyleSheet("""
            QWidget#sensorFilterPanel {
                background-color: #ffffff;
                border: 1px solid #d9e0e6;
                border-radius: 10px;
            }
            QCheckBox {
                color: #1f2a37;
                spacing: 12px;
                font-size: 14px;
                padding-top: 4px;
                padding-bottom: 4px;
            }
            QCheckBox::indicator {
                width: 32px;
                height: 32px;
                border: 2px solid #94a3b8;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:unchecked:hover {
                border: 2px solid #3b82f6;
                background-color: #eff6ff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #2563eb;
                border-radius: 6px;
                background-color: #2563eb;
                image: none;
            }
            QCheckBox::indicator:checked:hover {
                border: 2px solid #1d4ed8;
                background-color: #1d4ed8;
            }
        """)

        content_layout.addWidget(self._sensor_filter_panel)
        layout.addLayout(content_layout)

    def _update_visible_sensors(self):
        visible_indices = [index for index, checkbox in enumerate(self._sensor_checkboxes) if checkbox.isChecked()]
        self._visible_sensor_indices = visible_indices
        self._redraw_plot()

    def set_visible_sensor_indices(self, indices: list[int] | tuple[int, ...], lock_selection: bool = False):
        normalized_indices = sorted({int(index) for index in indices if 0 <= int(index) < len(self._sensor_checkboxes)})
        for index, checkbox in enumerate(self._sensor_checkboxes):
            checkbox.blockSignals(True)
            checkbox.setChecked(index in normalized_indices)
            checkbox.setEnabled(not lock_selection or index in normalized_indices)
            checkbox.blockSignals(False)
        self._visible_sensor_indices = normalized_indices
        self._redraw_plot()

    def reset_sensor_filter(self):
        self.set_visible_sensor_indices(tuple(range(len(self._sensor_checkboxes))), lock_selection=False)

    def _redraw_plot(self):
        if self._visible_sensor_indices:
            x_values = np.arange(1, len(self._visible_sensor_indices) + 1, dtype=float)
            heights = np.clip(self._latest_values[self._visible_sensor_indices], 0.0, None)
            ticks = [
                (position, f"R{sensor_index + 1}")
                for position, sensor_index in enumerate(self._visible_sensor_indices, start=1)
            ]
            x_min = 0.5
            x_max = len(self._visible_sensor_indices) + 0.5
        else:
            x_values = np.array([], dtype=float)
            heights = np.array([], dtype=float)
            ticks = [(index, f"R{index}") for index in range(1, 9)]
            x_min = 0.5
            x_max = 8.5

        y_max = max(
            4.0,
            self.baseline_value + self.threshold + 0.25,
            float(np.max(heights)) + 0.25 if heights.size else 0.0,
        )
        self.plot_widget.setYRange(0.0, y_max, padding=0.02)
        self.plot_widget.setXRange(x_min, x_max, padding=0.0)
        self.plot_widget.getPlotItem().getViewBox().setLimits(xMin=0.5, xMax=max(8.5, x_max), yMin=0.0)
        self.plot_widget.getAxis("bottom").setTicks([ticks])
        self._threshold_band.setRegion((self.baseline_value - self.threshold, self.baseline_value + self.threshold))
        self._baseline_line.setValue(self.baseline_value)
        self._bar_item.setOpts(x=x_values, height=heights)

    def update_values(self, values):
        values = np.asarray(values, dtype=float)
        self._latest_values = np.zeros(8, dtype=float)
        if values.size:
            self._latest_values[: min(8, values.size)] = values[:8]
        self._redraw_plot()

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)
        self._redraw_plot()

    def set_baseline_value(self, baseline_value: float):
        self.baseline_value = float(baseline_value)
        self._redraw_plot()


class LiveForcePlot(QWidget):
    def __init__(
        self,
        history_seconds: float = 5.0,
        target_force: float = 0.0,
        threshold: float = 0.5,
        hide_x_tick_labels: bool = False,
        fixed_grid: bool = False,
        center_latest_value: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is not installed.")
        self.history_seconds = history_seconds
        self.target_force = float(target_force)
        self.threshold = float(threshold)
        self.hide_x_tick_labels = hide_x_tick_labels
        self.fixed_grid = fixed_grid
        self.center_latest_value = center_latest_value
        self.timestamps: deque[datetime] = deque()
        self.values: deque[float] = deque()
        self.current_time: datetime | None = None
        self.stream_active = False
        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("left", "Kraft [N]")
        self.plot_widget.setLabel("bottom", "Zeit seit Start [s]")
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("#344054"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#344054"))
        if self.hide_x_tick_labels:
            self.plot_widget.getAxis("bottom").setStyle(showValues=False, tickLength=0)
            self.plot_widget.setLabel("bottom", "")
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._band_item = pg.LinearRegionItem(
            values=(self.target_force - self.threshold, self.target_force + self.threshold),
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(15, 139, 109, 28),
            pen=pg.mkPen(None),
        )
        self._target_line = pg.InfiniteLine(
            pos=self.target_force,
            angle=0,
            movable=False,
            pen=pg.mkPen("#0f8b6d", width=1.5, style=Qt.DashLine),
        )
        self._curve = self.plot_widget.plot([], [], pen=pg.mkPen("#5f6b7a", width=2))
        self._marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#b42318"), pen=pg.mkPen("#b42318"))
        self.plot_widget.addItem(self._band_item)
        self.plot_widget.addItem(self._target_line)
        self.plot_widget.addItem(self._marker)
        self._build_layout()
        self._update_static_ranges()
        self._redraw()

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

    def _update_static_ranges(self):
        min_force = min(0.0, self.target_force - max(2.0, 2.0 * self.threshold))
        max_force = max(25.0, self.target_force + max(2.0, 2.0 * self.threshold))
        self.plot_widget.setYRange(min_force, max_force, padding=0.02)
        self._band_item.setRegion((self.target_force - self.threshold, self.target_force + self.threshold))
        self._target_line.setValue(self.target_force)

    def append_value(self, timestamp: datetime, value: float):
        self.append_values([(timestamp, value)])

    def append_values(self, samples: list[tuple[datetime, float]]):
        if not samples:
            return

        last_timestamp = samples[-1][0]
        self.current_time = last_timestamp

        for timestamp, value in samples:
            self.timestamps.append(timestamp)
            self.values.append(float(value))

        cutoff = last_timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.values.popleft()
        self._redraw()

    def advance_time(self, timestamp: datetime):
        self.current_time = timestamp
        cutoff = timestamp.timestamp() - self.history_seconds
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.values.popleft()
        self._redraw()

    def set_stream_active(self, active: bool):
        active = bool(active)
        if self.stream_active == active:
            return
        self.stream_active = active
        self._redraw()

    def set_target_force(self, target_force: float):
        target_force = float(target_force)
        if self.target_force == target_force:
            return
        self.target_force = target_force
        self._update_static_ranges()
        self._redraw()

    def set_threshold(self, threshold: float):
        threshold = float(threshold)
        if self.threshold == threshold:
            return
        self.threshold = threshold
        self._update_static_ranges()
        self._redraw()

    def _redraw(self):
        self._update_static_ranges()
        reference_time = self.current_time.timestamp() if self.current_time is not None else None

        if self.timestamps and reference_time is not None:
            x_values = np.array([timestamp.timestamp() - reference_time for timestamp in self.timestamps], dtype=float)
            if not self.center_latest_value:
                x_values += self.history_seconds
            y_values = np.array(self.values, dtype=float)
            latest_value = y_values[-1]
            within_threshold = abs(latest_value - self.target_force) <= self.threshold
            latest_color = "#0f8b6d" if self.stream_active and within_threshold else "#b42318"
            trace_color = "#5f6b7a" if self.stream_active else "#98a2ad"
            self._curve.setPen(pg.mkPen(trace_color, width=2))
            self._curve.setData(x_values, y_values)
            marker_x = 0.0 if self.center_latest_value else self.history_seconds
            self._marker.setBrush(pg.mkBrush(latest_color))
            self._marker.setPen(pg.mkPen(latest_color))
            self._marker.setData([marker_x], [latest_value])

            if self.center_latest_value:
                half_window = self.history_seconds / 2.0
                self.plot_widget.setXRange(-half_window, half_window, padding=0.0)
            else:
                self.plot_widget.setXRange(0.0, self.history_seconds, padding=0.0)
        else:
            self._curve.setData([], [])
            self._marker.setData([], [])
            if self.center_latest_value:
                half_window = self.history_seconds / 2.0
                self.plot_widget.setXRange(-half_window, half_window, padding=0.0)
            else:
                self.plot_widget.setXRange(0.0, self.history_seconds, padding=0.0)


class XYGridPlot(QWidget):
    def __init__(
        self,
        x_limits,
        y_limits,
        grid_spacing: float,
        corner_marker_size: float | None = None,
        membrane_size: float = 450.0 * math.sqrt(2.0),
        max_abs_coordinate: float | None = None,
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
        self.max_abs_coordinate = max_abs_coordinate
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

    @staticmethod
    def _rotate_point_ccw(point: tuple[float, float]) -> tuple[float, float]:
        x_value, y_value = float(point[0]), float(point[1])
        return (-y_value, x_value)

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
            center_x, center_y = self._rotate_point_ccw((center_x, center_y))
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

    def _draw_monitor(self):
        monitor_width = 150.0
        monitor_height = 45.0
        monitor_offset = 72.0
        center_x = 0.0
        center_y = (self.membrane_size / 2.0) + monitor_offset
        center_x, center_y = self._rotate_point_ccw((center_x, center_y))
        monitor = Rectangle(
            (center_x - (monitor_width / 2.0), center_y - (monitor_height / 2.0)),
            monitor_width,
            monitor_height,
            facecolor="#c9d3df",
            edgecolor="#4b5563",
            linewidth=1.4,
            alpha=0.95,
            clip_on=False,
            zorder=2.4,
        )
        self.axis.add_patch(monitor)

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
            start_x, start_y = self._rotate_point_ccw((x_value, -half_size))
            end_x, end_y = self._rotate_point_ccw((x_value, half_size))
            line = self.axis.plot(
                [start_x, end_x],
                [start_y, end_y],
                color="#1f4f82",
                linewidth=0.9,
                alpha=0.7,
                zorder=0.8,
            )[0]
            line.set_clip_path(membrane_patch)

        for y_value in sample_y_positions:
            start_x, start_y = self._rotate_point_ccw((-half_size, y_value))
            end_x, end_y = self._rotate_point_ccw((half_size, y_value))
            line = self.axis.plot(
                [start_x, end_x],
                [start_y, end_y],
                color="#1f4f82",
                linewidth=0.9,
                alpha=0.7,
                zorder=0.8,
            )[0]
            line.set_clip_path(membrane_patch)

        self._draw_sample_axes(membrane_patch)

    def _draw_sample_axes(self, membrane_patch):
        profile_offset = 20.0
        start_x, start_y = self._rotate_point_ccw((0.0, -(self.membrane_size / 2.0) - profile_offset))
        end_x, end_y = self._rotate_point_ccw((0.0, (self.membrane_size / 2.0) + profile_offset))
        line = self.axis.plot(
            [start_x, end_x],
            [start_y, end_y],
            color="black",
            linewidth=1.6,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            zorder=1.2,
        )[0]
        line.set_clip_on(False)

    def _draw_eyelets(self):
        ring_radius = 7.0
        for corner_x, corner_y in self._eyelet_positions():
            corner_x, corner_y = self._rotate_point_ccw((corner_x, corner_y))
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
                if self.max_abs_coordinate is not None and max(abs(point[0]), abs(point[1])) > self.max_abs_coordinate:
                    continue
                if abs(point[0]) + abs(point[1]) <= half_size + 1e-9 and self._point_within_eyelet_bounds(point):
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
            selected_x, selected_y = self._rotate_point_ccw(selected_point)
            self.axis.scatter(
                [selected_x],
                [selected_y],
                color="green" if selected_is_saved else "red",
                edgecolors="black",
                linewidths=1.2,
                s=90,
                zorder=5,
            )
        else:
            active_point = self.get_active_point()
            if active_point is not None:
                active_x, active_y = self._rotate_point_ccw(active_point)
                self.axis.scatter(
                    [active_x],
                    [active_y],
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
            Patch(facecolor="#c9d3df", edgecolor="#4b5563", alpha=0.95, label="Monitor"),
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
        self.axis.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
        self.axis.set_xlabel("")
        self.axis.set_ylabel("")
        self.axis.spines["left"].set_visible(False)
        self.axis.spines["bottom"].set_visible(False)
        self.axis.spines["left"].set_color("#54606e")
        self.axis.spines["bottom"].set_color("#54606e")
        self.axis.spines["left"].set_linewidth(1.0)
        self.axis.spines["bottom"].set_linewidth(1.0)
        self.axis.spines["top"].set_visible(False)
        self.axis.spines["right"].set_visible(False)

    def _draw_grid(self):
        self.axis.clear()
        self.figure.subplots_adjust(right=0.78)
        self.axis.set_title("XY Draufsicht", pad=25)                                                                                                                                                                                                                                                                                
        self.axis.set_xlim(*self.x_limits)
        self.axis.set_ylim(*self.y_limits)
        self._configure_axes()

        membrane_patch = self._draw_membrane()
        self._draw_membrane_grid(membrane_patch)
        self._draw_sensors()
        self._draw_monitor()
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("Kalibrierung / Basisspannung")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.summary_label = QLabel("Kalibrierung nicht gestartet")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(title)
        layout.addWidget(self.summary_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("Sensor"), 0, 0)
        grid.addWidget(QLabel("Basis"), 0, 1)
        grid.addWidget(QLabel("Live"), 0, 2)
        grid.addWidget(QLabel("Status"), 0, 3)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)
        grid.setColumnMinimumWidth(0, 46)
        grid.setColumnMinimumWidth(1, 62)
        grid.setColumnMinimumWidth(2, 62)
        grid.setColumnMinimumWidth(3, 110)

        for index, baseline in enumerate(self.baseline_values, start=1):
            sensor_label = QLabel(f"R{index}")
            baseline_label = QLabel(f"{baseline:.3f}")
            live_label = QLabel("-")
            status_label = QLabel("Nicht kalibriert")
            sensor_label.setStyleSheet("font-size: 12px;")
            baseline_label.setStyleSheet("font-size: 12px;")
            live_label.setStyleSheet("font-size: 12px;")
            status_label.setMinimumWidth(110)
            status_label.setStyleSheet(
                "background-color: #f8d7da; color: #721c24; padding: 3px 6px; font-size: 11px;"
            )
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

    def set_baseline_values(self, baseline_values):
        self.baseline_values = np.asarray(baseline_values, dtype=float)
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
                status_label.setStyleSheet(
                    "background-color: #e2e3e5; color: #41464b; padding: 3px 6px; font-size: 11px;"
                )
                continue

            live_label.setText(f"{live_value:.3f}")
            is_calibrated = self.calibration_active and abs(live_value - baseline_value) <= self.tolerance
            if is_calibrated:
                matched_count += 1
                status_label.setText("Kalibriert")
                status_label.setStyleSheet(
                    "background-color: #d4edda; color: #155724; padding: 3px 6px; font-size: 11px;"
                )
            elif self.calibration_active:
                status_label.setText("Nicht kal.")
                status_label.setStyleSheet(
                    "background-color: #f8d7da; color: #721c24; padding: 3px 6px; font-size: 11px;"
                )
            else:
                status_label.setText("Bereit")
                status_label.setStyleSheet(
                    "background-color: #e2e3e5; color: #41464b; padding: 3px 6px; font-size: 11px;"
                )

        if self.calibration_active:
            self.summary_label.setText(
                f"Kalibrierung aktiv: {matched_count}/{len(self.baseline_values)} Sensoren im Toleranzbereich (+/- {self.tolerance:.3f})"
            )

    def all_calibrated(self) -> bool:
        if not self.calibration_active or len(self.live_values) != len(self.baseline_values):
            return False
        return bool(np.all(np.abs(self.live_values - self.baseline_values) <= self.tolerance))
