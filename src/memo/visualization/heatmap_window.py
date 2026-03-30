from __future__ import annotations

import argparse
import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


MEMBRANE_SIZE = 350.0
OFFSET = 30.0
X_LIMITS = (-MEMBRANE_SIZE / 2 - OFFSET, MEMBRANE_SIZE / 2 + OFFSET)
Y_LIMITS = (-MEMBRANE_SIZE / 2 - OFFSET, MEMBRANE_SIZE / 2 + OFFSET)
GRID_SPACING = 50.0
FORCE_RANGE = (0.0, 20.0)
UPDATE_MS = 140
ISO_UPDATE_EVERY = 3
DEMO_SEED = 7
HEATMAP_VMAX = 2.6
ENABLE_XY_INTERACTION = False


class XYHeatmapPlot(QWidget):
    def __init__(
        self,
        x_limits=X_LIMITS,
        y_limits=Y_LIMITS,
        membrane_size: float = MEMBRANE_SIZE,
        grid_spacing: float = GRID_SPACING,
        resolution: int = 160,
        parent=None,
    ):
        super().__init__(parent)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.membrane_size = membrane_size
        self.grid_spacing = grid_spacing
        self.resolution = resolution
        self.figure = Figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_axes([0.10, 0.10, 0.84, 0.84])

        self._x_values = np.linspace(self.x_limits[0], self.x_limits[1], self.resolution)
        self._y_values = np.linspace(self.y_limits[0], self.y_limits[1], self.resolution)
        self._xx, self._yy = np.meshgrid(self._x_values, self._y_values)
        self._half_size = self.membrane_size / 2.0
        self._inside_membrane = (
            (self._xx >= -self._half_size)
            & (self._xx <= self._half_size)
            & (self._yy >= -self._half_size)
            & (self._yy <= self._half_size)
        )

        self.on_point_selected = None
        self.interaction_enabled = ENABLE_XY_INTERACTION
        self._build_layout()
        self._initialize_plot()
        if self.interaction_enabled:
            self.canvas.mpl_connect('button_press_event', self._handle_click)
        self.set_heatmap(0.0, 0.0)

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _grid_ticks(self, limits):
        start = np.ceil(limits[0] / self.grid_spacing) * self.grid_spacing
        stop = np.floor(limits[1] / self.grid_spacing) * self.grid_spacing
        ticks = np.arange(start, stop + (self.grid_spacing * 0.5), self.grid_spacing)
        if limits[0] <= 0 <= limits[1] and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        return ticks

    def _build_field(self, center_x: float, center_y: float, intensity: float, sigma: float):
        distance_sq = (self._xx - center_x) ** 2 + (self._yy - center_y) ** 2
        field = intensity * np.exp(-distance_sq / (2.0 * sigma ** 2))
        return np.where(self._inside_membrane, field, np.nan)

    def _initialize_plot(self):
        empty_field = np.full((self.resolution, self.resolution), np.nan, dtype=float)
        self.image = self.axis.imshow(
            empty_field,
            origin='lower',
            extent=(self.x_limits[0], self.x_limits[1], self.y_limits[0], self.y_limits[1]),
            cmap='turbo',
            interpolation='bilinear',
            alpha=0.92,
            vmin=0.0,
            vmax=HEATMAP_VMAX,
            zorder=1,
        )

        self.axis.add_patch(
            Rectangle(
                (-self._half_size, -self._half_size),
                self.membrane_size,
                self.membrane_size,
                fill=False,
                edgecolor='#0f3d75',
                linewidth=2.0,
                zorder=3,
            )
        )

        self.axis.set_title('XY Heatmap / Draufsicht')
        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_xlim(*self.x_limits)
        self.axis.set_ylim(*self.y_limits)
        self.axis.set_xticks(self._grid_ticks(self.x_limits))
        self.axis.set_yticks(self._grid_ticks(self.y_limits))
        self.axis.tick_params(axis='x', pad=10)
        self.axis.tick_params(axis='y', pad=10)
        self.axis.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
        self.axis.axvline(0, color='black', linewidth=0.8, alpha=0.35, zorder=2)
        self.axis.axhline(0, color='black', linewidth=0.8, alpha=0.35, zorder=2)
        self.axis.set_aspect('equal', adjustable='box')

    def set_heatmap(self, center_x: float, center_y: float, intensity: float = 1.0, sigma: float = 35.0):
        sigma = max(float(sigma), 1.0)
        intensity = max(float(intensity), 0.0)
        field = self._build_field(float(center_x), float(center_y), intensity, sigma)
        self.image.set_data(field)
        self.canvas.draw_idle()

    def _handle_click(self, event):
        if not self.interaction_enabled:
            return
        if event.inaxes != self.axis or event.xdata is None or event.ydata is None:
            return

        x_value = float(np.clip(event.xdata, -self._half_size, self._half_size))
        y_value = float(np.clip(event.ydata, -self._half_size, self._half_size))

        if self.on_point_selected is not None:
            self.on_point_selected(x_value, y_value)


class IsometricMembranePlot(QWidget):
    def __init__(self, membrane_size: float = MEMBRANE_SIZE, resolution: int = 48, parent=None):
        super().__init__(parent)
        self.membrane_size = membrane_size
        self.resolution = resolution
        self.figure = Figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111, projection='3d')
        self._half_size = self.membrane_size / 2.0
        self._x_values = np.linspace(-self._half_size, self._half_size, self.resolution)
        self._y_values = np.linspace(-self._half_size, self._half_size, self.resolution)
        self._xx, self._yy = np.meshgrid(self._x_values, self._y_values)
        self._inside_membrane = (
            (self._xx >= -self._half_size)
            & (self._xx <= self._half_size)
            & (self._yy >= -self._half_size)
            & (self._yy <= self._half_size)
        )
        self._build_layout()
        self._initialize_plot()
        self.axis.disable_mouse_rotation()
        self.canvas.setFocusPolicy(0)
        self.update_heatmap(0.0, 0.0, 1.0, 35.0)

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _base_surface(self):
        # Symmetric saddle shape over a square membrane:
        # (+x, -y) and (-x, +y) are positive z,
        # (+x, +y) and (-x, -y) are negative z.
        x_norm = self._xx / self._half_size
        y_norm = self._yy / self._half_size
        z = -64.0 * x_norm * y_norm
        z *= 1.0 - 0.10 * (x_norm ** 2 + y_norm ** 2)
        z -= 4.0 * (x_norm ** 2 + y_norm ** 2)
        return z

    def _build_heat_field(self, center_x: float, center_y: float, intensity: float, sigma: float):
        sigma = max(float(sigma), 1.0)
        intensity = max(float(intensity), 0.0)
        distance_sq = (self._xx - center_x) ** 2 + (self._yy - center_y) ** 2
        field = intensity * np.exp(-distance_sq / (2.0 * sigma ** 2))
        return np.where(self._inside_membrane, field, np.nan)

    def _build_surface(self):
        base = self._base_surface()
        return np.where(self._inside_membrane, base, np.nan)

    def _field_to_facecolors(self, field):
        cell_values = np.nanmean(
            np.stack([
                field[:-1, :-1],
                field[1:, :-1],
                field[:-1, 1:],
                field[1:, 1:],
            ]),
            axis=0,
        )
        rgba = self.colormap(self.norm(np.nan_to_num(cell_values, nan=0.0)))
        rgba[np.isnan(cell_values)] = (0.0, 0.0, 0.0, 0.0)
        return rgba

    def _initialize_plot(self):
        surface = self._build_surface()
        self.colormap = colormaps['turbo']
        self.norm = Normalize(vmin=0.0, vmax=HEATMAP_VMAX)
        initial_field = self._build_heat_field(0.0, 0.0, 1.0, 35.0)
        self.surface = self.axis.plot_surface(
            self._xx,
            self._yy,
            surface,
            facecolors=self._field_to_facecolors(initial_field),
            linewidth=0.0,
            antialiased=False,
            shade=False,
            rcount=self.resolution,
            ccount=self.resolution,
        )
        self.axis.set_title('Membran / Isometrische Sicht')
        self.axis.set_xlabel('X')
        self.axis.set_ylabel('Y')
        self.axis.set_zlabel('Z')
        self.axis.view_init(elev=45, azim=-135)
        self.current_elev = 45
        self.current_azim = -135
        self.axis.set_xlim(-self._half_size, self._half_size)
        self.axis.set_ylim(-self._half_size, self._half_size)
        self.axis.set_zlim(-110.0, 80.0)
        self.axis.set_box_aspect((1.0, 1.0, 0.7))
        self.axis.xaxis.pane.set_alpha(0.0)
        self.axis.yaxis.pane.set_alpha(0.0)
        self.axis.zaxis.pane.set_alpha(0.0)

    def update_heatmap(self, center_x: float, center_y: float, intensity: float, sigma: float):
        field = self._build_heat_field(float(center_x), float(center_y), float(intensity), float(sigma))
        facecolors = self._field_to_facecolors(field).reshape(-1, 4)
        self.surface.set_facecolors(facecolors)
        self.surface.set_edgecolors(facecolors)
        self.axis.view_init(elev=self.current_elev, azim=self.current_azim)
        self.canvas.draw_idle()

    def set_view(self, elev: int, azim: int):
        self.current_elev = int(elev)
        self.current_azim = int(azim)
        self.axis.view_init(elev=self.current_elev, azim=self.current_azim)
        self.canvas.draw_idle()


class ForceBarPlot(QWidget):
    def __init__(self, force_limits=FORCE_RANGE, parent=None):
        super().__init__(parent)
        self.force_limits = force_limits
        self.figure = Figure(figsize=(3.0, 5.0))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self._build_layout()
        self._initialize_plot()
        self.set_force(0.0)

    def _build_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _initialize_plot(self):
        minimum, maximum = self.force_limits
        self.bar = self.axis.bar([0], [0.0], color='#0f8b6d', width=0.5)[0]
        self.force_text = self.axis.text(0, minimum + 0.4, '0.0 N', ha='center', va='bottom', fontsize=11, fontweight='bold')
        self.axis.set_xlim(-0.8, 0.8)
        self.axis.set_ylim(minimum, maximum)
        self.axis.set_xticks([])
        self.axis.set_yticks(np.arange(minimum, maximum + 0.1, 5.0))
        self.axis.set_ylabel('Kraft [N]')
        self.axis.set_title('Live Kraft')
        self.axis.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

    def set_force(self, force_value: float):
        minimum, maximum = self.force_limits
        force_value = min(max(float(force_value), minimum), maximum)
        self.bar.set_height(force_value)
        self.force_text.set_position((0, min(force_value + 0.4, maximum - 0.2)))
        self.force_text.set_text(f'{force_value:.1f} N')
        self.canvas.draw_idle()


class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MeMo Heatmap Visualisierung')
        self.resize(1850, 980)

        self.plot = XYHeatmapPlot()
        self.plot.on_point_selected = self._handle_xy_click
        self.iso_plot = IsometricMembranePlot()
        self.force_plot = ForceBarPlot()
        self.position_label = QLabel('-')
        self.force_label = QLabel('-')
        self.status_label = QLabel('Demo aktiv')
        self.toggle_button = QPushButton('Live Demo stoppen')
        self.elev_input = QSpinBox()
        self.azim_input = QSpinBox()

        self.demo_samples = self._build_demo_samples()
        self.demo_index = 0
        self.iso_frame_counter = 0
        self.timer = QTimer(self)
        self.timer.setInterval(UPDATE_MS)

        self._build_ui()
        self._apply_styles()
        self._connect_signals()
        self._step_demo()
        self.timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        layout.addWidget(self.plot, stretch=4)
        layout.addWidget(self.iso_plot, stretch=4)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setSpacing(14)
        side_layout.addWidget(self.force_plot, stretch=3)

        info_card = QWidget()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(10)
        info_layout.addWidget(self._label_pair('Position', self.position_label))
        info_layout.addWidget(self._label_pair('Kraft', self.force_label))
        info_layout.addWidget(self._label_pair('Status', self.status_label))
        info_layout.addWidget(self._build_view_controls())
        info_layout.addWidget(self.toggle_button)
        info_layout.addStretch(1)
        info_card.setObjectName('infoCard')

        side_layout.addWidget(info_card, stretch=2)
        layout.addWidget(side_panel, stretch=2)

    def _label_pair(self, caption: str, value_label: QLabel) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        title = QLabel(caption)
        title.setObjectName('captionLabel')
        value_label.setObjectName('valueLabel')
        layout.addWidget(title)
        layout.addWidget(value_label)
        return widget

    def _build_view_controls(self) -> QWidget:
        self.elev_input.setRange(-180, 180)
        self.elev_input.setValue(self.iso_plot.current_elev)
        self.azim_input.setRange(-180, 180)
        self.azim_input.setValue(self.iso_plot.current_azim)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel('Kamera')
        title.setObjectName('captionLabel')
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow('elev', self.elev_input)
        form.addRow('azim', self.azim_input)
        layout.addWidget(title)
        layout.addLayout(form)
        return widget

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f5f7;
            }
            QWidget#infoCard {
                background: #ffffff;
                border: 1px solid #d9e0e6;
                border-radius: 12px;
            }
            QLabel#captionLabel {
                color: #5b6470;
                font-weight: 600;
            }
            QLabel#valueLabel {
                color: #1f2a37;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton {
                background-color: #0f8b6d;
                color: white;
                border: 1px solid #0b6b54;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 700;
                min-height: 42px;
            }
            QPushButton:hover {
                background-color: #0c7a60;
            }
        """)

    def _connect_signals(self):
        self.timer.timeout.connect(self._step_demo)
        self.toggle_button.clicked.connect(self._toggle_demo)
        self.elev_input.valueChanged.connect(self._update_view)
        self.azim_input.valueChanged.connect(self._update_view)

    def _build_demo_samples(self):
        rng = np.random.default_rng(DEMO_SEED)
        half = MEMBRANE_SIZE / 2.0 - 24.0
        anchor_count = 9
        anchor_points = np.column_stack((
            rng.uniform(-half * 0.75, half * 0.75, size=anchor_count),
            rng.uniform(-half * 0.75, half * 0.75, size=anchor_count),
        ))
        anchor_points[0] = np.array([0.0, 110.0])
        anchor_points[-1] = anchor_points[0]

        frames = []
        steps_between = 16
        for start, end in zip(anchor_points[:-1], anchor_points[1:]):
            for alpha in np.linspace(0.0, 1.0, steps_between, endpoint=False):
                point = (1.0 - alpha) * start + alpha * end
                phase = len(frames) / 8.0
                force_value = 9.5 + 6.0 * np.sin(phase) + 1.8 * np.cos(phase * 0.7)
                force_value = float(np.clip(force_value, FORCE_RANGE[0], FORCE_RANGE[1]))
                intensity = 0.9 + (force_value / FORCE_RANGE[1]) * 1.5
                sigma = 26.0 + 12.0 * (0.5 + 0.5 * np.sin(phase * 0.5))
                frames.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'force': force_value,
                    'intensity': float(intensity),
                    'sigma': float(sigma),
                })
        return frames

    def _toggle_demo(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText('Demo pausiert')
            self.toggle_button.setText('Live Demo starten')
            return

        self.timer.start()
        self.status_label.setText('Demo aktiv')
        self.toggle_button.setText('Live Demo stoppen')

    def _update_view(self):
        self.iso_plot.set_view(self.elev_input.value(), self.azim_input.value())

    def _handle_xy_click(self, x_value: float, y_value: float):
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_button.setText('Live Demo starten')

        self.plot.set_heatmap(x_value, y_value, 1.4, 32.0)
        self.iso_plot.update_heatmap(x_value, y_value, 1.4, 32.0)
        self.position_label.setText(f"X={x_value:.1f}, Y={y_value:.1f}")
        self.status_label.setText('Manueller Punkt aus XY-Plot')

    def _step_demo(self):
        if not self.demo_samples:
            return
        sample = self.demo_samples[self.demo_index]
        self.demo_index = (self.demo_index + 1) % len(self.demo_samples)

        self.plot.set_heatmap(sample['x'], sample['y'], sample['intensity'], sample['sigma'])
        if self.iso_frame_counter % ISO_UPDATE_EVERY == 0:
            self.iso_plot.update_heatmap(sample['x'], sample['y'], sample['intensity'], sample['sigma'])
        self.iso_frame_counter += 1
        self.force_plot.set_force(sample['force'])
        self.position_label.setText(f"X={sample['x']:.1f}, Y={sample['y']:.1f}")
        self.force_label.setText(f"{sample['force']:.1f} N")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MeMo heatmap visualization window')
    return parser.parse_args(argv)


def run_heatmap_window(argv=None):
    parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = HeatmapWindow()
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(run_heatmap_window())
