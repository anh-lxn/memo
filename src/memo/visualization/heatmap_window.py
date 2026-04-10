from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
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

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from memo.acquisition.readers import Ads1115Reader, CsvReplayReader, MockReader, SerialSensorReader, UnavailableSensorReader
from memo.ml.inference import ModelPredictor
from memo.ml.models import MembraneModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_models" / "model_xyf.pt"
DEFAULT_NORMALIZATION_CSV = PROJECT_ROOT / "data" / "recorded_samples" / "3D_Messung_01_10N.csv"
SENSOR_COLUMNS = [f"Sensor R{i}" for i in range(1, 9)]
MEMBRANE_SIZE = 350.0
OFFSET = 30.0
X_LIMITS = (-MEMBRANE_SIZE / 2 - OFFSET, MEMBRANE_SIZE / 2 + OFFSET)
Y_LIMITS = (-MEMBRANE_SIZE / 2 - OFFSET, MEMBRANE_SIZE / 2 + OFFSET)
GRID_SPACING = 50.0
FORCE_RANGE = (0.0, 20.0)
UPDATE_MS = 140
ISO_UPDATE_EVERY = 3
HEATMAP_VMAX = 2.6
ENABLE_XY_INTERACTION = False
TEMPORAL_SMOOTHING_ALPHA = 0.22


def _load_normalization_stats(csv_path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if csv_path is None or not csv_path.exists():
        return None, None

    data = pd.read_csv(csv_path)
    missing_columns = [column for column in SENSOR_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV fuer Normalisierung fehlt Sensor-Spalten: {', '.join(missing_columns)}")
    sensor_data = data[SENSOR_COLUMNS]
    return sensor_data.min().to_numpy(dtype=float), sensor_data.max().to_numpy(dtype=float)


def build_reader(replay_csv: str | None, sensor_port: str | None, sensor_baudrate: int, sensor_timeout: float, use_mock: bool):
    if use_mock:
        return MockReader(seed=7, baseline=2.5, noise_std=0.08)
    if replay_csv:
        return CsvReplayReader(replay_csv, loop=True)
    if sensor_port:
        return SerialSensorReader(
            port=sensor_port,
            baudrate=sensor_baudrate,
            timeout=sensor_timeout,
        )
    try:
        return Ads1115Reader()
    except RuntimeError as exc:
        return UnavailableSensorReader(
            reason=(
                "ADS1115 ist in dieser Umgebung nicht verfuegbar. "
                "Bitte mit --mock, --replay-csv oder --sensor-port starten. "
                f"Details: {exc}"
            )
        )


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
    def __init__(self, reader, predictor: ModelPredictor, update_ms: int = UPDATE_MS):
        super().__init__()
        self.reader = reader
        self.predictor = predictor
        self.update_ms = int(update_ms)
        self._smoothed_prediction: np.ndarray | None = None
        self._last_error: str | None = None

        self.setWindowTitle('MeMo Heatmap Visualisierung')
        self.resize(1850, 980)

        self.plot = XYHeatmapPlot()
        self.plot.on_point_selected = self._handle_xy_click
        self.iso_plot = IsometricMembranePlot()
        self.force_plot = ForceBarPlot()
        self.position_label = QLabel('-')
        self.force_label = QLabel('-')
        self.status_label = QLabel('Modell wird geladen')
        self.toggle_button = QPushButton('Live-Prediction stoppen')
        self.elev_input = QSpinBox()
        self.azim_input = QSpinBox()
        self.iso_frame_counter = 0
        self.timer = QTimer(self)
        self.timer.setInterval(self.update_ms)

        self._build_ui()
        self._apply_styles()
        self._connect_signals()
        self._step_prediction()
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
        self.timer.timeout.connect(self._step_prediction)
        self.toggle_button.clicked.connect(self._toggle_prediction)
        self.elev_input.valueChanged.connect(self._update_view)
        self.azim_input.valueChanged.connect(self._update_view)

    def _toggle_prediction(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText('Live-Prediction pausiert')
            self.toggle_button.setText('Live-Prediction starten')
            return

        self.timer.start()
        self.status_label.setText('Live-Prediction aktiv')
        self.toggle_button.setText('Live-Prediction stoppen')

    def _update_view(self):
        self.iso_plot.set_view(self.elev_input.value(), self.azim_input.value())

    def _handle_xy_click(self, x_value: float, y_value: float):
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_button.setText('Live-Prediction starten')

        self.plot.set_heatmap(x_value, y_value, 1.4, 32.0)
        self.iso_plot.update_heatmap(x_value, y_value, 1.4, 32.0)
        self.position_label.setText(f"X={x_value:.1f}, Y={y_value:.1f}")
        self.status_label.setText('Manueller Punkt aus XY-Plot')

    def _smooth_prediction(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        if self._smoothed_prediction is None:
            self._smoothed_prediction = prediction.copy()
        else:
            alpha = TEMPORAL_SMOOTHING_ALPHA
            self._smoothed_prediction = ((1.0 - alpha) * self._smoothed_prediction) + (alpha * prediction)
        return self._smoothed_prediction

    def _step_prediction(self):
        try:
            frame = self.reader.read()
            sensor_values = np.asarray(frame.sensors, dtype=float)
            prediction = np.asarray(self.predictor.predict(sensor_values), dtype=float).reshape(-1)
            if prediction.size < 3:
                raise ValueError('Das model_xyf liefert weniger als drei Ausgabewerte.')

            smoothed = self._smooth_prediction(prediction[:3])
            x_value = float(np.clip(smoothed[0], X_LIMITS[0], X_LIMITS[1]))
            y_value = float(np.clip(smoothed[1], Y_LIMITS[0], Y_LIMITS[1]))
            force_value = float(np.clip(smoothed[2], FORCE_RANGE[0], FORCE_RANGE[1]))
            intensity = 0.9 + (force_value / max(FORCE_RANGE[1], 1.0)) * 1.5
            sigma = 28.0 + 10.0 * (force_value / max(FORCE_RANGE[1], 1.0))

            self.plot.set_heatmap(x_value, y_value, intensity, sigma)
            if self.iso_frame_counter % ISO_UPDATE_EVERY == 0:
                self.iso_plot.update_heatmap(x_value, y_value, intensity, sigma)
            self.iso_frame_counter += 1

            self.force_plot.set_force(force_value)
            self.position_label.setText(f"X={x_value:.1f}, Y={y_value:.1f}")
            self.force_label.setText(f"{force_value:.1f} N")
            self.status_label.setText(f"Live-Vorhersage aktiv ({frame.source})")
            self._last_error = None
        except StopIteration:
            self.timer.stop()
            self.status_label.setText('Replay beendet')
            self.toggle_button.setText('Live-Prediction starten')
        except Exception as exc:
            message = str(exc)
            if message != self._last_error:
                self.status_label.setText(f'Fehler: {message}')
                self._last_error = message

    def closeEvent(self, event: QCloseEvent):
        self.timer.stop()
        close_method = getattr(self.reader, 'close', None)
        if callable(close_method):
            close_method()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MeMo heatmap visualization window')
    parser.add_argument('--model-path', default=str(DEFAULT_MODEL_PATH), help='Pfad zu model_xyf.pt')
    parser.add_argument(
        '--normalization-csv',
        default=str(DEFAULT_NORMALIZATION_CSV),
        help='CSV fuer Min/Max-Normalisierung der acht Sensorspalten',
    )
    parser.add_argument('--replay-csv', default=None, help='CSV-Replay statt Live-Hardware')
    parser.add_argument('--sensor-port', default=None, help='Serieller Port fuer 8 Sensorsignale')
    parser.add_argument('--sensor-baudrate', type=int, default=57600, help='Baudrate des seriellen Sensorreaders')
    parser.add_argument('--sensor-timeout', type=float, default=0.2, help='Timeout des seriellen Sensorreaders')
    parser.add_argument('--mock', action='store_true', help='Verwendet Mock-Sensordaten statt echter Hardware')
    parser.add_argument('--cpu', action='store_true', help='Erzwingt Vorhersage auf CPU')
    parser.add_argument('--update-ms', type=int, default=UPDATE_MS, help='UI-Updateintervall in Millisekunden')
    return parser.parse_args(argv)


def run_heatmap_window(argv=None):
    args = parse_args(argv)
    mins, maxs = _load_normalization_stats(Path(args.normalization_csv) if args.normalization_csv else None)
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    predictor = ModelPredictor(
        model_class=MembraneModel,
        model_path=str(Path(args.model_path)),
        output_dim=3,
        device=device,
        mins=mins,
        maxs=maxs,
    )
    reader = build_reader(
        replay_csv=args.replay_csv,
        sensor_port=args.sensor_port,
        sensor_baudrate=args.sensor_baudrate,
        sensor_timeout=args.sensor_timeout,
        use_mock=args.mock,
    )
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = HeatmapWindow(reader=reader, predictor=predictor, update_ms=args.update_ms)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(run_heatmap_window())
