# Imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
import helper_fns_ki_model as h_fn_ki

# Definition der Sensorpositionen für Plot
sensor_pos = [(-315, 315), (0, 315), (315, 315), (-315, 0), (315, 0), (-315, -315), (0, -315), (315, -315)]
model = h_fn_ki.load_model(path='../resources/models/model_demonstrator_30_05_2024_15-52-32.pth')
# Funktion zum Berechnen der Lastkoordinate aus
def calc_loadpoint(model):
    # 1. Auslesen der Sensorwerte über i2C
    sensor_01 = np.random.uniform(0,2)
    sensor_02 = np.random.uniform(0,2)
    sensor_03 = np.random.uniform(0, 2)
    sensor_04 = np.random.uniform(0, 2)
    sensor_05 = np.random.uniform(0, 2)
    sensor_06 = np.random.uniform(0, 2)
    sensor_07 = np.random.uniform(0, 2)
    sensor_08 = np.random.uniform(0, 2)

    # Beispielhafte Eingabewerte für X_curr
    X_curr = torch.tensor([[sensor_01, sensor_02, sensor_03, sensor_04, sensor_05, sensor_06, sensor_07, sensor_08]], dtype=torch.float32)

    # Berechnung Vorhersage aus trainiertem KI-Model
    model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_curr)

    x_values_pred = y_pred[:,0].numpy()
    y_values_pred = y_pred[:, 1].numpy()

    return  x_values_pred[0], y_values_pred[0]


def create_live_scatterplot():
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y

    fig, ax = plt.subplots(figsize=(8, 8))  # Größere Figur für bessere Sichtbarkeit
    fig.tight_layout(pad=2.0)  # Damit der Plot innerhalb des Fensters bleibt

    # Hintergrundfläche einfärben
    ax.fill([-400, 400, 400, -400], [-400, -400, 400, 400], color='yellow', alpha=0.4)

    # Sensorpositionen plotten
    ax.scatter(sensor_x, sensor_y, c='red', alpha=1.0, marker='x', label='Sensor Positions')  # Sensorpositionen in Rot
    load_scatter = ax.scatter([], [], c='blue', alpha=1.0, marker='o', label='Calculated Load Position')

    # Sensorpositionen beschriften
    for i, (sx, sy) in enumerate(sensor_pos, start=1):
        ax.text(sx, sy, "Sensor "+str(i)+"  ", fontsize=12, fontweight='bold', ha='right')

    ax.set_xlim(-500, 500)  # Setzt die x-Achse auf den Bereich -500 bis 500
    ax.set_ylim(-500, 500)  # Setzt die y-Achse auf den Bereich -500 bis 500
    ax.set_xlabel('X-Achse [mm]', fontsize=18)
    ax.set_ylabel('Y-Achse [mm]',fontsize=18)
    ax.set_title('XY-Plot - Lastpunkt (Blau), Sensorpunkte (Rot)', fontsize=18)
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()

    ax.set_aspect('equal', adjustable='box')  # Verhältnisse der Achsen beibehalten

    # Set the figure to full screen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    # Kreise initialisieren
    circle0 = plt.Circle((0, 0), 45, color='blue', fill=True, linewidth=1, alpha=0.1)
    circle = plt.Circle((0, 0), 30, color='blue', fill=False, linewidth=3, alpha=0.9)
    circle2 = plt.Circle((0, 0), 45, color='blue', fill=False, linewidth=2, alpha=0.6)

    ax.add_patch(circle0)
    ax.add_patch(circle)
    ax.add_patch(circle2)

    def update(frame):
        load_pos_x, load_pos_y = calc_loadpoint(model)

        # Update scatter plot
        load_scatter.set_offsets([load_pos_x, load_pos_y])

        # Update circle positions
        circle0.set_center((load_pos_x, load_pos_y))
        circle.set_center((load_pos_x, load_pos_y))
        circle2.set_center((load_pos_x, load_pos_y))

        return load_scatter, circle0, circle, circle2

    ani = animation.FuncAnimation(fig, update, frames=None, interval=250, blit=True)

    plt.show()

create_live_scatterplot()
