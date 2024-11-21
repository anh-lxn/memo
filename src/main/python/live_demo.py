# Imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
import time
import helper_fns_ki_model as h_fn_ki
import board                # Für die GPIO- und I2C-Steuerung auf dem Raspberry Pi
import busio                # Für die I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Für die Verwendung des ADS1115 AD-Wandlers
from adafruit_ads1x15.analog_in import AnalogIn  # Für die Spannungsmessung an ADS1115-Kanälen

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert den I2C-Bus mit den SCL- und SDA-Pins
ads0 = ADS.ADS1115(i2cbus, address=0x4b)  # Erstellt ein Objekt für den ADS1115 mit Adresse 0x48
ads1 = ADS.ADS1115(i2cbus, address=0x49)  # Erstellt ein zweites Objekt für den ADS1115 mit Adresse 0x49

# Einrichtung der analogen Kanäle
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)  # Kanäle des ersten ADS1115
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)  # Kanäle des zweiten ADS1115

# Definition der Sensorpositionen für Plot
sensor_pos = [(-315, 315), (0, 315), (315, 315), (-315, 0), (315, 0), (-315, -315), (0, -315), (315, -315)]
model = h_fn_ki.load_model(path='../resources/models/model_demonstrator_21_11_2024_13-30-17.pth')

# Funktion zum Berechnen der Lastkoordinate aus
def calc_loadpoint(model):
    # 1. Auslesen der Sensorwerte über i2C
    sensor_01 = ch1.voltage
    sensor_02 = ch2.voltage
    sensor_03 = ch3.voltage
    sensor_04 = ch4.voltage
    sensor_05 = ch5.voltage
    sensor_06 = ch6.voltage
    sensor_07 = ch7.voltage
    sensor_08 = ch8.voltage
    """
    sensor_01 = np.random.uniform(0,1)
    sensor_02 = np.random.uniform(0,1)
    sensor_03 = np.random.uniform(0, 1)
    sensor_04 = np.random.uniform(0, 1)
    sensor_05 = np.random.uniform(0, 1)
    sensor_06 = np.random.uniform(0, 1)
    sensor_07 = np.random.uniform(0, 1)
    sensor_08 = np.random.uniform(0, 1)
    """

    # Werte in einer Liste speichern
    sensor_values = [sensor_01, sensor_02, sensor_03, sensor_04, sensor_05, sensor_06, sensor_07, sensor_08]

    # Maximalen Wert in der Liste finden
    max_value = max(sensor_values)

    # Werte normieren
    normalized_sensor_values = [value / max_value for value in sensor_values]

    # Summe der Sensorwerte ermitteln
    total_sum = sensor_01 + sensor_02 + sensor_03 + sensor_04 + sensor_05 + sensor_06 + sensor_07 + sensor_08

    # Definieren eines Thresholds um Lastpunkt nur zu plotten, wenn gedrückt wird
    if total_sum < 1:
        x_value_pred = [1000]
        y_value_pred = [1000]
    else:
        # Beispielhafte Eingabewerte für X_curr
        X_curr = torch.tensor([[normalized_sensor_values[0], normalized_sensor_values[1],
                                normalized_sensor_values[2], normalized_sensor_values[3], normalized_sensor_values[4],
                                normalized_sensor_values[5], normalized_sensor_values[6],
                                normalized_sensor_values[7]]], dtype=torch.float32)

        # Berechnung Vorhersage aus trainiertem KI-Model
        model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
        with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
            y_pred = model(X_curr) # y_pred enthält die vorhergesagten x- und y-Koordinaten des Lastpunktes

        # Extrahieren der x- und y-Koordinaten, welche das KI-Modell vorhergesagt hat
        x_value_pred = y_pred[:,0].numpy()
        y_value_pred = y_pred[:, 1].numpy()

    return  x_value_pred[0], y_value_pred[0]

def create_live_scatterplot_text():
    try:
        while True:
            # Berechne die Werte für load_pos_x und load_pos_y
            load_pos_x, load_pos_y = calc_loadpoint(model)
            
            # Gib die Werte im Terminal aus
            print(f"Load Position X: {load_pos_x}, Load Position Y: {load_pos_y}")
            
            # Warte eine Sekunde, bevor die Werte erneut ausgegeben werden
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        print("\nProgramm wurde beendet.")


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
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()

    # Kreise initialisieren für Plot des Lastpunktes
    circle0 = plt.Circle((0, 0), 45, color='blue', fill=True, linewidth=1, alpha=0.1)
    circle = plt.Circle((0, 0), 30, color='blue', fill=False, linewidth=3, alpha=0.9)
    circle2 = plt.Circle((0, 0), 45, color='blue', fill=False, linewidth=2, alpha=0.6)

    ax.add_patch(circle0)
    ax.add_patch(circle)
    ax.add_patch(circle2)
    
    # Funktion zum Updaten des Plots zur Liverealisation
    def update(frame):
        # Aufruf der Funktion zum Berechnen der Lastkoordinaten
        load_pos_x, load_pos_y = calc_loadpoint(model)

        # Update scatter plot
        load_scatter.set_offsets([load_pos_x, load_pos_y])

        # Update circle positions
        circle0.set_center((load_pos_x, load_pos_y))
        circle.set_center((load_pos_x, load_pos_y))
        circle2.set_center((load_pos_x, load_pos_y))

        return load_scatter, circle0, circle, circle2
    # interval -> Angabe der Zeit in ms, wie oft Plot aktualisiert werden soll
    ani = animation.FuncAnimation(fig, update, frames=None, interval=250, blit=True)

    plt.show()


create_live_scatterplot()
