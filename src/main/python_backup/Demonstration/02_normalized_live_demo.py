# Imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mplcyberpunk
import numpy as np
import pandas as pd
import torch
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
import helper_fns_ki_model as h_fn_ki
from matplotlib.widgets import Button
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
sensor_pos = [(-315, -315), (-315, 0), (-315, 315), (0, 315), (315, 315), (315, 0), (315, -315), (0, -315)]

# Model laden
model = h_fn_ki.load_model(path='model_demonstrator_28_11_2024_13-49-38.pth')

# Plot im Cybepunk-Stil
plt.style.use("cyberpunk")

# Hintergrundfarbe und Textfarben für den Cyberpunk-Stil
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # blaugrauer Hintergrund

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # sehr helles Grau für Text und Achsen

# Funktion zum Berechnen der Lastkoordinate aus
def calc_loadpoint(model):
    """
    Berechnet die Lastkoordinate aus den Sensorwerten.
    
    Args:
        model (torch.nn.Module): Das trainierte KI-Modell.
    
    Returns:
        x_value_pred (float): Die vorhergesagte x-Koordinate des Lastpunktes.
        y_value_pred (float): Die vorhergesagte y-Koordinate des Lastpunktes.
        sensor_values (list): Die Sensorwerte.
    """
    
    # 1. Auslesen der Sensorwerte über i2C'
    sensor_R2 = ch1.voltage
    sensor_R3 = ch2.voltage
    sensor_R4 = ch3.voltage
    sensor_R1 = ch4.voltage
    sensor_R8 = ch5.voltage
    sensor_R7 = ch6.voltage
    sensor_R6 = ch7.voltage
    sensor_R5 = ch8.voltage
    
    #sensor_values = [sensor_R1, sensor_R2, sensor_R3, sensor_R4, sensor_R5, sensor_R6, sensor_R7, sensor_R8]
    sensor_values = [round(sensor_R2, 3), round(sensor_R3, 3), round(sensor_R4, 3), round(sensor_R1, 3), round(sensor_R8, 3), round(sensor_R7, 3), round(sensor_R6, 3), round(sensor_R5, 3)]
    # sensor_values, x_value, y_value = get_sensor_values_by_id(np.random.randint(0, 3))
    
    # Wenn nicht gedrückt wird, übergebe nicht sichtbaren Punkt
    total_sum = 0 
    for spannung in sensor_values:
       total_sum += spannung
    if total_sum > 25:
       return 10000, 10000, sensor_values
    
    # Maximalen Wert in der Liste finden
    max_value = max(sensor_values)

    # Werte normieren
    normalized_sensor_values = [value / max_value for value in sensor_values]

    # Beispielhafte Eingabewerte für X_curr
    X_curr = torch.tensor([[normalized_sensor_values[0], normalized_sensor_values[1],
                            normalized_sensor_values[2], normalized_sensor_values[3], 
                            normalized_sensor_values[4], normalized_sensor_values[5], 
                            normalized_sensor_values[6], normalized_sensor_values[7]]], dtype=torch.float32)
                            

    # Berechnung Vorhersage aus trainiertem KI-Model
    model.eval()  # Setze das Modell in den Auswertungsmodus (deaktiviere Dropout und BatchNorm)
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_curr)  # y_pred enthält die vorhergesagten x- und y-Koordinaten des Lastpunktes

    # Extrahieren der x- und y-Koordinaten, welche das KI-Modell vorhergesagt hat
    x_value_pred = y_pred[:, 0].numpy()
    y_value_pred = y_pred[:, 1].numpy()

    return x_value_pred[0], y_value_pred[0], sensor_values

# Funktion zum Abrufen der Sensorwerte und X/Y-Werte basierend auf der Datei-ID
def get_sensor_values_by_id(file_id):
    df = pd.read_csv('../resources/messungen/auswertung/28_11/auswertung_gesamt_2024-11-28_13-37.csv')
    # Suchen nach der Zeile, die der angegebenen Datei-ID entspricht
    row = df[df['Datei_ID'] == file_id]
    
    if row.empty:
        print(f"Keine Daten für Datei_ID {file_id} gefunden.")
        return None
    else:
        # Extrahieren der Sensorwerte (Sensor 1 bis Sensor 8)
        sensor_values = row.iloc[0, 4:12].tolist()  # Spalten 4 bis 11 (Sensor 1 bis Sensor 8)
        
        # Extrahieren der X- und Y-Werte
        x_value = row.iloc[0]['X']  # X-Wert
        y_value = row.iloc[0]['Y']  # Y-Wert
        
        # Rückgabe der Sensorwerte und X, Y als Tupel
        return sensor_values, x_value, y_value

# Funktion zum Erstellen des Live-Scatterplots
def create_live_scatterplot(rectangle_width=50, rectangle_height=20, rotation_angles=None):
    """
    Erstellt einen Live-Scatterplot mit Balken, die die Sensorwerte darstellen,
    und platziert sie direkt an den Positionen der Sensoren.
    """
    # Sensorpositionen
    sensor_x, sensor_y = zip(*sensor_pos)  # Entpacke die Tupel in separate Listen für x und y
    
    # Standardrotationen, falls keine angegeben sind
    if rotation_angles is None:
        #rotation_angels = [R1, R2, R3, R4, R5, R6, R7, R8]
        rotation_angles = [45, 0, -45, -90, -135, -180, 135, 90]

    # Erstellung eines neuen Plots
    fig, ax = plt.subplots(figsize=(8, 8))  # Größere Figur für Übersichtlichkeit
    fig.tight_layout(pad=2.0)  # Abstand zwischen Plot und Rändern

    # Make the plot fullscreen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # Rechtecke als Sensoren
    sensor_rects = []
    sensor_bars = []
    for (sx, sy), angle in zip(sensor_pos, rotation_angles):
        # Sensor (statisches Rechteck)
        trans = Affine2D().rotate_deg_around(sx, sy, angle) + plt.gca().transData
        rect = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),  # Linke untere Ecke
            rectangle_width,
            rectangle_height,
            edgecolor='#00bfff',
            facecolor='#00bfff',
            alpha=0.9,
            transform=trans
        )
        ax.add_patch(rect)
        sensor_rects.append(rect)

        # Balken (initiale Höhe = 1)
        bar = patches.Rectangle(
            (sx - rectangle_width / 2, sy - rectangle_height / 2),  # Schmale Balkenbreite, direkt über der Sensorposition
            rectangle_width,  # Breite des Balkens
            rectangle_height,   # Anfangshöhe des Balkens
            edgecolor='black',
            facecolor='black',
            alpha=0.8,
            transform=trans
        )
        ax.add_patch(bar)
        sensor_bars.append(bar)

        # Textbeschriftung des Sensors
        text_x, text_y = sx, sy + 40  # Standard: Text über dem Sensor
        ha, va = 'center', 'bottom'  # Textausrichtung (zentriert)

        # Anpassung der Textposition
        if sy < 0:  # Text unterhalb des Sensors
            text_y = sy - 40
            va = 'top'
        elif sx > 0:  # Text rechts vom Sensor
            text_x, text_y = sx + 40, sy
            ha, va = 'left', 'center'
        elif sx < 0:  # Text links vom Sensor
            text_x, text_y = sx - 40, sy
            ha, va = 'right', 'center'

        plt.text(
            text_x, text_y, f"Sensor R{sensor_pos.index((sx, sy)) + 1}",  # Sensor-Nummer
            fontsize=11, fontweight='bold', ha=ha, va=va, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9)
        )

    # Achsenbegrenzungen und Gitterlinien setzen
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)

    # Grid 
    for i in range(-290, 291, 58):
        ax.axhline(i, color='gray', linewidth=0.5, xmin=0.1, xmax=0.9)
        ax.axvline(i, color='gray', linewidth=0.5, ymin=0.1, ymax=0.9)

    # Rand
    ax.axhline(-400, color='black', linewidth=4, xmin=0.1, xmax=0.9) # Horizontale Linie bei y=-400
    ax.axhline(400, color='black', linewidth=4, xmin=0.1, xmax=0.9) # Horizontale Linie bei y=400
    ax.axvline(-400, color='black', linewidth=4, ymin=0.1, ymax=0.9) # Vertikale Linie bei x=-400
    ax.axvline(400, color='black', linewidth=4, ymin=0.1, ymax=0.9) # Vertikale Linie bei x=400    

    ax.set_xlabel('X-Achse [mm]', fontsize=18)
    ax.set_ylabel('Y-Achse [mm]', fontsize=18)
    ax.set_title('MeMo - Echtzeit Lasterkennung', fontsize=18)
    ax.grid(False)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # Heatmap erstellen
    x_min, x_max, y_min, y_max = -400, 400, -400, 400
    X, Y = np.meshgrid(np.linspace(-400, 400, 50), np.linspace(-400, 400, 50))
    Z = np.sqrt((1*X)**2 + (1*Y)**2)  # Z-Wert ist die Distanz vom Ursprung (Kreis)
    heatmap = ax.imshow(Z, cmap="viridis", interpolation='bilinear', origin='lower', extent=[x_min, x_max, y_min, y_max])  # Erstellen einer neuen Heatmap
    #color_bar = fig.colorbar(heatmap)

    # Scatterplot für Lastpunkt (blauer Kreis, initial leer)
    # circle = plt.Circle((0, 0), 5, color='black', fill=True, linewidth=1, alpha=1)
    # ax.add_patch(circle)
       
    # Update-Funktion für Animation
    def update(frame):
        """
        Aktualisiert den Plot mit den neu berechneten Lastkoordinaten und Sensorwerten.
        """

        # Funktion zur Berechnung der Lastkoordinaten und Sensorwerte aufrufen
        load_pos_x, load_pos_y, sensor_values = calc_loadpoint(model)

        # Sensorwerte zu den zugehörigen Sensoren zuordnen
        sensor_labels = ['R2', 'R3', 'R4', 'R1', 'R8', 'R7', 'R6', 'R5']
        sensor_value_mapping = dict(zip(sensor_labels, sensor_values))  # Mapping von Bezeichnungen zu Werten

        # Maximalwert zur Skalierung der Balkenhöhen
        max_sensor_value = max(sensor_values)
        normalized_sensor_values = [
            value / max_sensor_value * 50 for value in sensor_values
        ]  # Skaliert für Balkenhöhen (0 bis 100)

        # Balkenhöhen entsprechend der zugeordneten Sensorwerte aktualisieren
        # bar, norm_value, (sx, sy), sensor_label in zip(sensor_bars, normalized_sensor_values, sensor_pos, sensor_labels):
         #   bar.set_width(norm_value)  # Höhe des Balkens aktualisieren
            
        
        # Manuelle Zuordnung der Balkenhöhen
        sensor_bars[0].set_width(normalized_sensor_values[3])  # Sensor R1
        sensor_bars[1].set_width(normalized_sensor_values[0])  # Sensor R2
        sensor_bars[2].set_width(normalized_sensor_values[1])  # Sensor R3
        sensor_bars[3].set_width(normalized_sensor_values[2])  # Sensor R4
        sensor_bars[4].set_width(normalized_sensor_values[7])  # Sensor R5
        sensor_bars[5].set_width(normalized_sensor_values[6])  # Sensor R6
        sensor_bars[6].set_width(normalized_sensor_values[5])  # Sensor R7
        sensor_bars[7].set_width(normalized_sensor_values[4])  # Sensor R8
        
        # Konsolenausgabe der Koordinaten und Sensorwerte
        # print(f"Load Position - X: {load_pos_x:.2f}, Y: {load_pos_y:.2f}")
        # print(f"Sensor Values: {sensor_value_mapping}")

        # Entfernen der vorherigen Heatmap (konturierte Flächen)
        for coll in ax.collections:
            coll.remove()

        # Aktualisieren der Heatmap
        Z = np.sqrt((0.6*(X - load_pos_x))**2 + (0.6*(Y - load_pos_y))**2)  # Z-Wert ist die Distanz vom Ursprung (Kreis)
        heatmap.set_data(Z)

        # Aktualisiere die Colorbar, indem die Mappable-Instanz geändert wird
        #color_bar.update_ticks()  # Update die Colorbar-Ticks

        # Aktualisiere die Scatterplot (Lastpunkt)
        #circle.set_center((load_pos_x, load_pos_y))
    
    # Einen Button hinzufügen
    ax_button = plt.axes([0.915, 0.015, 0.075, 0.030])  # Position des Buttons
    ax_button.set_frame_on(False)
    button = Button(ax_button, 'Beenden')
    button.label.set_fontsize(18)
    button.label.set_color("black")
    button.label.set_fontweight("bold")
    box = patches.FancyBboxPatch((0,0), 1,1, 
                                edgecolor="black",
                                facecolor="0.9",
                                boxstyle="round,pad=0.1", 
                                mutation_aspect=3, 
                                transform=ax_button.transAxes, clip_on=False)
    ax_button.add_patch(box)
    # Funktion zum Beenden des Plots
    def close(event):
        plt.close()  # Beendet das Plot-Fenster
    #Die close-Funktion wird beim Klicken auf den Button aufgerufen
    button.on_clicked(close)

    # Animation starten (Intervall = 100 ms)
    ani = animation.FuncAnimation(fig, update, frames=None, interval=10, blit=False)

    # Plot anzeigen
    plt.show()

# Live-Plot starten
create_live_scatterplot()
