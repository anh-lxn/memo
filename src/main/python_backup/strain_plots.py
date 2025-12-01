import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import csv

# Datei und Daten laden
datei_pfad = "../resources/messungen/auswertung/28_11/auswertung_gesamt_2024-11-28_13-37_sorted.csv"

# Listen zum Speichern der Werte
load_pos_x, load_pos_y = [], []
strain_sensors = [[] for _ in range(8)]  # Liste für die 8 Sensoren

# Die Datei zeilenweise einlesen
with open(datei_pfad, mode='r', newline='') as csv_datei:
    reader = csv.reader(csv_datei)

    # Kopfzeile extrahieren
    kopfzeile = next(reader)
    print("Kopfzeile:", kopfzeile)

    # Jede weitere Zeile lesen und verarbeiten
    for zeile in reader:
        load_pos_x.append(float(zeile[1]))
        load_pos_y.append(float(zeile[2]))
        for i in range(8):
            strain_sensors[i].append(float(zeile[4 + i]))

# Daten in NumPy-Arrays umwandeln
x = np.array(load_pos_x)
y = np.array(load_pos_y)

# Variable zur Auswahl des Sensors (Index von 0 bis 7)
sensor_index = 5  # Ändern Sie diesen Wert, um einen anderen Sensor auszuwählen (0 = Sensor 1, 1 = Sensor 2, etc.)

# Daten für den ausgewählten Sensor
z = np.array(strain_sensors[sensor_index])

# Interpolation
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method='cubic')

# Plot erstellen
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Interpolierte Fläche plotten
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Achsentitel und Farbbalken
ax.set_xlabel('X Achse')
ax.set_ylabel('Y Achse')
ax.set_zlabel(f'Spannungswerte für Sensor R{sensor_index + 1}')
ax.set_zlim(0, 4)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Interpolierte Spannungswerte')

ax.set_title(f"3D-Plot für Sensor R{sensor_index + 1}")
plt.legend()
plt.show()
