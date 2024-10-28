"""
Dateiname: 04_train_demonstrator_model.py
Autor: Florian Schmidt
Datum: 30.05.2024

Beschreibung:
Dieses Skript führt die Verarbeitung, Normalisierung und Vorhersage von Dehnungsdaten durch. Die Daten werden
mithilfe eines neuronalen Netzmodells trainiert und getestet. Das Modell kann gespeichert und später für Vorhersagen
wieder geladen werden.
"""

import helper_fns_data as h_fn
import helper_fns_ki_model as h_fn_ki
import glob
import numpy as np
import csv

# Pfad zur Gesamtdatei angeben
datei_pfad = "../resources/messungen/auswertung/auswertung_gesamt_2024-10-28_15-42.csv"  # <-- Passe diesen Pfad an

# Listen zum Speichern der Werte (load_pos_x, load_pos_y sind Positionen der Lasteinleitung, strain_1 bis strain_8
# sind die resultierenden Dehnungswerte der Sensoren 1-8 (Auf Sensoranordnung achten!))
load_pos_x, load_pos_y, load_value, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8 = [], [], [], [], [], [], [], [], [], [], []

# Die Datei zeilenweise einlesen
with open(datei_pfad, mode='r', newline='') as csv_datei:
    reader = csv.reader(csv_datei)

    # Kopfzeile extrahieren (erste Zeile)
    kopfzeile = next(reader)
    print("Kopfzeile:", kopfzeile)

    # Jede weitere Zeile lesen und verarbeiten
    for zeile in reader:
        print("Zeile:", zeile)
        load_pos_x.append(float(zeile[1]))
        load_pos_y.append(float(zeile[2]))
        load_value.append(float(zeile[3]))
        strain_1.append(float(zeile[4]))
        strain_2.append(float(zeile[5]))
        strain_3.append(float(zeile[6]))
        strain_4.append(float(zeile[7]))
        strain_5.append(float(zeile[8]))
        strain_6.append(float(zeile[9]))
        strain_7.append(float(zeile[10]))
        strain_8.append(float(zeile[11]))


strains = [strain_1,strain_2,strain_3,strain_4,strain_5,strain_6,strain_7,strain_8]

#### Normalisierung der Strain Listen
# Umwandeln in ein numpy array für einfachere Handhabung
strains_array = np.array(strains)

# Normieren der Werte für jeden Index
min_vals = np.min(strains_array, axis=0)
max_vals = np.max(strains_array, axis=0)

# Vermeidung von Division durch Null
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1  # Falls min und max gleich sind, setze den Bereich auf 1 um Division durch Null zu vermeiden

# Normierte Werte berechnen
normalized_strains = (strains_array - min_vals) / range_vals

# In Listen zurück konvertieren
normalized_strains_list = normalized_strains.tolist()
strain_1_norm = normalized_strains_list[0]
strain_2_norm = normalized_strains_list[1]
strain_3_norm = normalized_strains_list[2]
strain_4_norm = normalized_strains_list[3]
strain_5_norm = normalized_strains_list[4]
strain_6_norm = normalized_strains_list[5]
strain_7_norm = normalized_strains_list[6]
strain_8_norm = normalized_strains_list[7]
# Ergebnis
print("normalized_strain_list: ", normalized_strains_list[:][:5])
########


# Listen in Daten-Hauptliste ablegen
#data = [load_pos_x,load_pos_y,load_value,strain_1_norm,strain_2_norm,strain_3_norm,strain_4_norm,strain_5_norm,strain_6_norm,strain_7_norm,strain_8_norm]

# Definition der Sensorpositionen für Plot
sensor_pos = [(-315, 315), (0, 315), (315, 315), (-315, 0), (315, 0), (-315, -315), (0, -315), (315, -315)]

# Positionen der extrahierten Daten visualisieren
h_fn.create_scatterplot(load_pos_x,load_pos_y,sensor_pos)

# Test und Trainingsdaten erstellen
X_train, X_val, X_test, y_train, y_val, y_test = h_fn_ki.prepare_data(
    [strain_1_norm, strain_2_norm, strain_3_norm, strain_4_norm, strain_5_norm, strain_6_norm, strain_7_norm, strain_8_norm],
    load_pos_x, load_pos_y
)

# Ki Modell Trainieren
model = h_fn_ki.train_model(X_train, X_test, y_train, y_test)

# Testing + Visualisierung der Daten
X_sample, y_sample, y_pred = h_fn_ki.test_random_samples(model,X_test,y_test, num_samples=4)

# Extrahiere die x- und y-Werte
x_values_sample = y_sample[:, 0].numpy()
y_values_sample = y_sample[:, 1].numpy()
x_values_pred = y_pred[:, 0].numpy()
y_values_pred = y_pred[:, 1].numpy()

h_fn.create_scatterplot_testing(x_values_sample,x_values_pred,y_values_sample,y_values_pred,sensor_pos)