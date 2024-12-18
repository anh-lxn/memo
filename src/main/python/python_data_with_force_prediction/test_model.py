import helper_fns_data_with_force as h_fn
#import helper_fns_ki_model_with_force_test as h_fn_ki
import helper_fns_ki_model_with_force as h_fn_ki
import numpy as np
import csv
from torch import tensor, stack

# Pfad zur Gesamtdatei angeben
datei_pfad = "../../resources/interpolation/auswertung_gesamt_8N_10N_12N_15N_17N_18N_20N.csv"  # <-- Pfad auf Raspberry anpassen

# Listen zum Speichern der Werte (load_pos_x, load_pos_y sind Positionen der Lasteinleitung, strain_1 bis strain_8
# sind die resultierenden Dehnungswerte der Sensoren 1-8 (Auf Sensoranordnung achten!))
ids, load_pos_x, load_pos_y, load_value, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8 = [], [], [], [], [], [], [], [], [], [], [], []

# Die Datei zeilenweise einlesen
with open(datei_pfad, mode='r', newline='') as csv_datei:
    reader = csv.reader(csv_datei)

    # Kopfzeile extrahieren (erste Zeile)
    kopfzeile = next(reader)

    # Jede weitere Zeile lesen und verarbeiten
    for zeile in reader:
        ids.append(int(zeile[0]))
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
#print("strains_array: ", strains_array)

# Normieren der Werte für jeden Index
min_vals = np.min(strains_array, axis=0)
#print("min_vals: ", min_vals)
max_vals = np.max(strains_array, axis=0)

# Vermeidung von Division durch Null
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1  # Falls min und max gleich sind, setze den Bereich auf 1 um Division durch Null zu vermeiden

# Normierte Werte berechnen
normalized_strains = (strains_array - min_vals) / range_vals
#print("normalized_strains: ", normalized_strains)

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
#print("normalized_strain_list: ", normalized_strains_list[:][:5])
########

# Test und Trainingsdaten erstellen
X_train, X_val, X_test, y_train, y_val, y_test = h_fn_ki.prepare_data(
    [strain_1_norm, strain_2_norm, strain_3_norm, strain_4_norm, strain_5_norm, strain_6_norm, strain_7_norm, strain_8_norm],
    load_pos_x, load_pos_y, load_value
)

# Ki Modell Trainieren
model = h_fn_ki.load_model(path='../../resources/models/model_demonstrator_normalized_12_12_2024_19-10-44.pth')

"""
# Testing + Visualisierung der Daten
X_sample, y_sample, y_pred = h_fn_ki.test_random_samples(model,X_test,y_test, num_samples=1)
# Extrahiere die x- und y-Werte und die Kraft
x_values_sample = y_sample[:, 0].cpu().numpy()
y_values_sample = y_sample[:, 1].cpu().numpy()
F_values_sample = y_sample[:, 2].cpu().numpy()
x_values_pred = y_pred[:, 0].cpu().numpy()
y_values_pred = y_pred[:, 1].cpu().numpy()
F_values_pred = y_pred[:, 2].cpu().numpy()

print(f"x_values_sample: {x_values_sample}")
print(f"x_values_pred: {x_values_pred}")
print(f"y_values_sample: {y_values_sample}")
print(f"y_values_pred: {y_values_pred}")
print(f"F_values_sample: {F_values_sample}")
print(f"F_values_pred: {F_values_pred}")
h_fn.create_scatterplot_testing(x_values_sample,x_values_pred,y_values_sample,y_values_pred,F_values_sample,F_values_pred,sensor_pos)
"""


# Testing
sensor_values_test = stack((tensor(strain_1_norm), tensor(strain_2_norm), tensor(strain_3_norm), tensor(strain_4_norm), tensor(strain_5_norm), tensor(strain_6_norm), tensor(strain_7_norm), tensor(strain_8_norm)), dim=1)
x_y_F_test = stack((tensor(load_pos_x), tensor(load_pos_y), tensor(load_value)), dim=1)
X_sample, y_sample, y_pred = h_fn_ki.test_all_samples(model,sensor_values_test,x_y_F_test)

# Extrahiere die x- und y-Werte und die Kraft
x_values_sample = y_sample[:, 0].cpu().numpy()
y_values_sample = y_sample[:, 1].cpu().numpy()
F_values_sample = y_sample[:, 2].cpu().numpy()
x_values_pred = y_pred[:, 0].cpu().numpy()
y_values_pred = y_pred[:, 1].cpu().numpy()
F_values_pred = y_pred[:, 2].cpu().numpy()

def print_colored_diff(id, real_x, real_y, real_f, pred_x, pred_y, pred_f):
    # Definiere eine Funktion, um Farben je nach Abweichung zu setzen
    def color_diff(real_value, pred_value, is_force=False):
        diff = real_value - pred_value
        # Für die Kraft (F) verwenden wir ein anderes Intervall
        if is_force:
            # Kraft hat ein Intervall von -2 bis 2
            if -2 <= diff <= 2:
                return f"\033[92m{diff:.2f}\033[0m"  # Grün für kleine Abweichung
            else:
                return f"\033[91m{diff:.2f}\033[0m"  # Rot für große Abweichung
        else:
            # Für X und Y verwenden wir das Intervall von -10 bis 10
            if -10 <= diff <= 10:
                return f"\033[92m{diff:.2f}\033[0m"  # Grün für kleine Abweichung
            else:
                return f"\033[91m{diff:.2f}\033[0m"  # Rot für große Abweichung

    # Berechne die farbigen Differenzen für X, Y und F
    x_diff = color_diff(real_x, pred_x)
    y_diff = color_diff(real_y, pred_y)
    f_diff = color_diff(real_f, pred_f, is_force=True)  # Kraft verwendet das Intervall von -2 bis 2

    # Zählen der roten Differenzen
    anz_delta_x = 0
    anz_delta_y = 0
    anz_delta_F = 0

    if "\033[91m" in x_diff:  # Wenn der Unterschied rot ist
        anz_delta_x += 1
    if "\033[91m" in y_diff:  # Wenn der Unterschied rot ist
        anz_delta_y += 1
    if "\033[91m" in f_diff:  # Wenn der Unterschied rot ist
        anz_delta_F += 1

    # Drucke die Ergebnisse im gewünschten Format
    print(f"ID: {id}, Real X: {real_x:.2f}, Real Y: {real_y:.2f}, Real F: {real_f:.2f}, Pred. X: {pred_x:.2f}, Pred. Y: {pred_y:.2f}, Pred. F: {pred_f:.2f} /// X Diff: {x_diff}, Y Diff: {y_diff}, F Diff: {f_diff}")
    
    return anz_delta_x, anz_delta_y, anz_delta_F


# Initialisiere Zähler für alle Differenzen
total_delta_x = 0
total_delta_y = 0
total_delta_F = 0

# Beispiel für die Verwendung mit den extrahierten Werten
for i in range(len(y_values_sample)):
    real_x = x_values_sample[i]
    real_y = y_values_sample[i]
    real_f = F_values_sample[i]
    pred_x = x_values_pred[i]
    pred_y = y_values_pred[i]
    pred_f = F_values_pred[i]
    
    # Fehler zählen
    anz_delta_x, anz_delta_y, anz_delta_F = print_colored_diff(ids[i], real_x, real_y, real_f, pred_x, pred_y, pred_f)

    # Akkumulieren der Differenzen
    total_delta_x += anz_delta_x
    total_delta_y += anz_delta_y
    total_delta_F += anz_delta_F

# Ausgabe der Gesamtzahl der roten Differenzen
print(f"Total deltaX (rot): {total_delta_x}, Fehler: {total_delta_x / len(y_values_sample) * 100:.2f} %")
print(f"Total deltaY (rot): {total_delta_y}, Fehler: {total_delta_y / len(y_values_sample) * 100:.2f} %")
print(f"Total deltaF (rot): {total_delta_F}, Fehler: {total_delta_F / len(y_values_sample) * 100:.2f} %")
