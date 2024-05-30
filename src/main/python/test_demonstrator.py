import helper_fns_ki_model as h_fn_ki
import helper_fns_data as h_fn_data
import glob
import numpy as np

# Pfad zu den CSV-Dateien (Auslesen von Dateien beginnend mit Nummerierung 000 bis 999)
file_pattern = '../resources/messdaten/[0-9][0-9][0-9]*.csv'
files = glob.glob(file_pattern)

# Listen zum Speichern der Werte (load_pos_x, load_pos_y sind Positionen der Lasteinleitung, strain_1 bis strain_8
# sind die resultierenden Dehnungswerte der Sensoren 1-8 (Auf Sensoranordnung achten!))
load_pos_x, load_pos_y, load_value, strain_1, strain_2, strain_3, strain_4, strain_5, strain_6, strain_7, strain_8 = [], [], [], [], [], [], [], [], [], [], []

# Daten in Listen schreiben
for file_path in files:
    last_row = h_fn_data.read_last_line(file_path)
    load_value.append(last_row[3])
    load_pos_x.append(last_row[1])
    load_pos_y.append(last_row[2])
    strain_1.append(last_row[4])
    strain_2.append(last_row[5])
    strain_3.append(last_row[6])
    strain_4.append(last_row[7])
    strain_5.append(last_row[8])
    strain_6.append(last_row[9])
    strain_7.append(last_row[10])
    strain_8.append(last_row[11])

# Definition der Sensorpositionen für Plot
sensor_pos = [(-315, 315), (0, 315), (315, 315), (-315, 0), (315, 0), (-315, -315), (0, -315), (315, -315)]

#### Normalisierung der Strain Listen
strains = [strain_1,strain_2,strain_3,strain_4,strain_5,strain_6,strain_7,strain_8]

# Normalisierung der Strain Daten durchführen
normalized_strains_list = h_fn_ki.normalize_strain_data(strains)

strain_1_norm = normalized_strains_list[0]
strain_2_norm = normalized_strains_list[1]
strain_3_norm = normalized_strains_list[2]
strain_4_norm = normalized_strains_list[3]
strain_5_norm = normalized_strains_list[4]
strain_6_norm = normalized_strains_list[5]
strain_7_norm = normalized_strains_list[6]
strain_8_norm = normalized_strains_list[7]
####

# Test und Trainingsdaten erstellen
X_train, X_test, y_train, y_test = h_fn_ki.prepare_data(strain_1_norm, strain_2_norm, strain_3_norm, strain_4_norm, strain_5_norm, strain_6_norm, strain_7_norm, strain_8_norm,
                       load_pos_x, load_pos_y)

model = h_fn_ki.load_model(path='../resources/models/model_demonstrator_v02_normalized.pth')

# Testing + Visualisierung der Daten
X_sample, y_sample, y_pred = h_fn_ki.test_random_samples(model,X_train,y_train, num_samples=10)

# Extrahiere die x- und y-Werte
x_values_sample = y_sample[:, 0].numpy()
y_values_sample = y_sample[:, 1].numpy()
x_values_pred = y_pred[:, 0].numpy()
y_values_pred = y_pred[:, 1].numpy()

h_fn_data.create_scatterplot_testing(x_values_sample,x_values_pred,y_values_sample,y_values_pred,sensor_pos)
