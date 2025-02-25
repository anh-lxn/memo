import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# CSV-Datei laden (ersetze 'deine_datei.csv' durch den Pfad zu deiner Datei)
data = pd.read_csv('gesamt_simulations_auswertung_edit_k=25.csv')

# Features (Sensorwerte) extrahieren: Spalten von 'Strain_1' bis 'Strain_8'
X = data[['Strain_1', 'Strain_2', 'Strain_3', 'Strain_4',
          'Strain_5', 'Strain_6', 'Strain_7', 'Strain_8']].values

# Die tatsächlichen X- und Y-Koordinaten als Zielwerte nehmen
true_coordinates = data[['X', 'Y']].values

# Daten in Trainings- und Testset aufteilen (95% Training, 5% Test)
X_train, X_test, coord_train, coord_test, indices_train, indices_test = train_test_split(
    X, true_coordinates, data.index, test_size=0.1, random_state=50)

# Feature-Skalierung für bessere Leistung des Modells
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Nearest Neighbors Modell mit Manhattan-Distanz
nn_model = NearestNeighbors(n_neighbors=1, metric='manhattan')

# Trainiere das Modell mit den Trainingsdaten
nn_model.fit(X_train)

# Zeitmessung für die Vorhersage
start = time.perf_counter()
# Finde den nächsten Nachbarn für jede Testprobe
distances, nearest_indices = nn_model.kneighbors(X_test)
end = time.perf_counter()

print(f"Laufzeit für NN-Suche: {(end-start)/len(X_test):.6f} Sekunden pro Testdatenset")

# Fehlerberechnung
abs_errors_x = []
abs_errors_y = []
sq_errors_x = []
sq_errors_y = []

# Durchlaufen der Testdaten und Berechnung des Fehlers
for test_idx, nearest_idx in zip(indices_test, nearest_indices):
    nearest_coord = coord_train[nearest_idx[0]]  # Vorhergesagte Koordinate aus Trainingsdaten
    actual_coord = data.iloc[test_idx][['X', 'Y']].values  # Tatsächliche Koordinate

    error_x = nearest_coord[0] - actual_coord[0]
    error_y = nearest_coord[1] - actual_coord[1]

    sq_errors_x.append(error_x ** 2)
    sq_errors_y.append(error_y ** 2)

    abs_errors_x.append(abs(error_x))
    abs_errors_y.append(abs(error_y))


    print(f"Testpunkt Index {test_idx}:")
    print(f" Tatsächliche Koordinate: {actual_coord}")
    print(f" Vorhergesagte Koordinate: {nearest_coord}")
    print(f" Fehler (X, Y): ({error_x:.2f}, {error_y:.2f})")
    print("-" * 80)

# Berechnung des MAE (Mean Absolute Error)
mae_x = np.mean(abs_errors_x)
mae_y = np.mean(abs_errors_y)
mse_x = np.mean(sq_errors_x)
mse_y = np.mean(sq_errors_y)

print(f"MAE_X={mae_x:.2f} mm, MAE_Y={mae_y:.2f} mm")
print(f"MSE_X={mse_x:.2f}, MSE_Y={mse_y:.2f}")

