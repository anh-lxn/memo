# Imports
import matplotlib.pyplot as plt
import torch
import pandas as pd
import helper_fns_ki_model as h_fn_ki

# Laden des KI-Modells
model = h_fn_ki.load_model(path='../resources/models/model_demonstrator_28_11_2024_13-49-38.pth')

# CSV-Datei einlesen
df = pd.read_csv('../resources/messungen/auswertung/28_11/auswertung_gesamt_2024-11-28_13-37.csv')

# Funktion zum Abrufen der Sensorwerte und X/Y-Werte basierend auf der Datei-ID
def get_sensor_values_by_id(file_id):
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


# Funktion zum Berechnen der Lastkoordinaten
def calc_loadpoint(sensor_values, model):
    # Maximalen Wert in der Liste finden
    max_value = max(sensor_values)

    # Werte normieren
    normalized_sensor_values = [value / max_value for value in sensor_values]

    # Beispielhafte Eingabewerte für X_curr
    X_curr = torch.tensor([normalized_sensor_values], dtype=torch.float32)

    # Berechnung der Vorhersage aus dem trainierten KI-Modell
    model.eval()  # Setze das Modell in den Auswertungsmodus
    with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
        y_pred = model(X_curr)  # y_pred enthält die vorhergesagten X- und Y-Koordinaten des Lastpunktes

    # Predicted Werte
    x_value_pred = y_pred[0, 0].item()
    y_value_pred = y_pred[0, 1].item()

    return x_value_pred, y_value_pred

# Listen zum Speichern der Ergebnisse
real_x_values = []
real_y_values = []
pred_x_values_1 = []
pred_y_values_1 = []

# Durchlaufe alle Datei-IDs (angenommen, die Datei-IDs gehen von 0 bis 9)
for file_id in range(0, 121):  # Von 0 bis 9 für Testzwecke
    # Sensorwerte und X/Y-Werte abrufen
    sensor_values, x_load_real, y_load_real = get_sensor_values_by_id(file_id)
    
    if sensor_values:
        # Berechnung der drei vorhergesagten Lastpunkte (mit verschiedenen Varianten)
        x_load_pred_1, y_load_pred_1 = calc_loadpoint(sensor_values, model)

        # Speichern der realen und vorhergesagten Werte
        real_x_values.append(x_load_real)
        real_y_values.append(y_load_real)
        pred_x_values_1.append(x_load_pred_1)
        pred_y_values_1.append(y_load_pred_1)

        # Ausgabe der Abweichung (Delta)
        print(f'File_ID {file_id}')
        print(f'real X: {x_load_real}, real Y: {y_load_real}')
        print(f'Predicted 1 - X: {x_load_pred_1}, Y: {y_load_pred_1}')

# Plotten der realen und vorhergesagten Lastpunkte
plt.figure(figsize=(10, 8))
plt.scatter(real_x_values, real_y_values, color='blue', label='Real', alpha=0.6)
plt.scatter(pred_x_values_1, pred_y_values_1, color='red', label='Predicted 1', alpha=0.6)

# Diagramm formatieren
plt.title('Real vs Predicted Load Points')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.xlim([-400, 400])
plt.ylim([-400, 400])
plt.legend()
plt.grid(True)
plt.show()
