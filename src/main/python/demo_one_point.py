# Imports
import matplotlib.pyplot as plt
import torch
import pandas as pd
import helper_fns_ki_model as h_fn_ki

# Laden des KI-Modells
model = h_fn_ki.load_model(path='../resources/models/model_demonstrator_28_11_2024_13-57-41.pth')

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
def calc_loadpoint(sensor_values, model, variant=1):
    # Maximalen Wert in der Liste finden
    max_value = max(sensor_values)

    # Werte normieren
    normalized_sensor_values = [value / max_value for value in sensor_values]

    # Berechnung der total_sum
    total_sum = sum(sensor_values)  # Summiere alle Sensorwerte

    # Definieren eines Thresholds, um Lastpunkt nur zu plotten, wenn gedrückt wird
    if total_sum < 1:
        return 1000, 1000  # Gibt einen "Platzhalter" zurück, wenn die Summe sehr niedrig ist
    else:
        # Beispielhafte Eingabewerte für X_curr
        X_curr = torch.tensor([normalized_sensor_values], dtype=torch.float32)

        # Berechnung der Vorhersage aus dem trainierten KI-Modell
        model.eval()  # Setze das Modell in den Auswertungsmodus
        with torch.no_grad():  # Deaktiviere das Gradienten-Tracking für die Vorhersage
            y_pred = model(X_curr)  # y_pred enthält die vorhergesagten X- und Y-Koordinaten des Lastpunktes

        # Unterschiedliche Varianten von Vorhersagen (hier als Platzhalter für mehrere Modelle oder Ansätze)
        if variant == 1:
            x_value_pred = y_pred[0, 0].item()
            y_value_pred = y_pred[0, 1].item()
        elif variant == 2:
            # Angenommene Modifikation oder alternative Vorhersage-Logik
            x_value_pred = y_pred[0, 0].item() + 5  # Beispiel für eine Veränderung
            y_value_pred = y_pred[0, 1].item() + 5
        elif variant == 3:
            # Eine andere Modifikation oder alternative Berechnung
            x_value_pred = y_pred[0, 0].item() - 5
            y_value_pred = y_pred[0, 1].item() - 5
        else:
            x_value_pred = y_pred[0, 0].item()
            y_value_pred = y_pred[0, 1].item()

    return x_value_pred, y_value_pred


# Listen zum Speichern der Ergebnisse
real_x_values = []
real_y_values = []
pred_x_values_1 = []
pred_y_values_1 = []
pred_x_values_2 = []
pred_y_values_2 = []
pred_x_values_3 = []
pred_y_values_3 = []

# Durchlaufe alle Datei-IDs (angenommen, die Datei-IDs gehen von 0 bis 9)
for file_id in range(60, 61):  # Von 0 bis 9 für Testzwecke
    # Sensorwerte und X/Y-Werte abrufen
    sensor_values, x_load_real, y_load_real = get_sensor_values_by_id(file_id)
    
    if sensor_values:
        # Berechnung der drei vorhergesagten Lastpunkte (mit verschiedenen Varianten)
        x_load_pred_1, y_load_pred_1 = calc_loadpoint(sensor_values, model, variant=1)
        x_load_pred_2, y_load_pred_2 = calc_loadpoint(sensor_values, model, variant=2)
        x_load_pred_3, y_load_pred_3 = calc_loadpoint(sensor_values, model, variant=3)

        # Speichern der realen und vorhergesagten Werte
        real_x_values.append(x_load_real)
        real_y_values.append(y_load_real)
        pred_x_values_1.append(x_load_pred_1)
        pred_y_values_1.append(y_load_pred_1)
        pred_x_values_2.append(x_load_pred_2)
        pred_y_values_2.append(y_load_pred_2)
        pred_x_values_3.append(x_load_pred_3)
        pred_y_values_3.append(y_load_pred_3)

        # Ausgabe der Abweichung (Delta)
        print(f'File_ID {file_id}')
        print(f'real X: {x_load_real}, real Y: {y_load_real}')
        print(f'Predicted 1 - X: {x_load_pred_1}, Y: {y_load_pred_1}')
        print(f'Predicted 2 - X: {x_load_pred_2}, Y: {y_load_pred_2}')
        print(f'Predicted 3 - X: {x_load_pred_3}, Y: {y_load_pred_3}\n')

# Plotten der realen und vorhergesagten Lastpunkte
plt.figure(figsize=(10, 8))
plt.scatter(real_x_values, real_y_values, color='blue', label='Real', alpha=0.6)
plt.scatter(pred_x_values_1, pred_y_values_1, color='red', label='Predicted 1', alpha=0.6)
plt.scatter(pred_x_values_2, pred_y_values_2, color='green', label='Predicted 2', alpha=0.6)
plt.scatter(pred_x_values_3, pred_y_values_3, color='purple', label='Predicted 3', alpha=0.6)

# Diagramm formatieren
plt.title('Real vs Predicted Load Points')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.xlim([-400, 400])
plt.ylim([-400, 400])
plt.legend()
plt.grid(True)
plt.show()
