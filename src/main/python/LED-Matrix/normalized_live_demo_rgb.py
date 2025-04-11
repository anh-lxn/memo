# Imports
import time
import numpy as np
import pandas as pd
import torch
import helper_fns_ki_model_with_force as h_fn_ki
import csv
import socket
import os
import threading
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


# Model laden
model = h_fn_ki.load_model(path='../../resources/models/model_demonstrator_normalized_49501_0.00205_41_better.pth')

# PARAMETER
file_name = f"live_data.csv"
FILE_PATH = f"../../resources/live_sync/{file_name}"

DEST_IP = "192.200.2.2"  # IP-Adresse des Pi 4
PORT = 5001  # Gleicher Port wie auf Pi 4

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
    
    

    sensor_values = [sensor_R1, sensor_R2, sensor_R3, sensor_R4, sensor_R5, sensor_R6, sensor_R7, sensor_R8]
    #sensor_values = [round(sensor_R2, 3), round(sensor_R3, 3), round(sensor_R4, 3), round(sensor_R1, 3), round(sensor_R8, 3), round(sensor_R7, 3), round(sensor_R6, 3), round(sensor_R5, 3)]
    #sensor_values, x_value, y_value, F_value = get_sensor_values_by_id(np.random.randint(0, 10))
    
    # Wenn nicht gedrückt wird, übergebe nicht sichtbaren Punkt
    total_sum = 0 
    for spannung in sensor_values:
       total_sum += spannung
    if total_sum > 25.5:
       return 10000, 10000, 0, sensor_values
    
    #### Normalisierung der Strain Listen
    # Umwandeln in ein numpy array für einfachere Handhabung
    strains_array = np.array(sensor_values)

    # Normieren der Werte für jeden Index
    min_vals = np.min(strains_array, axis=0)
    max_vals = np.max(strains_array, axis=0)

    # Vermeidung von Division durch Null
    range_vals = max_vals - min_vals

    # Normierte Werte berechnen
    normalized_sensor_values = (strains_array - min_vals) / range_vals

    # In Listen zurueck konvertieren
    normalized_sensor_values = normalized_sensor_values.tolist()

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
    F_values_pred = y_pred[:, 2].numpy()

    return x_value_pred[0], y_value_pred[0], F_values_pred[0], sensor_values

# Funktion zum Abrufen der Sensorwerte und X/Y-Werte basierend auf der Datei-ID
def get_sensor_values_by_id(file_id):
    df = pd.read_csv('../../resources/interpolation/auswertung_gesamt_8N_10N_12N_15N_17N_18N_20N.csv')
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
        F_value = row.iloc[0]['F']  # F-Wert
        
        # Rückgabe der Sensorwerte und X, Y als Tupel
        return sensor_values, x_value, y_value, F_value

# Funktion zum kontinuierlichen Senden der berechneten Werte an Pi4
def stream_data_to_pi4():
    """
    Berechnet kontinuierlich Lastdaten mit dem KI-Modell und sendet sie direkt an Pi4.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 👈 WICHTIG
            s.connect((DEST_IP, PORT))
            print(f"📡 Verbunden mit Pi4 ({DEST_IP}:{PORT})")

            while True:
                load_pos_x, load_pos_y, load_value, sensor_values = calc_loadpoint(model)
                data_line = f"{load_pos_x},{load_pos_y},{load_value},{sensor_values[0]},{sensor_values[1]},{sensor_values[2]},{sensor_values[3]},{sensor_values[4]},{sensor_values[5]},{sensor_values[6]},{sensor_values[7]}\n"
                s.sendall(data_line.encode('utf-8'))
                print(f"📤 Gesendet an Pi4: {data_line.strip()}")
                time.sleep(0.01)
    except Exception as e:
        print(f"❌ Verbindung zu Pi4 fehlgeschlagen: {e}")

# Starte das Streaming in einem separaten Thread
stream_thread = threading.Thread(target=stream_data_to_pi4, daemon=True)
stream_thread.start()

# Hauptthread läuft weiter, um `CTRL + C` abzufangen
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🔴 Beenden mit STRG + C erkannt. Skript wird gestoppt.")


  
