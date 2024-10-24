"""
Dateiname: reading_sensordata_pc.py
Autor: Florian Schmidt (angepasst)
Datum: 05.06.2024 (angepasst am 24.10.2024)

Beschreibung:
Diese Datei dient dem Testen der Funktionalität des Programms auf einem PC. Die Sensordaten werden zufällig generiert
und die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""

# Imports
import os
import threading
import time
import csv
from datetime import datetime
import random

# Funktion zur Generierung zufälliger Spannungswerte (zwischen 0 und 4 Volt)
def generate_random_voltages():
    return [round(random.uniform(0, 4), 3) for _ in range(8)]

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data, output_dir, id, x, y, F):
    output_csv_path = os.path.join(output_dir, f'0{id}_strain_values_{x}_{y}_{F}.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'dt','Voltage1', 'Voltage2', 'Voltage3', 'Voltage4', 'Voltage5', 'Voltage6', 'Voltage7', 'Voltage8'])
        writer.writerows(data)
    print(f'Daten wurden in {output_csv_path} gespeichert.')

# Funktion zum kontinuierlichen Auslesen der Sensordaten
def read_sensors():
    global data, dt
    while not stop_thread:
        curr_voltages = generate_random_voltages()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append([timestamp] + [dt] + curr_voltages)
        time.sleep(dt)

# Funktion zur Eingabe der Parameter über die Kommandozeile
def get_user_inputs():
    id = int(input("Geben Sie die ID des Lastpunkts ein: "))
    x = int(input("Geben Sie die X-Position ein: "))
    y = int(input("Geben Sie die Y-Position ein: "))
    F = int(input("Geben Sie die Kraft (in Newton) ein: "))
    return id, x, y, F

# Hauptprogramm
if __name__ == "__main__":
    # Benutzerparameter abfragen
    id, x, y, F = get_user_inputs()

    # Benutzerdefinierte Variablen
    dt = 0.25  # sleep time between saved data
    data = []

    # Erstelle den Ordner "messungen_aktuelles-datum-zeit"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'../../resources/messungen/messung_pc'
    os.makedirs(output_dir, exist_ok=True)

    # Starte Sensor auslesen (hier Zufallsgenerierung von Zahlen zwischen 4 und 0)
    stop_thread = False
    thread = threading.Thread(target=read_sensors)
    thread.start()

    # Warte auf Benutzereingabe, um das Programm zu beenden
    input("Drücke 'Enter', um das Programm zu beenden und die Daten zu speichern.")
    stop_thread = True  # Beende die Schleife
    thread.join()  # Warte, bis der Thread beendet ist

    # Speichere die Daten, wenn das Programm beendet wird
    save_to_csv(data, output_dir, id, x, y, F)