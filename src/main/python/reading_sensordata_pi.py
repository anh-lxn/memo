"""
Dateiname: reading_sensordata_pi.py
Autor: Florian Schmidt
Datum: 05.06.2024

Beschreibung:
Diese Datei dient dem Aufnehmen von Sensordaten (8 Sensoren) mittels Raspberry Pi und Adafruit 1115 AD-Wandlern.
Die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""

# Imports
import os
import time
import csv
from datetime import datetime
import random
import threading

# User Inputs
id = 33 # id for the position of the load point
x = 290 # x-position of the load point
y = 174 # y-position of the load point
F = 50 # force (load) in Newton
dt = 0.25 # sleep time between saved data

# Liste zur Speicherung der Daten (8 Sensoren)
data = []

# Funktion zum Generieren von 8 zufälligen Werten (zum Testen)
def generate_random_values():
    return [random.uniform(10.0, 12.0) for _ in range(8)]

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data, output_dir):
    output_csv_path = os.path.join(output_dir, f'0{id}_strain_values_{x}_{y}_{F}.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4', 'Voltage5', 'Voltage6', 'Voltage7', 'Voltage8'])
        writer.writerows(data)
    print(f'Daten wurden in {output_csv_path} gespeichert.')

# Funktion zum kontinuierlichen Auslesen der Sensordaten
def read_sensors():
    while not stop_thread:
        voltages = generate_random_values()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append([timestamp] + voltages)
        time.sleep(dt)

# Variable zum Stoppen des Threads
stop_thread = False

# Erstelle den Ordner "messungen_aktuelles-datum-zeit"
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f'../resources/messungen/messung_pi'
os.makedirs(output_dir, exist_ok=True)

# Starten des Sensorlesethreads
thread = threading.Thread(target=read_sensors)
thread.start()

print("Drücke 'Enter', um das Programm zu beenden und die Daten zu speichern.")
input()  # Warten auf Benutzereingabe
stop_thread = True  # Signalisiere dem Thread, dass er stoppen soll
thread.join()  # Warte, bis der Thread beendet ist

# Speichere die Daten, wenn das Programm beendet wird
save_to_csv(data, output_dir)
