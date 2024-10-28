"""
Dateiname: reading_sensordata_pi.py
Autor: Florian Schmidt
Datum: 28.10.2024

Beschreibung:
Diese Datei dient dem Aufnehmen von Sensordaten (8 Sensoren) mittels Raspberry Pi und Adafruit 1115 AD-Wandlern.
Die ausgelesenen Daten werden in einer .csv Datei abgelegt.
"""

# Imports
import os
import csv
from datetime import datetime
import threading

# User Inputs
id = 33 # id for the position of the load point
x = 290 # x-position of the load point
y = 174 # y-position of the load point
F = 50 # force (load) in Newton
dt = 0.25 # sleep time between saved data

# Liste zur Speicherung der Daten (8 Sensoren)
data = []

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup (ads1115 uses following i2c adresses: 0x48, 0x49, 0x4A, 0x4B)
i2cbus = busio.I2C(board.SCL, board.SDA)
ads0 = ADS.ADS1115(i2cbus, address=0x48)
ads1 = ADS.ADS1115(i2cbus, address=0x49)

# Set up analog channels
ch1 = AnalogIn(ads0, ADS.P0) # entspricht Sensor x
ch2 = AnalogIn(ads0, ADS.P1) # entspricht Sensor x
ch3 = AnalogIn(ads0, ADS.P2) # entspricht Sensor x
ch4 = AnalogIn(ads0, ADS.P3) # entspricht Sensor x
ch5 = AnalogIn(ads1, ADS.P0) # entspricht Sensor x
ch6 = AnalogIn(ads1, ADS.P1) # entspricht Sensor x
ch7 = AnalogIn(ads1, ADS.P2) # entspricht Sensor x
ch8 = AnalogIn(ads1, ADS.P3) # entspricht Sensor x


# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data, output_dir, ids, x, y, F):
    output_csv_path = os.path.join(output_dir, f'0{ids}_strain_values_{x}_{y}_{F}.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'dt','Voltage1', 'Voltage2', 'Voltage3', 'Voltage4', 'Voltage5', 'Voltage6', 'Voltage7', 'Voltage8'])
        writer.writerows(data)
    print(f'Daten wurden in {output_csv_path} gespeichert.')

# Funktion zum kontinuierlichen Auslesen der Sensordaten
def read_sensors():
    global dt, ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
    while not stop_thread:
        curr_voltages = [ch1.voltage,ch2.voltage,ch3.voltage,ch4.voltage,ch5.voltage,ch6.voltage,ch7.voltage,ch8.voltage]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append([timestamp, dt] + curr_voltages)
        time.sleep(dt)

# Variable zum Stoppen des Threads
stop_thread = False

# Funktion zur Eingabe der Parameter über die Kommandozeile
def get_user_inputs():
    global ids, x, y, F
    ids = int(input("Geben Sie die ID des Lastpunkts ein: "))
    x = int(input("Geben Sie die X-Position ein: "))
    y = int(input("Geben Sie die Y-Position ein: "))
    F = int(input("Geben Sie die Kraft (in Newton) ein: "))
    return ids, x, y, F

# Hauptprogramm Ausführung
if __name__ == "__main__":
    # Benutzerparameter abfragen
    get_user_inputs()

    # Benutzerdefinierte Variablen
    dt = 0.25  # sleep time between saved data
    data = []

    # Erstelle den Ordner "messungen_aktuelles-datum-zeit"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'../../resources/messungen/messung_{current_time}'
    os.makedirs(output_dir, exist_ok=True)

    # Starte Sensor auslesen
    stop_event.clear()
    thread = threading.Thread(target=read_sensors)
    thread.start()

    # Warte auf Benutzereingabe, um das Programm zu beenden
    input("Drücke 'Enter', um das Programm zu beenden und die Daten zu speichern.")
    stop_event.set()  # Beende die Schleife
    thread.join()  # Warte, bis der Thread beendet ist

    # Speichere die Daten, wenn das Programm beendet wird
    save_to_csv(data, output_dir, id, x, y, F)

