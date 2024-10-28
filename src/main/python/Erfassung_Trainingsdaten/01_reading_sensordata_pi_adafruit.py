# Imports
import os
import csv
from datetime import datetime
import threading
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup
i2cbus = busio.I2C(board.SCL, board.SDA)
ads0 = ADS.ADS1115(i2cbus, address=0x48)
ads1 = ADS.ADS1115(i2cbus, address=0x49)

# Set up analog channels
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)

# Variable zum Stoppen des Threads
stop_event = threading.Event()

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data, output_dir, ids, x, y, F):
    output_csv_path = os.path.join(output_dir, f'0{ids}_strain_values_{x}_{y}_{F}.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'dt', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4', 'Voltage5', 'Voltage6', 'Voltage7', 'Voltage8'])
        writer.writerows(data)
    print(f'Daten wurden in {output_csv_path} gespeichert.')

# Funktion zum kontinuierlichen Auslesen der Sensordaten
def read_sensors():
    while not stop_event.is_set():
        curr_voltages = [ch1.voltage, ch2.voltage, ch3.voltage, ch4.voltage, ch5.voltage, ch6.voltage, ch7.voltage, ch8.voltage]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append([timestamp, dt] + curr_voltages)
        time.sleep(dt)

# Funktion zur Eingabe der Parameter über die Kommandozeile
def get_user_inputs():
    ids = int(input("Geben Sie die ID des Lastpunkts ein: "))
    x = int(input("Geben Sie die X-Position ein: "))
    y = int(input("Geben Sie die Y-Position ein: "))
    F = int(input("Geben Sie die Kraft (in Newton) ein: "))
    return ids, x, y, F

# Hauptprogramm Ausführung
if __name__ == "__main__":
    # Benutzerparameter abfragen
    ids, x, y, F = get_user_inputs()
    dt = 0.25
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
    save_to_csv(data, output_dir, ids, x, y, F)
