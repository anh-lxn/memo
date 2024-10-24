import time
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import board
import busio
import csv

# Definieren von ID, x- und y-Position, Kraft, Zeitintervall und Zeitpunkten
id = 33
x = 290
y = 174
F = 50
dt = 0.25
# Erstellen des Dateipfads basierend auf den oben definierten Parametern
filepath = f'../resources/messungen/messung_pi/0{id}_strain_values_{x}_{y}_{F}.csv'

# Initialisierung I2C-Bus und ADS1115
i2c = busio.I2C(board.SCL, board.SDA)

# Specify i2c Address (0x48, 0x49, 0x4a)
ads_address_0 = 0x4a

# ADS1115 Initialisieren
ads_01 = ADS.ADS1115(i2c, address=ads_address_0)

# Konfiguration des ADC-Kanals
chan = AnalogIn(ads_01, ADS.P0)

print("Messungen beginnen in 3 s...")
time.sleep(2)
# Liste für die Speicherung der Messwerte
measurements = []

# Startzeit setzen
start_time = time.time()

# Messungen für 5 Sekunden durchführen
while time.time() - start_time < 5:
    # Lese Sensordaten aus
    voltage = chan.voltage
    print(f"Spannung: {voltage:.6f} V")

    # Zeitstempel hinzufügen
    timestamp = time.time() - start_time
    measurements.append((timestamp, voltage))

    # Warte 0.25 Sekunden, bevor die nächsten Daten gelesen werden
    time.sleep(dt)

# Daten in CSV-Datei speichern
with open(filepath, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Zeit (s)", "Spannung 1 (V)"])  # Kopfzeile schreiben
    csvwriter.writerows(measurements)

print(f"Messungen abgeschlossen und unter {filepath} gespeichert.")
