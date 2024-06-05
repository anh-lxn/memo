import time
import csv
import keyboard
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from datetime import datetime
import board
import busio

# Initialisierung I2C-Bus und ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)

# Konfiguration des ADC-Kanals
chan = AnalogIn(ads, ADS.P0)

# Liste zur Speicherung der Daten
data = []

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data):
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_sensordaten.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Voltage'])
        writer.writerows(data)
    print(f'Daten wurden in {filename} gespeichert.')

# Abtastrate einstellen (in Sekunden)
sampling_rate = 1  # Ändere dies auf die gewünschte Abtastrate

print("Drücke 'q', um das Programm zu beenden und die Daten zu speichern.")

try:
    while True:
        # Lese Sensordaten aus
        voltage = chan.voltage
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append([timestamp, voltage])

        # Warte bis zur nächsten Abtastung
        time.sleep(sampling_rate)

        # Überprüfe, ob 'q' gedrückt wurde
        if keyboard.is_pressed('q'):
            save_to_csv(data)
            break

except KeyboardInterrupt:
    # Speichere die Daten, wenn das Skript manuell beendet wird
    save_to_csv(data)
