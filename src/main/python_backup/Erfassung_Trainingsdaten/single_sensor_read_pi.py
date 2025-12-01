"""
Dateiname: single_sensor_read_pi.py
Autor: Florian Schmidt
Datum: 12.06.2024

Beschreibung:
Diese Datei dient dem testweisen Auslesen von Sensordaten (1 Sensor) mittels Raspberry Pi und Adafruit 1115 AD-Wandler.

Zusätzlich muss der I2C-Bus aktiviert werden über:
1. `sudo raspi-config`  ---> Interface Options ---> I2C --> aktivieren
2. `sudo i2cdetect -y 1` ausführen, um die genutzte I2C-Adresse zu ermitteln (z.B. 0x...).

Für das Ausführen dieser Datei müssen folgende Bibliotheken auf dem Raspberry Pi installiert werden (virtuelle Umgebung):
1. Virtuelle Umgebung erstellen und aktivieren
2. `pip3 install adafruit-circuitpython-ads1x15` (Bibliothek für ADS1115)
3. `pip3 install gpiod`

Zum Anschließen des ADS1115-Boards an den Raspberry Pi müssen die Pins wie folgt verbunden werden:
- SCL (Serial Clock) des ADS1115 --> GPIO3 Pin des Raspberry Pi
- SDA (Serial Data) des ADS1115 --> GPIO2 Pin des Raspberry Pi
"""

# Imports
import time                      # Importiert die Zeitbibliothek für Wartezeiten zwischen Messungen
import board                     # Importiert die Hardware-Bibliothek für GPIO und I2C-Pins des Raspberry Pi
import busio                     # Importiert die I2C-Bibliothek für die Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Importiert das ADS1115-Modul zur analogen Spannungsmessung
from adafruit_ads1x15.analog_in import AnalogIn  # Importiert die Klasse zur Kanal-Auswahl für die Spannungsmessung

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert die I2C-Schnittstelle mit den SCL- und SDA-Pins
ads = ADS.ADS1115(i2cbus, address=0x49)   # Erstellt ein Objekt für den ADS1115-AD-Wandler mit der Adresse 0x49
ads.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Spannungsmessbereich festzulegen

# Einrichten der analogen Kanäle
ch0 = AnalogIn(ads, ADS.P0)  # Erstellt ein Objekt für Kanal 0 des ADS1115
ch1 = AnalogIn(ads, ADS.P1)  # Erstellt ein Objekt für Kanal 1 des ADS1115
ch2 = AnalogIn(ads, ADS.P2)  # Erstellt ein Objekt für Kanal 2 des ADS1115
ch3 = AnalogIn(ads, ADS.P3)  # Erstellt ein Objekt für Kanal 3 des ADS1115

# Endlosschleife zum kontinuierlichen Auslesen der Spannung
while True:
    # print("Voltage 0: ", round(ch0.voltage, 3))  # Ausgabe der Spannung am Kanal 0 (auskommentiert)
    # print("Voltage 1: ", round(ch1.voltage, 3))  # Ausgabe der Spannung am Kanal 1 (auskommentiert)
    # print("Voltage 2: ", round(ch2.voltage, 3))  # Ausgabe der Spannung am Kanal 2 (auskommentiert)
    print("Voltage 3: ", round(ch3.voltage, 3))  # Ausgabe der Spannung am Kanal 3 des ADS1115
    time.sleep(0.3)  # Wartezeit von 0,3 Sekunden zwischen den Messungen
