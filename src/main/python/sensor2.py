"""
Dateiname: sensor2.py
Autor: Florian Schmidt
Datum: 12.06.2024

Beschreibung:
Diese Datei dient dem Testweisen auslesen von Sensordaten (1 Sensor) mittels Raspberry Pi und Adafruit 1115 AD-Wandler.

Zusätzlich muss der i2c Bus aktiviert werden über:
sudo raspi-config  ---> Interface Options ---> I2C --> aktivieren
sudo i2cdetect -y 1 ausführen, um genutzte i2c Adresse zu ermitteln 0x...

Es müssen für das Ausführen dieser Datei folgende Bibliotheken auf dem Raspberry Pi installiert werden (virtuelle Umgebung)
Virtuelle Entwicklung erstellen + aktivieren
pip3 install adafruit-circuitpython-ads1x15
pip3 install gpiod

Zum Anschließen des ADS1115 Boards an den Raspberry Pi müssen die SCL (Serial Clock) mit dem GPIO3 Pin des Raspberry und der SDA (Serial Data) Pin mit dem GPIO2 Pin des Raspberrys verbunden werden.#
SCL --> GPIO3
SDA --> GPIO2
"""
# Imports
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup
i2cbus = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2cbus, address=0x49)
ads.gain = 1

# Set up analog channels
ch0 = AnalogIn(ads, ADS.P0)
ch1 = AnalogIn(ads, ADS.P1)
ch2 = AnalogIn(ads, ADS.P2)
ch3 = AnalogIn(ads, ADS.P3)

while True:
    #print("Voltage 0: ", round(ch0.voltage, 3))
    #print("Voltage 1: ", round(ch1.voltage, 3))
    #print("Voltage 2: ", round(ch2.voltage, 3))
    print("Voltage 3: ", round(ch3.voltage, 3)) # anschluss A3 an ads1115 board
    time.sleep(0.3)
