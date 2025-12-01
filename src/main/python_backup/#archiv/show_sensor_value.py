import time
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import board
import busio

# Initialisierung I2C-Bus und ADS1115
i2c = busio.I2C(board.SCL, board.SDA)

# Adressen auslesen über konsole (Raspberry Pi)
# sudo apt install i2c-tools
# Aktivieren von i2c: sudo raspi-config -> Interfacing Options -> I2C -> aktivieren
# sudo i2cdetect -y 1


# Specify i2c Adress (0x48, 0x49, 0x4a)
ads_address_0 = 0x4a


ads_01 = ADS.ADS1115(i2c, adress=ads_address_0)

# Konfiguration des ADC-Kanals
chan = AnalogIn(ads_01, ADS.P0)

print("Drücke 'Strg+C', um das Programm zu beenden.")

try:
    while True:
        # Lese Sensordaten aus
        voltage = chan.voltage
        print(f"Spannung: {voltage:.6f} V")
        # Warte kurz, bevor die nächsten Daten gelesen werden
        time.sleep(1)  # Ändere dies, um die Abtastrate anzupassen

except KeyboardInterrupt:
    print("Programm beendet.")