import board                # Für die GPIO- und I2C-Steuerung auf dem Raspberry Pi
import busio                # Für die I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Für die Verwendung des ADS1115 AD-Wandlers
from adafruit_ads1x15.analog_in import AnalogIn  # Für die Spannungsmessung an ADS1115-Kanälen

# ANSI Escape Codes für Farben
RESET = "\033[0m"     # Zum Zurücksetzen auf die Standardfarbe
GREEN = "\033[32m"    # Grüne Schriftfarbe
WHITE = "\033[37m"    # Weiße Schriftfarbe (normal)

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert den I2C-Bus mit den SCL- und SDA-Pins
ads0 = ADS.ADS1115(i2cbus, address=0x4b)  # Erstellt ein Objekt für den ADS1115 mit Adresse 0x48
ads1 = ADS.ADS1115(i2cbus, address=0x49)  # Erstellt ein zweites Objekt für den ADS1115 mit Adresse 0x49

# Einrichtung der analogen Kanäle
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)  # Kanäle des ersten ADS1115
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)  # Kanäle des zweiten ADS1115

# Definiere Toleranzbereiche für jeden Sensor (min, max)
tolerance_ranges = {
    'R1 (A1)': (3.33, 3.53),
    'R2 (B2)': (3.55, 3.75),
    'R3 (B1)': (3.10, 3.30),
    'R4 (A2)': (3.27, 3.47),
    'R5 (C1)': (3.45, 3.65),
    'R6 (C2)': (3.52, 3.72),
    'R7 (D1)': (3.19, 3.39),
    'R8 (D2)': (3.19, 3.39),
}

while True:
    # 1. Auslesen der Sensorwerte über i2C
    sensor_R1 = ch4.voltage  # Zuerst R1, dann die restlichen Sensoren
    sensor_R2 = ch1.voltage
    sensor_R3 = ch2.voltage
    sensor_R4 = ch3.voltage
    sensor_R5 = ch8.voltage
    sensor_R6 = ch7.voltage
    sensor_R7 = ch6.voltage
    sensor_R8 = ch5.voltage

    # Die Sensorwerte in der gewünschten Reihenfolge zuweisen
    sensor_values = [round(sensor_R1, 3), round(sensor_R2, 3), round(sensor_R3, 3), round(sensor_R4, 3), 
                     round(sensor_R5, 3), round(sensor_R6, 3), round(sensor_R7, 3), round(sensor_R8, 3)]

    # Die Sensorbezeichner in der gewünschten Reihenfolge definieren
    sensor_labels = ['R1 (A1)', 'R2 (B2)', 'R3 (B1)', 'R4 (A2)', 'R5 (C1)', 'R6 (C2)', 'R7 (D1)', 'R8 (D2)']

    # Zuordnung der Sensorbezeichner zu den Sensorwerten in der gewünschten Reihenfolge
    sensor_value_mapping = dict(zip(sensor_labels, sensor_values))
    
    # Terminal leeren (Wisch-Effekt)
    print("\033[H\033[J", end="")


    # Konsolenausgabe der Sensorwerte mit Farbanpassung je nach Wert
    for label, value in sensor_value_mapping.items():
        # Holen des Toleranzbereichs für den aktuellen Sensor
        min_value, max_value = tolerance_ranges[label]

        # Standardfarbe ist Weiß
        color = WHITE

        # Wenn der Wert innerhalb des Toleranzbereichs liegt, setze die Farbe auf Grün
        if min_value <= value <= max_value:
            color = GREEN

        # Ausgabe des Sensorwertes mit der entsprechenden Farbe
        print(f"{color}{label}: {value}{RESET}")
