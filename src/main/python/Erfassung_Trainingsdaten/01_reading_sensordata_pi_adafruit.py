"""
Dateiname: 01_reading_sensordata_pi_adafruit.py
Autor: Florian Schmidt
Datum: 07.11.2024

Beschreibung:
Dieses Skript führt ein Auslesen von 8 Sensoren mittels Adafruit ADS1115 AD-Wandlern druch.
Die Werte werden strukturiert in einer .csv Datei abgelegt.
"""

# Imports
import os                   # Zum Arbeiten mit Verzeichnissen und Dateipfaden
import csv                  # Zum Schreiben der Sensordaten in eine CSV-Datei
from datetime import datetime  # Für Zeitstempel und Datumsformatierungen
import threading            # Zum parallelen Ausführen der Sensordaten-Erfassung
import time                 # Zum Hinzufügen von Wartezeiten
import board                # Für die GPIO- und I2C-Steuerung auf dem Raspberry Pi
import busio                # Für die I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Für die Verwendung des ADS1115 AD-Wandlers
from adafruit_ads1x15.analog_in import AnalogIn  # Für die Spannungsmessung an ADS1115-Kanälen

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert den I2C-Bus mit den SCL- und SDA-Pins
ads0 = ADS.ADS1115(i2cbus, address=0x4b)  # Erstellt ein Objekt für den ADS1115 mit Adresse 0x48
ads1 = ADS.ADS1115(i2cbus, address=0x49)  # Erstellt ein zweites Objekt für den ADS1115 mit Adresse 0x49

# Einrichtung der analogen Kanäle
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)  # Kanäle des ersten ADS1115
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)  # Kanäle des zweiten ADS1115

# Sensorzuordnung
sensor_R2 = ch1.voltage
sensor_R3 = ch2.voltage
sensor_R4 = ch3.voltage
sensor_R1 = ch4.voltage
sensor_R8 = ch5.voltage
sensor_R7 = ch6.voltage
sensor_R6 = ch7.voltage
sensor_R5 = ch8.voltage

# Variable zum Stoppen des Threads
stop_event = threading.Event()  # Stop-Event für die Steuerung des Sensor-Lese-Threads

# Funktion zum Speichern der Daten in eine CSV-Datei
def save_to_csv(data, output_dir, ids, x, y, F):
    # Legt den Pfad für die CSV-Datei fest und erstellt die Datei
    output_csv_path = os.path.join(output_dir, f'{ids:03}_strain_values_{x}_{y}_{F}.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)  # CSV-Schreiber-Objekt erstellen
        writer.writerow(['Timestamp', 'dt', 'Voltage1', 'Voltage2', 'Voltage3', 'Voltage4', 'Voltage5', 'Voltage6', 'Voltage7', 'Voltage8'])  # Header schreiben
        writer.writerows(data)  # Alle gesammelten Daten in die Datei schreiben
    print(f'Daten wurden in {output_csv_path} gespeichert.')  # Rückmeldung an den Benutzer

# Funktion zum kontinuierlichen Auslesen der Sensordaten
def read_sensors():
    while not stop_event.is_set():  # Schleife, solange das Stop-Event nicht gesetzt ist
        curr_voltages = [sensor_R1, sensor_R2, sensor_R3, sensor_R4, sensor_R5, sensor_R6, sensor_R7, sensor_R8]  # Spannungen an den Kanälen auslesen
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Zeitstempel generieren
        data.append([timestamp, dt] + curr_voltages)  # Zeitstempel und Spannungsdaten zur Datensammlung hinzufügen
        time.sleep(dt)  # Wartezeit zwischen den Messungen

# Funktion zur Eingabe der Parameter über die Kommandozeile
def get_user_inputs():
    # Definiere feste Werte
    F = 20  # Kraft
    y_start = 290  # Startwert für die Y-Position
    y_value_list = []  # Liste für Y-Werte
    ids_list = []  # Liste für IDs

    # Eingabe der Start-ID und der Y-Position
    id_start = int(input("Geben Sie die Start-ID ein: "))
    x = int(input("Geben Sie die X-Position ein: "))

    # Anzahl der Messpunkte (eine Reihe mit y=const.)
    anz_messwerte = 11  
    abstand = -58

    # Schleife, um IDs und X-Werte hinzuzufügen
    for i in range(anz_messwerte):
        # Füge die aktuelle ID zur Liste hinzu (wird bei jedem Schritt um 1 erhöht)
        ids_list.append(id_start + i)

        # Berechne den X-Wert und füge ihn zur Liste hinzu
        y_value = y_start + i * abstand
        y_value_list.append(y_value)

    return ids_list, y_value_list, x, F, anz_messwerte  # Gibt die Listen und Werte zurück


# Hauptprogramm Ausführung
if __name__ == "__main__":
    # Benutzerparameter abfragen
    ids_list, y_value_list, x, F, anz_messwerte = get_user_inputs()  # Benutzereingaben abrufen
    dt = 0.25  # Zeitintervall für das Lesen der Sensoren

    # Erstelle den Ordner "messungen_aktuelles-datum-zeit"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Aktueller Zeitstempel für den Ordnernamen
    output_dir = f'../../resources/messungen/messung_pi_28_11'  # Pfad für den Output-Ordner
    os.makedirs(output_dir, exist_ok=True)  # Erstelle den Ordner, falls er noch nicht existiert

    for i in range(anz_messwerte): #Schleife zur Messung von einer Reihe (11 Messwerte)
        data = []  # Liste zum Speichern der Sensordaten
        
        # Starte Sensor auslesen
        stop_event.clear()  # Setzt das Stop-Event zurück
        thread = threading.Thread(target=read_sensors)  # Erstellt einen Thread für das kontinuierliche Auslesen der Sensordaten
        thread.start()  # Startet den Sensor-Lese-Thread
        
        #Gibt die aktuellen x, y, und ids aus
        print(f"X: {x}, Y: {y_value_list[i]}, ID: {ids_list[i]}, F: {F}")

        # Warte auf Benutzereingabe, um das Programm zu beenden
        input("Drücke 'Enter', um die Daten zu speichern.")  # Warten auf Eingabe
        stop_event.set()  # Stop-Event setzen, um die Schleife im Sensor-Lese-Thread zu beenden
        thread.join()  # Warten, bis der Thread beendet ist

        # Speichere die Daten, wenn das Programm beendet wird
        save_to_csv(data, output_dir, ids_list[i], x, y_value_list[i], F)  # Speichert die Daten in der CSV-Datei
