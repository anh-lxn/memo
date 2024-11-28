import time                      # Importiert die Zeitbibliothek, um Zeitstempel für die Messungen zu generieren
import board                     # Importiert die Hardware-Bibliothek, um auf die I2C-Pins des Raspberry Pi oder Mikrocontrollers zuzugreifen
import busio                     # Importiert die Bibliothek für I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Importiert das ADS1115-Modul von Adafruit zur Spannungsmessung
from adafruit_ads1x15.analog_in import AnalogIn  # Importiert die Klasse zur Spannungsmessung an einzelnen Kanälen
import matplotlib.pyplot as plt  # Importiert Matplotlib für die grafische Darstellung der Messdaten
import matplotlib.animation as animation  # Importiert die Animationsfunktion von Matplotlib für das Live-Plotten der Messdaten

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert die I2C-Schnittstelle mit den SCL- und SDA-Pins des Boards
ads0 = ADS.ADS1115(i2cbus, address=0x4b)  # Erstellt ein Objekt für den ADS1115 mit Adresse 0x48
ads1 = ADS.ADS1115(i2cbus, address=0x49)  # Erstellt ein Objekt für das ADS1115-Modul mit der I2C-Adresse 0x49

# Gain (Verstärkung) einstellen
ads0.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Messbereich des ADS1115 mit Adresse 0x48 festzulegen
ads1.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Messbereich des ADS1115 mit der I2C-Adresse 0x49

# Einrichtung der analogen Kanäle
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)  # Kanäle des ersten ADS1115
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)  # Kanäle des zweiten ADS1115

# Plot Setup
fig, ax = plt.subplots()  # Erstellt ein neues Diagramm und Achsenobjekt für die grafische Darstellung
xs = []  # Leere Liste zum Speichern der Zeitwerte
ys_ch1 = []  # Leere Liste für Kanal 1 zum Speichern der Spannungswerte
ys_ch2 = []  # Leere Liste für Kanal 2 zum Speichern der Spannungswerte
ys_ch3 = []  # Leere Liste für Kanal 3 zum Speichern der Spannungswerte
ys_ch4 = []  # Leere Liste für Kanal 4 zum Speichern der Spannungswerte
ys_ch5 = []  # Leere Liste für Kanal 5 zum Speichern der Spannungswerte
ys_ch6 = []  # Leere Liste für Kanal 6 zum Speichern der Spannungswerte
ys_ch7 = []  # Leere Liste für Kanal 7 zum Speichern der Spannungswerte
ys_ch8 = []  # Leere Liste für Kanal 8 zum Speichern der Spannungswerte

def animate(i, xs, ys_ch1, ys_ch2, ys_ch3, ys_ch4, ys_ch5, ys_ch6, ys_ch7, ys_ch8):
    # Spannung an Kanal 3 auslesen
    voltage_ch1 = round(ch1.voltage, 3)  # Liest die Spannung an Kanal 1 und rundet sie auf 3 Dezimalstellen
    voltage_ch2 = round(ch2.voltage, 3)  # Liest die Spannung an Kanal 2 und rundet sie auf 3 Dezimalstellen
    voltage_ch3 = round(ch3.voltage, 3)  # Liest die Spannung an Kanal 3 und rundet sie auf 3 Dezimalstellen
    voltage_ch4 = round(ch4.voltage, 3)  # Liest die Spannung an Kanal 4 und rundet sie auf 3 Dezimalstellen
    voltage_ch5 = round(ch5.voltage, 3)  # Liest die Spannung an Kanal 5 und rundet sie auf 3 Dezimalstellen
    voltage_ch6 = round(ch6.voltage, 3)  # Liest die Spannung an Kanal 6 und rundet sie auf 3 Dezimalstellen
    voltage_ch7 = round(ch7.voltage, 3)  # Liest die Spannung an Kanal 7 und rundet sie auf 3 Dezimalstellen
    voltage_ch8 = round(ch8.voltage, 3)  # Liest die Spannung an Kanal 8 und rundet sie auf 3 Dezimalstellen

    
    # Aktuelle Zeit und Spannung hinzufügen
    xs.append(time.time())  # Fügt den aktuellen Zeitstempel zur xs-Liste hinzu
    ys_ch1.append(voltage_ch1)  # Fügt den Spannungswert von Kanal 1 hinzu
    ys_ch2.append(voltage_ch2)  # Fügt den Spannungswert von Kanal 2 hinzu
    ys_ch3.append(voltage_ch3)  # Fügt den Spannungswert von Kanal 3 hinzu
    ys_ch4.append(voltage_ch4)  # Fügt den Spannungswert von Kanal 4 hinzu
    ys_ch5.append(voltage_ch5)  # Fügt den Spannungswert von Kanal 5 hinzu
    ys_ch6.append(voltage_ch6)  # Fügt den Spannungswert von Kanal 6 hinzu
    ys_ch7.append(voltage_ch7)  # Fügt den Spannungswert von Kanal 7 hinzu
    ys_ch8.append(voltage_ch8)  # Fügt den Spannungswert von Kanal 8 hinzu
	
    # Nur die letzten 50 Messungen anzeigen
    xs = xs[-50:]  # Begrenzen der Liste xs auf die letzten 50 Einträge
    ys_ch1 = ys_ch1[-50:]  # Begrenzen der Liste ys_ch1 auf die letzten 50 Einträge
    ys_ch2 = ys_ch2[-50:]  # Begrenzen der Liste ys_ch2 auf die letzten 50 Einträge
    ys_ch3 = ys_ch3[-50:]  # Begrenzen der Liste ys_ch3 auf die letzten 50 Einträge
    ys_ch4 = ys_ch4[-50:]  # Begrenzen der Liste ys_ch4 auf die letzten 50 Einträge
    ys_ch5 = ys_ch5[-50:]  # Begrenzen der Liste ys_ch5 auf die letzten 50 Einträge
    ys_ch6 = ys_ch6[-50:]  # Begrenzen der Liste ys_ch6 auf die letzten 50 Einträge
    ys_ch7 = ys_ch7[-50:]  # Begrenzen der Liste ys_ch7 auf die letzten 50 Einträge
    ys_ch8 = ys_ch8[-50:]  # Begrenzen der Liste ys_ch8 auf die letzten 50 Einträge

    # Plot aktualisieren
    ax.clear()  # Löscht den aktuellen Inhalt des Diagramms
    ax.plot(xs, ys_ch1, label="Strain 1 : R2")  # Plottet die Werte für Kanal 1
    ax.plot(xs, ys_ch2, label="Strain 2 : R3")  # Plottet die Werte für Kanal 2
    ax.plot(xs, ys_ch3, label="Strain 3 : R4")  # Plottet die Werte für Kanal 3
    ax.plot(xs, ys_ch4, label="Strain 4 : R1")  # Plottet die Werte für Kanal 4
    ax.plot(xs, ys_ch5, label="Strain 5 : R8")  # Plottet die Werte für Kanal 5
    ax.plot(xs, ys_ch6, label="Strain 6 : R7")  # Plottet die Werte für Kanal 6
    ax.plot(xs, ys_ch7, label="Strain 7 : R6")  # Plottet die Werte für Kanal 7
    ax.plot(xs, ys_ch8, label="Strain 8 : R5")  # Plottet die Werte für Kanal 8
    

    plt.xlabel('Zeit (s)')  # Beschriftet die x-Achse
    plt.ylabel('Spannung (V)')  # Beschriftet die y-Achse
    plt.title('Live Plot der Spannung an Kanal 3')  # Setzt den Titel für das Diagramm
    plt.ylim([0, 4])  # Legt den y-Achsenbereich auf 0 bis 4 Volt fest
    plt.legend()  # Fügt eine Legende hinzu, die die Kanäle beschreibt

# Startet die Animation, die alle 100 ms die Funktion 'animate' aufruft
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys_ch1, ys_ch2, ys_ch3, ys_ch4, ys_ch5, ys_ch6, ys_ch7, ys_ch8), interval=100)


# Zeigt das Diagramm an
plt.show()
