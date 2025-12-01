import time                      # Importiert die Zeitbibliothek, um Zeitstempel für die Messungen zu generieren
import board                     # Importiert die Hardware-Bibliothek, um auf die I2C-Pins des Raspberry Pi oder Mikrocontrollers zuzugreifen
import busio                     # Importiert die Bibliothek für I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Importiert das ADS1115-Modul von Adafruit zur Spannungsmessung
from adafruit_ads1x15.analog_in import AnalogIn  # Importiert die Klasse zur Spannungsmessung an einzelnen Kanälen
import matplotlib.pyplot as plt  # Importiert Matplotlib für die grafische Darstellung der Messdaten
import matplotlib.animation as animation  # Importiert die Animationsfunktion von Matplotlib für das Live-Plotten der Messdaten

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert den I2C-Bus mit den SCL- und SDA-Pins
ads0 = ADS.ADS1115(i2cbus, address=0x4b)  # Erstellt ein Objekt für den ADS1115 mit Adresse 0x48
ads1 = ADS.ADS1115(i2cbus, address=0x49)  # Erstellt ein zweites Objekt für den ADS1115 mit Adresse 0x49

# Einrichtung der analogen Kanäle
ch1, ch2, ch3, ch4 = AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)  # Kanäle des ersten ADS1115
ch5, ch6, ch7, ch8 = AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)  # Kanäle des zweiten ADS1115

#ch1 = R2, #ch2 = R3, #ch3 = R4, #ch4 = R1
#ch5 = R8, #ch6 = R7, #ch7 = R6, #ch8 = R5

# Gain (Verstärkung) einstellen
ads0.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Messbereich des ADS1115 festzulegen
ads1.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Messbereich des ADS1115 festzulegen


# Plot Setup
fig, ax = plt.subplots()  # Erstellt ein neues Diagramm und Achsenobjekt für die grafische Darstellung
xs = []  # Leere Liste zum Speichern der Zeitwerte
ys = []  # Leere Liste zum Speichern der Spannungswertesingle_sensor_liveplot_pi.py


def animate(i, xs, ys):
    # Spannung an Kanal 3 auslesen
    voltage = round(ch8.voltage, 3)  # Liest die Spannung an Kanal 3 und rundet sie auf 3 Dezimalstellen

    # Aktuelle Zeit und Spannung hinzufügen
    xs.append(time.time())  # Fügt den aktuellen Zeitstempel zur xs-Liste hinzu
    ys.append(voltage)      # Fügt den gemessenen Spannungswert zur ys-Liste hinzu

    # Nur die letzten 50 Messungen anzeigen
    xs = xs[-50:]  # Begrenzen der Liste xs auf die letzten 50 Einträge
    ys = ys[-50:]  # Begrenzen der Liste ys auf die letzten 50 Einträge

    # Plot aktualisieren
    ax.clear()  # Löscht den aktuellen Inhalt des Diagramms
    ax.plot(xs, ys)  # Plottet die Werte in xs und ys neu

    plt.xlabel('Zeit (s)')  # Beschriftet die x-Achse
    plt.ylabel('Spannung (V)')  # Beschriftet die y-Achse
    plt.title('Live Plot der Spannung')  # Setzt den Titel für das Diagramm
    plt.ylim([0, 4])  # Legt den y-Achsenbereich auf 0 bis 4 Volt fest

# Startet die Animation, die alle 50 ms die Funktion 'animate' aufruft
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=50)

# Zeigt das Diagramm an
plt.show()
