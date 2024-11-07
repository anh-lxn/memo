import time                      # Importiert die Zeitbibliothek, um Zeitstempel für die Messungen zu generieren
import board                     # Importiert die Hardware-Bibliothek, um auf die I2C-Pins des Raspberry Pi oder Mikrocontrollers zuzugreifen
import busio                     # Importiert die Bibliothek für I2C-Kommunikation
import adafruit_ads1x15.ads1115 as ADS  # Importiert das ADS1115-Modul von Adafruit zur Spannungsmessung
from adafruit_ads1x15.analog_in import AnalogIn  # Importiert die Klasse zur Spannungsmessung an einzelnen Kanälen
import matplotlib.pyplot as plt  # Importiert Matplotlib für die grafische Darstellung der Messdaten
import matplotlib.animation as animation  # Importiert die Animationsfunktion von Matplotlib für das Live-Plotten der Messdaten

# I2C-Setup
i2cbus = busio.I2C(board.SCL, board.SDA)  # Initialisiert die I2C-Schnittstelle mit den SCL- und SDA-Pins des Boards
ads = ADS.ADS1115(i2cbus, address=0x49)   # Erstellt ein Objekt für das ADS1115-Modul mit der I2C-Adresse 0x49

# Gain (Verstärkung) einstellen
ads.gain = 1  # Setzt den Verstärkungsfaktor auf 1, um den Messbereich des ADS1115 festzulegen

# Kanal 3 als Eingangsquelle festlegen
ch3 = AnalogIn(ads, ADS.P3)  # Erstellt ein Objekt für den Kanal 3 des ADS1115, um die Spannung an diesem Eingang zu messen

# Plot Setup
fig, ax = plt.subplots()  # Erstellt ein neues Diagramm und Achsenobjekt für die grafische Darstellung
xs = []  # Leere Liste zum Speichern der Zeitwerte
ys = []  # Leere Liste zum Speichern der Spannungswerte

def animate(i, xs, ys):
    # Spannung an Kanal 3 auslesen
    voltage = round(ch3.voltage, 3)  # Liest die Spannung an Kanal 3 und rundet sie auf 3 Dezimalstellen

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
    plt.title('Live Plot der Spannung an Kanal 3')  # Setzt den Titel für das Diagramm
    plt.ylim([0, 4])  # Legt den y-Achsenbereich auf 0 bis 4 Volt fest

# Startet die Animation, die alle 50 ms die Funktion 'animate' aufruft
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=50)

# Zeigt das Diagramm an
plt.show()
