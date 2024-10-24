import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# I2C setup
i2cbus = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2cbus, address=0x49)

# Gain einstellen
ads.gain = 1

# Kanal 3 als Eingangsquelle
ch3 = AnalogIn(ads, ADS.P3)

# Plot Setup
fig, ax = plt.subplots()
xs = []
ys = []

def animate(i, xs, ys):
    # Spannung an Kanal 3 auslesen
    voltage = round(ch3.voltage, 3)

    # Aktuelle Zeit und Spannung hinzufügen
    xs.append(time.time())
    ys.append(voltage)

    # Nur die letzten 50 Messungen anzeigen
    xs = xs[-50:]
    ys = ys[-50:]

    # Plot aktualisieren
    ax.clear()
    ax.plot(xs, ys)

    plt.xlabel('Zeit (s)')
    plt.ylabel('Spannung (V)')
    plt.title('Live Plot der Spannung an Kanal 3')
    plt.ylim([0, 4])  # Y-Achsen-Skala festlegen

ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=50)
plt.show()
