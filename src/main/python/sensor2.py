# Virtuelle Entwicklung erstellen + aktivieren
# pip3 install adafruit-circuitpython-ads1x15
# pip3 install gpiod

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup
i2cbus = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2cbus, address=0x48)

# Set up analog channels
ch0 = AnalogIn(ads, ADS.P0)
ch1 = AnalogIn(ads, ADS.P1)
ch2 = AnalogIn(ads, ADS.P2)
ch3 = AnalogIn(ads, ADS.P3)

while True:
    #print("Voltage 0: ", round(ch0.voltage, 3))
    #print("Voltage 1: ", round(ch1.voltage, 3))
    #print("Voltage 2: ", round(ch2.voltage, 3))
    print("Voltage 3: ", round(ch3.voltage, 3)) # anschluss 3 an ads1115 board
    time.sleep(0.3)
