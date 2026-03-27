import serial

ser = serial.Serial("COM3", 57600, timeout=0.2)

while True:
    line = ser.readline()   # liest bis \n
    if not line:
        continue

    text = line.decode("ascii", errors="ignore").strip()
    print(repr(text))