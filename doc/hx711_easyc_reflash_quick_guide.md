# HX711 easyC ATTINY404 Reflash Quick Guide

Kurzablauf, um ein einzelnes Soldered HX711 easyC/Qwiic Board neu zu flashen.

## Wichtig

- Geflasht wird der onboard ATTINY404 easyC/I2C-Controller, nicht der HX711.
- Immer nur ein HX711 easyC Board anschliessen.
- Qwiic/I2C zum Raspberry Pi waehrend des Flashens trennen.
- Keine Firmware inhaltlich veraendern.
- Nach jedem Board per I2C testen.

## Verkabelung

```text
Arduino Uno GND  -> HX711 Debug GND
Arduino Uno D6   -> 4.7 kOhm -> HX711 Debug UPDI
Arduino Uno 3.3V -> HX711 Debug 3V3
```

## Vorbereitung

Der Workspace enthaelt nach dem ersten Setup:

```text
tools/arduino-cli/arduino-cli.exe
tools/arduino-cli-config/arduino-cli.yaml
external/jtag2updi/
external/Soldered-HX711-ADC-For-Weight-Scales-Arduino-Library/
```

Arduino Uno Port pruefen:

```powershell
.\venv\Scripts\python.exe -m serial.tools.list_ports -v
```

Bei mir war der Uno `COM3`.

## 1. jtag2updi auf den Uno flashen

Nur noetig, wenn auf dem Uno nicht schon jtag2updi laeuft.

```powershell
.\tools\arduino-cli\arduino-cli.exe --config-file .\tools\arduino-cli-config\arduino-cli.yaml upload -b arduino:avr:uno -p COM3 -i .\external\jtag2updi\build\JTAG2UPDI.hex -v
```

Erfolg sieht ungefaehr so aus:

```text
Device signature = 1E 95 0F (ATmega328P)
bytes of flash written
Avrdude done. Thank you.
```

## 2. Soldered Firmware kompilieren

```powershell
.\tools\arduino-cli\arduino-cli.exe --config-file .\tools\arduino-cli-config\arduino-cli.yaml compile -b "megaTinyCore:megaavr:atxy4:chip=404" --build-path .\build\attiny404_firmware2 .\external\Soldered-HX711-ADC-For-Weight-Scales-Arduino-Library\extras\attiny_firmware
```

Die wichtige Datei ist:

```text
build/attiny404_firmware2/attiny_firmware.ino.hex
```

## 3. ATTINY404 auf dem HX711 easyC Board flashen

Board einzeln anschliessen, dann:

```powershell
.\tools\arduino-cli\arduino-cli.exe --config-file .\tools\arduino-cli-config\arduino-cli.yaml upload -b "megaTinyCore:megaavr:atxy4:chip=404" -p COM3 -P jtag2updi -i .\build\attiny404_firmware2\attiny_firmware.ino.hex -v -t
```

Erfolg sieht ungefaehr so aus:

```text
Device signature = 0x1e9226 (probably t404)
1768 bytes of flash written
1768 bytes of flash verified
avrdude done. Thank you.
```

Hinweis: megaTinyCore fuegt bei diesem Upload automatisch Fuse-Parameter in die avrdude-Commandline ein. Fuer einen reinen Flash ohne Fuse-Writes muesste man avrdude manuell mit nur `-Uflash:w:...` aufrufen.

## 4. Nach dem Flash testen

Board kurz stromlos machen, danach an den Raspberry Pi I2C anschliessen.

```bash
i2cdetect -y -r 1 0x30 0x37
```

Wenn das Board repariert ist, sollte eine Adresse erscheinen, z. B.:

```text
30: 30 -- -- -- -- -- -- --
```

Danach direkten Readout testen, z. B. mit dem vorhandenen Python-Testskript:

```bash
python src/memo/test_cases/hx711_qwiic_minimal.py --address 0x30
```

Adresse entsprechend anpassen.

## Wiederholung fuer weitere Boards

1. Strom ab.
2. Naechstes einzelnes Board an UPDI anschliessen.
3. Flash-Befehl aus Schritt 3 ausfuehren.
4. Board stromlos machen.
5. I2C-Test aus Schritt 4 ausfuehren.
6. Adresse und Ergebnis notieren.
