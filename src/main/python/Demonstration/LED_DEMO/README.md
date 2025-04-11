# Memo: KI-gestützte Lastpunkt-Berechnung mit Visualisierung über LED-Matrix

Dieses Demonstrator-System nutzt 8 Sensoren, ein trainiertes KI-Modell und eine LED-Matrix zur Echtzeit-Visualisierung von Kraftpunkten auf einer Membran.  
Zwei Raspberry Pis übernehmen jeweils spezifische Rollen:

- Raspberry Pi 5: Führt das KI-Modell aus, berechnet Lastpunkte (x, y) und Kraft (in Newton)
- Raspberry Pi 4: Steuert die LED-Matrix, um die berechneten Punkte visuell darzustellen

## Komponenten

### Raspberry Pi 5 (Recheneinheit und Sensoranbindung)
- Verantwortlich für die Berechnung der Lastpunkte mittels KI
- Verbunden mit einer elektronischen Schaltung und 8 Sensoren auf einer Membran
- Führt das Skript `normalized_live_demo_rgb.py` aus

### Raspberry Pi 4 (LED-Matrix-Controller)
- Steuert die 4x4 LED-Matrix mit insgesamt 16 Panels über eine Daisy-Chain
- Empfängt Positionsdaten vom Pi5 und visualisiert diese auf der Matrix
- Stromversorgung der Matrix erfolgt separat mit 5 V (nicht über den Pi4)
- Stromaufnahme pro Panel: zwischen 0.1 A und max. 4 A, je nach Anzahl und Farbe der Pixel

### Membran mit 8 Sensoren
- Dient als Eingabegerät für das KI-Modell
- Auf diese Membran wurde das Modell trainiert
- Die Sensoren liefern Spannungsveränderungen die zur Kraft- und Punktbestimmung

### LED-Matrix (4x4 Grid)
- Besteht aus 16 Panels, physisch zu einem Gitter verbunden
- Angesteuert über den Pi4 mit entsprechender LED-Hardware

## Start des Demonstrators
Um den Demonstrator erfolgreich zu starten, müssen drei Schritte in der richtigen Reihenfolge ausgeführt werden:

### 1. Kalibrierung der Sensoren (`01_Calibration_of_Sensors.py`)
Dieses Skript zeigt die aktuellen Spannugnswerte der 8 Sensoren an. Vor dem Start des Systems sollten die Sensoren **manuell kalibriert** werden:

- Öffne ein Terminal und starte das Kalibrierungsskript:
  ```
  cd /home/pi/Documents/Memo/git/memo/src/main/python/Demonstration/LED_DEMO
  /home/pi/Documents/Memo/git/memo/src/venv/bin/python 01_Calibration_of_Sensors.py 
  ```
- An der elektronischen Konstruktion befindet sich unten rechts ein kleiner Potentiometer (Widerstand).

- Die Kalibrierung erfolgt durch langsames Drehen mit einem kleinen Schlitzschraubenzieher.

- Die Einstellung ist sehr sensibel – daher vorsichtig justieren, bis alle Sensorwerte korrekt angezeigt werden.

### 2. Start der LED-Matrix auf dem Pi4 (02_Start_LED_Matrix.sh)
Dieses Shell-Skript baut eine SSH-Verbindung vom Pi5 zum Pi4 auf und startet dort das den LED Plot der Matrix

- Im gleichen Terminal ausführen: 
  ```
  ./02_Start_LED_Matrix.sh
  ```
- Sobald das Skript läuft, befindet sich der Pi4 im Wartemodus und wartet auf eingehende Daten vom Pi5.

### 3. Start des KI-Modells auf dem Pi5 (03_Start_Model.sh)
Sobald der Pi4 bereit ist, kann das KI-Modell auf dem Pi5 gestartet werden:

- Im gleichen Terminal ausführen: 
  ```
  ./03_Start_Model.sh
  ```
- Das Skript führt das trainierte Modell aus, berechnet die Lastpunkte (x, y) sowie die Kraft in Newton basierend auf den Sensorwerten und sendet die Ergebnisse an den Pi4 zur Darstellung auf der LED-Matrix.

### Wichtige Hinweise und mögliche Probleme

- Die Skripte müssen in der **korrekten Reihenfolge** gestartet werden, damit die Kommunikation zwischen Pi5 und Pi4 funktioniert:
  1. Zuerst `02_Start_LED_Matrix.sh` (startet die Anzeige auf dem Pi4)
  2. Danach `03_Start_Model.sh` (startet die Berechnung auf dem Pi5)

- **LED-Matrix aktualisiert zu langsam (weniger als 25 Hz):**
  - In diesem Fall den **Pi4 neu starten**.
  - Die normale Bildwiederholrate liegt zwischen **70 und 85 Hz**.

- **SSH-Verbindung zum Pi4 kann nicht aufgebaut werden (bei `02_Start_LED_Matrix.sh`):**
  - Prüfe die **Ethernetverbindung zwischen Pi5 und Pi4**.
  - Stelle sicher, dass beide Geräte korrekt im Netzwerk verbunden sind.

- **LED-Matrix bleibt dunkel:**
  - Prüfen, ob die **separate Stromversorgung** der LED-Panels eingeschaltet ist.