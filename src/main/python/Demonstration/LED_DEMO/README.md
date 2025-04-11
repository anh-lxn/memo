# Memo KI-Lastpunkten Berechnung mit Visualisierung über LED-Matrix 

Dieser Demonstrator funktioniert mit einem 8 Sensoren, Pi4 (led controller), pi5 (zum laufen der trainierten KI um Punkstlasten anhand von 8 Sensoren auf der Membran auszurechnen und vorherzusagen)
es werden x, y koordinaten und auch die kraft in Newton ausgerechnet.

## Komponenten

#### pi5
ist der zum ausführen des KI models zur berechnung
#### Pi4
#### Membran mit 8 Sensoren
#### LED-Matrix als 4x4 Grid mit 16 Panels über eine chain verbunden.

## Start des Demonstrators
1. 01_Calibration_of_Sensors.py starten (das zeigt die aktuellen sensorwerte an, diese müssen manuell angepasst werden, es muss an der elektrischen Konstruktion am unteren rechten widerstand gedreht werden, es ist sehr sensitive also langsam drehen, es wird ein kleiner schlitzschraubenzieher benötigt)
2. 02_Start_LED_Matrix.sh starten (startet verbindung vom pi5 zum pi4, also der pi4 lässt sich über den pi5 steuern, dort wird die led matrix gestartet, sobald gestartet wartet der pi4 auf die berehcneten werte vom pi5 über die KI-Model)
3. 03_Start_Model.sh starten (startet auf dem pi5 die KI und berechnet die Lastpunkte anhand der Sensorwerte und sendet diese an den Pi4)
!!! wichtig die dateien 02 und 03 müssen in der richtigen reihenfolge gestartet werden damit es funktioniert!