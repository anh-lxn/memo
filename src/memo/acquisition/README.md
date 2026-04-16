# Memo Demonstrator

Passwort: memo123
Betrieb 1: - ein aktiver Sensor (standard: R6)
            - alle anderen Sensoren passen sich der Spannungsänderung des Referenzsensors an
            - in diesem Betrieb bitte nur in der Nähe von R6 in der Mitte drücken und nicht bei den anderen Sensoren.

Betrieb 2: - jeder Sensor (R1 bis R8) ist aktiver Sensor, d.h. jeder Spannungswert der angezeigt wird, ist realer Spannungswert!

Kalibriermodus: - diesen aktivieren, wenn der aktive Sensor in Betrieb 1 nicht kalibriert ist, also nicht im Spannungstoleranzbereich liegt!
                - jetzt einfach an den Potis an der Seite drehen bis der Spannungswert des aktiven Sensors den Spannungsbreich erreicht hat

Admin: - hier kann man für Betrieb 1 den Referenzsensor einstellen (standard: R6) -> R5 funktioniert auch sehr gut.

# Starten
Bitte einfach nur die start.sh ausführen auf dem Desktop.

# Probleme
Falls die SD-Karte abschmiert:
- aktuelle Python Version ist 3.13.5
- und dann venv erstellen mit der requirements.txt