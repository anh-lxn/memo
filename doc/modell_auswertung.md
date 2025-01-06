# Dokumentation des Neuronalen Netzes zur Vorhersage von Lasten und Lasteinleitungspositionen

## Modellaufbau

- hier Modellstruktur beschreiben (Anzahl Layer, Aktivierungsfunktion(en), Optimierer, ....)

## Evaluierungsmethoden 

Die Evaluierung des Modells kann an folgenden Metriken durchgeführt werden:

1. **Mean Squared Error (MSE), Mean Absolute Error (MAE), Bestimmtheitsmaß (R²)**:
   - Berechnet für Trainings- und Testdaten, um die allgemeine Leistungsfähigkeit und Überanpassung des Modells zu überwachen.

2. **Untersuchung Under- und Overfitting**:
   - Visualisierung der Test- und Trainingsloss über Epochen, um Konvergenz und Stabilität nachzuweisen
   - Training- und Test-MSE/MAE über die Epochen hinweg visualisieren, um zu erkennen, 
   ob das Modell zu sehr auf den Trainingsdaten angepasst wird und auf den Testdaten schlecht abschneidet (Overfitting)

## Zu untersuchende Parameter

- Anzahl Trainingsdaten
- Modellkomplexität-, struktur
- Learningrate
- Epochenanzahl
- Datenaugmentierung (Datenerweiterung)
- 