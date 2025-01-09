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


# Modellbewertungstabelle (zu untersuchende Parameter können noch gewählt werden) - gefüllt mit Beispielwerten

| Trainingsdaten (%) | Parameterkombination                                                    | MAE (x) | MAE (y) | MAE (Kraft) | RMSE (x) | RMSE (y) | RMSE (Kraft) | R² (x) | R² (y) | R² (Kraft) |
|--------------------|-------------------------------------------------------------------------|---------|---------|-------------|----------|----------|--------------|-------|-------|------------|
| 10                 | LR=0.0001, Batch-Size=16, Optimizer=SDG, Activation-Function=ReLu, .... | 0.05    | 0.06    | 0.07        | 0.08     | 0.09     | 0.10         | 0.85  | 0.87  | 0.88       |
| 50                 | Param1=0.1, Param2=10                                                   | 0.04    | 0.05    | 0.06        | 0.07     | 0.08     | 0.09         | 0.88  | 0.89  | 0.90       |
| 100                | Param1=0.1, Param2=10                                                   | 0.01    | 0.02    | 0.03        | 0.04     | 0.05     | 0.06         | 0.95  | 0.96  | 0.97       |
| 10                 | Param1=0.2, Param2=20                                                   | 0.06    | 0.07    | 0.08        | 0.09     | 0.10     | 0.11         | 0.82  | 0.83  | 0.84       |
| 50                 | Param1=0.2, Param2=20                                                   | 0.05    | 0.06    | 0.07        | 0.08     | 0.09     | 0.10         | 0.85  | 0.86  | 0.87       |
| 100                | Param1=0.2, Param2=20                                                   | 0.02    | 0.03    | 0.04        | 0.05     | 0.06     | 0.07         | 0.93  | 0.94  | 0.95       |

## Metriken zur Modellbewertung

- **R² (Bestimmtheitsmaß)**: Beschreibt, wie gut das Modell die Variation der Zielgröße anhand der Eingabedaten erklärt.
- **MAE (Mean Absolute Error)**: Misst den durchschnittlichen absoluten Unterschied zwischen den tatsächlichen und den vorhergesagten Werten.
- **RMSE (Root Mean Squared Error)**: Gibt die Quadratwurzel des durchschnittlichen quadratischen Fehlers und betont größere Abweichungen stärker als kleinere.
