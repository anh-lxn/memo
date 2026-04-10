# Testmodel fuer XYF-Praediktion

Das Skript `train_xyf_model.py` trainiert aus den Sensoren `R1` bis `R8` zwei Regressionsmodelle:

- ein Modell fuer `X` und `Y`
- ein separates Modell fuer `F`

Warum getrennt:

- Die Positionsvorhersage und die Kraftvorhersage haben hier unterschiedliche Fehlercharakteristik.
- In ersten Vergleichen war ein separates Kraftmodell genauer als ein gemeinsames Dreifach-Modell.

Die Daten werden positionsbasiert aufgeteilt:

- Gruppen sind die eindeutigen `(X, Y)`-Koordinaten.
- Dieselbe Position taucht damit nicht gleichzeitig in Train und Test auf.
- So ist die Testauswertung ehrlicher als ein zufaelliger Zeilen-Split.

Ausfuehren:

```powershell
& "C:\Users\anh\Documents\Repositories\memo\venv\Scripts\python.exe" "C:\Users\anh\Documents\Repositories\memo\test\train_xyf_model.py"
```

Artefakte:

- `test/artifacts/xy_extra_trees.joblib`
- `test/artifacts/f_extra_trees.joblib`
- `test/artifacts/test_predictions.csv`
- `test/artifacts/results.json`

Die `tolerance_accuracy_xy` in `results.json` bedeutet:

- Anteil der Testpunkte, bei denen `X` und `Y` jeweils innerhalb von `+/- tolerance` Millimetern liegen.

Eine zweite, staerkere Variante liegt in `train_xyf_model_v2.py`:

- `X,Y`: kNN auf kraftnormalisierten und differenziellen Features
- danach Snap auf das bekannte 40-mm-Messraster
- `F`: ExtraTrees-Regressor auf den Rohsensoren

Artefakte der zweiten Variante:

- `test/artifacts_v2/xy_model.joblib`
- `test/artifacts_v2/f_model.joblib`
- `test/artifacts_v2/test_predictions.csv`
- `test/artifacts_v2/results.json`

Fuer ein reines Innenbereichsmodell gibt es `train_xyf_model_inner.py`:

- filtert standardmaessig auf `max(|X|, |Y|) <= 120 mm`
- entfernt damit die aeusseren Randpunkte
- trainiert nur auf den inneren Messpunkten

Artefakte der Innenbereichsvariante:

- `test/artifacts_inner/xy_model.joblib`
- `test/artifacts_inner/f_model.joblib`
- `test/artifacts_inner/test_predictions.csv`
- `test/artifacts_inner/results.json`
