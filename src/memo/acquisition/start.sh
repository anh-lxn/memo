#!/bin/bash

# In dein Projektverzeichnis wechseln
cd ~/Documents/Repositories/memo || exit

# Virtual Environment aktivieren
source venv/bin/activate

# Python Script starten
python src/memo/acquisition/ui.py