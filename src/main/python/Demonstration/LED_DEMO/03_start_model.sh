#!/bin/bash

cd /home/pi/Documents/Memo/git/memo/src/main/python/LED-Matrix || exit 1

/home/pi/Documents/Memo/git/memo/src/venv/bin/python normalized_live_demo_rgb.py

echo ""
echo "KI-Model wird beendet."
for i in 3 2 1; do
  echo "$i..."
  sleep 1
done
