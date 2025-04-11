#!/bin/bash

# Terminal vom Pi4 (Controller der LED Matrix)

lxterminal -t "Remote LED" -e bash -c '
ssh -t memo@192.200.2.2 "
  cd /home/memo/Documents/remote_rsync/src/main/python/LED-Matrix/
  sudo python led_plot.py
"
echo ""
echo "Terminal schließt sich in:"
for i in 3 2 1; do
  echo "$i..."
  sleep 1
done
' &


