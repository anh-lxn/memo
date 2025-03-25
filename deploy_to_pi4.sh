#!/bin/bash

# Variablen
LOCAL_DIR="/home/pi/Documents/Memo/git/memo/"
REMOTE_USER="memo"
REMOTE_HOST="192.200.2.2"
REMOTE_DIR="/home/memo/Documents/remote_rsync"

echo "📦 Starte Übertragung von $LOCAL_DIR nach $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# Zielverzeichnis auf dem Pi 4 anlegen (falls nicht vorhanden)
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Übertragen mit rsync (ohne .git)
rsync -avz --delete --exclude='.git' "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "✅ Übertragung abgeschlossen."
