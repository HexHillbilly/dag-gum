#!/bin/bash
# Swap the brain corpus, recompile the FAISS index, and restart the proxy.
#
# If the systemd user service (scp-proxy.service) is present, restart it;
# otherwise fall back to the original nohup behavior.

set -euo pipefail

VENV_PY="${VENV_PY:-python3}"
SERVICE="scp-proxy"

# Check if the user provided a file
if [ -z "${1:-}" ]; then
    echo "Usage: ./swap_brain.sh <path_to_new_corpus.txt>"
    exit 1
fi

NEW_CORPUS="$1"

# Check if the file actually exists
if [ ! -f "$NEW_CORPUS" ]; then
    echo "[!] Error: File '$NEW_CORPUS' not found."
    exit 1
fi

echo "[*] 🧠 Swapping brain corpus to: $NEW_CORPUS"
cp "$NEW_CORPUS" brain_corpus.txt

echo "[*] ⚙️  Recompiling the FAISS index..."
if ! "$VENV_PY" compile_brain.py; then
    echo "[!] Engine failed to compile the new index. Aborting restart."
    exit 1
fi

if systemctl --user cat "$SERVICE" >/dev/null 2>&1; then
    echo "[*] 🔄 Restarting the systemd service '$SERVICE'..."
    systemctl --user restart "$SERVICE"
    echo "[*] Engine is online (systemd)."
    systemctl --user status "$SERVICE" --no-pager | head -n 6
else
    echo "[*] 🚀 No systemd service found — igniting the proxy with nohup..."
    nohup "$VENV_PY" proxy.py > proxy.log 2>&1 &
    echo "[*] Engine is online (nohup)."
    sleep 2
    tail -n 10 proxy.log
fi
