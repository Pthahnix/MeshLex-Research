#!/bin/bash
# MeshLex Monitoring Daemon
# Runs disk alert and pipeline monitor on intervals
# Started by: nohup ./scripts/monitor_daemon.sh &

set -e
cd /workspace/MeshLex-Research

DISK_INTERVAL=900    # 15 minutes
MONITOR_INTERVAL=900 # 15 minutes
LOG="/tmp/monitor_daemon.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG"
}

log "Monitor daemon started (PID $$)"

last_disk=0
last_monitor=0

while true; do
    now=$(date +%s)

    # Disk alert: every 15 min at :07, :22, :37, :52
    # Check if we should run (approximate timing)
    if (( now - last_disk >= DISK_INTERVAL )); then
        log "Running disk alert..."
        /usr/bin/python3 scripts/disk_alert.py >> /tmp/disk_alert.log 2>&1 || true
        last_disk=$now
    fi

    # Pipeline monitor: every 15 min at :03, :18, :33, :48
    if (( now - last_monitor >= MONITOR_INTERVAL )); then
        log "Running pipeline monitor..."
        PYTHONPATH=/workspace/MeshLex-Research \
        HF_TOKEN="${HF_TOKEN:-}" \
        /usr/bin/python3 scripts/monitor_pipeline.py >> /tmp/pipeline_monitor.log 2>&1 || true
        last_monitor=$now
    fi

    sleep 60
done
