#!/usr/bin/env bash
# Starts a training run and its external watcher as one unit.
#
# Both halves must exist for a run to be trustworthy. The trainer's own
# EarlyAbort only fires on a metric the trainer computes and only while
# the trainer is alive, so it cannot catch a hang, a kernel deadlock, or
# a kill without a traceback -- all three happened while developing
# Veda2. The watcher catches those from outside. Launching them
# separately means forgetting the second one, so this does both.
#
# Usage:
#   scripts/launch_training.sh <config.yaml> <tag> [extra train.py args...]
#
# Environment:
#   METRIC     metric the watcher follows (default kept_over_ceiling)
#   PATIENCE   updates without a new best before stopping (0 = off)
#   MAX_DROP   fall below the starting value that counts as divergence
#   STALL      seconds of log silence that counts as hung
#   NPROC      processes for torchrun (default 1)
#   PY         interpreter (default .venv/bin/python)
set -u -o pipefail

if [ $# -lt 2 ]; then
  sed -n '1,20p' "$0" >&2
  exit 2
fi
config=$1; tag=$2; shift 2

PY=${PY:-.venv/bin/python}
NPROC=${NPROC:-1}
METRIC=${METRIC:-kept_over_ceiling}
PATIENCE=${PATIENCE:-0}
MAX_DROP=${MAX_DROP:-0.12}
STALL=${STALL:-2400}

mkdir -p runs/logs
train_log=runs/logs/train_${tag}.log
watch_log=runs/logs/watch_${tag}.log
for f in "$train_log" "$watch_log"; do
  # Never silently append to a previous run's numbers: the watcher reads
  # the first update in the file as the baseline for --max-drop.
  if [ -e "$f" ]; then
    mv "$f" "${f%.log}.$(date +%Y%m%d-%H%M%S).log"
  fi
done

nohup "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
  scripts/train.py --config "$config" "$@" > "$train_log" 2>&1 &
train_pid=$!

nohup "$PY" scripts/watch_training.py --log "$train_log" \
  --metric "$METRIC" --patience "$PATIENCE" --max-drop "$MAX_DROP" \
  --stall "$STALL" --kill > "$watch_log" 2>&1 &
watch_pid=$!

echo "train pid $train_pid -> $train_log"
echo "watch pid $watch_pid -> $watch_log (metric $METRIC, patience"\
     "$PATIENCE, max_drop $MAX_DROP, stall ${STALL}s)"
