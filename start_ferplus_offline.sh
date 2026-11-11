#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/mnt/data/yanyi2025/cyj/ferplus/fer_img"
OUTPUT_ROOT="/mnt/data/yanyi2025/cyj/MobieleNetMy/ferplus/fer_img_8class"
WHICH="${1:-all}"
GPU_ID="${2:-}"
LOG_ROOT="/mnt/data/yanyi2025/cyj/MobileNetV4/logs"

if [[ -z "$GPU_ID" || ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <experiment> <gpu-id>" >&2
  echo "Example: $0 all 1" >&2
  echo "Experiments: all, baseline, ca, pconv, shuffleatt, ca_pconv, ca_shuffleatt, pconv_shuffleatt, ca_pconv_shuffleatt" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/${WHICH}_gpu${GPU_ID}_${STAMP}.log"
PID_FILE="${LOG_ROOT}/${WHICH}_gpu${GPU_ID}_${STAMP}.pid"

cd "$SCRIPT_DIR"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
  bash "$SCRIPT_DIR/run_fer2013_ablation.sh" \
  "$DATA_ROOT" "$OUTPUT_ROOT" "$WHICH" \
  >"$LOG_FILE" 2>&1 </dev/null &

TRAIN_PID=$!
echo "$TRAIN_PID" >"$PID_FILE"
ln -sfn "$LOG_FILE" "${LOG_ROOT}/latest.log"
ln -sfn "$PID_FILE" "${LOG_ROOT}/latest.pid"

echo "FER+ training started in background."
echo "GPU (physical index): $GPU_ID"
echo "Experiment selection: $WHICH"
echo "PID: $TRAIN_PID"
echo "Log: $LOG_FILE"
echo "Follow log: tail -f '$LOG_FILE'"
echo "Latest log shortcut: tail -f '${LOG_ROOT}/latest.log'"
echo "Check process: ps -fp $TRAIN_PID"
