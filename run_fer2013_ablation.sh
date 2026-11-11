#!/usr/bin/env bash
set -euo pipefail

# FER+ data path supplied by the user. Passing FER2013Test is supported:
# datasets/mydataset.py resolves the sibling FER2013Train and FER2013Valid.
DATA_ROOT="${1:-/mnt/data/yanyi2025/cyj/ferplus/FER2013Test}"
OUTPUT_ROOT="${2:-/mnt/data/yanyi2025/cyj/MobieleNetMy/ferplus}"
WHICH="${3:-all}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_PATH="${SCRIPT_DIR}/pretrain/model.safetensors"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --nb_classes 7
  --model mobilenetv4_conv_large
  --input-size 224
  --batch-size 64
  --epochs 300
  --num_workers 8
  --mixup 0.8
  --cutmix 1.0
  --lr 0.001
)

run_one() {
  local name="$1"
  local script="$2"
  echo "===== ${name} (${script}) ====="
  local pretrained_args=()
  if [[ "$script" != "m_fast.py" ]]; then
    pretrained_args=(--pretrained_path "$PRETRAINED_PATH")
  fi
  "$PYTHON" "$script" "${COMMON_ARGS[@]}" "${pretrained_args[@]}" \
    --output_dir "${OUTPUT_ROOT}/${name}" \
    --writer_output "${OUTPUT_ROOT}/${name}"
}

case "$WHICH" in
  baseline) run_one baseline m_fast.py ;;
  ca) run_one baseline_ca 'm+ca.py' ;;
  pconv) run_one baseline_pconv '训练代码副本/有效改进点/m+pconv.py' ;;
  shuffleatt) run_one baseline_shuffleatt 'm+shuffleatt.py' ;;
  ca_pconv) run_one baseline_ca_pconv 'm+ca+pconv.py' ;;
  ca_shuffleatt) run_one baseline_ca_shuffleatt 'm+ca+shuffleatt.py' ;;
  pconv_shuffleatt) run_one baseline_pconv_shuffleatt 'm+pconv+shuffleatt.py' ;;
  all)
    run_one baseline m_fast.py
    run_one baseline_ca 'm+ca.py'
    run_one baseline_pconv '训练代码副本/有效改进点/m+pconv.py'
    run_one baseline_shuffleatt 'm+shuffleatt.py'
    run_one baseline_ca_pconv 'm+ca+pconv.py'
    run_one baseline_ca_shuffleatt 'm+ca+shuffleatt.py'
    run_one baseline_pconv_shuffleatt 'm+pconv+shuffleatt.py'
    run_one baseline_ca_pconv_shuffleatt 'm+ca+pconv+shuffleatt.py'
    ;;
  *)
    echo "Usage: $0 [FER2013Test-or-data-root] [output-root] [all|baseline|ca|pconv|shuffleatt|ca_pconv|ca_shuffleatt|pconv_shuffleatt]" >&2
    exit 2
    ;;
esac
