#!/usr/bin/env bash
# Reproduce the FLOWER paper's Table 4 inference-efficiency measurement
# on local hardware (RTX 3070 Ti Laptop GPU). The paper's setup:
#   - 1000 measurement passes (vs. our default 200)
#   - bf16, chunk length 50 (we cannot exceed the trained chunk length —
#     LIBERO checkpoints are chunk 10, pretrained CALVIN is chunk 20)
#   - VLM + DiT + head, all decomposed
# See docs/latency_methodology.md for the full reconciliation argument.
#
# Usage:
#   bash scripts/bench_paper_setup.sh rf                    # FLOWER baseline, LIBERO-10, chunk 10
#   bash scripts/bench_paper_setup.sh imf                   # iMF, LIBERO-10, chunk 10
#   bash scripts/bench_paper_setup.sh pretrained_rf         # CALVIN pretrained FLOWER, chunk 20
#   bash scripts/bench_paper_setup.sh rf compile            # same + torch.compile (default mode)
#   bash scripts/bench_paper_setup.sh imf compile reduce-overhead  # custom compile mode
#
# Output: tools/results/_cache/latency_synth/latency_<variant>_paper[_compile].json

set -euo pipefail

VARIANT="${1:-rf}"
COMPILE_ARG="${2:-}"
COMPILE_MODE="${3:-default}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_ROOT="${REPO_ROOT}/checkpoints"
OUT_DIR="${REPO_ROOT}/tools/results/_cache/latency_synth"
mkdir -p "$OUT_DIR"

case "$VARIANT" in
    rf|flower)
        CKPT_DIR="${CKPT_ROOT}/flower/libero_10"
        CKPT_FILE="${CKPT_DIR}/avg_seq_len=0.90.ckpt"
        VARIANT_LABEL="rf_paper"
        ;;
    imf)
        CKPT_DIR="${CKPT_ROOT}/imf/libero_10"
        CKPT_FILE="${CKPT_DIR}/avg_seq_len=0.92.ckpt"
        VARIANT_LABEL="imf_paper"
        ;;
    pretrained_rf)
        CKPT_DIR="${CKPT_ROOT}/pretrained"
        CKPT_FILE="$(find "$CKPT_DIR" -maxdepth 2 -name '*.ckpt' | head -1)"
        VARIANT_LABEL="rf_pretrained_paper"
        ;;
    *)
        echo "Unknown variant: $VARIANT (expected: rf | flower | imf | pretrained_rf)" >&2
        exit 1
        ;;
esac

COMPILE_FLAG="false"
if [[ "$COMPILE_ARG" == "compile" ]]; then
    COMPILE_FLAG="true"
    VARIANT_LABEL="${VARIANT_LABEL}_compile"
fi

TRAIN_FOLDER="${CKPT_DIR}/.hydra/config.yaml"
if [ ! -f "$TRAIN_FOLDER" ]; then
    echo "ERROR: hydra config not found at $TRAIN_FOLDER" >&2
    exit 1
fi
if [ ! -f "$CKPT_FILE" ]; then
    echo "ERROR: checkpoint not found: $CKPT_FILE" >&2
    exit 1
fi

OUT_JSON="${OUT_DIR}/latency_${VARIANT_LABEL}.json"

echo "== Paper-setup inference benchmark =="
echo "  variant      : $VARIANT_LABEL"
echo "  train_folder : $TRAIN_FOLDER"
echo "  checkpoint   : $CKPT_FILE"
echo "  output       : $OUT_JSON"

cd "$REPO_ROOT"
# Quote values that contain '=' so Hydra's override grammar doesn't misparse
# them (LIBERO checkpoint filenames are like "avg_seq_len=0.92.ckpt").
python flower/evaluation/bench_inference.py \
    train_folder="'$TRAIN_FOLDER'" \
    checkpoint="'$CKPT_FILE'" \
    +bench.n_passes=1000 \
    +bench.n_warmup=50 \
    +bench.compile="$COMPILE_FLAG" \
    +bench.compile_mode="$COMPILE_MODE" \
    +bench.out_json="'$OUT_JSON'" \
    +variant_label="$VARIANT_LABEL"
