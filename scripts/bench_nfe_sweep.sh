#!/usr/bin/env bash
# Sweep RF inference latency at NFE = 1, 2, 3, 4 on whatever GPU is current.
# Mirrors the LIBERO NFE-sweep latency table (Table R5) so the 4090 column has
# the same NFE rows as the 3070 Ti Laptop column.
#
# Output: tools/results/_cache/latency_synth/latency_rf_paper_nfe{1,2,3,4}.json
#
# Usage (after activating the venv):
#   source .venv/bin/activate
#   bash scripts/bench_nfe_sweep.sh                  # all 4 NFE values
#   bash scripts/bench_nfe_sweep.sh 1 2              # only NFE=1 and NFE=2
#
# Per-run cost on RTX 4090: ~2-4 min (NFE=1 fastest, NFE=4 slowest). Full
# sweep ~12 min.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/checkpoints/flower/libero_10"
CKPT_FILE="${CKPT_DIR}/avg_seq_len=0.90.ckpt"
TRAIN_FOLDER="${CKPT_DIR}/.hydra/config.yaml"
OUT_DIR="${REPO_ROOT}/tools/results/_cache/latency_synth"
mkdir -p "$OUT_DIR"

if [ ! -f "$TRAIN_FOLDER" ]; then
    echo "ERROR: $TRAIN_FOLDER not found" >&2
    exit 1
fi
if [ ! -f "$CKPT_FILE" ]; then
    echo "ERROR: $CKPT_FILE not found" >&2
    exit 1
fi

# Default sweep range; override by passing NFE values as positional args.
if [ "$#" -gt 0 ]; then
    NFE_VALUES=("$@")
else
    NFE_VALUES=(1 2 3 4)
fi

cd "$REPO_ROOT"

for NFE in "${NFE_VALUES[@]}"; do
    OUT_JSON="${OUT_DIR}/latency_rf_paper_nfe${NFE}.json"
    LABEL="rf_paper_nfe${NFE}"
    echo ""
    echo "=========================================="
    echo " RF NFE=${NFE}  ->  $(basename "$OUT_JSON")"
    echo "=========================================="
    python flower/evaluation/bench_inference.py \
        train_folder="'$TRAIN_FOLDER'" \
        checkpoint="'$CKPT_FILE'" \
        eval_cfg_overwrite.model.num_sampling_steps="$NFE" \
        +bench.n_passes=1000 \
        +bench.n_warmup=50 \
        +bench.compile=false \
        +bench.compile_mode=default \
        +bench.out_json="'$OUT_JSON'" \
        +variant_label="$LABEL"
done

echo ""
echo "Sweep complete. JSONs in: $OUT_DIR/"
echo "Quick summary:"
python - <<'PY'
import json
from pathlib import Path
out_dir = Path("tools/results/_cache/latency_synth")
for nfe in (1, 2, 3, 4):
    p = out_dir / f"latency_rf_paper_nfe{nfe}.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    s = d["summary"]
    print(f"  NFE={nfe}  total {s['total']['mean']:6.2f} ± {s['total']['std']:5.2f} ms   "
          f"head {s['head']['mean']:6.2f}   vlm {s['vlm']['mean']:5.2f}   "
          f"(hw={d['hardware']}, runtime_NFE={d.get('n_sampling','?')})")
PY
