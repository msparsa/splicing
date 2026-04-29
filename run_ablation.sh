#!/bin/bash
# SpliceMamba Ablation Experiment Runner
# ======================================
# Varies ONE parameter at a time to identify performance bottlenecks.
# All experiments use: 10 epochs, lr=3e-4, warmup=500, patience=3
#
# Usage:
#   bash run_ablation.sh --tier 1        # Tier 1 only (5 runs, most informative)
#   bash run_ablation.sh --tier 2        # Tier 2 only (4 runs, confirmation)
#   bash run_ablation.sh --tier 3        # Tier 3 only (4 runs, fill gaps)
#   bash run_ablation.sh --tier all      # All tiers (13 runs)

set -euo pipefail

ABLATION_DIR="checkpoints-ablation"
COMMON_ARGS="--max-epochs 10 --lr 3e-4 --warmup-steps 500 --early-stopping-patience 3"

run_experiment() {
    local name="$1"
    local extra_args="$2"
    local ckpt_dir="${ABLATION_DIR}/${name}"

    # Skip if already completed (best.pt exists)
    if [ -f "${ckpt_dir}/best.pt" ]; then
        echo "=== SKIP ${name}: best.pt already exists ==="
        return 0
    fi

    echo "============================================================"
    echo "=== RUNNING: ${name}"
    echo "=== Checkpoint dir: ${ckpt_dir}"
    echo "=== Started: $(date)"
    echo "============================================================"

    # Resume from last.pt if interrupted
    local resume_arg=""
    if [ -f "${ckpt_dir}/last.pt" ]; then
        resume_arg="--resume ${ckpt_dir}/last.pt"
        echo "=== Resuming from ${ckpt_dir}/last.pt ==="
    fi

    python train.py \
        --checkpoint-dir "${ckpt_dir}" \
        --wandb-name "ablation-${name}" \
        ${COMMON_ARGS} \
        ${extra_args} \
        ${resume_arg}

    echo "=== COMPLETED: ${name} at $(date) ==="
    echo ""
}

# Parse arguments
TIER_NUM="1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_ablation.sh --tier [1|2|3|all]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "SpliceMamba Ablation Experiments -- Tier ${TIER_NUM}"
echo "Checkpoint root: ${ABLATION_DIR}/"
echo "Started: $(date)"
echo "============================================================"
echo ""

# --- Baseline under ablation conditions (always runs first) ---
run_experiment "baseline-10ep" ""

# --- Tier 1: Most informative experiments ---
if [[ "$TIER_NUM" == "1" || "$TIER_NUM" == "all" ]]; then
    echo ">>> Starting Tier 1 experiments <<<"
    run_experiment "d_model-128"         "--d-model 128"
    run_experiment "window_radius-1600"  "--window-radius 1600"
    run_experiment "n_mamba-4"           "--n-mamba-layers 4"
    run_experiment "n_attn-8"            "--n-attn-layers 8"
fi

# --- Tier 2: Confirmation experiments ---
if [[ "$TIER_NUM" == "2" || "$TIER_NUM" == "all" ]]; then
    echo ">>> Starting Tier 2 experiments <<<"
    run_experiment "n_mamba-16"          "--n-mamba-layers 16 --micro-batch-size 8 --grad-accum-steps 16"
    run_experiment "d_model-512"         "--d-model 512 --micro-batch-size 8 --grad-accum-steps 16"
    run_experiment "window_radius-200"   "--window-radius 200"
    run_experiment "n_attn-2"            "--n-attn-layers 2"
fi

# --- Tier 3: Fill in the picture ---
if [[ "$TIER_NUM" == "3" || "$TIER_NUM" == "all" ]]; then
    echo ">>> Starting Tier 3 experiments <<<"
    run_experiment "n_mamba-12"          "--n-mamba-layers 12"
    run_experiment "window_radius-800"   "--window-radius 800"
    run_experiment "n_attn-6"            "--n-attn-layers 6"
    run_experiment "d_model-384"         "--d-model 384"
fi

echo ""
echo "============================================================"
echo "ALL REQUESTED ABLATION EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Compare results on W&B (project: splice-mamba, runs: ablation-*)"
echo "  2. Evaluate best checkpoints:"
echo "     for dir in ${ABLATION_DIR}/*/; do"
echo "       name=\$(basename \"\$dir\")"
echo "       python evaluation/evaluate.py --checkpoint \"\${dir}best.pt\" \\"
echo "         --output-dir evaluation/results-ablation/\${name}"
echo "     done"
