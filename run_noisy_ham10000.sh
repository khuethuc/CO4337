#!/usr/bin/env bash
# Noisy experiments — HAM10000  (Dirichlet label-noise attack)
# 72 configs (4 optimizers × 2 archs × 3 topologies × 3 world sizes)
# × 3 seeds → 216 runs total
#
# Noise setting: exactly 20% clean / 20% medium / 60% noisy  (Dirichlet label noise, rate=0.4)
# --quality-mode tiered uses tier_size = world_size // 5 → exact 20/20/60 for all world sizes
#   5  nodes → 3 noisy : ranks 2,3,4   (tier_size=1)
#   10 nodes → 6 noisy : ranks 4-9     (tier_size=2)
#   20 nodes → 12 noisy: ranks 8-19    (tier_size=4)
#
# Outputs: outputs_noisy_ham10000/seed_{SEED}/{run_id}/
#          Each run saves: training_log.txt, excel_data/, PNG plots
#
# NOTE: Run scripts in parallel only if different PORT ranges are used.
#       Default starts at 25500 and increments per experiment.

set -u

PORT=25500

COMMON=(
    --lr 0.01
    --batch-size 160
    --skew 1.0
    --gamma 0.1
    --normtype evonorm
    --epochs 100
    --momentum 0.9
    --alpha 1.0
    --nesterov
    --weight_decay 1e-4
    --steplr
)

DATASET=(--data-dir ../data/ham10000 --dataset ham10000 --classes 7)

OPT_ENGC=(--optimizer engc)
OPT_NGC=(--optimizer ngc)
OPT_CGA=(--optimizer cga)
OPT_MURMURA=(--optimizer murmura)

# Noise setup: 60% noisy / 20% medium / 20% clean  (exact for all world sizes)
# tiered now uses tier_size = world_size // 5  →  exact 20/20/60 split
# --noise-agents aligns with the poor tier; --noise-rate overrides their label_noise_rate
#
#  5 nodes  (tier_size=1): clean=[0]     (20%), medium=[1]     (20%), noisy=[2,3,4]    (60%)
# 10 nodes  (tier_size=2): clean=[0,1]   (20%), medium=[2,3]   (20%), noisy=[4-9]      (60%)
# 20 nodes  (tier_size=4): clean=[0-3]   (20%), medium=[4-7]   (20%), noisy=[8-19]     (60%)
NOISE5=(--quality-mode tiered --noise-type dirichlet --noise-agents "2,3,4"                              --noise-rate 0.4)
NOISE10=(--quality-mode tiered --noise-type dirichlet --noise-agents "4,5,6,7,8,9"                      --noise-rate 0.4)
NOISE20=(--quality-mode tiered --noise-type dirichlet --noise-agents "8,9,10,11,12,13,14,15,16,17,18,19" --noise-rate 0.4)

# run_exp <opt_name> <nodes> <arch> <graph> <seed> <outdir> [extra trainer args...]
run_exp() {
    local opt="$1" nodes="$2" arch="$3" graph="$4" seed="$5" outdir="$6"
    shift 6

    local run_id="${opt}_${arch}_nodes_${nodes}_evonorm_lr_0.01_gamma_0.1_alpha_1.0_skew_1.0_${graph}"
    local run_dir="${outdir}/${run_id}"
    mkdir -p "${run_dir}/excel_data"

    printf '\n======================================================================\n'
    printf '  STARTING : %s  (seed=%s)\n' "$run_id" "$seed"
    printf '  Time     : %s\n' "$(date)"
    printf '======================================================================\n\n'

    python3.12 trainer.py "${COMMON[@]}" "${DATASET[@]}" "$@" \
        --arch "$arch" \
        --world_size "$nodes" \
        --seed "$seed" \
        --save-dir "$outdir" \
        --port "$PORT" \
        2>&1 | tee "${run_dir}/training_log.txt"
    local exit_code=${PIPESTATUS[0]}
    PORT=$((PORT + 1))

    if [ "$exit_code" -ne 0 ]; then
        printf '\n  [FAILED] %s  seed=%s  (exit=%d)\n' "$run_id" "$seed" "$exit_code"
    else
        printf '\n  [OK]     %s  seed=%s  | %s\n' "$run_id" "$seed" "$(date)"
    fi
}

for SEED in 42 123 321; do
    OUTDIR="outputs_noisy_ham10000/seed_${SEED}"
    printf '\n\n######################################################################\n'
    printf '##  SEED = %s  |  HAM10000  |  Noisy (Dirichlet 20%%)\n' "$SEED"
    printf '######################################################################\n'

    # ── 5 nodes ──────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### HAM10000 | 5 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_ENGC[@]}"    "${NOISE5[@]}"
        run_exp ngc     5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_NGC[@]}"     "${NOISE5[@]}"
        run_exp cga     5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_CGA[@]}"     "${NOISE5[@]}"
        run_exp murmura 5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_MURMURA[@]}" "${NOISE5[@]}"

        printf '\n### HAM10000 | 5 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_ENGC[@]}"    "${NOISE5[@]}"
        run_exp ngc     5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_NGC[@]}"     "${NOISE5[@]}"
        run_exp cga     5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_CGA[@]}"     "${NOISE5[@]}"
        run_exp murmura 5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_MURMURA[@]}" "${NOISE5[@]}"

        printf '\n### HAM10000 | 5 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_ENGC[@]}"    "${NOISE5[@]}"
        run_exp ngc     5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_NGC[@]}"     "${NOISE5[@]}"
        run_exp cga     5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_CGA[@]}"     "${NOISE5[@]}"
        run_exp murmura 5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_MURMURA[@]}" "${NOISE5[@]}"
    done

    # ── 10 nodes ─────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### HAM10000 | 10 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_ENGC[@]}"    "${NOISE10[@]}"
        run_exp ngc     10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_NGC[@]}"     "${NOISE10[@]}"
        run_exp cga     10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_CGA[@]}"     "${NOISE10[@]}"
        run_exp murmura 10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_MURMURA[@]}" "${NOISE10[@]}"

        printf '\n### HAM10000 | 10 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_ENGC[@]}"    "${NOISE10[@]}"
        run_exp ngc     10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_NGC[@]}"     "${NOISE10[@]}"
        run_exp cga     10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_CGA[@]}"     "${NOISE10[@]}"
        run_exp murmura 10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_MURMURA[@]}" "${NOISE10[@]}"

        printf '\n### HAM10000 | 10 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_ENGC[@]}"    "${NOISE10[@]}"
        run_exp ngc     10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_NGC[@]}"     "${NOISE10[@]}"
        run_exp cga     10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_CGA[@]}"     "${NOISE10[@]}"
        run_exp murmura 10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_MURMURA[@]}" "${NOISE10[@]}"
    done

    # ── 20 nodes ─────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### HAM10000 | 20 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_ENGC[@]}"    "${NOISE20[@]}"
        run_exp ngc     20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_NGC[@]}"     "${NOISE20[@]}"
        run_exp cga     20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_CGA[@]}"     "${NOISE20[@]}"
        run_exp murmura 20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_MURMURA[@]}" "${NOISE20[@]}"

        printf '\n### HAM10000 | 20 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_ENGC[@]}"    "${NOISE20[@]}"
        run_exp ngc     20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_NGC[@]}"     "${NOISE20[@]}"
        run_exp cga     20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_CGA[@]}"     "${NOISE20[@]}"
        run_exp murmura 20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_MURMURA[@]}" "${NOISE20[@]}"

        printf '\n### HAM10000 | 20 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_ENGC[@]}"    "${NOISE20[@]}"
        run_exp ngc     20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_NGC[@]}"     "${NOISE20[@]}"
        run_exp cga     20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_CGA[@]}"     "${NOISE20[@]}"
        run_exp murmura 20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_MURMURA[@]}" "${NOISE20[@]}"
    done

done

printf '\n\nAll experiments finished: %s\n' "$(date)"
