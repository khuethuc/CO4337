#!/usr/bin/env bash
# Clean experiments — FedISIC-2019
# 72 configs (4 optimizers × 2 archs × 3 topologies × 3 world sizes)
# × 3 seeds → 216 runs total
#
# Outputs: outputs_clean_fedisic2019/seed_{SEED}/{run_id}/
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

DATASET=(--data-dir ../data/fedisic2019 --dataset fedisic2019 --classes 8)

OPT_ENGC=(--optimizer engc)
OPT_NGC=(--optimizer ngc)
OPT_CGA=(--optimizer cga)
OPT_MURMURA=(--optimizer murmura)

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
    OUTDIR="outputs_clean_fedisic2019/seed_${SEED}"
    printf '\n\n######################################################################\n'
    printf '##  SEED = %s  |  FedISIC-2019  |  Clean\n' "$SEED"
    printf '######################################################################\n'

    # ── 5 nodes ──────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### FedISIC-2019 | 5 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_ENGC[@]}"
        run_exp ngc     5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_NGC[@]}"
        run_exp cga     5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_CGA[@]}"
        run_exp murmura 5 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2 "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 5 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_ENGC[@]}"
        run_exp ngc     5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_NGC[@]}"
        run_exp cga     5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_CGA[@]}"
        run_exp murmura 5 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 4 "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 5 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_ENGC[@]}"
        run_exp ngc     5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_NGC[@]}"
        run_exp cga     5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_CGA[@]}"
        run_exp murmura 5 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4 "${OPT_MURMURA[@]}"
    done

    # ── 10 nodes ─────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### FedISIC-2019 | 10 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_ENGC[@]}"
        run_exp ngc     10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_NGC[@]}"
        run_exp cga     10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_CGA[@]}"
        run_exp murmura 10 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 10 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_ENGC[@]}"
        run_exp ngc     10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_NGC[@]}"
        run_exp cga     10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_CGA[@]}"
        run_exp murmura 10 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 9  "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 10 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_ENGC[@]}"
        run_exp ngc     10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_NGC[@]}"
        run_exp cga     10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_CGA[@]}"
        run_exp murmura 10 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_MURMURA[@]}"
    done

    # ── 20 nodes ─────────────────────────────────────────────────────────────
    for ARCH in cganet resnet; do
        printf '\n### FedISIC-2019 | 20 nodes | Ring | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_ENGC[@]}"
        run_exp ngc     20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_NGC[@]}"
        run_exp cga     20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_CGA[@]}"
        run_exp murmura 20 "$ARCH" ring  "$SEED" "$OUTDIR" --graph ring  --neighbors 2  "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 20 nodes | Full | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_ENGC[@]}"
        run_exp ngc     20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_NGC[@]}"
        run_exp cga     20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_CGA[@]}"
        run_exp murmura 20 "$ARCH" full  "$SEED" "$OUTDIR" --graph full  --neighbors 19 "${OPT_MURMURA[@]}"

        printf '\n### FedISIC-2019 | 20 nodes | Torus | arch=%s | seed=%s ###\n' "$ARCH" "$SEED"
        run_exp engc    20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_ENGC[@]}"
        run_exp ngc     20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_NGC[@]}"
        run_exp cga     20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_CGA[@]}"
        run_exp murmura 20 "$ARCH" torus "$SEED" "$OUTDIR" --graph torus --neighbors 4  "${OPT_MURMURA[@]}"
    done

done

printf '\n\nAll experiments finished: %s\n' "$(date)"
