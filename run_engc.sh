#!/bin/bash

BASE_NAME="engc_cganet_nodes_5_evonorm_lr_0.01_gamma_0.1_alpha_1.0_skew_1_ring"

# ============================================================
# ENGC — Clean setting (uniform quality, no label noise)
# ============================================================
SAVE_BASE_CLEAN="outputs/clean"
OUT_DIR_CLEAN="${SAVE_BASE_CLEAN}/${BASE_NAME}"
mkdir -p "${OUT_DIR_CLEAN}/excel_data"

echo "========== [ENGC] Clean setting =========="
python trainer.py \
  --data-dir ../data/ham10000 --dataset ham10000 --classes 7 \
  --lr 0.01 --batch-size 160 \
  --world_size 5 --skew 1 \
  --gamma 0.1 --normtype evonorm \
  --epochs 100 \
  --optimizer engc \
  --arch cganet \
  --momentum 0.9 \
  --alpha 1.0 \
  --graph ring \
  --neighbors 2 \
  --nesterov \
  --quality-mode uniform \
  --weight_decay 1e-4 \
  --steplr \
  --save-dir "${SAVE_BASE_CLEAN}" \
  2>&1 | tee "${OUT_DIR_CLEAN}/training_log.txt"

echo ""
echo "========== [ENGC] Clean setting DONE =========="
echo ""

# ============================================================
# ENGC — Noisy setting (tiered quality)
#   60% agents (ranks 2,3,4): label noise 15%  [poor tier]
#   20% agents (rank  1    ): label noise  5%  [medium tier]
#   20% agents (rank  0    ): clean             [good tier]
# ============================================================
SAVE_BASE_NOISY="outputs/noisy"
OUT_DIR_NOISY="${SAVE_BASE_NOISY}/${BASE_NAME}"
mkdir -p "${OUT_DIR_NOISY}/excel_data"

echo "========== [ENGC] Noisy setting =========="
python trainer.py \
  --data-dir ../data/ham10000 --dataset ham10000 --classes 7 \
  --lr 0.01 --batch-size 160 \
  --world_size 5 --skew 1 \
  --gamma 0.1 --normtype evonorm \
  --epochs 100 \
  --optimizer engc \
  --arch cganet \
  --momentum 0.9 \
  --alpha 1.0 \
  --graph ring \
  --neighbors 2 \
  --nesterov \
  --quality-mode tiered \
  --weight_decay 1e-4 \
  --steplr \
  --save-dir "${SAVE_BASE_NOISY}" \
  2>&1 | tee "${OUT_DIR_NOISY}/training_log.txt"

echo ""
echo "========== [ENGC] Noisy setting DONE =========="
