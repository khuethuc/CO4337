python trainer.py \
  --data-dir ../data/ham10000 \
  --dataset ham10000 \
  --classes 7 \
  --lr 0.01 \
  --batch-size 160 \
  --world_size 5 \
  --skew 1 \
  --gamma 0.1 \
  --normtype evonorm \
  --optimizer engc \
  --epochs 50 \
  --arch cganet \
  --momentum 0.9 \
  --alpha 1.0 \
  --graph full \
  --neighbors 4 \
  --nesterov \
  --use-edl \
  --quality-mode tiered \
  --noise-rate 0.15 \
  # --noise-type dirichlet \
  # --noise-alpha 0.1


### MURMURA ###
# python trainer.py \
#   --data-dir ../data/ham10000 \
#   --dataset ham10000 \
#   --classes 7 \
#   --lr 0.01 \
#   --batch-size 160 \
#   --world_size 5 \
#   --skew 1 \
#   --gamma 0.1 \
#   --normtype evonorm \
#   --optimizer murmura \
#   --epochs 50 \
#   --arch cganet \
#   --momentum 0.9 \
#   --graph full \
#   --neighbors 4 \
#   --nesterov \
#   --quality-mode tiered \
#   --murmura-self-weight 0.5 \
#   --murmura-vacuity-threshold 0.5 \
#   --murmura-trust-threshold 0.1
