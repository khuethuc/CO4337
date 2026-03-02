python trainer.py --data-dir ../data/ham10000 --dataset ham10000 \
--lr 0.01 --batch-size 160 \
--world_size 5 --classes 7 \
--skew 1 \
--gamma 0.1 --normtype evonorm \
--optimizer medhengc --epoch 10 \
--arch cganet --momentum 0.9 --alpha 1.0 \
--graph ring --neighbors 2 \
--nesterov \
--sparsity 0.05 --sparsity_warmup 3 --sparsity_ramp 3 --tau_alpha 0.9