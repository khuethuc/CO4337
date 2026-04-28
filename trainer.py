import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import argparse
import os
import shutil
import time
import math
from tkinter import E
import numpy as np
import statistics
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from math import ceil
import random
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn

# # # Import custom modules # # #
from gossip import GossipDataParallel
from gossip import RingGraph, GridGraph, FullGraph
from gossip import UniformMixing
from gossip import *
from models import *
from optimizers import *
from dataloader import *

# # # Add arguments # # #
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cganet', help='resnet or vgg or resquant')
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype', default='evonorm', help='none or batchnorm or groupnorm or evonorm')
parser.add_argument('--data-dir', dest='data_dir', help='The directory used to save the trained models', default='../../data', type=str)
parser.add_argument('--dataset', dest='dataset', help='available datasets: cifar10, cifar100, imagenette, ham10000', default='cifar10', type=str)
parser.add_argument('--skew', default=1.0, type=float, help='belongs to [0,1] where 0=completely iid and 1=completely non-iid')
parser.add_argument('--classes', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=160, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, metavar='AR', help='averaging rate')
parser.add_argument('--alpha', default=1.0, type=float, help='NGC mixing weight')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight_decay')
parser.add_argument('-world_size', '--world_size', default=10, type=int, help='total number of nodes')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--optimizer', default='ngc', type=str, help='global optimizer = [d-psgd, cga, ngc, compcga, compngc, topkngc, engc, adaptive_ngc]')
parser.add_argument('--graph', '-g', default='ring', help='graph structure - [ring, torus, full, chain]')
parser.add_argument('--neighbors', default=2, type=int, help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int, help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=321, type=int, help='set seed')
parser.add_argument('--print-freq', '-p', default=100, type=int, help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default='outputs', type=str)
parser.add_argument('--port', dest='port', help='between 3000 to 65000', default='25500', type=str)
parser.add_argument("--steplr", action="store_true", help="Uses step lr scheduler for training.")
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--qgm', action='store_true', help='quasi global momentum')

# --- NGC legacy args (kept for backward compat, unused in ENGC H3) ---
parser.add_argument('--engc-tau-u', dest='engc_tau_u', default=0.5, type=float,
                    help='[legacy] uncertainty threshold for old trust score')
parser.add_argument('--engc-wa', dest='engc_wa', default=0.5, type=float,
                    help='[legacy] accuracy weight for old trust score')
parser.add_argument('--ngc-self-weight', dest='ngc_self_weight', default=0.60, type=float,
                    help='reserved weight for local self-gradient inside each NGC branch')
parser.add_argument('--ngc-score-momentum', dest='ngc_score_momentum', default=0.90, type=float,
                    help='EMA momentum for branch-wise neighbor compatibility scores')
parser.add_argument('--ngc-temperature', dest='ngc_temperature', default=0.20, type=float,
                    help='softmax temperature for neighbor weighting')
parser.add_argument('--ngc-align-weight', dest='ngc_align_weight', default=0.75, type=float,
                    help='weight of cosine-alignment score in soft weighting')
parser.add_argument('--ngc-norm-weight', dest='ngc_norm_weight', default=0.25, type=float,
                    help='weight of norm-agreement score in soft weighting')
parser.add_argument('--ngc-min-peer-weight', dest='ngc_min_peer_weight', default=0.00, type=float,
                    help='minimum peer weight floor after normalization')

# --- EDL / vacuity args (ENGC) ---
parser.add_argument('--use-edl', dest='use_edl', action='store_true',
                    help='enable vacuity-adaptive cross-gradient weighting (ENGC)')
parser.add_argument('--edl-main', dest='edl_main', action='store_true',
                    help='use EDL loss for main local training (not just cross-gradients)')
parser.add_argument('--edl-lambda', dest='edl_lambda', default=1.0, type=float,
                    help='KL weight lambda in EDL loss (applies to both main and cross-gradient loss)')
parser.add_argument('--edl-aux-weight', dest='edl_aux_weight', default=0.0, type=float,
                    help='weight for auxiliary EDL loss added to CE: loss = CE + w * EDL')

# --- MURMURA args ---
parser.add_argument('--murmura-self-weight', dest='murmura_self_weight', default=0.5, type=float,
                    help='MURMURA: weight for own model in aggregation (0=full neighbor, 1=no aggregation)')
parser.add_argument('--murmura-vacuity-threshold', dest='murmura_vacuity_threshold', default=0.5, type=float,
                    help='MURMURA: vacuity above this incurs exponential trust penalty')
parser.add_argument('--murmura-accuracy-weight', dest='murmura_accuracy_weight', default=0.5, type=float,
                    help='MURMURA: weight of accuracy vs baseline in trust score')
parser.add_argument('--murmura-trust-threshold', dest='murmura_trust_threshold', default=0.1, type=float,
                    help='MURMURA: minimum trust to accept a neighbor')
parser.add_argument('--murmura-trust-momentum', dest='murmura_trust_momentum', default=0.7, type=float,
                    help='MURMURA: EMA momentum for trust score smoothing')
parser.add_argument('--murmura-tightening-gamma', dest='murmura_tightening_gamma', default=0.5, type=float,
                    help='MURMURA: initial leniency factor for threshold tightening')
parser.add_argument('--murmura-tightening-kappa', dest='murmura_tightening_kappa', default=1.0, type=float,
                    help='MURMURA: tightening rate (higher = faster tightening)')
parser.add_argument('--murmura-max-eval', dest='murmura_max_eval', default=100, type=int,
                    help='MURMURA: max samples per neighbor for trust evaluation')

# --- Data quality / noise args ---
parser.add_argument('--noise-rate', dest='noise_rate', default=0.0, type=float,
                    help='label noise rate for designated agents (0.0 = no noise)')
parser.add_argument('--noise-agents', dest='noise_agents', default='', type=str,
                    help='comma-separated ranks to inject noise, e.g. "0,1"')
parser.add_argument('--quality-mode', dest='quality_mode', default='uniform', type=str,
                    help='heterogeneous data quality: uniform | tiered | random')
parser.add_argument('--noise-type', dest='noise_type', default='uniform', type=str,
                    help='label noise distribution: uniform (USN) | dirichlet')
parser.add_argument('--noise-alpha', dest='noise_alpha', default=0.1, type=float,
                    help='Dirichlet concentration for label noise: '
                         'small (0.01) = near pair-flip; large (10) = near uniform')

args = parser.parse_args()
args.devices = torch.cuda.device_count()

# # # Check the save_dir exists or not # # #
args.save_dir = os.path.join(
    args.save_dir,
    args.optimizer + "_" + args.arch + "_nodes_" + str(args.world_size) + "_"
    + args.normtype + "_lr_" + str(args.lr) + "_gamma_" + str(args.gamma)
    + "_alpha_" + str(args.alpha) + "_skew_" + str(args.skew) + "_" + args.graph
)
if not os.path.exists(os.path.join(args.save_dir, "excel_data")):
    os.makedirs(os.path.join(args.save_dir, "excel_data"))
torch.save(args, os.path.join(args.save_dir, "training_args.bin"))


# # # Run training # # #
def run(rank, size):
    global args, best_prec1, global_steps
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    best_prec1 = 0
    data_transferred = 0
    global_steps = 0

    # Accuracy and loss lists
    train_acc_list  = []
    train_loss_list = []
    val_acc_list    = []
    val_loss_list   = []

    # Per-epoch resource / communication tracking
    train_time_list  = []   # seconds per epoch
    cpu_pct_list     = []   # %CPU per epoch
    gpu_pct_list     = []   # %GPU per epoch
    comm_bytes_list  = []   # bytes transferred per epoch
    comm_pkgs_list   = []   # total comm packages per epoch

    # ENGC per-epoch diagnostic tracking (empty for non-ENGC optimizers)
    # engc_trust_list[ep][r] = mean weight assigned to neighbor r in epoch ep
    engc_trust_list = []

    # --- Build base model ---
    if args.arch.lower() == 'resnet':
        base_model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'vgg11':
        base_model = vgg11(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'mobilenet':
        base_model = MobileNetV2(num_classes=args.classes, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'cganet':
        base_model = cganet5(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'lenet5':
        base_model = LeNet5()
    else:
        raise NotImplementedError

    if rank == 0:
        print(args)
        print('Printing model summary...')
        if args.dataset == "fmnist":
            print(summary(base_model, (1, 28, 28), batch_size=int(args.batch_size / size), device='cpu'))
        elif args.dataset in ["imagenette_full", "imagenet"]:
            print(summary(base_model, (3, 224, 224), batch_size=int(args.batch_size / size), device='cpu'))
        else:
            print(summary(base_model, (3, 32, 32), batch_size=int(args.batch_size / size), device='cpu'))

    # --- Data loading ---
    train_loader, bsz_train = partition_trainDataset(
        args.dataset, args.data_dir, args.skew, args.seed, args.batch_size,
        args.classes,
        quality_mode=args.quality_mode,
        noise_rate=args.noise_rate,
        noise_type=args.noise_type,
        noise_alpha=args.noise_alpha,
    )

    if rank == 0:
        from dataloader import make_quality_profiles
        profiles = make_quality_profiles(size, mode=args.quality_mode, seed=args.seed)
        print(f"\n[QualityMode={args.quality_mode}] Data quality profiles:")
        for r, p in enumerate(profiles):
            print(f"  Rank {r}: {p}")

    val_loader, bsz_val = test_Dataset(args.dataset, args.data_dir, seed=args.seed)

    check_noniid(train_loader, rank, args.world_size)

    local_class_weights = compute_local_class_weights(
        train_loader=train_loader,
        num_classes=args.classes,
        device=device,
    )
    if rank == 0:
        print(f"[Info] local_class_weights rank {rank}: {local_class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss().to(device)

    # --- Build sender ---
    if args.optimizer.lower() == 'cga':
        sender = CGA_sender(base_model, device)
    elif args.optimizer.lower() == 'ngc':
        sender = NGC_sender(base_model, device)
    elif args.optimizer.lower() == 'compcga':
        sender = CompCGA_sender(base_model, device)
    elif args.optimizer.lower() == 'compngc':
        sender = CompNGC_sender(base_model, device)
    elif args.optimizer.lower() == 'topkngc':
        sender = Topk_NGC_sender(base_model, device)
    elif args.optimizer.lower() == 'engc':
        sender = ENGC_sender(
            base_model,
            device,
            num_classes=args.classes,
            edl_lambda=args.edl_lambda,
        )
    elif args.optimizer.lower() == 'adaptive_ngc':
        sender = Adaptive_NGC_sender(base_model, device)
    elif args.optimizer.lower() == 'murmura':
        sender = MURMURA_sender(
            base_model, device,
            num_classes=args.classes,
            vacuity_threshold=args.murmura_vacuity_threshold,
            accuracy_weight=args.murmura_accuracy_weight,
            trust_momentum=args.murmura_trust_momentum,
            max_eval_samples=args.murmura_max_eval,
        )
    else:
        sender = None

    # --- Build graph and gossip model ---
    if args.graph.lower() == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    elif args.graph.lower() == 'torus':
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size - 1)
    elif args.graph.lower() == 'chain':
        graph = ChainGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    else:
        raise NotImplementedError

    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(
        base_model,
        device_ids=[rank],
        rank=rank,
        world_size=size,
        graph=graph,
        mixing=mixing,
        comm_device=device,
        level=32,
        biased=False,
        eta=args.gamma,
        compress_ratio=0.0,
        compress_fn='quantize',
        compress_op='top_k',
        momentum=args.momentum,
        lr=args.lr,
    )
    model.to(device)

    # --- Build receiver ---
    if args.optimizer.lower() == 'cga':
        receiver = CGA_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower() == 'compcga':
        receiver = CompCGA_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower() == 'ngc':
        receiver = NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha=args.alpha)
    elif args.optimizer.lower() == 'compngc':
        receiver = CompNGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha=args.alpha)
    elif args.optimizer.lower() == 'topkngc':
        receiver = Topk_NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha=args.alpha)
    elif args.optimizer.lower() == 'engc':
        # ENGC Hướng 3: receiver uniform averaging giống NGC, không cần trust score
        receiver = ENGC_receiver(
            model,
            device,
            rank,
            args.lr,
            args.momentum,
            args.qgm,
            args.nesterov,
            weight_decay=args.weight_decay,
            neighbors=args.neighbors,
            alpha=args.alpha,
        )
    elif args.optimizer.lower() == 'adaptive_ngc':
        receiver = Adaptive_NGC_receiver(
            model,
            device,
            rank,
            args.lr,
            args.momentum,
            args.qgm,
            args.nesterov,
            weight_decay=args.weight_decay,
            neighbors=args.neighbors,
            alpha=args.alpha,
        )
    elif args.optimizer.lower() == 'murmura':
        receiver = MURMURA_receiver(
            model, device, rank,
            args.lr, args.momentum, args.qgm, args.nesterov,
            weight_decay=args.weight_decay,
            neighbors=args.neighbors,
            self_weight=args.murmura_self_weight,
            trust_threshold=args.murmura_trust_threshold,
            tightening_gamma=args.murmura_tightening_gamma,
            tightening_kappa=args.murmura_tightening_kappa,
            total_rounds=args.epochs,
        )
    else:
        receiver = DSGD_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay)

    optimizer = optim.SGD(model.parameters(), args.lr)

    if args.steplr:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.981, step_size=1)
    else:
        if args.dataset == 'imagenet':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, gamma=0.1,
                milestones=[int(args.epochs * 0.33), int(args.epochs * 0.67), int(args.epochs * 0.89)]
            )
        else:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, gamma=0.1,
                milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)]
            )

    # --- Training loop ---
    for epoch in range(0, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()

        dt, prec1, loss, m = train(
            train_loader      = train_loader,
            val_loader        = val_loader,
            model             = model,
            criterion         = criterion,
            optimizer         = optimizer,
            epoch             = epoch,
            batch_size        = bsz_train,
            lr                = optimizer.param_groups[0]['lr'],
            device            = device,
            receiver          = receiver,
            sender            = sender,
            gpu_index         = rank,
            monitor_every     = max(10, args.print_freq),
        )
        data_transferred += dt

        train_acc_list.append(float(prec1))
        train_loss_list.append(float(loss))

        train_time_list.append(float(m["train_time_s"]))
        cpu_pct_list.append(float(m["cpu_pct_avg"]))
        gpu_pct_list.append(float(m["gpu_pct_avg"]))
        comm_bytes_list.append(int(m["payload_bytes_from_calls"]))
        comm_pkgs_list.append(
            int(m["comm_calls_transfer_params"]) +
            int(m["comm_calls_transfer_additional"])
        )

        # ENGC per-epoch diagnostics
        engc_trust_list.append(m.get("engc_trust", {}))

        lr_scheduler.step()

        prec1, loss = validate(val_loader, model, criterion, bsz_val, device, epoch)
        is_best    = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}.th'.format(rank)))

        val_acc_list.append(float(prec1))
        val_loss_list.append(float(loss))

        if rank == 0:
            t_loss = train_loss_list[-1]
            v_loss = val_loss_list[-1]
            t_acc  = train_acc_list[-1]
            gap    = v_loss - t_loss
            print(
                f"[Loss][Epoch {epoch:>3}] "
                f"Train={t_loss:.4f}  Val={v_loss:.4f}  "
                f"Gap={gap:+.4f}  TrainAcc={t_acc:.2f}%  ValAcc={prec1:.2f}%"
            )

    average_parameters(model)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val, device, epoch)

    # --- ENGC: print per-epoch trust score evolution (rank 0 only) ---
    if args.optimizer.lower() == 'engc' and rank == 0 and engc_trust_list:
        noise_ranks = set(
            int(x) for x in args.noise_agents.split(',') if x.strip()
        ) if args.noise_agents else set()

        # Collect all neighbor ranks that appeared
        all_nbr_ranks = sorted({
            r for ep_d in engc_trust_list for r in ep_d if r != 'self'
        })

        W = 72
        print("\n" + "=" * W)
        print(f"  ENGC DIAGNOSTIC SUMMARY  [Rank {rank} | {args.graph} | skew={args.skew}]")
        print(f"  Noisy agents: {sorted(noise_ranks) if noise_ranks else 'none'}")
        print("=" * W)

        # --- Table 1: per-epoch trust score per neighbor ---
        hdr = f"  {'Epoch':>5}"
        hdr += f"  {'self':>6}"
        for r in all_nbr_ranks:
            tag = '[N]' if r in noise_ranks else '[c]'
            hdr += f"  peer{r}{tag:>3}"
        print(hdr)
        print("  " + "-" * (W - 2))
        for ep, ep_trust in enumerate(engc_trust_list):
            row = f"  {ep+1:>5}  {ep_trust.get('self', float('nan')):>6.3f}"
            for r in all_nbr_ranks:
                row += f"  {ep_trust.get(r, float('nan')):>9.3f}"
            print(row)
        print("=" * W)

        print("=" * W)

        # --- Table 3: per-epoch trust scores with checklist ---
        # PASS: noisy peers [N] have lower final weight than clean peers [c]
        # FAIL: weights flat / noisy peers not downweighted
        print(f"\n  Trust score evolution — noisy peers [N] should trend lower than [c]")
        hdr3 = f"  {'Epoch':>5}  {'self':>6}"
        for r in all_nbr_ranks:
            tag = '[N]' if r in noise_ranks else '[c]'
            hdr3 += f"  peer{r}{tag}"
        print(hdr3)
        print("  " + "-" * (W - 2))
        for ep, ep_trust in enumerate(engc_trust_list):
            row = f"  {ep+1:>5}  {ep_trust.get('self', float('nan')):>6.3f}"
            for r in all_nbr_ranks:
                row += f"  {ep_trust.get(r, float('nan')):>8.3f}"
            print(row)

        # Summary: compare mean weight of noisy vs clean peers over last 10 epochs
        if noise_ranks and len(engc_trust_list) >= 2:
            window = engc_trust_list[-10:]
            clean_ranks = [r for r in all_nbr_ranks if r not in noise_ranks]
            for r in all_nbr_ranks:
                mean_w = sum(ep.get(r, 0.0) for ep in window) / len(window)
                tag = '[N]' if r in noise_ranks else '[c]'
                print(f"  Mean weight last {len(window)} ep — peer{r}{tag}: {mean_w:.4f}")
        print("=" * W + "\n")

    total_comm_gb   = data_transferred / 1.0e9
    total_time_s    = sum(train_time_list)
    avg_cpu_pct     = sum(cpu_pct_list)  / max(len(cpu_pct_list),  1)
    avg_gpu_pct     = sum(gpu_pct_list)  / max(len(gpu_pct_list),  1)
    total_comm_pkgs = sum(comm_pkgs_list)
    last_comm_mb    = comm_bytes_list[-1] / 1e6 if comm_bytes_list else 0.0
    last_time_s     = train_time_list[-1]        if train_time_list else 0.0

    # Print in rank order so output is readable with multiple processes
    for r in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == r:
            W = 62
            print("\n" + "=" * W)
            print(f"  RANK {rank} — POST-TRAINING METRICS  "
                  f"[{args.optimizer.upper()} | {args.graph} | skew={args.skew}]")
            print("=" * W)
            print(f"  {'Metric':<30} {'Last epoch':>12}  {'Total / Avg':>12}")
            print("  " + "-" * (W - 2))
            print(f"  {'Training time (s)':<30} {last_time_s:>12.2f}  {total_time_s:>11.1f}s")
            print(f"  {'CPU usage (%)':<30} {cpu_pct_list[-1] if cpu_pct_list else 0:>12.1f}  {avg_cpu_pct:>11.1f}%")
            print(f"  {'GPU usage (%)':<30} {gpu_pct_list[-1] if gpu_pct_list else 0:>12.1f}  {avg_gpu_pct:>11.1f}%")
            print(f"  {'Comm payload (MB)':<30} {last_comm_mb:>12.2f}  {total_comm_gb*1e3:>10.2f}MB")
            print(f"  {'Comm packages':<30} {comm_pkgs_list[-1] if comm_pkgs_list else 0:>12}  {total_comm_pkgs:>12}")
            print(f"  {'Val accuracy (%)':<30} {'—':>12}  {prec1_final:>11.2f}%")
            print("=" * W + "\n")

    result = {
        "acc_last_epoch":  float(prec1),
        "acc_final":       float(prec1_final),
        "payload_gb":      float(total_comm_gb),
        "train_time_s":    float(total_time_s),
        "cpu_pct_avg":     float(avg_cpu_pct),
        "gpu_pct_avg":     float(avg_gpu_pct),
        "comm_pkgs_total": int(total_comm_pkgs),
        # per-epoch lists
        "train_acc_list":   train_acc_list,
        "train_loss_list":  train_loss_list,
        "val_acc_list":     val_acc_list,
        "val_loss_list":    val_loss_list,
        "train_time_list":  train_time_list,
        "cpu_pct_list":     cpu_pct_list,
        "gpu_pct_list":     gpu_pct_list,
        "comm_bytes_list":  comm_bytes_list,
        "comm_pkgs_list":   comm_pkgs_list,
    }
    torch.save(result, os.path.join(args.save_dir, "excel_data", f"rank_{rank}.sp"))


# # # Train function # # #
def train(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    epoch,
    batch_size,
    lr,
    device,
    receiver          = None,
    sender            = None,
    gpu_index         : int   = 0,
    monitor_every     : int   = 50,
):
    global global_steps

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    data_transferred = 0

    all_outputs = []
    all_targets = []

    model.train()

    torch.cuda.synchronize(device)
    t_epoch_start = time.perf_counter()

    last_wall = time.perf_counter()
    last_cpu  = time.process_time()
    cpu_samples = []
    gpu_samples = []

    comm_calls_transfer_params      = 0
    comm_calls_transfer_additional  = 0
    payload_bytes_from_calls        = 0

    # ENGC per-batch diagnostic accumulators (empty for non-ENGC)
    # Keys are neighbor ranks (int); 'self' for own weight
    _engc_trust = {}   # {rank/'self' -> [float]}  weight assigned each step

    end   = time.time()
    step  = len(train_loader) * batch_size * epoch

    val_iter = iter(val_loader)

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var  = Variable(input).to(device)
        target_var = Variable(target).to(device)

        # val batch vẫn load để backward compat với các optimizer khác
        (val_input, val_target), val_iter = get_next_batch(val_iter, val_loader)
        val_input_var  = Variable(val_input).to(device)
        val_target_var = Variable(val_target).to(device)

        _, amt_data_transfer, cross_weights = model.transfer_params(epoch=epoch + (1e-3 * i), lr=lr)
        comm_calls_transfer_params += 1
        payload_bytes_from_calls   += amt_data_transfer
        data_transferred           += amt_data_transfer

        # ----------------------------------------------------------------
        # ENGC
        # ----------------------------------------------------------------
        if args.optimizer.lower() == 'engc':
            output = model(input_var)

            # Per-sample loss of local model B on D_B.
            # Passed to sender so it can compute epistemic gap
            # U_epi(x_i) = max(0, loss_A(x_i) - loss_B(x_i)) per sample.
            with torch.no_grad():
                loss_b_per_sample = edl_loss(
                    output.detach(), target_var, args.classes,
                    lambda_kl=args.edl_lambda, return_per_sample=True,
                )

            cross_grad, ref_buf = sender(
                cross_weights, input_var, target_var,
                loss_b_per_sample=loss_b_per_sample,
            )

            if args.edl_main:
                loss = edl_loss(output, target_var, args.classes,
                                lambda_kl=args.edl_lambda)
            elif args.edl_aux_weight > 0.0:
                loss = criterion(output, target_var) + args.edl_aux_weight * edl_loss(
                    output, target_var, args.classes, lambda_kl=args.edl_lambda
                )
            else:
                loss = criterion(output, target_var)

            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

            loss.backward()

            cross_grad_copy = copy.deepcopy(cross_grad)
            _, amt_data_transfer, received_cross_grad = model.transfer_additional(cross_grad)
            comm_calls_transfer_additional += 1
            payload_bytes_from_calls       += amt_data_transfer
            data_transferred               += amt_data_transfer

            # U_epi_scores: per-neighbor epistemic gap computed in sender.
            # aleatoric_scores: extracted from received gradient scalars inside receiver.
            U_epi_scores   = sender.U_epi_scores   if args.use_edl else None
            vacuity_scores = sender.vacuity_scores  if args.use_edl else None
            loss_scores    = sender.loss_scores     if args.use_edl else None
            acc_scores     = sender.acc_scores      if args.use_edl else None

            if vacuity_scores and i % args.print_freq == 0:
                from optimizers.engc import log_vacuity_stats
                log_vacuity_stats(dist.get_rank(), global_steps, vacuity_scores)

            receiver(received_cross_grad, cross_grad_copy, ref_buf,
                     U_epi_scores=U_epi_scores,
                     vacuity_scores=vacuity_scores, loss_scores=loss_scores,
                     acc_scores=acc_scores)
            receiver.project_gradients(lr)

            # --- Accumulate ENGC diagnostics ---
            for r, w in receiver.last_neighbor_weights.items():
                _engc_trust.setdefault(r, []).append(w)

            # --- Periodic diagnostic print ---
            if i % args.print_freq == 0 and sender.U_epi_scores:
                noise_ranks = set(
                    int(x) for x in args.noise_agents.split(',') if x.strip()
                ) if args.noise_agents else set()

                header = (
                    f"[ENGC-Diag][Rank {dist.get_rank()}]"
                    f"[Ep {epoch}][{i}/{len(train_loader)}]"
                )

                # ── Per-neighbor table ──────────────────────────────────────
                # Columns: U_epi | U_ale | raw_trust | final_weight
                #          sw_mean | sw_frac_low | vac_ok | vac_err
                # Red flags:
                #   U_epi ≈ 0 for all  → loss comparison not firing
                #   U_ale same for all → aleatoric scalar not varying
                #   vac_err ≤ vac_ok   → EDL not calibrated
                #   sw_frac_low ≈ 0   → no samples being downweighted
                print(header)
                # Columns: U_epi | U_ale | raw_trust | final_weight | sw_mean | sw_frac_low
                # Red flags:
                #   U_epi ≈ 0 for all   → loss comparison not firing (models too similar?)
                #   U_ale same for all  → aleatoric scalar not varying across neighbors
                #   sw_frac_low ≈ 0    → per-sample filter not downweighting anything
                #   noisy [N] w ≈ [c] w → trust not discriminating
                fmt = (
                    "  {tag:<8} U_epi={uepi:>6.3f}  U_ale={ale:>6.3f}"
                    "  trust={tr:>6.3f}  w={tw:>6.3f}"
                    "  sw_mean={swm:>5.3f}  sw_frac_low={swf:>5.3f}"
                )
                for r in sorted(sender.U_epi_scores):
                    tag = f"peer{r}{'[N]' if r in noise_ranks else '[c]'}"
                    sw  = sender.sample_weight_stats.get(r, {})
                    print(fmt.format(
                        tag  = tag,
                        uepi = sender.U_epi_scores.get(r, float('nan')),
                        ale  = receiver.last_aleatoric_scores.get(r, float('nan')),
                        tr   = receiver.last_trust_scores.get(r, float('nan')),
                        tw   = receiver.last_neighbor_weights.get(r, float('nan')),
                        swm  = sw.get('mean',     float('nan')),
                        swf  = sw.get('frac_low', float('nan')),
                    ))

            global_steps += 1

        # ----------------------------------------------------------------
        # MURMURA — model averaging, no cross-gradient transfer
        # ----------------------------------------------------------------
        elif args.optimizer.lower() == 'murmura':
            output = model(input_var)
            loss   = criterion(output, target_var)

            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

            loss.backward()

            # Trust scores from neighbor model eval (no 2nd comm round)
            trust_scores = sender(cross_weights, input_var, target_var)
            receiver.prepare(cross_weights, trust_scores,
                             step=epoch + 1e-3 * i)
            receiver.project_gradients(lr)   # no-op

            if i % args.print_freq == 0:
                noise_ranks = set(
                    int(x) for x in args.noise_agents.split(',') if x.strip()
                ) if args.noise_agents else set()
                parts = [f"tau={receiver.last_threshold:.3f}"]
                for r in sorted(trust_scores):
                    tag = '[N]' if r in noise_ranks else '[c]'
                    parts.append(
                        f"peer{r}{tag} "
                        f"trust={trust_scores[r]:.3f} "
                        f"vac={sender.last_vacuities.get(r, float('nan')):.3f} "
                        f"acc={sender.last_accuracies.get(r, float('nan')):.3f}"
                    )
                print(
                    f"[MURMURA-Diag][Rank {dist.get_rank()}]"
                    f"[Ep {epoch}][{i}/{len(train_loader)}] "
                    + " | ".join(parts)
                )

            global_steps += 1

        # ----------------------------------------------------------------
        # NGC, CGA, và các optimizer khác — giữ nguyên hoàn toàn
        # ----------------------------------------------------------------
        else:
            output = model(input_var)
            loss   = criterion(output, target_var)

            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

            loss.backward()

            if 'cga' in args.optimizer.lower() or 'ngc' in args.optimizer.lower():
                cross_grad, ref_buf = sender(cross_weights, input_var, target_var)
                cross_grad_copy     = copy.deepcopy(cross_grad)
                _, amt_data_transfer, received_cross_grad = model.transfer_additional(cross_grad)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls       += amt_data_transfer
                data_transferred               += amt_data_transfer
                receiver(received_cross_grad, cross_grad_copy, ref_buf)
                receiver.project_gradients(lr)

            elif args.optimizer.lower() == 'd-psgd':
                receiver.update_gradients(lr)

        # ----------------------------------------------------------------
        # Optimizer step — chung cho tất cả
        # ----------------------------------------------------------------
        optimizer.step()
        optimizer.zero_grad()

        # MURMURA: model averaging happens AFTER local gradient update
        if args.optimizer.lower() == 'murmura':
            receiver.post_step_aggregate()

        output = output.float()
        loss   = loss.float()

        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i % monitor_every) == 0:
            now_wall = time.perf_counter()
            now_cpu  = time.process_time()
            dt_wall  = max(now_wall - last_wall, 1e-9)
            dt_cpu   = max(now_cpu - last_cpu, 0.0)
            cpu_pct  = (dt_cpu / dt_wall) * 100.0
            cpu_samples.append(cpu_pct)
            last_wall, last_cpu = now_wall, now_cpu

            gu = get_gpu_util_percent(gpu_index)
            if gu is not None:
                gpu_samples.append(gu)

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses, top1=top1))
        step += batch_size

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    prec, rec, f1 = precision_recall_f1(all_outputs, all_targets, num_classes=args.classes)

    if dist.get_rank() == 0:
        print(
            f"[Train][Epoch {epoch}] "
            f"Precision = {prec:.2f}  "
            f"Recall = {rec:.2f}  "
            f"F1 = {f1:.2f}"
        )

    torch.cuda.synchronize(device)
    train_time_s = time.perf_counter() - t_epoch_start

    cpu_avg = float(sum(cpu_samples) / max(len(cpu_samples), 1)) if cpu_samples else 0.0
    gpu_avg = float(sum(gpu_samples) / max(len(gpu_samples), 1)) if gpu_samples else 0.0

    def _emean(d):
        return {r: float(sum(v) / len(v)) if v else float('nan') for r, v in d.items()}

    metrics = {
        "train_time_s":                   train_time_s,
        "cpu_pct_avg":                    cpu_avg,
        "gpu_pct_avg":                    gpu_avg,
        "comm_calls_transfer_params":     comm_calls_transfer_params,
        "comm_calls_transfer_additional": comm_calls_transfer_additional,
        "payload_bytes_from_calls":       int(payload_bytes_from_calls),
        # ENGC diagnostics (empty dict for non-ENGC optimizers)
        "engc_trust": _emean(_engc_trust),
    }
    return data_transferred, top1.avg, losses.avg, metrics


# # # Validation function # # #
def validate(val_loader, model, criterion, batch_size, device, epoch=0):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()

    model.eval()

    all_outputs = []
    all_targets = []

    step = len(val_loader) * batch_size * epoch
    end  = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var  = Variable(input).to(device)
            target_var = Variable(target).to(device)

            output = model(input_var)
            loss   = criterion(output, target_var)
            output = output.float()
            loss   = loss.float()

            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())

            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(), i, len(val_loader),
                          loss=losses, top1=top1))
            step += batch_size

    print('Rank:{0}, Prec@1 {top1.avg:.3f}'.format(dist.get_rank(), top1=top1))

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    prec, rec, f1 = precision_recall_f1(all_outputs, all_targets, num_classes=args.classes)

    ece = compute_ece(all_outputs, all_targets)

    mean_vacuity = float('nan')
    if args.optimizer.lower() == 'engc':
        mean_vacuity = compute_edl_vacuity(all_outputs).mean().item()

    if dist.get_rank() == 0:
        vac_str = f"  Vacuity = {mean_vacuity:.4f}" if not math.isnan(mean_vacuity) else ""
        print(
            f"[Val][Epoch {epoch}] "
            f"Precision = {prec:.2f}  "
            f"Recall = {rec:.2f}  "
            f"F1 = {f1:.2f}  "
            f"ECE = {ece:.4f}"
            f"{vac_str}"
        )

    return top1.avg, losses.avg


# # # Helper functions # # #
def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error — measures gap between confidence and accuracy.
    Lower is better. Perfect calibration = 0."""
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()
    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean().item()
        if prop > 0:
            acc  = correct[in_bin].mean().item()
            conf = confidences[in_bin].mean().item()
            ece += prop * abs(conf - acc)
    return ece


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def get_gpu_util_percent(gpu_index: int):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def average_parameters(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk       = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision_recall_f1(output, target, num_classes):
    pred           = output.argmax(dim=1)
    precision_list = []
    recall_list    = []
    f1_list        = []

    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        sum(precision_list) / num_classes * 100,
        sum(recall_list)    / num_classes * 100,
        sum(f1_list)        / num_classes * 100,
    )


def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat


def init_process(rank, size, fn, backend='nccl'):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    try:
        fn(rank, size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def check_noniid(train_loader, rank, world_size):
    local_counter = Counter()
    for _, targets in train_loader:
        local_counter.update(targets.tolist())
    local_total        = sum(local_counter.values())
    local_distribution = {int(label): count / local_total for label, count in local_counter.items()}

    all_distributions = [None for _ in range(world_size)]
    dist.all_gather_object(all_distributions, local_distribution)

    print("= = = = = CHECK NON-IID DISTRIBUTION ACROSS AGENTS = = = = =")
    if rank == 0:
        for r, dist_r in enumerate(all_distributions):
            print(f"Rank {r} label distribution: {dist_r}")
        all_labels = sorted({label for d in all_distributions for label in d.keys()})
        matrix = np.array([
            [d.get(label, 0.0) for label in all_labels]
            for d in all_distributions
        ])
        mean_distribution = matrix.mean(axis=0)
        l1_distance       = np.abs(matrix - mean_distribution).sum(axis=1)
        print("\nL1 distance:")
        for r, value in enumerate(l1_distance):
            print(f"Rank {r}: {value:.3f}")


def compute_local_class_weights(train_loader, num_classes, device, eps=1e-8):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, targets in train_loader:
        if isinstance(targets, torch.Tensor):
            t = targets.view(-1).cpu()
            counts += torch.bincount(t, minlength=num_classes).float()
    counts  = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=3.0)
    return weights.to(device)


def get_next_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch       = next(loader_iter)
    return batch, loader_iter


# # # Main # # #
if __name__ == '__main__':
    size = args.world_size

    spawn(init_process, args=(size, run), nprocs=size, join=True)

    # Read stored data
    excel_data = {
        'data':          args.dataset,
        "graph":         args.graph,
        "nodes":         size,
        'arch':          args.arch,
        "norm":          args.normtype,
        'depth':         args.depth,
        'optimizer':     args.optimizer,
        "learning rate": args.lr,
        "momentum":      args.momentum,
        "qgm":           args.qgm,
        "nesterov":      args.nesterov,
        "weight_decay":  args.weight_decay,
        "skew":          args.skew,
        "gamma":         args.gamma,
        "alpha":         args.alpha,
        "epochs":        args.epochs,
        "seed":          args.seed,
        # scalar per-rank
        "avg test acc":       [0.0 for _ in range(size)],
        "avg test acc final": [0.0 for _ in range(size)],
        "data transferred":   [0.0 for _ in range(size)],
        "train_time_s":       [0.0 for _ in range(size)],
        "cpu_pct_avg":        [0.0 for _ in range(size)],
        "gpu_pct_avg":        [0.0 for _ in range(size)],
        "comm_pkgs_total":    [0   for _ in range(size)],
        # per-epoch lists per-rank
        "train_acc_list":   [[] for _ in range(size)],
        "train_loss_list":  [[] for _ in range(size)],
        "val_acc_list":     [[] for _ in range(size)],
        "val_loss_list":    [[] for _ in range(size)],
        "train_time_list":  [[] for _ in range(size)],
        "cpu_pct_list":     [[] for _ in range(size)],
        "gpu_pct_list":     [[] for _ in range(size)],
        "comm_bytes_list":  [[] for _ in range(size)],
        "comm_pkgs_list":   [[] for _ in range(size)],
    }

    for i in range(size):
        r = torch.load(os.path.join(args.save_dir, "excel_data", f"rank_{i}.sp"))
        excel_data["avg test acc"][i]       = r["acc_last_epoch"]
        excel_data["avg test acc final"][i] = r["acc_final"]
        excel_data["data transferred"][i]   = r["payload_gb"]
        excel_data["train_time_s"][i]       = r["train_time_s"]
        excel_data["cpu_pct_avg"][i]        = r["cpu_pct_avg"]
        excel_data["gpu_pct_avg"][i]        = r["gpu_pct_avg"]
        excel_data["comm_pkgs_total"][i]    = r["comm_pkgs_total"]
        excel_data["train_acc_list"][i]     = r.get("train_acc_list",  [])
        excel_data["train_loss_list"][i]    = r.get("train_loss_list", [])
        excel_data["val_acc_list"][i]       = r.get("val_acc_list",    [])
        excel_data["val_loss_list"][i]      = r.get("val_loss_list",   [])
        excel_data["train_time_list"][i]    = r.get("train_time_list", [])
        excel_data["cpu_pct_list"][i]       = r.get("cpu_pct_list",    [])
        excel_data["gpu_pct_list"][i]       = r.get("gpu_pct_list",    [])
        excel_data["comm_bytes_list"][i]    = r.get("comm_bytes_list", [])
        excel_data["comm_pkgs_list"][i]     = r.get("comm_pkgs_list",  [])

    torch.save(excel_data, os.path.join(args.save_dir, "excel_data", "dict"))

    def _build_matrix(lists):
        """Stack per-rank lists into a (n_ranks x n_epochs) numpy matrix."""
        max_len = max((len(l) for l in lists if l), default=0)
        if max_len == 0:
            return None, max_len
        mat = np.full((len(lists), max_len), np.nan, dtype=float)
        for i, l in enumerate(lists):
            mat[i, :len(l)] = np.array(l, dtype=float)
        return mat, max_len

    def plot_epoch_metric(lists, out_dir, filename, ylabel, title=None):
        mat, max_len = _build_matrix(lists)
        if mat is None:
            return
        mean   = np.nanmean(mat, axis=0)
        epochs = np.arange(1, max_len + 1)
        plt.figure()
        for i in range(mat.shape[0]):
            plt.plot(epochs, mat[i], alpha=0.25)
        plt.plot(epochs, mean, linewidth=2.5, label="mean")
        if title:
            plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=200)
        plt.close()

    # Accuracy plots
    plot_epoch_metric(excel_data["train_acc_list"], args.save_dir,
                      "train_accuracy_vs_epoch.png", "Train accuracy (%)")
    plot_epoch_metric(excel_data["val_acc_list"],   args.save_dir,
                      "val_accuracy_vs_epoch.png",   "Val accuracy (%)")

    # Resource / communication plots
    plot_epoch_metric(excel_data["train_time_list"], args.save_dir,
                      "train_time_vs_epoch.png", "Training time (s/epoch)",
                      title="Training time per epoch")
    plot_epoch_metric(excel_data["cpu_pct_list"], args.save_dir,
                      "cpu_usage_vs_epoch.png", "CPU usage (%)",
                      title="CPU usage per epoch")
    plot_epoch_metric(excel_data["gpu_pct_list"], args.save_dir,
                      "gpu_usage_vs_epoch.png", "GPU usage (%)",
                      title="GPU usage per epoch")
    plot_epoch_metric(
        [[b / 1e6 for b in row] for row in excel_data["comm_bytes_list"]],
        args.save_dir, "comm_cost_vs_epoch.png", "Communication cost (MB/epoch)",
        title="Communication cost per epoch",
    )
    plot_epoch_metric(excel_data["comm_pkgs_list"], args.save_dir,
                      "comm_packages_vs_epoch.png", "# comm packages/epoch",
                      title="Communication packages per epoch")

    # Print cross-rank summary table
    print("\n" + "=" * 72)
    print(f"  FINAL SUMMARY  [{args.optimizer.upper()} | {args.graph} | "
          f"{size} nodes | skew={args.skew}]")
    print("=" * 72)
    print(f"  {'Rank':<6} {'ValAcc%':>8} {'FinalAcc%':>10} "
          f"{'Time(s)':>9} {'CPU%':>7} {'GPU%':>7} "
          f"{'CommGB':>8} {'Pkgs':>7}")
    print("  " + "-" * 68)
    for i in range(size):
        print(f"  {i:<6} "
              f"{excel_data['avg test acc'][i]:>8.2f} "
              f"{excel_data['avg test acc final'][i]:>10.2f} "
              f"{excel_data['train_time_s'][i]:>9.1f} "
              f"{excel_data['cpu_pct_avg'][i]:>7.1f} "
              f"{excel_data['gpu_pct_avg'][i]:>7.1f} "
              f"{excel_data['data transferred'][i]:>8.4f} "
              f"{excel_data['comm_pkgs_total'][i]:>7}")
    print("=" * 72 + "\n")

    # --- Loss curve (rank 0 only) ---
    r0_train = excel_data["train_loss_list"][0]
    r0_val   = excel_data["val_loss_list"][0]
    if r0_train and r0_val:
        print("=" * 72)
        print(f"  LOSS CURVE  [Rank 0 — {args.optimizer.upper()}]")
        print("=" * 72)
        print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'Gap(V-T)':>10}  {'Overfit?':>9}")
        print("  " + "-" * 60)
        for ep, (tl, vl) in enumerate(zip(r0_train, r0_val)):
            gap      = vl - tl
            overfit  = "YES" if gap > 0.1 else ""
            print(f"  {ep:>6}  {tl:>10.4f}  {vl:>10.4f}  {gap:>+10.4f}  {overfit:>9}")
        print("=" * 72 + "\n")