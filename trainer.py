import argparse
import os
import shutil
import time
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
import torch.nn.functional as F
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
parser.add_argument('--arch', '-a', metavar='ARCH', default='cganet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype',   default='evonorm', help = 'none or batchnorm or groupnorm or evonorm' )
parser.add_argument('--data-dir', dest='data_dir',    help='The directory used to save the trained models',   default='../../data', type=str)
parser.add_argument('--dataset', dest='dataset',     help='available datasets: cifar10, cifar100, imagenette, ham10000', default='cifar10', type=str)
parser.add_argument('--skew', default=1.0, type=float,     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=160, type=int,  help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--gamma',  default=0.1, type=float,  metavar='AR', help='averaging rate')
parser.add_argument('--alpha',  default=1.0, type=float, help='NGC mixing weight')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',     help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float,     help='weight_decay')
parser.add_argument('-world_size', '--world_size', default=10, type=int, help='total number of nodes')
parser.add_argument('--epochs', default=100, type=int, metavar='N',   help='number of total epochs to run')
parser.add_argument('--optimizer', default='ngc', type=str,  help='global optimizer = [d-psgd, cga, ngc, compcga, compngc, topkngc, engc]')
parser.add_argument('--graph', '-g',  default='ring', help = 'graph structure - [ring, torus]' )
parser.add_argument('--neighbors', default=2, type=int,     help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int, help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int,  help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=321, type=int,   help='set seed')
parser.add_argument('--print-freq', '-p', default=100, type=int,  help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='outputs', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='25500' , type=str)
parser.add_argument("--steplr", action="store_true", help="Uses step lr schedular for training.")
parser.add_argument('--nesterov', action='store_true', )
parser.add_argument('--qgm', action='store_true', help='quasi global momentum')
# engc arguments
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

args = parser.parse_args()
args.devices = torch.cuda.device_count()

# # # Check the save_dir exists or not # # #
args.save_dir = os.path.join(args.save_dir, args.optimizer+"_"+args.arch+"_nodes_"+str(args.world_size)+"_"+ args.normtype+"_lr_"+ str(args.lr)+"_gamma_"+str(args.gamma)+"_alpha_"+str(args.alpha)+"_skew_"+str(args.skew)+"_"+args.graph )
if not os.path.exists(os.path.join(args.save_dir, "excel_data") ):
    os.makedirs(os.path.join(args.save_dir, "excel_data") )
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
    # Train time - %CPU - %GPU
    total_train_time_s = 0.0
    total_cpu_pct_sum = 0.0
    total_gpu_pct_sum = 0.0
    total_cpu_cnt = 0
    total_gpu_cnt = 0
    # Accuracy and loss lists for plotting
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

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

    train_loader, bsz_train = partition_trainDataset(
        args.dataset, args.data_dir, args.skew, args.seed, args.batch_size
    )
    val_loader, bsz_val = test_Dataset(args.dataset, args.data_dir, seed=args.seed)

    check_noniid(train_loader, rank, args.world_size)

    local_class_weights = compute_local_class_weights(
        train_loader=train_loader,
        num_classes=args.classes,
        device=device,
    )
    if rank == 0:
        print(f"[ENGC] local_class_weights rank {rank}: {local_class_weights.detach().cpu().tolist()}")

    if args.optimizer.lower() == 'engc' and local_class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=local_class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer.lower() == 'cga':
        sender = CGA_sender(base_model, device)
    elif args.optimizer.lower() == "ngc":
        sender = NGC_sender(base_model, device)
    elif args.optimizer.lower() == 'compcga':
        sender = CompCGA_sender(base_model, device)
    elif args.optimizer.lower() == "compngc":
        sender = CompNGC_sender(base_model, device)
    elif args.optimizer.lower() == "topkngc":
        sender = Topk_NGC_sender(base_model, device)
    elif args.optimizer.lower() == 'engc':        
        sender = ENGC_sender(
            base_model,
            device,
            criterion=criterion,
            num_classes=args.classes,
            total_rounds=args.epochs * len(train_loader),
            tau_u=args.engc_tau_u,
            w_a=args.engc_wa
        )
    else:
        sender = None

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
            self_weight=args.ngc_self_weight,
            score_momentum=args.ngc_score_momentum,
            temperature=args.ngc_temperature,
            align_weight=args.ngc_align_weight,
            norm_weight=args.ngc_norm_weight,
            min_peer_weight=args.ngc_min_peer_weight,
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

    for epoch in range(0, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()

        dt, prec1, loss, m = train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=bsz_train,
            lr=optimizer.param_groups[0]['lr'],
            device=device,
            receiver=receiver,
            sender=sender,
            gpu_index=rank,
            monitor_every=max(10, args.print_freq),
        )
        data_transferred += dt

        train_acc_list.append(float(prec1))
        train_loss_list.append(float(loss))

        total_train_time_s += m["train_time_s"]
        total_cpu_pct_sum += m["cpu_pct_avg"]
        total_gpu_pct_sum += m["gpu_pct_avg"]
        total_cpu_cnt += 1
        total_gpu_cnt += 1

        lr_scheduler.step()

        prec1, loss = validate(val_loader, model, criterion, bsz_val, device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}.th'.format(rank)))

        val_acc_list.append(float(prec1))
        val_loss_list.append(float(loss))

    average_parameters(model)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val, device, epoch)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred / 1.0e9, "\n")

    result = {
        "acc_last_epoch": float(prec1),
        "acc_final": float(prec1_final),
        "payload_gb": float(data_transferred / 1.0e9),
        "train_time_s": float(total_train_time_s),
        "cpu_pct_avg": float(total_cpu_pct_sum / max(total_cpu_cnt, 1)),
        "gpu_pct_avg": float(total_gpu_pct_sum / max(total_gpu_cnt, 1)),
        "train_acc_list": train_acc_list,
        "train_loss_list": train_loss_list,
        "val_acc_list": val_acc_list,
        "val_loss_list": val_loss_list,
    }
    torch.save(result, os.path.join(args.save_dir, "excel_data", f"rank_{rank}.sp"))
    
# # # Train functions # # #
def train(train_loader, val_loader, model, criterion, optimizer, epoch, batch_size, lr, device,
          receiver=None, sender=None, gpu_index: int = 0, monitor_every: int = 50):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    all_outputs = []
    all_targets = []

    model.train()

    torch.cuda.synchronize(device)
    t_epoch_start = time.perf_counter()

    last_wall = time.perf_counter()
    last_cpu = time.process_time()
    cpu_samples = []
    gpu_samples = []

    comm_calls_transfer_params = 0
    comm_calls_transfer_additional = 0
    payload_bytes_from_calls = 0

    end = time.time()
    step = len(train_loader) * batch_size * epoch
    total_rounds = args.epochs * len(train_loader)

    val_iter = iter(val_loader)

    for i, (input, target) in enumerate(train_loader):
        current_round = epoch * len(train_loader) + i
        data_time.update(time.time() - end)

        input_var = Variable(input).to(device)
        target_var = Variable(target).to(device)

        (val_input, val_target), val_iter = get_next_batch(val_iter, val_loader)
        val_input_var = Variable(val_input).to(device)
        val_target_var = Variable(val_target).to(device)

        _, amt_data_transfer, cross_weights = model.transfer_params(epoch=epoch + (1e-3 * i), lr=lr)
        comm_calls_transfer_params += 1
        payload_bytes_from_calls += amt_data_transfer
        data_transferred += amt_data_transfer

        output = model(input_var)
        loss = criterion(output, target_var)

        all_outputs.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())

        loss.backward()

        if 'cga' in args.optimizer.lower() or 'ngc' in args.optimizer.lower():
            if args.optimizer.lower() == 'engc':
                cross_grad, ref_buf, trust_scores = sender(
                    cross_weights,
                    input_var,
                    target_var,
                    val_input_var,
                    val_target_var,
                    current_round
                )

                cross_grad_copy = copy.deepcopy(cross_grad)

                _, amt_data_transfer, received_cross_grad = model.transfer_additional(cross_grad)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls += amt_data_transfer
                data_transferred += amt_data_transfer

                self_eval_local = evaluate_self_evidential(
                    model,
                    val_input_var,
                    val_target_var,
                    args.classes,
                )

                receiver(
                    received_cross_grad, # neighbor_grads_comm
                    cross_grad_copy,     # neighbor_grads_comp
                    ref_buf,             # self gradient
                    trust_scores 
                )
                receiver.project_gradients(lr)
            else:
                cross_grad, ref_buf = sender(cross_weights, input_var, target_var)
                cross_grad_copy = copy.deepcopy(cross_grad)
                _, amt_data_transfer, received_cross_grad = model.transfer_additional(cross_grad)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls += amt_data_transfer
                data_transferred += amt_data_transfer
                receiver(received_cross_grad, cross_grad_copy, ref_buf)
                receiver.project_gradients(lr)

        elif args.optimizer.lower() == 'd-psgd':
            receiver.update_gradients(lr)

        optimizer.step()
        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i % monitor_every) == 0:
            now_wall = time.perf_counter()
            now_cpu = time.process_time()
            dt_wall = max(now_wall - last_wall, 1e-9)
            dt_cpu = max(now_cpu - last_cpu, 0.0)
            cpu_pct = (dt_cpu / dt_wall) * 100.0
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

    metrics = {
        "train_time_s": train_time_s,
        "cpu_pct_avg": cpu_avg,
        "gpu_pct_avg": gpu_avg,
        "comm_calls_transfer_params": comm_calls_transfer_params,
        "comm_calls_transfer_additional": comm_calls_transfer_additional,
        "payload_bytes_from_calls": int(payload_bytes_from_calls),
    }
    return data_transferred, top1.avg, losses.avg, metrics
    
# # # Validation function # # #
def validate(val_loader, model, criterion, batch_size, device, epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_outputs = []
    all_targets = []

    step = len(val_loader)*batch_size*epoch
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)
            # compute output and loss
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), 
                          #batch_time=batch_time, 
                          loss=losses,
                          top1=top1))
            step += batch_size
    print('Rank:{0}, Prec@1 {top1.avg:.3f}'.format(dist.get_rank(),top1=top1))
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    prec, rec, f1 = precision_recall_f1(
        all_outputs, all_targets, num_classes=args.classes
    )

    if dist.get_rank() == 0:
        print(
            f"[Val][Epoch {epoch}] "
            f"Precision = {prec:.2f}  "
            f"Recall = {rec:.2f}  "
            f"F1 = {f1:.2f}  "
        )
    return top1.avg, losses.avg

# # # Helper functions # # #
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_gpu_util_percent(gpu_index: int):
    """
    Return GPU utilization (%) via nvidia-smi, or None if failed.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        # may return multiple lines; take first
        return float(out.splitlines()[0])
    except Exception:
        return None

def average_parameters(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def precision_recall_f1(output, target, num_classes):
    pred = output.argmax(dim=1)
    precision_list, recall_list, f1_list = [], [], []

    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        sum(precision_list)/num_classes * 100,
        sum(recall_list)/num_classes * 100,
        sum(f1_list)/num_classes * 100,
    )

def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def average_parameters(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

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
    # 1. Counting local label distribution
    local_counter = Counter()
    for _, targets in train_loader: # train_loader returns (input, targets)
        local_counter.update(targets.tolist()) # Counter({label: count, ...})
    # 2. Convert to probability
    local_total = sum(local_counter.values())
    local_distribution = {int(label): count / local_total for label, count in local_counter.items()}
    # 3. Gather all local distributions
    ## all_distributions[rank] = local_distribution
    all_distributions = [None for _ in range(world_size)] # [None,...] with length = world_size
    dist.all_gather_object(all_distributions, local_distribution) # [{local_distribution},...]
    # 4. Print distributions
    print("= = = = = CHECK NON-IID DISTRIBUTION ACROSS AGENTS = = = = =")
    if rank == 0:
        for rank, distribution_rank in enumerate(all_distributions):
            print(f"Rank {rank} label distribution: {distribution_rank}")
        # 5. Computing L1 distance
        all_labels = sorted({label for distribution in all_distributions for label in distribution.keys()})
        matrix = np.array([
            [distribution.get(label, 0.0) for label in all_labels]
            for distribution in all_distributions
        ]) # [[...],...]
        mean_distribution = matrix.mean(axis=0)
        l1_distance = np.abs(matrix - mean_distribution).sum(axis=1)
        print("\nL1 distance:")
        for rank, value in enumerate(l1_distance):
                print(f"Rank {rank}: {value:.3f}")

def compute_local_class_weights(train_loader, num_classes, device, eps=1e-8):
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, targets in train_loader:
        if isinstance(targets, torch.Tensor):
            t = targets.view(-1).cpu()
            counts += torch.bincount(t, minlength=num_classes).float()

    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=3.0)
    return weights.to(device)


def get_next_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


@torch.no_grad()
def evaluate_self_evidential(model, x, y, num_classes):
    was_training = model.training
    model.eval()

    output = model(x)
    evidence = F.softplus(output)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / torch.clamp(S, min=1e-8)

    pred = probs.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    unc = (float(num_classes) / torch.clamp(S.squeeze(1), min=1e-8)).mean().item()

    if was_training:
        model.train()

    return {
        "accuracy": float(acc),
        "uncertainty": float(unc),
    }   

# # # Main function # # #
if __name__ == '__main__':
    size = args.world_size
    
    spawn(init_process, args=(size,run), nprocs=size, join=True)
    #read stored data
    excel_data = {
        'data': args.dataset,
        "graph" : args.graph,
        "nodes": size,
        'arch': args.arch,
        "norm" : args.normtype,
        'depth':args.depth,
        'optimizer' : args.optimizer,
        "learning rate": args.lr,
        "momentum":args.momentum,
        "qgm":args.qgm,
        "nesterov":args.nesterov,
        "weight_decay":args.weight_decay,
        "skew" : args.skew,
        "gamma" : args.gamma,
        "alpha" : args.alpha,
        "epochs": args.epochs,
        "avg test acc":[0.0 for _ in range(size)],
        "avg test acc final":[0.0 for _ in range(size)],
        "data transferred": [0.0 for _ in range(size)],
         "seed" :args.seed,
    }
    excel_data.update({
        "train_acc_list": [[] for _ in range(size)],
        "train_loss_list": [[] for _ in range(size)],
        "val_acc_list":   [[] for _ in range(size)],
        "val_loss_list":  [[] for _ in range(size)],
    })
         
    for i in range(size):
        r = torch.load(os.path.join(args.save_dir, "excel_data", f"rank_{i}.sp"))
        excel_data["avg test acc"][i] = r["acc_last_epoch"]
        excel_data["avg test acc final"][i] = r["acc_final"]
        excel_data["data transferred"][i] = r["payload_gb"]
        excel_data["train_acc_list"][i] = r.get("train_acc_list", [])
        excel_data["train_loss_list"][i] = r.get("train_loss_list", [])
        excel_data["val_acc_list"][i]   = r.get("val_acc_list", [])
        excel_data["val_loss_list"][i]  = r.get("val_loss_list", [])
        
    torch.save(excel_data, os.path.join(args.save_dir, "excel_data","dict"))

    def plot_epoch_acc(excel_data, out_dir, which="val"):
        key = f"{which}_acc_list"
        lists = excel_data.get(key, [])
        max_len = max((len(l) for l in lists if l), default=0)
        if max_len == 0:
            print(f"[plot] No data for {key}")
            return

        mat = np.full((len(lists), max_len), np.nan, dtype=float)
        for i, l in enumerate(lists):
            mat[i, :len(l)] = np.array(l, dtype=float)

        mean = np.nanmean(mat, axis=0)
        epochs = np.arange(1, max_len + 1)

        plt.figure()
        for i in range(mat.shape[0]):
            plt.plot(epochs, mat[i], alpha=0.25)
        plt.plot(epochs, mean, linewidth=2.5, label="mean")
        plt.xlabel("Epoch")
        plt.ylabel(f"{which} accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{which}_accuracy_vs_epoch.png"), dpi=200)
        plt.close()

    plot_epoch_acc(excel_data, args.save_dir, which="train")
    plot_epoch_acc(excel_data, args.save_dir, which="val")
    