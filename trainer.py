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
parser.add_argument('--optimizer', default='ngc', type=str,  help='global optimizer = [d-psgd, cga, ngc, compcga, compngc, topkngc, edlngc]')
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
    #torch.use_deterministic_algorithms(True)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
	##############
    best_prec1 = 0
    data_transferred = 0
    global_steps = 0

    total_train_time_s = 0.0
    total_cpu_pct_sum = 0.0
    total_gpu_pct_sum = 0.0
    total_cpu_cnt = 0
    total_gpu_cnt = 0

    total_tx_bytes = 0
    total_tx_packets = 0

    # track per-epoch metrics (for plotting)
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    
    
    if args.arch.lower()=='resnet':
        model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'vgg11':
        model = vgg11(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'mobilenet':
        model = MobileNetV2(num_classes=args.classes, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'cganet':
        model = cganet5(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'lenet5':
        model = LeNet5()
    else:
        raise NotImplementedError
    
    if rank==0: 
        print(args)
        print('Printing model summary...')
        if args.dataset=="fmnist":
            print(summary(model, (1,28,28), batch_size=int(args.batch_size/size), device='cpu'))
        elif args.dataset=="imagenette_full":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        elif args.dataset=="imagenet":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        else: 
            print(summary(model, (3, 32, 32), batch_size=int(args.batch_size/size), device='cpu'))
        
    if args.optimizer.lower()=='cga':
        sender = CGA_sender(model, device)
    elif args.optimizer.lower()=="ngc":
        sender = NGC_sender(model, device)
    elif args.optimizer.lower()=='compcga':
        sender = CompCGA_sender(model, device)
    elif args.optimizer.lower()=="compngc":
        sender = CompNGC_sender(model, device)
    elif args.optimizer.lower()=="topkngc":
        sender = Topk_NGC_sender(model, device)
    elif args.optimizer.lower()=='edlngc':
        sender = EDL_NGC_sender(model, device, num_classes=args.classes)
    else:
        sender=None

    if args.graph.lower() == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors) #undirected ring structure => neighbors = 2 ; directed ring => neighbors=1
    elif args.graph.lower() == 'torus':   
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors) # torus graph structure
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size-1) # torus graph structure  
    elif args.graph.lower() == 'chain':   
        graph = ChainGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    else:
        raise NotImplementedError
    
    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device, 
                level = 32,
                biased = False,
                eta = args.gamma,
                compress_ratio=0.0,
                compress_fn = 'quantize', 
                compress_op = 'top_k', 
                momentum=args.momentum,
                lr = args.lr) 
    model.to(device)
 
    train_loader, bsz_train = partition_trainDataset(args.dataset, args.data_dir, args.skew, args.seed, args.batch_size)
    val_loader, bsz_val     = test_Dataset(args.dataset, args.data_dir, seed=args.seed)
   
   # Check non-iid distribution
    check_noniid(train_loader, rank, args.world_size)

    if args.optimizer.lower()=='cga':
        receiver  = CGA_receiver(model, device, rank,  args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower()=='compcga':
        receiver = CompCGA_receiver(model, device, rank,  args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors)
    elif args.optimizer.lower()=='ngc':
        receiver  = NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower()=='compngc':
        receiver = CompNGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower()=='topkngc':
        receiver  = Topk_NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha = args.alpha)
    elif args.optimizer.lower() == 'edlngc':
        receiver = EDL_NGC_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay, neighbors=args.neighbors, alpha=args.alpha)
    else:
        receiver = DSGD_receiver(model, device, rank, args.lr, args.momentum, args.qgm, args.nesterov, weight_decay=args.weight_decay)
    

    optimizer = optim.SGD(model.parameters(), args.lr)
    
    if args.optimizer.lower() == 'edlngc':
        from optimizers.edlngc import EDLLoss 
        criterion = EDLLoss(num_classes=args.classes).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if args.steplr:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma = 0.981, step_size=1)
    else:
        if args.dataset=='imagenet':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[int(args.epochs*0.33), int(args.epochs*0.67), int(args.epochs*0.89)])
        else:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)])
            
    for epoch in range(0, args.epochs):  
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()
        # (NIC) đo trước epoch train
        net_before = read_net_dev() if rank == 0 else None

        dt, prec1, loss, m = train(
            train_loader, model, criterion, optimizer, epoch,
            bsz_train, optimizer.param_groups[0]['lr'],
            device,
            receiver=receiver,
            sender=sender,
            gpu_index=rank,
            monitor_every=max(10, args.print_freq),
        )
        data_transferred += dt

        train_acc_list.append(float(prec1))
        train_loss_list.append(float(loss))

        # (NIC) đo sau epoch train
        if rank == 0:
            net_after = read_net_dev()
            net_delta = diff_stats(net_after, net_before)
        else:
            net_delta = {"rx_bytes":0,"rx_packets":0,"tx_bytes":0,"tx_packets":0}

        total_train_time_s += m["train_time_s"]
        total_cpu_pct_sum += m["cpu_pct_avg"]
        total_gpu_pct_sum += m["gpu_pct_avg"]
        total_cpu_cnt += 1
        total_gpu_cnt += 1

        total_tx_bytes += net_delta["tx_bytes"]
        total_tx_packets += net_delta["tx_packets"]
        
        if epoch>=0: lr_scheduler.step()
        prec1, loss = validate(val_loader, model, criterion, bsz_val,device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}.th'.format(rank)))

        val_acc_list.append(float(prec1))
        val_loss_list.append(float(loss))
      
    #############################
    average_parameters(model)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val,device, epoch)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred/1.0e9, "\n")
    #Store processed data
    result = {
        "acc_last_epoch": float(prec1),
        "acc_final": float(prec1_final),
        "payload_gb": float(data_transferred / 1.0e9),   # payload theo gossip (đã có)
        "train_time_s": float(total_train_time_s),
        "cpu_pct_avg": float(total_cpu_pct_sum / max(total_cpu_cnt, 1)),
        "gpu_pct_avg": float(total_gpu_pct_sum / max(total_gpu_cnt, 1)),
        "tx_bytes": int(total_tx_bytes),                 # NIC-level (rank0)
        "tx_packets": int(total_tx_packets),             # NIC-level (rank0)
        "train_acc_list": train_acc_list,
        "train_loss_list": train_loss_list,
        "val_acc_list": val_acc_list,
        "val_loss_list": val_loss_list,
    }
    torch.save(result, os.path.join(args.save_dir, "excel_data", f"rank_{rank}.sp"))

# # # Train functions # # #
def train(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, receiver=None, sender=None, gpu_index: int = 0, monitor_every: int = 50):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    all_outputs = []
    all_targets = []

    # switch to train mode
    model.train()

    # timing (GPU-accurate)
    torch.cuda.synchronize(device)
    t_epoch_start = time.perf_counter()

    # CPU% (process) sampling: cpu_time / wall_time
    last_wall = time.perf_counter()
    last_cpu  = time.process_time()
    cpu_samples = []

    # GPU util% sampling
    gpu_samples = []

    # logical comm counters (rounds/calls)
    comm_calls_transfer_params = 0
    comm_calls_transfer_additional = 0
    payload_bytes_from_calls = 0  # sum of amt_data_transfer returned by gossip

    end = time.time()
    step = len(train_loader)*batch_size*epoch

    total_rounds = args.epochs * len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        current_round = epoch * len(train_loader) + i
        #print(dist.get_rank(), torch.unique(target))
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # gossip the weights
        _, amt_data_transfer, cross_weights = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        comm_calls_transfer_params += 1
        payload_bytes_from_calls += amt_data_transfer
        data_transferred += amt_data_transfer
        # do global update (gossip average step) in the pre forward hook, 
        # then compute output in the forward pass
        output = model(input_var)
        loss = criterion(output, target_var)

        if args.optimizer.lower() == 'edlngc':
            lambda_t = min(1.0, current_round / (total_rounds / 2.0))
            criterion.lambda_t = lambda_t
            sender.criterion.lambda_t = lambda_t
            evidence = F.softplus(output)
            loss = criterion(evidence, target_var)
        else:
            loss = criterion(output, target_var)

        all_outputs.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())
        # compute gradient 
        loss.backward()

        if 'cga' in args.optimizer.lower() or 'ngc' in args.optimizer.lower():
            if args.optimizer.lower() == 'edlngc':
                cross_grad, cross_unc, ref_buf = sender(cross_weights, input_var, target_var) 
                cross_grad_copy = copy.deepcopy(cross_grad)
                cross_uncertainty_copy  = copy.deepcopy(cross_unc)
                # Transmit gradients through topology
                _, amt_data_transfer, recieved_cross_grad = model.transfer_additional(cross_grad)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls += amt_data_transfer
                # Transmit uncertainty through topology
                _, amt_data_transfer_unc, recieved_cross_uncertainty = model.transfer_additional(cross_unc)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls += amt_data_transfer_unc
                
                data_transferred += (amt_data_transfer + amt_data_transfer_unc)
                
                # Call receiver
                receiver(recieved_cross_grad, cross_grad_copy, recieved_cross_uncertainty, cross_uncertainty_copy, ref_buf, current_round, total_rounds)
                receiver.project_gradients(lr)
            else:
                #send and recieve cross gradients
                cross_grad, ref_buf                       = sender(cross_weights, input_var, target_var) 
                cross_grad_copy                           = copy.deepcopy(cross_grad)
                _, amt_data_transfer, recieved_cross_grad = model.transfer_additional(cross_grad)
                comm_calls_transfer_additional += 1
                payload_bytes_from_calls += amt_data_transfer
                receiver(recieved_cross_grad, cross_grad_copy, ref_buf)
                data_transferred    +=amt_data_transfer
                #project the gradients
                receiver.project_gradients(lr)
        elif args.optimizer.lower() == 'd-psgd':
            receiver.update_gradients(lr)

        # do local update
        optimizer.step()
        #zero out the gradients
        optimizer.zero_grad() 
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i % monitor_every) == 0:
            now_wall = time.perf_counter()
            now_cpu  = time.process_time()
            dt_wall = max(now_wall - last_wall, 1e-9)
            dt_cpu  = max(now_cpu - last_cpu, 0.0)
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
                      dist.get_rank(), epoch, i, len(train_loader),  batch_time=batch_time,
                      loss=losses, top1=top1))
        step += batch_size
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    prec, rec, f1 = precision_recall_f1(all_outputs, all_targets, num_classes=args.classes)

    if dist.get_rank() == 0:
        print(
            f"[Train][Epoch {epoch}] "
            f"Precision = {prec:.2f}  "
            f"Recall = {rec:.2f}  "
            f"F1 = {f1:.2f}  "
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
            if args.optimizer.lower() == 'edlngc':
                evidence = F.softplus(output)
                loss = criterion(evidence, target_var)
            else:
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

def read_net_dev(iface: str | None = None):
    """
    Return dict: rx_bytes, rx_packets, tx_bytes, tx_packets
    If iface is None -> sum all non-loopback/non-virtual interfaces.
    Linux only.
    """
    stats = {}
    with open("/proc/net/dev", "r") as f:
        lines = f.readlines()[2:]  # skip headers

    for ln in lines:
        if ":" not in ln:
            continue
        name, data = ln.split(":", 1)
        name = name.strip()
        fields = data.split()
        # fields layout: rx_bytes rx_packets ... tx_bytes tx_packets ...
        rx_bytes = int(fields[0]); rx_packets = int(fields[1])
        tx_bytes = int(fields[8]); tx_packets = int(fields[9])
        stats[name] = (rx_bytes, rx_packets, tx_bytes, tx_packets)

    if iface:
        rx_b, rx_p, tx_b, tx_p = stats.get(iface, (0, 0, 0, 0))
        return {"rx_bytes": rx_b, "rx_packets": rx_p, "tx_bytes": tx_b, "tx_packets": tx_p}

    # auto-sum (exclude common virtual interfaces)
    rx_b = rx_p = tx_b = tx_p = 0
    for name, (a, b, c, d) in stats.items():
        if name == "lo":
            continue
        if name.startswith(("docker", "veth", "br-", "virbr", "vmnet")):
            continue
        rx_b += a; rx_p += b; tx_b += c; tx_p += d
    return {"rx_bytes": rx_b, "rx_packets": rx_p, "tx_bytes": tx_b, "tx_packets": tx_p}

def diff_stats(after: dict, before: dict):
    return {k: int(after[k]) - int(before.get(k, 0)) for k in after.keys()}

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
        # thêm cột mới nếu muốn:
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
    