
import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import random
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


_HAM_CACHE = {}

def _load_ham10000_rgb(data_dir, filename="hmnist_28_28_RGB.csv"):
    """
      - pixel0000..pixel2351: 28*28*3 = 2352 pixels
      - label: int 0..6
    """
    path = os.path.join(data_dir, filename)
    key = os.path.abspath(path)

    if key in _HAM_CACHE:
        return _HAM_CACHE[key]

    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=np.int64)  # (N,)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)  # (N, 2352)
    X = X.reshape(-1, 28, 28, 3)  # (N, H, W, C)

    _HAM_CACHE[key] = (X, y)
    return X, y


class HAM10000CSVDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, indices=None, transform=None):
        self.X = X
        self.y = y
        self.indices = np.array(indices) if indices is not None else np.arange(len(y))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = Image.fromarray(self.X[idx])  # PIL (H, W, C)
        target = int(self.y[idx])

        if self.transform is not None:
            img = self.transform(img)
        return img, target

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
                
class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""
    def __init__(self, data, sizes, skew, seed, dataset_name):
        
        self.data = data
        self.partitions = []
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False, num_workers=32)
        labels = []

        cache_file = f"labels_{dataset_name}_n{len(data)}_seed{seed}.npy"
        try:
            labels = np.load(cache_file).tolist()
        except:
            for _, targets in dataset:
                labels = labels + targets.tolist()
            np.save(cache_file, np.array(labels, dtype=np.int64))
        
        rng = random.Random()
        rng.seed(seed)
        indices_rand = np.arange(len(labels)).tolist()
        rng.shuffle(indices_rand)
        sort_index   = np.argsort(np.array(labels))
        sort_indices = sort_index.tolist()
        
        for i, frac in enumerate(sizes):
            if skew==1:
                part_len = int(frac*data_len)
                self.partitions.append(sort_indices[0:part_len])
                if len(sizes)>10 and i<10:
                    #print('here', i, len(sizes), len(indices))
                    sort_indices = sort_indices[2*part_len:]+sort_indices[part_len:2*part_len]
                else:
                    sort_indices = sort_indices[part_len:]
            elif skew==0:
                part_len = int(frac*data_len)
                self.partitions.append(indices_rand[0:part_len])
                indices_rand = indices_rand[part_len:] 
            else:
                # 0 < skew < 1: mix sorted and random
                n = len(labels)
                n_sorted = int(skew * n)
                chosen = set()
                mixed = []
                # first part is sorted (label-skew)
                for idx in sort_indices[:n_sorted]:
                    mixed.append(idx)
                    chosen.add(idx)
                # second part is random
                for idx in indices_rand:
                    if idx not in chosen:
                        mixed.append(idx)

                part_len = int(frac * data_len)
                self.partitions.append(mixed[:part_len])

                sort_indices = mixed[part_len:]


    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_trainDataset(dataset_name, data_dir, skew, seed, batch_size):
    """Partitioning dataset""" 
    if dataset_name== 'cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
    elif dataset_name== 'fmnist':
        normalize  = transforms.Normalize((0.5,), (0.5,))
        dataset = datasets.FashionMNIST(root=data_dir, train = True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
    elif dataset_name== 'cifar100':
        normalize  = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
    elif dataset_name== 'imagenette':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(32),
                                 transforms.RandomResizedCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)

    elif dataset_name== 'imagenette_full':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)

    elif dataset_name== 'imagenet':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)

    elif dataset_name == "ham10000":
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        train_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])

        X, y = _load_ham10000_rgb(data_dir)

        all_idx = np.arange(len(y))
        train_idx, _ = train_test_split(
            all_idx, test_size=0.2, random_state=seed, stratify=y
        )
        dataset = HAM10000CSVDataset(X, y, indices=train_idx, transform=train_tf)          
       
    size = dist.get_world_size()
    #print(size)
    bsz = int((batch_size) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    #print(partition_sizes, len(dataset))
    partition = DataPartitioner(dataset, partition_sizes, skew=skew, seed=seed, dataset_name=dataset_name)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
    return train_set, bsz


def test_Dataset(dataset_name, data_dir, seed=321):
    if dataset_name=='cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset_name=='fmnist':
        normalize = transforms.Normalize((0.5,), (0.5,))
        dataset   = datasets.FashionMNIST(root=data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset_name=='cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset_name== 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(32),
                                 transforms.CenterCrop(32),
                                 transforms.ToTensor(), normalize,])

        data_dir = data_dir

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)
    elif dataset_name== 'imagenette_full':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize,])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)

    elif dataset_name== 'imagenet':
        #/local/a/imagenet/imagenet2012/
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize,])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)
    elif dataset_name == "ham10000":
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        val_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
        X, y = _load_ham10000_rgb(data_dir)
        all_idx = np.arange(len(y))
        _, val_idx = train_test_split(
            all_idx, test_size=0.2, random_state=seed, stratify=y
        )
        dataset = HAM10000CSVDataset(X, y, indices=val_idx, transform=val_tf)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)

    return val_set, val_bsz