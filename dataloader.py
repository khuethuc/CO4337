
import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import random
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import kagglehub

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
        try:
            labels = np.load('labels'+str(dataset_name)+'.npy')
        except:
            for batch_idx, (inputs, targets) in enumerate(dataset):
                labels = labels+targets.tolist()
            np.save('labels'+str(dataset_name)+'.npy', labels)
        
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


    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class MIMICDataset(Dataset):
    def __init__(self, dataframe, root_dir, path_col="path", label_cols=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        if path_col not in self.df.columns:
            raise ValueError(f"Image path column not found: '{path_col}'. Columns: {list(self.df.columns)}")

        self.path_col = path_col

        if label_cols is None:
            ignore = set([path_col, "patient_id", "study_id", "dicom_id", "subject_id",
                          "ViewPosition", "Portable", "StudyDate", "StudyYear", "_group", "_portable", "_view"])
            label_cols = [c for c in self.df.columns if c not in ignore]

        self.label_cols = label_cols

        self.image_paths = self.df[self.path_col].astype(str).values
        self.labels = self.df[self.label_cols].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_rel)

        if not os.path.exists(img_path) and os.path.exists(img_rel):
            img_path = img_rel

        image = Image.open(img_path).convert("RGB")
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, y
    
def _find_csv(root_dir, preferred_names=("mimic_cxr_aug_train.csv", "mimic_cxr_aug_validate.csv")):
    hits = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            low = fn.lower()
            for p in preferred_names:
                if low == p.lower():
                    hits[p] = os.path.join(dirpath, fn)
    return hits

def _infer_col(df, candidates):
    """Finding existing column in df from candidate list (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _make_cxr_transforms(train=True, img_size=224):
    """Basic transforms for CXR. (You can further customize)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

def _build_realistic_groups(df, portable_col, view_col):
    """
    Build realistic key groups from Portable x ViewPosition columns.
    - Portable: 1/0, True/False, 'Y'/'N', 'portable'/'fixed'... convert to 'P'/'NP'
    - View: 'AP', 'PA', others -> 'UNK'
    """
    def norm_portable(v):
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "y", "yes", "portable", "p"):
            return "P"
        if s in ("0", "false", "f", "n", "no", "fixed", "non-portable", "np"):
            return "NP"
        return "UNK"

    def norm_view(v):
        s = str(v).strip().upper()
        if s in ("AP", "PA"):
            return s
        return "UNK"

    if portable_col is None:
        df["_portable"] = "UNK"
    else:
        df["_portable"] = df[portable_col].apply(norm_portable)

    if view_col is None:
        df["_view"] = "UNK"
    else:
        df["_view"] = df[view_col].apply(norm_view)

    df["_group"] = df["_portable"] + "_" + df["_view"]
    return df


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
        #data_dir = "/local/a/imagenet/imagenet2012/" #
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
        #print(len(dataset))  

    elif dataset_name == 'mimic_cxr':
        # Finding csv files
        csv_map = _find_csv(data_dir, preferred_names=("mimic_cxr_aug_train.csv", "mimic_cxr_aug_validate.csv"))
        if "mimic_cxr_aug_train.csv" not in csv_map:
            raise FileNotFoundError(f"mimic_cxr_aug_train.csv not found in {data_dir}. Found: {csv_map}")

        train_csv_path = csv_map["mimic_cxr_aug_train.csv"]
        df = pd.read_csv(train_csv_path)

        # Determining path column
        path_col = _infer_col(df, ["path", "image_path", "img_path", "filepath", "file_path"])
        if path_col is None:
            raise ValueError(f"Image path column not found in mimic_cxr_aug_train.csv. Columns: {list(df.columns)}")
        
        # Determining metadata columns to create group (portable/view)
        portable_col = _infer_col(df, ["Portable", "portable", "is_portable"])
        view_col     = _infer_col(df, ["ViewPosition", "view_position", "View", "view"])

        df = _build_realistic_groups(df, portable_col, view_col)

        # Map group -> rank/agent/node (realistic cross-silo)
        size = dist.get_world_size()
        rank = dist.get_rank()

        groups = df["_group"].value_counts().index.tolist()
        groups = sorted(groups)

        # round-robin assign group for rank
        group_to_rank = {g: (i % size) for i, g in enumerate(groups)}
        df_rank = df[df["_group"].map(group_to_rank) == rank].copy()

        # Shuffle in rank to avoid ordering bias
        df_rank = df_rank.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # Label columns
        train_tf = _make_cxr_transforms(train=True, img_size=224)
        dataset = MIMICDataset(df_rank, root_dir=data_dir, path_col=path_col, label_cols=None, transform=train_tf)

        # DataLoader according to batch of NGC
        bsz = int((batch_size) / float(size))
        train_set = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, num_workers=2)
        return train_set, bsz
       
       
    size = dist.get_world_size()
    #print(size)
    bsz = int((batch_size) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    #print(partition_sizes, len(dataset))
    partition = DataPartitioner(dataset, partition_sizes, skew=skew, seed=seed, dataset_name=dataset_name)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
    return train_set, bsz


def test_Dataset(dataset_name, data_dir):
  
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
    
    elif dataset_name == 'mimic_cxr':
        csv_map = _find_csv(data_dir, preferred_names=("mimic_cxr_aug_train.csv", "mimic_cxr_aug_validate.csv"))
        if "mimic_cxr_aug_validate.csv" not in csv_map:
            raise FileNotFoundError(f"File mimic_cxr_aug_validate.csv not found in {data_dir}. Found: {csv_map}")

        test_csv_path = csv_map["mimic_cxr_aug_validate.csv"]
        df = pd.read_csv(test_csv_path)

        path_col = _infer_col(df, ["path", "image_path", "img_path", "filepath", "file_path"])
        if path_col is None:
            raise ValueError(f"Image path column not found in mimic_cxr_aug_validate.csv. Columns: {list(df.columns)}")

        test_tf = _make_cxr_transforms(train=False, img_size=224)
        dataset = MIMICDataset(df, root_dir=data_dir, path_col=path_col, label_cols=None, transform=test_tf)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)

    return val_set, val_bsz