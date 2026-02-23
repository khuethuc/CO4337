
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


_HAM10000_DX_TO_LABEL = {
    # canonical HAM10000 short codes
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,

    # common long-form aliases (ISIC-style)
    "actinic keratosis": 0,
    "bowen": 0,
    "bowen's disease": 0,
    "intraepithelial carcinoma": 0,
    "squamous cell carcinoma in situ": 0,

    "basal cell carcinoma": 1,

    "benign keratosis": 2,
    "seborrheic keratosis": 2,
    "solar lentigo": 2,
    "lichen planus-like keratosis": 2,

    "dermatofibroma": 3,

    "melanoma": 4,

    "nevus": 5,
    "melanocytic nevus": 5,

    "vascular lesion": 6,
    "angioma": 6,
    "hemangioma": 6,
    "pyogenic granuloma": 6,
}

def _find_metadata_path(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "HAM10000_metadata.csv"),
        os.path.join(data_dir, "isic_ham10000_metadata.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find metadata. Need 1 file in: {', '.join(os.path.basename(x) for x in candidates)}"
    )

def _extract_ids_labels(df: pd.DataFrame):
    # id column
    id_candidates = ["isic_id", "image_id", "name", "image", "_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"Cannot find id column in metadata. Columns: {list(df.columns)}")
    ids_all = df[id_col].astype(str).str.strip().tolist()

    # if numeric label exists, use it
    if "label" in df.columns:
        y_all = df["label"].to_numpy(dtype=np.int64)
        return ids_all, y_all

    # choose the "most specific" diagnosis column available: 3 -> 2 -> 1
    diag_cols = [c for c in ["diagnosis_3", "diagnosis_2", "diagnosis_1", "dx", "diagnosis", "diagnosis_name"] if c in df.columns]
    if not diag_cols:
        raise ValueError(
            "Metadata has no label and no diagnosis columns. "
            f"Columns: {list(df.columns)}"
        )

    # helper to normalize text
    def norm(x: str) -> str:
        return str(x).strip().lower()

    # values that are too coarse (can't map to 7 classes)
    coarse = {"benign", "malignant", "unknown", "nan", "none", ""}

    keep_ids = []
    keep_y = []

    for i in range(len(df)):
        isic_id = ids_all[i]

        # pick first non-coarse diagnosis among diag_cols (prefer diagnosis_3 then 2 then 1)
        chosen = ""
        for c in diag_cols:
            v = norm(df.iloc[i][c])
            if v not in coarse:
                chosen = v
                break

        if not chosen:
            # skip if we only have benign/malignant/unknown
            continue

        # direct match or substring match
        if chosen in _HAM10000_DX_TO_LABEL:
            lab = _HAM10000_DX_TO_LABEL[chosen]
            keep_ids.append(isic_id)
            keep_y.append(lab)
            continue

        matched = None
        for k, lab in _HAM10000_DX_TO_LABEL.items():
            if k in chosen:
                matched = lab
                break

        if matched is None:
            # skip unknown diagnosis values instead of crashing
            continue

        keep_ids.append(isic_id)
        keep_y.append(matched)

    if len(keep_ids) == 0:
        raise ValueError(
            "After filtering, no samples could be mapped to 7 HAM10000 classes. "
            "Your metadata may not contain 7-class diagnosis information. "
            "Use HAM10000_metadata.csv (dx) or provide a mapping."
        )

    return keep_ids, np.array(keep_y, dtype=np.int64)

class HAM10000ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, ids, y, indices=None, transform=None):
        self.images_dir = images_dir
        self.ids = list(ids)
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices) if indices is not None else np.arange(len(self.y))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        isic_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, f"{isic_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        target = int(self.y[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def _load_ham10000_images(data_dir: str, seed: int, train: bool, transform):
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Cannot find folder images/: {images_dir}")

    meta_path = _find_metadata_path(data_dir)
    df = pd.read_csv(meta_path)
    ids, y = _extract_ids_labels(df)

    keep_ids, keep_y = [], []
    for _id, _lab in zip(ids, y):
        p = os.path.join(images_dir, f"{_id}.jpg")
        if os.path.exists(p):
            keep_ids.append(_id)
            keep_y.append(int(_lab))

    if len(keep_ids) == 0:
        raise FileNotFoundError(f"No .jpg matched after mapping in {images_dir}")

    y2 = np.array(keep_y, dtype=np.int64)
    all_idx = np.arange(len(y2))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=seed, stratify=y2)
    indices = train_idx if train else val_idx
    return HAM10000ImageDataset(images_dir, keep_ids, y2, indices=indices, transform=transform)

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
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = _load_ham10000_images(data_dir=data_dir, seed=seed, train=True, transform=train_tf)    
       
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
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = _load_ham10000_images(data_dir=data_dir, seed=seed, train=False, transform=val_tf)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)

    return val_set, val_bsz