
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


# # # Helper functions for Dirichlet attack # # #
def _get_labels_cached(dataset, dataset_name: str, cache_dir: str = "."):
    """Return numpy array of labels for a torchvision-style dataset.
    Cache to labels{dataset_name}.npy for speed.
    """
    cache_path = os.path.join(cache_dir, f"labels{str(dataset_name)}.npy")
    try:
        labels = np.load(cache_path, allow_pickle=True)
        return np.asarray(labels, dtype=np.int64)
    except Exception:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=32)
        labels = []
        for _, targets in loader:
            labels += targets.tolist()
        np.save(cache_path, np.asarray(labels, dtype=np.int64))
        return np.asarray(labels, dtype=np.int64)

def dirichlet_partition_indices(labels: np.ndarray, n_clients: int, n_classes: int, alpha: float,
                                seed: int = 0, min_size: int = 10):
    """Partition dataset indices among clients using Dirichlet(alpha) per class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    while True:
        client_indices = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            if len(idx_by_class[c]) == 0:
                continue
            proportions = rng.dirichlet(np.ones(n_clients) * float(alpha))
            splits = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
            parts = np.split(idx_by_class[c], splits)
            for i in range(n_clients):
                client_indices[i].extend(parts[i].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break

    for i in range(n_clients):
        rng.shuffle(client_indices[i])
    return client_indices

def apply_dirichlet_attack_to_partition(labels: np.ndarray,
                                       base_partition: list,
                                       adv_client_ids: list,
                                       n_classes: int,
                                       alpha_attack: float,
                                       seed: int = 123):
    """Re-partition only samples already assigned to malicious clients with smaller alpha."""
    if not adv_client_ids:
        return base_partition

    adv_client_ids = sorted(set(int(x) for x in adv_client_ids))
    n_clients = len(base_partition)
    for cid in adv_client_ids:
        if cid < 0 or cid >= n_clients:
            raise ValueError(f"adv_client_id {cid} out of range [0,{n_clients-1}]")

    adv_pool = []
    for cid in adv_client_ids:
        adv_pool.extend(base_partition[cid])
    adv_pool = np.asarray(adv_pool, dtype=np.int64)

    rng = np.random.default_rng(seed)
    adv_labels = labels[adv_pool]
    idx_by_class = [adv_pool[np.where(adv_labels == c)[0]] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    k = len(adv_client_ids)
    new_parts = {cid: [] for cid in adv_client_ids}

    for c in range(n_classes):
        if len(idx_by_class[c]) == 0:
            continue
        proportions = rng.dirichlet(np.ones(k) * float(alpha_attack))
        splits = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        parts = np.split(idx_by_class[c], splits)
        for j, cid in enumerate(adv_client_ids):
            new_parts[cid].extend(parts[j].tolist())

    out = [list(x) for x in base_partition]
    for cid in adv_client_ids:
        rng.shuffle(new_parts[cid])
        out[cid] = new_parts[cid]
    return out

def get_adv_clients(n_clients: int, adv_ratio: float, seed: int,
                    contiguous: bool = False, start: int = 0):
    adv_n = int(round(float(adv_ratio) * n_clients))
    adv_n = max(0, min(n_clients, adv_n))
    if adv_n == 0:
        return []
    if contiguous:
        start = int(start) % n_clients
        return [(start + i) % n_clients for i in range(adv_n)]
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n_clients), size=adv_n, replace=False).astype(int).tolist()


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
        # -------- labels: take directly from dataset --------
        if hasattr(data, "targets"):
            labels = [int(x) for x in data.targets]
        elif hasattr(data, "y") and hasattr(data, "indices"):
            # IMPORTANT: data.y is global labels, but len(data) is subset length
            labels = [int(data.y[int(idx)]) for idx in data.indices]
        elif hasattr(data, "y"):
            labels = [int(x) for x in data.y]
        elif hasattr(data, "labels"):
            labels = [int(x) for x in data.labels]
        else:
            loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False, num_workers=0)
            labels = []
            for _, targets in loader:
                labels.extend([int(t) for t in targets])
        
        assert len(labels) == len(data), f"labels({len(labels)}) != len(data)({len(data)})"
        # -------- end labels --------
        
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

    
def partition_trainDataset(dataset_name, data_dir, skew, seed, batch_size,
                           partition_mode=None, dirichlet_alpha=0.3, n_classes=10, min_size=10,
                           dirichlet_attack=False, adv_ratio=0.0, adv_contiguous=False,
                           adv_start=0, attack_alpha=0.05):
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
    bsz = int((batch_size) / float(size))

    # Choose partition mode: skew < 0.5 is iid, skew >= 0.5 is sort
    if partition_mode is None:
        partition_mode = "sort" if float(skew) >= 0.5 else "iid"

    if partition_mode in ["iid", "random"]:
        partition_sizes = [1.0/size for _ in range(size)]
        partition = DataPartitioner(dataset, partition_sizes, skew=0, seed=seed, dataset_name=dataset_name)
        partition = partition.use(dist.get_rank())
        train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
        return train_set, bsz

    if partition_mode in ["sort", "label", "non-iid"]:
        partition_sizes = [1.0/size for _ in range(size)]
        partition = DataPartitioner(dataset, partition_sizes, skew=1, seed=seed, dataset_name=dataset_name)
        partition = partition.use(dist.get_rank())
        train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
        return train_set, bsz

    # Dirichlet partition + optional Dirichlet attack
    labels = _get_labels_cached(dataset, dataset_name)
    client_indices = dirichlet_partition_indices(
        labels=labels,
        n_clients=size,
        n_classes=n_classes,
        alpha=dirichlet_alpha,
        seed=seed,
        min_size=min_size,
    )

    if dirichlet_attack and adv_ratio > 0:
        adv_clients = get_adv_clients(size, adv_ratio, seed + 2027,
                                    contiguous=adv_contiguous, start=adv_start)
        client_indices = apply_dirichlet_attack_to_partition(
            labels=labels,
            base_partition=client_indices,
            adv_client_ids=adv_clients,
            n_classes=n_classes,
            alpha_attack=attack_alpha,
            seed=seed + 9999,
        )

    partition = Partition(dataset, client_indices[dist.get_rank()])
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