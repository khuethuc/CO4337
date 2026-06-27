
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
import torchvision.transforms.functional as TF
import math

# # # Fed-ISIC2019 # # # 
_FEDISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
_FEDISIC2019_NUM_CLASSES = len(_FEDISIC2019_CLASSES)  # 8

_FEDISIC2019_MEAN = [0.6681, 0.5301, 0.5247]
_FEDISIC2019_STD  = [0.1337, 0.1480, 0.1595]

class FedISIC2019Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, meta_path: str, transform=None):
        self.images_dir = images_dir
        self.transform  = transform

        df = pd.read_csv(meta_path)
        ids     = df["isic_id"].astype(str).str.strip().tolist()
        labels  = df["label"].to_numpy(dtype=np.int64)
        centers = df["center_id"].tolist() if "center_id" in df.columns else [-1] * len(ids)

        keep = [i for i, id_ in enumerate(ids)
                if os.path.exists(os.path.join(images_dir, f"{id_}.jpg"))]
        if len(keep) < len(ids):
            print(f"[FedISIC2019] {len(ids) - len(keep)} images missing on disk — skipped")

        self.ids     = [ids[i]     for i in keep]
        self.y       = labels[keep]
        self.centers = [centers[i] for i in keep]
        self.indices = np.arange(len(self.y))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx      = int(self.indices[i])
        img_path = os.path.join(self.images_dir, f"{self.ids[idx]}.jpg")
        img      = Image.open(img_path).convert("RGB")
        target   = int(self.y[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _load_fedisic2019_images(data_dir: str, train: bool, transform) -> FedISIC2019Dataset:
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"images/ not found in {data_dir}\n"
            "Please download dataset Fed-ISIC-2019"
        )
    meta_fname = "metadata_train.csv" if train else "metadata_test.csv"
    meta_path  = os.path.join(data_dir, meta_fname)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            "Please download dataset Fed-ISIC-2019"
        )
    return FedISIC2019Dataset(images_dir, meta_path, transform=transform)


def _fedisic2019_center_split(dataset: FedISIC2019Dataset, rank: int, world_size: int, seed: int) -> list:
    unique_centers = sorted(set(dataset.centers))
    num_centers    = len(unique_centers)
    center_idx     = rank % num_centers
    sub_rank       = rank // num_centers
    my_center      = unique_centers[center_idx]

    center_flat = [i for i, c in enumerate(dataset.centers) if c == my_center]

    agents_this_center = math.ceil((world_size - center_idx) / num_centers)

    rng = np.random.RandomState(seed + center_idx)
    arr = np.array(center_flat)
    rng.shuffle(arr)
    splits = np.array_split(arr, agents_this_center)

    my_indices = splits[sub_rank].tolist()
    print(f"[FedISIC2019][Rank {rank}] center={my_center} "
          f"sub={sub_rank}/{agents_this_center} samples={len(my_indices)}")
    return my_indices


# # # HAM10000 # # #
_HAM10000_DX_TO_LABEL = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,

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

    if "label" in df.columns:
        y_all = df["label"].to_numpy(dtype=np.int64)
        return ids_all, y_all

    diag_cols = [c for c in ["diagnosis_3", "diagnosis_2", "diagnosis_1", "dx", "diagnosis", "diagnosis_name"] if c in df.columns]
    if not diag_cols:
        raise ValueError(
            "Metadata has no label and no diagnosis columns. "
            f"Columns: {list(df.columns)}"
        )

    def norm(x: str) -> str:
        return str(x).strip().lower()

    coarse = {"benign", "malignant", "unknown", "nan", "none", ""}

    keep_ids = []
    keep_y = []

    for i in range(len(df)):
        isic_id = ids_all[i]

        chosen = ""
        for c in diag_cols:
            v = norm(df.iloc[i][c])
            if v not in coarse:
                chosen = v
                break

        if not chosen:
            continue

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

class CorruptionTransform:
    def __init__(self, noise_std: float = 0.0, blur_radius: int = 0):
        self.noise_std   = noise_std
        self.blur_radius = blur_radius

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (C, H, W)
        if self.noise_std > 0.0:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise
        if self.blur_radius > 0:
            k = self.blur_radius * 2 + 1
            tensor = TF.gaussian_blur(tensor, kernel_size=k, sigma=self.blur_radius)
        return tensor

class QualityPartition(object):
    def __init__(self, data, index,
                 corruption_transform=None,
                 retain_ratio: float = 1.0,
                 label_noise_rate: float = 0.0,
                 num_classes: int = 7,
                 seed: int = 0,
                 rank: int = 0,
                 noise_type: str = 'uniform',
                 noise_alpha: float = 0.1):
        self.data = data
        self.corruption = corruption_transform

        rng = np.random.RandomState(seed + rank + 42)

        index = list(index)
        if retain_ratio < 1.0:
            keep_n = max(1, int(len(index) * retain_ratio))
            index  = rng.choice(index, size=keep_n, replace=False).tolist()
        self.index = index

        if label_noise_rate > 0.0:
            if hasattr(self.data, 'y'):
                y_arr = self.data.y
            elif hasattr(self.data, 'targets'):
                y_arr = self.data.targets
            else:
                raise AttributeError("Dataset has neither .y nor .targets")

            if noise_type == 'dirichlet':
                """
                Dirichlet label noise
                Each true class i has transition row T[i]:
                    T[i, i]   = 1 - noise_rate  # keep original label
                    T[i, j != i] = noise_rate x w_j # flip to class j with Dirichlet weight
                    w ~ Dirichlet(alpha x 1_{K-1})
                small alpha -> close to pair-flip
                large alpha -> close to USN
                """
                T = _build_dirichlet_transition(num_classes, label_noise_rate,
                                                noise_alpha, rng)
                if rank == 0:
                    print(f"[DirichletNoise][Rank {rank}] "
                          f"alpha={noise_alpha:.3f}  noise_rate={label_noise_rate:.2f}")
                    print("  Transition matrix T (rows = true class, cols = noisy class):")
                    header = "       " + "".join(f"  c{k:<3}" for k in range(num_classes))
                    print(header)
                    for c in range(num_classes):
                        row_str = "".join(f"  {T[c,k]:.3f}" for k in range(num_classes))
                        print(f"  c{c} → {row_str}")

                n_flipped = 0
                for pos in range(len(self.index)):
                    dataset_pos = self.index[pos]
                    if hasattr(self.data, 'indices'):
                        actual_idx = int(self.data.indices[dataset_pos])
                    else:
                        actual_idx = dataset_pos
                    orig      = int(y_arr[actual_idx])
                    new_label = int(rng.choice(num_classes, p=T[orig]))
                    if new_label != orig:
                        y_arr[actual_idx] = new_label
                        n_flipped += 1
                actual_rate = n_flipped / max(len(self.index), 1)
                print(f"[DirichletNoise][Rank {rank}] "
                      f"flipped={n_flipped}/{len(self.index)} "
                      f"(actual {actual_rate*100:.1f}%, "
                      f"expected ~{label_noise_rate*100:.0f}%)")

            else:
                # Uniform Symmetric Noise (USN)
                n_noisy   = int(label_noise_rate * len(self.index))
                noisy_pos = rng.choice(len(self.index), size=n_noisy, replace=False)
                for pos in noisy_pos:
                    dataset_pos = self.index[pos]
                    if hasattr(self.data, 'indices'):
                        actual_idx = int(self.data.indices[dataset_pos])
                    else:
                        actual_idx = dataset_pos
                    orig   = int(y_arr[actual_idx])
                    others = [c for c in range(num_classes) if c != orig]
                    y_arr[actual_idx] = int(rng.choice(others))
                print(f"[UniformNoise][Rank {rank}] label noise {n_noisy}/{len(self.index)} "
                      f"({label_noise_rate*100:.0f}%)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        data_idx = self.index[i]
        img, target = self.data[data_idx]
        if self.corruption is not None:
            img = self.corruption(img)
        return img, target


def _build_dirichlet_transition(
    num_classes: int,
    noise_rate: float,
    alpha: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    T = np.zeros((num_classes, num_classes), dtype=np.float64)
    for c in range(num_classes):
        alpha_vec        = np.ones(num_classes - 1) * alpha
        off_diag_weights = rng.dirichlet(alpha_vec)   # shape (K-1,), sums to 1
        j = 0
        for k in range(num_classes):
            if k == c:
                T[c, k] = 1.0 - noise_rate
            else:
                T[c, k] = noise_rate * off_diag_weights[j]
                j += 1
    return T

def make_quality_profiles(
    world_size: int,
    mode: str = "uniform",
    seed: int = 0,
    noise_type: str = "uniform",
    noise_alpha: float = 0.1,
) -> list[dict]:

    rng = np.random.RandomState(seed)

    def _prof(**kw):
        return dict(**kw, noise_type=noise_type, noise_alpha=noise_alpha)

    if mode == "uniform":
        return [_prof(noise_std=0.0, blur_radius=0, retain_ratio=1.0, label_noise_rate=0.0)
                for _ in range(world_size)]

    if mode == "tiered":
        profiles = []
        tier_size = world_size // 5
        for i in range(world_size):
            if i < tier_size: # good nodes
                p = _prof(noise_std=0.0,  blur_radius=0, retain_ratio=1.0, label_noise_rate=0.0)
            elif i < 2 * tier_size: # medium nodes
                p = _prof(noise_std=0.08, blur_radius=1, retain_ratio=1.0, label_noise_rate=0.05)
            else: # poor nodes
                p = _prof(noise_std=0.20, blur_radius=2, retain_ratio=1.0, label_noise_rate=0.15)
            profiles.append(p)
        return profiles

    if mode == "random":
        profiles = []
        for _ in range(world_size):
            p = _prof(
                noise_std        = float(rng.uniform(0.0, 0.25)),
                blur_radius      = int(rng.choice([0, 1, 2])),
                retain_ratio     = 1.0,
                label_noise_rate = float(rng.uniform(0.0, 0.20)),
            )
            profiles.append(p)
        return profiles

    raise ValueError(f"Unknown mode: {mode}")


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
    def __init__(self, data, sizes, skew, seed, dataset_name):
        
        self.data = data
        self.partitions = []
        data_len = len(data)
        if hasattr(data, "targets"):
            labels = [int(x) for x in data.targets]
        elif hasattr(data, "y") and hasattr(data, "indices"):
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
                assigned = set(mixed[:part_len])
                self.partitions.append(mixed[:part_len])

                sort_indices = mixed[part_len:]
                indices_rand = [x for x in indices_rand if x not in assigned]


    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_trainDataset(dataset_name, data_dir, skew, seed, batch_size,
                        num_classes, noise_rate=0.0, noise_agents=None,
                        quality_mode: str = "uniform", quality_profiles: list = None,
                        noise_type: str = "uniform", noise_alpha: float = 0.1):
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
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = _load_ham10000_images(data_dir=data_dir, seed=seed, train=True, transform=train_tf)

    elif dataset_name == "fedisic2019":
        normalize = transforms.Normalize(
            mean=_FEDISIC2019_MEAN, std=_FEDISIC2019_STD
        )
        train_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = _load_fedisic2019_images(data_dir=data_dir, train=True, transform=train_tf)

    rank = dist.get_rank()
    size = dist.get_world_size()

    partition_sizes = [1.0 / size for _ in range(size)]
    dp = DataPartitioner(dataset, partition_sizes, skew=skew,
                         seed=seed, dataset_name=dataset_name)
    raw_partition = dp.use(rank)

    if quality_profiles is None:
        quality_profiles = make_quality_profiles(
            size, mode=quality_mode, seed=seed,
            noise_type=noise_type, noise_alpha=noise_alpha,
        )
    prof = quality_profiles[rank]

    if noise_rate > 0.0 and (not noise_agents or rank in noise_agents):
        prof = dict(prof, label_noise_rate=noise_rate)

    corruption = CorruptionTransform(
        noise_std  = prof["noise_std"],
        blur_radius= prof["blur_radius"],
    ) if (prof["noise_std"] > 0 or prof["blur_radius"] > 0) else None

    partition = QualityPartition(
        data                 = raw_partition.data,
        index                = raw_partition.index,
        corruption_transform = corruption,
        retain_ratio         = prof["retain_ratio"],
        label_noise_rate     = prof["label_noise_rate"],
        num_classes          = num_classes,
        seed                 = seed,
        rank                 = rank,
        noise_type           = prof.get("noise_type", "uniform"),
        noise_alpha          = prof.get("noise_alpha", 0.1),
    )

    print(f"[Rank {rank}] quality={prof}, data_size={len(partition)}")

    bsz       = int(batch_size / float(size))
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True, num_workers=4
    )
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

    elif dataset_name == "fedisic2019":
        normalize = transforms.Normalize(
            mean=_FEDISIC2019_MEAN, std=_FEDISIC2019_STD
        )
        val_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = _load_fedisic2019_images(data_dir=data_dir, train=False, transform=val_tf)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=0)

    return val_set, val_bsz