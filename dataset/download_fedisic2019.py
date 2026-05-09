"""
Download Fed-ISIC2019 from HuggingFace (flwrlabs/fed-isic2019) and cache
images + metadata to disk for use with dataloader.py.

Usage:
    python dataset/download_fedisic2019.py --out-dir ../data/fedisic2019

Output layout:
    ../data/fedisic2019/
        images/                 # flat JPEG directory
            ISIC_XXXXXXX.jpg
            ...
        metadata_train.csv      # columns: isic_id, label, center_id
        metadata_test.csv

Requirements:
    pip install datasets pillow tqdm pandas
"""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

DATASET_ID = "flwrlabs/fed-isic2019"

# ISIC 2019: 8 diagnostic classes (same order used in the challenge)
ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
_STR_TO_LABEL: dict[str, int] = {c: i for i, c in enumerate(ISIC2019_CLASSES)}
_STR_TO_LABEL.update({c.lower(): i for c in ISIC2019_CLASSES})
# numeric string fall-through handled in _to_label()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_label(val) -> int:
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    upper = s.upper()
    if upper in _STR_TO_LABEL:
        return _STR_TO_LABEL[upper]
    if s.lower() in _STR_TO_LABEL:
        return _STR_TO_LABEL[s.lower()]
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"Cannot map label value to int: {val!r}")


def _to_pil(val) -> Image.Image:
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(BytesIO(val)).convert("RGB")
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return Image.fromarray(val).convert("RGB")
    except ImportError:
        pass
    # HuggingFace datasets may wrap images in a dict {'bytes': ..., 'path': ...}
    if isinstance(val, dict):
        if val.get("bytes"):
            return Image.open(BytesIO(val["bytes"])).convert("RGB")
        if val.get("path") and os.path.exists(val["path"]):
            return Image.open(val["path"]).convert("RGB")
    raise TypeError(f"Cannot convert {type(val).__name__} to PIL Image")


def _detect_cols(row: dict):
    """Auto-detect column names from a sample row dict."""
    keys = set(row.keys())
    id_col = next(
        (c for c in ["isic_id", "image_id", "id", "name", "_id"] if c in keys), None
    )
    label_col = next(
        (c for c in ["label", "dx", "diagnosis", "class", "target"] if c in keys), None
    )
    image_col = next(
        (c for c in ["image", "img", "pixel_values", "pixels"] if c in keys), None
    )
    center_col = next(
        (c for c in ["center_id", "center", "site_id", "site", "hospital_id", "hospital"]
         if c in keys),
        None,
    )
    return id_col, label_col, image_col, center_col


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _process_split(
    ds_split,
    images_dir: str,
    split_tag: str,
    center_override=None,
) -> list[dict]:
    """Iterate one HuggingFace dataset split, save JPEGs, return metadata rows."""
    id_col = label_col = image_col = center_col = None
    records: list[dict] = []

    for i, row in enumerate(tqdm(ds_split, desc=f"  {split_tag}", unit="img")):
        # Detect columns on first row
        if id_col is None:
            id_col, label_col, image_col, center_col = _detect_cols(row)
            if image_col is None:
                raise ValueError(
                    f"[{split_tag}] Cannot find image column. "
                    f"Available keys: {list(row.keys())}"
                )
            if label_col is None:
                raise ValueError(
                    f"[{split_tag}] Cannot find label column. "
                    f"Available keys: {list(row.keys())}"
                )
            print(
                f"  Columns detected → id={id_col!r}  label={label_col!r}  "
                f"image={image_col!r}  center={center_col!r}"
            )

        # --- Image ID ---
        if id_col is not None:
            img_id = str(row[id_col]).strip()
        else:
            img_id = f"{split_tag}_{i:07d}"

        # --- Label ---
        label = _to_label(row[label_col])

        # --- Center ID ---
        if center_override is not None:
            center_id = center_override
        elif center_col is not None:
            center_id = row[center_col]
        else:
            center_id = -1

        # --- Save image (skip if already present) ---
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            try:
                pil = _to_pil(row[image_col])
                pil.save(img_path, "JPEG", quality=95)
            except Exception as exc:
                print(f"  [WARN] Cannot save {img_id}: {exc}", file=sys.stderr)
                continue

        records.append({"isic_id": img_id, "label": label, "center_id": center_id})

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _infer_center_from_split_name(split_name: str):
    """Try to parse an integer center ID from a split name like 'train_client_3'."""
    parts = split_name.replace("-", "_").split("_")
    for j, part in enumerate(parts):
        if part in ("site", "center", "client", "hospital", "node") and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download flwrlabs/fed-isic2019 from HuggingFace to disk."
    )
    ap.add_argument(
        "--out-dir", default="../data/fedisic2019",
        help="Root output directory (default: ../data/fedisic2019)"
    )
    ap.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace datasets cache directory (optional)"
    )
    ap.add_argument(
        "--num-proc", type=int, default=1,
        help="Parallel workers for HuggingFace download"
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print(
            "[ERROR] HuggingFace datasets not installed.\n"
            "        Run:  pip install datasets pillow tqdm pandas",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} → {out_dir}")

    # --- Discover configs ---
    try:
        configs = get_dataset_config_names(DATASET_ID)
    except Exception:
        configs = ["default"]
    print(f"Configs: {configs}")

    train_records: list[dict] = []
    test_records:  list[dict] = []

    if len(configs) <= 1:
        # Single config — may have multiple splits (one per center)
        ds = load_dataset(
            DATASET_ID,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
        )
        print(f"Splits: {list(ds.keys())}")

        for split_name, ds_split in ds.items():
            is_train = "train" in split_name.lower()
            center_override = _infer_center_from_split_name(split_name)

            records = _process_split(
                ds_split,
                str(images_dir),
                split_tag=split_name,
                center_override=center_override,
            )
            (train_records if is_train else test_records).extend(records)

    else:
        # Multiple configs — one per center/hospital
        for c_idx, cfg in enumerate(configs):
            ds = load_dataset(
                DATASET_ID, cfg,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
            )
            print(f"Config {cfg!r}: splits = {list(ds.keys())}")
            for split_name, ds_split in ds.items():
                is_train = "train" in split_name.lower()
                records = _process_split(
                    ds_split,
                    str(images_dir),
                    split_tag=f"{cfg}_{split_name}",
                    center_override=c_idx,
                )
                (train_records if is_train else test_records).extend(records)

    # --- Save metadata CSVs ---
    if not train_records and not test_records:
        print("[ERROR] No records saved — check dataset structure.", file=sys.stderr)
        return 1

    def _save_meta(records, fname):
        if not records:
            return
        df = pd.DataFrame(records)
        path = out_dir / fname
        df.to_csv(path, index=False)
        print(
            f"\n{fname}: {len(df)} images | "
            f"{df['label'].nunique()} classes | "
            f"{df['center_id'].nunique()} centers"
        )
        counts = df["label"].value_counts().sort_index()
        for lab, cnt in counts.items():
            class_name = ISIC2019_CLASSES[lab] if lab < len(ISIC2019_CLASSES) else f"class_{lab}"
            print(f"  label {lab} ({class_name:>4s}): {cnt:>5d}")

    _save_meta(train_records, "metadata_train.csv")
    _save_meta(test_records,  "metadata_test.csv")

    print(f"\nDone. Images → {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
