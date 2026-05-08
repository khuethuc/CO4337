"""
Download Camelyon17 from Kaggle (mahdibonab/camelyon17).

Requirements:
  pip install kaggle tqdm
  Set up Kaggle API credentials: ~/.kaggle/kaggle.json
  (Account → Settings → Create New Token at https://www.kaggle.com/account)

Usage:
  python download_camelyon17.py --out ../../data/camelyon17
  python download_camelyon17.py --out ../../data/camelyon17 --resume

Output structure:
  <out>/
    images/          ← all patch images (.png)
    metadata.csv     ← columns: image_id, label (0=no tumor, 1=tumor)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

KAGGLE_DATASET = "mahdibonab/camelyon17"
IMG_EXTS       = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Column name aliases used in various Camelyon17 releases
_ID_CANDIDATES    = ["image_id", "id", "name", "filename", "patch_id", "file"]
_LABEL_CANDIDATES = ["label", "tumor", "target", "class", "y", "is_tumor"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _check_kaggle_cli() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _check_kaggle_python() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def download_via_python(dataset: str, out_dir: Path) -> None:
    import kaggle
    print(f"[INFO] Downloading {dataset} via kaggle Python API …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=str(out_dir), unzip=True)


def download_via_cli(dataset: str, out_dir: Path) -> None:
    print(f"[INFO] Downloading {dataset} via kaggle CLI …")
    cmd = ["kaggle", "datasets", "download",
           "-d", dataset,
           "-p", str(out_dir),
           "--unzip"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("kaggle CLI download failed.")


# ---------------------------------------------------------------------------
# Post-download reorganisation
# ---------------------------------------------------------------------------

def _find_label_csv(root: Path) -> Path | None:
    """Find the first CSV that contains a recognisable label column."""
    for p in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=3)
            has_label = any(c in df.columns for c in _LABEL_CANDIDATES)
            if has_label:
                return p
        except Exception:
            continue
    return None


def _collect_images(root: Path) -> list[Path]:
    """Collect all image files under root."""
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            imgs.append(p)
    return imgs


def _normalise_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Normalise an arbitrary Camelyon17 CSV to columns (image_id, label).
    label: 0 = no tumor, 1 = tumor.
    """
    df = pd.read_csv(csv_path)

    # --- id column ---
    id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    if id_col is None:
        # Use first column as id if it looks like strings
        id_col = df.columns[0]
    df = df.rename(columns={id_col: "image_id"})
    df["image_id"] = df["image_id"].astype(str).str.strip()

    # --- label column ---
    label_col = next((c for c in _LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Cannot find a label column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={label_col: "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # strip extension from image_id if present
    df["image_id"] = df["image_id"].apply(
        lambda x: Path(x).stem if Path(x).suffix.lower() in IMG_EXTS else x
    )

    return df[["image_id", "label"]]


def reorganise(raw_dir: Path, out_dir: Path, resume: bool) -> None:
    """
    Move all images to out_dir/images/ and write out_dir/metadata.csv.
    Skips images already present when resume=True.
    """
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # --- find and copy images ---
    all_imgs = _collect_images(raw_dir)
    if not all_imgs:
        print("[WARN] No image files found after download. "
              "Check the Kaggle dataset structure manually.", file=sys.stderr)
        return

    print(f"[INFO] Found {len(all_imgs)} image files — copying to {images_out} …")
    skipped = 0
    for src in tqdm(all_imgs, desc="Copying images", unit="img"):
        dst = images_out / src.name
        if resume and dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        shutil.copy2(src, dst)
    if skipped:
        print(f"[INFO] Skipped {skipped} already-present images (--resume).")

    # --- find and normalise metadata CSV ---
    csv_path = _find_label_csv(raw_dir)
    if csv_path is None:
        print("[WARN] No label CSV found. Generating labels from folder names …")
        _generate_labels_from_folders(raw_dir, images_out, out_dir)
        return

    print(f"[INFO] Found label CSV: {csv_path}")
    try:
        df = _normalise_metadata(csv_path)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # keep only rows with a matching image file
    present = {p.stem for p in images_out.iterdir() if p.suffix.lower() in IMG_EXTS}
    before  = len(df)
    df      = df[df["image_id"].isin(present)]
    print(f"[INFO] Metadata: {before} rows → {len(df)} matched images.")

    meta_out = out_dir / "metadata.csv"
    df.to_csv(meta_out, index=False)
    print(f"[INFO] Saved metadata: {meta_out}")
    _print_label_stats(df)


def _generate_labels_from_folders(raw_dir: Path, images_out: Path, out_dir: Path) -> None:
    """
    Fallback: infer labels from subdirectory names (0/ and 1/ ImageFolder layout).
    Writes metadata.csv from the discovered structure.
    """
    rows = []
    for cls_dir in sorted(raw_dir.rglob("*")):
        if not cls_dir.is_dir():
            continue
        name = cls_dir.name
        if name in ("0", "1"):
            label = int(name)
            for img in cls_dir.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    rows.append({"image_id": img.stem, "label": label})

    if not rows:
        print("[WARN] Cannot infer labels from folder structure either. "
              "Please provide metadata.csv manually.")
        return

    df = pd.DataFrame(rows)
    meta_out = out_dir / "metadata.csv"
    df.to_csv(meta_out, index=False)
    print(f"[INFO] Generated metadata from folder structure: {meta_out}")
    _print_label_stats(df)


def _print_label_stats(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    print("[INFO] Label distribution:")
    for lbl, cnt in counts.items():
        name = "no tumor" if lbl == 0 else "tumor"
        print(f"  label {lbl} ({name}): {cnt:,}  ({cnt/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download Camelyon17 from Kaggle and prepare for training."
    )
    ap.add_argument("--out",     required=True,  help="Output directory (e.g. ../../data/camelyon17)")
    ap.add_argument("--resume",  action="store_true", help="Skip images that already exist")
    ap.add_argument("--keep-raw", dest="keep_raw", action="store_true",
                    help="Keep raw downloaded files after reorganisation")
    return ap.parse_args()


def main() -> int:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "_raw_download"
    raw_dir.mkdir(exist_ok=True)

    # --- Step 1: download ---
    if _check_kaggle_python():
        try:
            download_via_python(KAGGLE_DATASET, raw_dir)
        except Exception as e:
            print(f"[WARN] Python API failed ({e}), falling back to CLI …")
            if not _check_kaggle_cli():
                print("[ERROR] Neither kaggle Python package nor CLI is available.\n"
                      "  Install: pip install kaggle\n"
                      "  Then put kaggle.json in ~/.kaggle/",
                      file=sys.stderr)
                return 1
            download_via_cli(KAGGLE_DATASET, raw_dir)
    elif _check_kaggle_cli():
        download_via_cli(KAGGLE_DATASET, raw_dir)
    else:
        print("[ERROR] Kaggle not found. Install with:  pip install kaggle\n"
              "  Then create ~/.kaggle/kaggle.json from https://www.kaggle.com/account",
              file=sys.stderr)
        return 1

    # --- Step 2: reorganise ---
    reorganise(raw_dir, out_dir, resume=args.resume)

    # --- Step 3: cleanup raw ---
    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("[INFO] Removed raw download directory.")

    print(f"\n[DONE] Camelyon17 ready at: {out_dir.resolve()}")
    print(f"  Run training with:  --dataset camelyon17 --data-dir {out_dir.resolve()} --classes 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
