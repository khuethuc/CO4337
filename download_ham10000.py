"""
download_ham10000.py

Download Skin Cancer / HAM10000 (CSV + optional images) into ../data.

Default Kaggle dataset:
  harish20ug0429/skin-cancer
This Kaggle dataset includes:
  - HAM10000_metadata.csv
  - hmnist_28_28_L.csv
  - hmnist_28_28_RGB.csv
  - hmnist_8_8_L.csv
  - hmnist_8_8_RGB.csv
  - Images/ ... (optional, depends on dataset package)

Usage:
  python download_ham10000.py
  python download_ham10000.py --out-root ../data --subdir ham10000
  python download_ham10000.py --force
  python download_ham10000.py --dataset <owner/dataset-slug>

Kaggle credentials:
  - Put kaggle.json at ~/.kaggle/kaggle.json
    (Linux/macOS) chmod 600 ~/.kaggle/kaggle.json
  OR
  - export KAGGLE_USERNAME=...
    export KAGGLE_KEY=...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


EXPECTED_FILES = [
    "HAM10000_metadata.csv",
    "hmnist_28_28_L.csv",
    "hmnist_28_28_RGB.csv",
    "hmnist_8_8_L.csv",
    "hmnist_8_8_RGB.csv",
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def resolve_default_out_dir(out_root: Optional[str], subdir: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    root = Path(out_root) if out_root is not None else (script_dir / ".." / "data")
    return (root / subdir).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def kaggle_api():
    """
    Import KaggleApi. Provide a helpful message if missing.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as ex:
        raise RuntimeError(
            "Missing Kaggle client. Install with:\n"
            "  pip install kaggle\n"
            "Then configure credentials:\n"
            "  ~/.kaggle/kaggle.json  (chmod 600)\n"
            "or env vars KAGGLE_USERNAME, KAGGLE_KEY\n"
        ) from ex
    return KaggleApi()


def authenticate(api) -> None:
    """
    Kaggle authentication with clear error instructions.
    """
    try:
        api.authenticate()
    except Exception as ex:
        raise RuntimeError(
            "Kaggle authentication failed.\n"
            "Fix by either:\n"
            "  1) Put kaggle.json at ~/.kaggle/kaggle.json and chmod 600 it\n"
            "     - Download from https://www.kaggle.com/settings/account (Create New API Token)\n"
            "  2) Or set env vars:\n"
            "     export KAGGLE_USERNAME=...\n"
            "     export KAGGLE_KEY=...\n"
        ) from ex


def already_have_all_files(dst_dir: Path) -> bool:
    return all((dst_dir / f).exists() for f in EXPECTED_FILES)


def find_files(root: Path, filenames: List[str]) -> Dict[str, Path]:
    """
    Search for filenames under root and return {filename: found_path}.
    """
    found: Dict[str, Path] = {}
    target_set = set(filenames)

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name in target_set and name not in found:
            found[name] = p
            if len(found) == len(filenames):
                break
    return found


def move_to_root(dst_dir: Path, found: Dict[str, Path], force: bool) -> None:
    """
    Move each found file into dst_dir root.
    If a file already exists:
      - force=True -> overwrite
      - else -> skip
    """
    for name, src in found.items():
        dst = dst_dir / name
        if dst.exists():
            if not force:
                print(f"[skip] {name} already exists at {dst}")
                continue
            dst.unlink()

        if src.resolve() == dst.resolve():
            continue

        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        print(f"[ok] moved {name} -> {dst}")


def download_from_kaggle(dataset: str, dst_dir: Path, unzip: bool) -> None:
    api = kaggle_api()
    authenticate(api)

    print(f"[download] kaggle dataset: {dataset}")
    print(f"[download] target dir: {dst_dir}")
    ensure_dir(dst_dir)

    # Kaggle downloads a zip; unzip=True extracts automatically
    api.dataset_download_files(dataset, path=str(dst_dir), unzip=unzip, quiet=False)
    print("[ok] kaggle download finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="harish20ug0429/skin-cancer",
        help="Kaggle dataset slug (owner/dataset). Default matches the CSV filenames used by your dataloader.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output root directory. Default: ../data (relative to this script).",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="ham10000",
        help="Subfolder under out-root. Default: ham10000",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not unzip downloaded archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing expected CSV files if present.",
    )
    args = parser.parse_args()

    dst_dir = resolve_default_out_dir(args.out_root, args.subdir)
    ensure_dir(dst_dir)

    if already_have_all_files(dst_dir) and not args.force:
        print(f"[ok] All expected files already exist in {dst_dir}")
        print("Files:")
        for f in EXPECTED_FILES:
            print(f"  - {dst_dir / f}")
        return

    # Step 1: Download from Kaggle
    try:
        download_from_kaggle(args.dataset, dst_dir, unzip=not args.no_unzip)
    except Exception as ex:
        eprint(str(ex))
        sys.exit(1)

    # Step 2: After download, ensure the expected CSVs are in dst_dir root
    found = find_files(dst_dir, EXPECTED_FILES)
    if len(found) == 0:
        eprint("[warn] Could not find any expected CSV files after download.")
    else:
        move_to_root(dst_dir, found, force=args.force)

    # Step 3: Verify
    missing = [f for f in EXPECTED_FILES if not (dst_dir / f).exists()]
    print("\n[verify]")
    if missing:
        eprint("[warn] Missing files:")
        for f in missing:
            eprint(f"  - {f}")
        eprint("\nTip: re-run with --force, or check Kaggle dataset contents / slug.")
        sys.exit(2)

    print("[ok] All expected files are ready:")
    for f in EXPECTED_FILES:
        print(f"  - {dst_dir / f}")
    print(f"\nDone. Dataset saved at: {dst_dir}")


if __name__ == "__main__":
    main()