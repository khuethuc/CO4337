"""
URL: https://isic-archive.s3.amazonaws.com/images/<ISIC_ID>.jpg

Command:
  python download_ham10000.py --out data/ham10000
  python download_ham10000.py --out data/ham10000 --csv /path/to/HAM10000_metadata.csv
  python download_ham10000.py --out data/ham10000 --workers 16 --resume
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import requests
from tqdm import tqdm


S3_IMAGE_URL = "https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download HAM10000 images based on local HAM10000_metadata.csv (no Kaggle).")
    ap.add_argument(
        "--csv",
        type=str,
        default="/mnt/data/HAM10000_metadata.csv",
        help="Path to HAM10000_metadata.csv (default: /mnt/data/HAM10000_metadata.csv).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder to save images + copy of metadata (ex: data/ham10000).",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--user-agent", default="ham10000-downloader/1.0")
    return ap.parse_args()


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def load_isic_ids(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find file CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "image_id" not in df.columns:
        raise ValueError(f"CSV has no column 'image_id'. Existed columns: {list(df.columns)}")

    ids = df["image_id"].astype(str).tolist()
    ids = [x.strip() for x in ids if x and x.strip().startswith("ISIC_")]

    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)

    if not out:
        raise ValueError("Cannot extract image_id started by 'ISIC_' from CSV.")
    return out


def download_one(
    s: requests.Session,
    isic_id: str,
    dst: Path,
    timeout: int,
    retries: int,
) -> Tuple[str, bool, str]:
    """
    Returns: (isic_id, ok, err_msg)
    """
    url = S3_IMAGE_URL.format(isic_id=isic_id)

    last_err = ""
    for attempt in range(retries + 1):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dst.with_suffix(dst.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dst)
            return isic_id, True, ""
        except Exception as e:
            last_err = str(e)
            try:
                tmp = dst.with_suffix(dst.suffix + ".part")
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

            if attempt < retries:
                continue
            return isic_id, False, last_err

    return isic_id, False, last_err


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        isic_ids = load_isic_ids(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    try:
        (out_dir / "HAM10000_metadata.csv").write_bytes(csv_path.read_bytes())
    except Exception:
        pass
    (out_dir / "isic_ids.txt").write_text("\n".join(isic_ids) + "\n", encoding="utf-8")

    print(f"[INFO] Read from CSV: {csv_path}")
    print(f"[INFO] Number of image_id (unique): {len(isic_ids)}")
    print(f"[INFO] Output: {out_dir.resolve()}")

    # Lọc jobs theo resume
    jobs = []
    skipped = 0
    for isic_id in isic_ids:
        dst = images_dir / f"{isic_id}.jpg"
        if args.resume and dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        jobs.append((isic_id, dst))

    print(f"[INFO] Need to download: {len(jobs)} image (skipped={skipped}, resume={args.resume})")

    s = make_session(args.user_agent)

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(download_one, s, isic_id, dst, args.timeout, args.retries)
            for (isic_id, dst) in jobs
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", unit="img"):
            isic_id, ok, err = fut.result()
            if not ok:
                failures.append((isic_id, err))

    if failures:
        fail_path = out_dir / "failed_ids.txt"
        fail_path.write_text("\n".join([f"{i}\t{e}" for i, e in failures]) + "\n", encoding="utf-8")
        print(f"[WARN] Error {len(failures)} image. Saved: {fail_path}")
        return 2

    print("[DONE] Download HAM10000 successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())