"""
URL: https://isic-archive.s3.amazonaws.com/images/<ISIC_ID>.jpg
Commands: python download_ham10000.py --out ../../data/ham10000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import requests
from tqdm import tqdm

COLLECTION_URL = "https://api.isic-archive.com/collections/212/"
S3_IMAGE_URL = "https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"

METADATA_CANDIDATES = [
    "https://api.isic-archive.com/collections/212/metadata/",
    "https://api.isic-archive.com/collections/212/metadata",
    "https://api.isic-archive.com/collections/212/metadata/?format=csv",
    "https://api.isic-archive.com/collections/212/metadata?format=csv",
]

ISIC_ID_RE = re.compile(r"\bISIC_\d{7}\b")
NEXT_HREF_RE = re.compile(r'href="([^"]*cursor=[^"]*)"[^>]*>\s*next\s*<', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download HAM10000 from ISIC Archive (no Kaggle).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--user-agent", default="ham10000-downloader/1.0")
    return ap.parse_args()


def session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def fetch_text(s: requests.Session, url: str, timeout: int) -> str:
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def crawl_isic_ids(s: requests.Session, max_images: int, timeout: int) -> list[str]:
    url = COLLECTION_URL
    seen = set()
    out: list[str] = []

    pbar = tqdm(total=(max_images if max_images > 0 else None), desc="Crawling IDs", unit="img")

    while True:
        html = fetch_text(s, url, timeout=timeout)

        ids = ISIC_ID_RE.findall(html)
        for isic_id in ids:
            if isic_id not in seen:
                seen.add(isic_id)
                out.append(isic_id)
                pbar.update(1)
                if max_images > 0 and len(out) >= max_images:
                    pbar.close()
                    return out

        m = NEXT_HREF_RE.search(html)
        if not m:
            break

        next_href = m.group(1)
        if next_href.startswith("http"):
            url = next_href
        else:
            url = "https://api.isic-archive.com" + next_href

    pbar.close()
    return out


def download_file(s: requests.Session, url: str, dst: Path, timeout: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with s.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def try_download_metadata(s: requests.Session, out_dir: Path, timeout: int) -> Path | None:
    out_path = out_dir / "isic_ham10000_metadata.csv"
    headers = {"Accept": "text/csv,application/octet-stream,*/*"}

    for url in METADATA_CANDIDATES:
        try:
            r = s.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            ct = (r.headers.get("Content-Type") or "").lower()
            if ("text/csv" not in ct) and ("csv" not in r.text[:200].lower()):
                # tránh lưu nhầm HTML
                continue
            out_path.write_bytes(r.content)
            return out_path
        except Exception:
            continue
    return None


def iter_jobs(isic_ids: Iterable[str], images_dir: Path) -> Iterable[tuple[str, str, Path]]:
    for isic_id in isic_ids:
        url = S3_IMAGE_URL.format(isic_id=isic_id)
        dst = images_dir / f"{isic_id}.jpg"
        yield isic_id, url, dst


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    s = session(args.user_agent)

    # Step 1: Crawl ID list
    try:
        isic_ids = crawl_isic_ids(s, max_images=args.max_images, timeout=args.timeout)
    except Exception as e:
        print(f"[ERROR] Crawl IDs fail: {e}", file=sys.stderr)
        return 1

    if not isic_ids:
        print("[ERROR] Cannot crawl ISIC_ID", file=sys.stderr)
        return 1

    (out_dir / "isic_ids.txt").write_text("\n".join(isic_ids) + "\n", encoding="utf-8")
    print(f"[INFO] Total get ID: {len(isic_ids)} (saved isic_ids.txt)")

    # Step 2: Download metadata (optional)
    if not args.no_metadata:
        meta_path = try_download_metadata(s, out_dir=out_dir, timeout=args.timeout)
        if meta_path is None:
            print("[WARN] Cannot download metadata through auto endpoint")
        else:
            print(f"[INFO] Saved metadata: {meta_path}")

    # Step 3: Download parallel
    jobs = list(iter_jobs(isic_ids, images_dir))
    to_download = []
    if args.resume:
        for isic_id, url, dst in jobs:
            if dst.exists() and dst.stat().st_size > 0:
                continue
            to_download.append((isic_id, url, dst))
    else:
        to_download = jobs

    print(f"[INFO] Number of images that need to be downloaded: {len(to_download)} (resume={args.resume})")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {
            ex.submit(download_file, s, url, dst, args.timeout): (isic_id, url, dst)
            for (isic_id, url, dst) in to_download
        }

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Downloading", unit="img"):
            isic_id, url, dst = future_map[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append(isic_id)
                try:
                    if dst.exists():
                        dst.unlink()
                except Exception:
                    pass

    if failures:
        (out_dir / "failed_ids.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[WARN] Error {len(failures)} images. List is saved at failed_ids.txt")
    else:
        print("[DONE] Download HAM10000 successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())