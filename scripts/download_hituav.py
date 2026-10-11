"""Download and extract the HIT-UAV infrared thermal dataset into data/hituav/.

The GitHub repository only holds labels; the full release (images and labels) is
archived on Zenodo. File name, size and md5 are read from the Zenodo API.

Usage:
    uv run python scripts/download_hituav.py --size-only   # print size, download nothing
    uv run python scripts/download_hituav.py               # download, verify, extract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "7633134"  # HIT-UAV v1.2.1
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "hituav"
CHUNK = 1 << 20


def fetch_file_info() -> dict:
    with urllib.request.urlopen(API_URL, timeout=30) as resp:
        record = json.load(resp)
    files = record["files"]
    if len(files) != 1:
        raise RuntimeError(f"expected 1 file in record {ZENODO_RECORD}, got {len(files)}")
    info = files[0]
    return {
        "name": Path(info["key"]).name,
        "size": int(info["size"]),
        "md5": info["checksum"].removeprefix("md5:"),
        "url": info["links"]["self"],
        "version": record["metadata"].get("version"),
    }


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    """Download to dest, resuming a partial file if present."""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    if done < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if done and resp.status != 206:  # server ignored Range; start over
                done = 0
            with part.open("ab" if done else "wb") as f:
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r{human(done)} / {human(size)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    part.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--size-only", action="store_true", help="print size and exit")
    parser.add_argument("--keep-zip", action="store_true", help="keep the archive after extracting")
    args = parser.parse_args()

    info = fetch_file_info()
    print(f"{info['name']} (Zenodo record {ZENODO_RECORD}, {info['version']})")
    print(f"download size: {human(info['size'])} ({info['size']:,} bytes)")
    print(f"md5: {info['md5']}")
    if args.size_only:
        return

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < 3 * info["size"]:
        sys.exit(f"not enough free space: {human(free)} free, need about {human(3 * info['size'])}")

    archive = dest / info["name"]
    if not archive.exists():
        download(info["url"], archive, info["size"])
    if md5sum(archive) != info["md5"]:
        sys.exit(f"md5 mismatch for {archive}; delete it and retry")
    print("md5 ok, extracting")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    if not args.keep_zip:
        archive.unlink()
    print(f"extracted to {dest}")


if __name__ == "__main__":
    main()
