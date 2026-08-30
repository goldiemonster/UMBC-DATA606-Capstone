#!/usr/bin/env python3
"""
Download the authoritative USAspending Award Data Archive full contract files.

By default, this script discovers the currently published
FY####_All_Contracts_Full_YYYYMMDD.zip files from USAspending and downloads
the latest snapshot for every available fiscal year into:

    data/raw/usaspending/contracts/

Examples
--------
List available fiscal-year archives without downloading:
    python data/download_usaspending_contracts.py --list-only

Download every available full contract fiscal-year archive:
    python data/download_usaspending_contracts.py

Download only FY2020-FY2025:
    python data/download_usaspending_contracts.py --years 2020-2025
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ARCHIVE_INDEX = "https://files.usaspending.gov/award_data_archive/"
FILE_RE = re.compile(
    r"^FY(?P<fy>\d{4})_All_Contracts_Full_(?P<snapshot>\d{8})\.zip$"
)
DEFAULT_OUT = Path("data/raw/usaspending/contracts")


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def fetch_archive_keys() -> List[str]:
    """Read USAspending's public archive listing and return object keys."""
    req = urllib.request.Request(
        ARCHIVE_INDEX,
        headers={"User-Agent": "UMBC-DATA606-BidIntel/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not read USAspending archive index: {exc}"
        ) from exc

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(
            "USAspending archive index did not return parseable XML."
        ) from exc

    keys: List[str] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) == "Key" and elem.text:
            keys.append(elem.text.strip())

    if not keys:
        raise RuntimeError(
            "No archive keys were discovered. USAspending may have changed "
            "its archive listing format."
        )
    return keys


def discover_latest_contract_archives() -> Dict[int, Tuple[str, str]]:
    """
    Return {fiscal_year: (snapshot_date, filename)} for the latest available
    full-contract file for each fiscal year.
    """
    discovered: Dict[int, Tuple[str, str]] = {}

    for key in fetch_archive_keys():
        name = key.rsplit("/", 1)[-1]
        match = FILE_RE.match(name)
        if not match:
            continue

        fy = int(match.group("fy"))
        snapshot = match.group("snapshot")

        current = discovered.get(fy)
        if current is None or snapshot > current[0]:
            discovered[fy] = (snapshot, name)

    if not discovered:
        raise RuntimeError(
            "No FY####_All_Contracts_Full_*.zip files were found in the "
            "USAspending archive listing."
        )
    return dict(sorted(discovered.items()))


def parse_years(spec: str | None, available: Iterable[int]) -> List[int]:
    available_set = set(available)
    if not spec:
        return sorted(available_set)

    requested = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            requested.update(range(start, end + 1))
        else:
            requested.add(int(token))

    missing = sorted(requested - available_set)
    if missing:
        print(
            "Warning: these requested fiscal years are not currently "
            f"available and will be skipped: {missing}",
            file=sys.stderr,
        )
    return sorted(requested & available_set)


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "UMBC-DATA606-BidIntel/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def human_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{size:,} B"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Skipping existing file: {destination}")
        return

    temp = destination.with_suffix(destination.suffix + ".part")
    if temp.exists():
        temp.unlink()

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "UMBC-DATA606-BidIntel/1.0"},
    )

    print(f"Downloading: {url}")
    try:
        with urllib.request.urlopen(req, timeout=120) as response, temp.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            copied = 0
            block = 8 * 1024 * 1024

            while True:
                chunk = response.read(block)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                if total:
                    pct = copied / total * 100
                    print(
                        f"\r  {human_bytes(copied)} / {human_bytes(total)} "
                        f"({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  {human_bytes(copied)} downloaded",
                        end="",
                        flush=True,
                    )
            print()

        temp.replace(destination)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def write_manifest(rows: List[dict], output_dir: Path) -> None:
    manifest = output_dir / "manifest.csv"
    fieldnames = [
        "fiscal_year",
        "snapshot_date",
        "filename",
        "source_url",
        "size_bytes",
        "sha256",
        "downloaded_at_utc",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest written: {manifest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        help="Fiscal years, e.g. 2020-2025 or 2020,2022,2025. "
             "Default: all available years.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Destination directory. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Discover and list current files without downloading them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist.",
    )
    args = parser.parse_args()

    print("Discovering current USAspending full contract archives...")
    archives = discover_latest_contract_archives()
    years = parse_years(args.years, archives.keys())

    if not years:
        print("No fiscal years selected.", file=sys.stderr)
        return 2

    print("\nSelected authoritative archives:")
    planned = []
    for fy in years:
        snapshot, filename = archives[fy]
        url = ARCHIVE_INDEX + filename
        size = remote_size(url)
        planned.append((fy, snapshot, filename, url, size))
        print(
            f"  FY{fy}: {filename}"
            + (f"  [{human_bytes(size)}]" if size is not None else "")
        )

    known_total = sum(x[4] or 0 for x in planned)
    if all(x[4] is not None for x in planned):
        print(f"\nEstimated download size: {human_bytes(known_total)}")
    else:
        print("\nSome remote file sizes could not be determined in advance.")

    if args.list_only:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[dict] = []

    for fy, snapshot, filename, url, size in planned:
        destination = args.output / filename
        download(url, destination, force=args.force)
        actual_size = destination.stat().st_size
        print(f"Hashing {filename}...")
        digest = sha256_file(destination)
        manifest_rows.append(
            {
                "fiscal_year": fy,
                "snapshot_date": snapshot,
                "filename": filename,
                "source_url": url,
                "size_bytes": actual_size,
                "sha256": digest,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    write_manifest(manifest_rows, args.output)
    print(
        "\nDone. The raw ZIP archives are intentionally excluded from Git "
        "version control; see data/README.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
