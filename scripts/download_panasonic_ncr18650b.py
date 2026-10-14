"""Download a named cell from the original CC-BY-4.0 Zenodo record.

One cell is explicit; there is no implicit 27 GB bulk download. Data are
accepted only after exact length and Zenodo MD5 match. Interrupted transfers
stay in a .part file and may resume on the next run.
"""

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import requests


RECORD_API = "https://zenodo.org/api/records/7896429"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/panasonic_ncr18650b"


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record(retries=5):
    """Retry transient transport faults, never authentication/HTTP failures."""
    for attempt in range(retries):
        try:
            response = requests.get(RECORD_API, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 10))


def download(cell, output_dir=DEFAULT_OUTPUT, retries=5):
    if not re.fullmatch(r"2_\d{3}", cell):
        raise ValueError("cell must be an original Zenodo key such as 2_097")
    record = fetch_record(retries)
    if record["metadata"]["license"]["id"].lower() != "cc-by-4.0":
        raise ValueError("source license changed; review before downloading")
    key = cell + ".h5"
    matches = [item for item in record["files"] if item["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"{key} is not in the original Zenodo record")
    item = matches[0]
    expected_md5 = item["checksum"].removeprefix("md5:").lower()
    expected_size = int(item["size"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / key
    part = output_dir / (key + ".part")
    if final.exists():
        if final.stat().st_size == expected_size and md5(final) == expected_md5:
            return final
        raise ValueError(f"existing {final} failed checksum; preserved for review")
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            raise ValueError(f"partial file is larger than Zenodo source: {part}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(item["links"]["self"], headers=headers,
                              stream=True, timeout=(30, 90)) as stream:
                stream.raise_for_status()
                if offset and stream.status_code == 206:
                    content_range = stream.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("server returned mismatched byte range")
                    mode = "ab"
                elif stream.status_code == 200:
                    mode = "wb"  # Server ignored Range; start over safely.
                else:
                    raise ValueError(f"unexpected HTTP {stream.status_code} for range request")
                with part.open(mode) as output:
                    for block in stream.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
            if part.stat().st_size == expected_size and md5(part) == expected_md5:
                os.replace(part, final)
                return final
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 10))
    raise ValueError(f"download incomplete or checksum mismatch: {part}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, help="Original Zenodo cell ID, e.g. 2_097")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(download(args.cell, args.output_dir))


if __name__ == "__main__":
    main()
