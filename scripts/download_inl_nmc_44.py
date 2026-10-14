"""Download and checksum the official INL NMC/Gr 44-cell Zenodo files.

The files are intake candidates only. This downloader never extracts an
archive, runs supplied code, or marks cells eligible for model training.
"""

import hashlib
import json
from pathlib import Path

import requests


RECORD_ID = 7250553
API = f"https://zenodo.org/api/records/{RECORD_ID}"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "inl_nmc_44"
REQUIRED = {"Battery raw data.zip", "README.txt", "Pouch cell_summary.xlsx"}
EXPECTED_MD5 = {
    "Battery raw data.zip": "a0ee7ee69285f23489e1f02b66114bc5",
    "README.txt": "c5c9455a196f407b5e7413a1c6ad8c59",
    "Pouch cell_summary.xlsx": "5730d75ec215f751c6b1f6911422e787",
}


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_files(session):
    response = session.get(API, timeout=30)
    response.raise_for_status()
    record = response.json()
    if (record["metadata"]["access_right"] != "open" or
            record["metadata"]["license"]["id"].lower() != "cc-by-4.0"):
        raise ValueError("source is not verified as open CC-BY-4.0")
    files = {row["key"]: row for row in record["files"]}
    if not REQUIRED.issubset(files):
        raise ValueError(f"official record missing files: {REQUIRED - set(files)}")
    for name in REQUIRED:
        if files[name]["checksum"] != "md5:" + EXPECTED_MD5[name]:
            raise ValueError(f"official checksum changed for {name}")
    return files


def download(session, name, file_record):
    destination = OUT / name
    if destination.exists():
        if md5_file(destination) != EXPECTED_MD5[name]:
            raise ValueError(f"existing local file has incorrect checksum: {destination}")
        return destination
    temporary = OUT / (name + ".part")
    if temporary.exists():
        raise ValueError(f"incomplete previous transfer exists: {temporary}")
    url = file_record["links"]["self"]
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("xb") as handle:
            for chunk in response.iter_content(chunk_size=2**20):
                if chunk:
                    handle.write(chunk)
    if temporary.stat().st_size != file_record["size"] or \
       md5_file(temporary) != EXPECTED_MD5[name]:
        raise ValueError(f"downloaded file failed official size/MD5 check: {temporary}")
    temporary.replace(destination)
    return destination


def main():
    with requests.Session() as session:
        files = metadata_files(session)
        OUT.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset_id": "INL_NMC_44", "official_url": API,
                    "license": "CC-BY-4.0", "doi": "10.5281/zenodo.7250553",
                    "training_eligible": False, "files": []}
        for name in sorted(REQUIRED):
            path = download(session, name, files[name])
            manifest["files"].append({"name": name, "bytes": path.stat().st_size,
                                      "md5": md5_file(path)})
            print(name, path.stat().st_size, flush=True)
        (OUT / "intake_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
