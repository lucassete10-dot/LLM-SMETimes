"""Package existing PNG evaluations without modifying the source artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images = sorted(args.source.rglob("*.png"))
    expected = {"MIT": 135, "NASA": 5, "XJTU": 56, "Zhu2022_NCA": 56}
    counts = {key: sum(p.parent.name == key for p in images) for key in expected}
    if len(images) != 252 or counts != expected:
        raise ValueError(f"Unexpected image inventory: {len(images)}, {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for path in images:
            name = path.relative_to(args.source).as_posix()
            payload = path.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        archive.write(args.source / "manifest.json", "evaluation_manifest.json")
        archive.writestr("image_inventory.json", json.dumps(
            {"png_count": 252, "counts_including_overviews": counts,
             "sha256": hashes}, ensure_ascii=False, indent=2))
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        assert sum(name.endswith(".png") for name in archive.namelist()) == 252
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(json.dumps({"output": str(args.output), "png_count": 252,
                      "counts": counts, "bytes": args.output.stat().st_size,
                      "integrity_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
