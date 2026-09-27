"""Download the canonical Olist archive without committing the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
SHA256 = "967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784"


def download(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "olist-brazilian-ecommerce.zip"
    if not archive.exists():
        request = Request(URL, headers={"User-Agent": "delivery-eta-ml-platform/1.0"})
        with urlopen(request, timeout=120) as response, archive.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected Olist archive SHA-256: {digest}")
    extracted = output_dir / "olist"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(extracted)
    manifest = {
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "source_url": URL,
        "license": "CC BY-NC-SA 4.0",
        "archive_sha256": digest,
        "raw_data_policy": "Download locally; do not commit or redistribute from this repository.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(download(args.output_dir))
