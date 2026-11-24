"""Fetch official public Hugging Face SaaS data into ignored data/raw/.

Usage: python data/download.py
Does not copy the downloaded source data into the public repository.
Only user_monthly.parquet is currently needed to train the models; users.parquet
is also downloaded for future EDA, but its lifetime aggregates may leak future data.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPO = "arti199919/synthetic-saas-churn-sample"
FILES = ("user_monthly.parquet", "users.parquet")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO)
    revision = info.sha
    downloaded = []
    for filename in FILES:
        result = hf_hub_download(repo_id=REPO, repo_type="dataset",
                                 filename=filename, revision=revision, local_dir=RAW)
        downloaded.append({"name": filename, "path": str(Path(result).relative_to(ROOT))})
        print(f"Downloaded {filename} to {result}")
    manifest = {
        "repository": REPO, "revision": revision,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": downloaded,
        "note": "Downloaded files remain ignored. Redistribution terms must be checked before sharing."
    }
    (RAW / "download_info.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Recorded download revision in data/raw/download_info.json")


if __name__ == "__main__":
    main()
