"""Transfer exact prepared shards and frozen tokenizer via a private HF dataset repo."""
import argparse
import json
from pathlib import Path

from .common import read_json, sha256


def safe_name(name):
    if not name or name in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("Use a simple bundle name, e.g. pilot-v1 or train-v1")
    return name


def files_from_manifest(data):
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    files = ["manifest.json"]
    for entries in manifest["splits"].values():
        for entry in entries:
            for key, hash_key in [("file", "sha256"), ("mask", "mask_sha256")]:
                if key not in entry:
                    continue
                name = entry[key]
                if Path(name).name != name or name in {".", ".."} or "\\" in name:
                    raise ValueError("Shard name must be a plain filename")
                if (data / name).is_symlink() or sha256(data / name) != entry[hash_key]:
                    raise ValueError(f"Corrupt or symlinked shard: {name}")
                files.append(name)
    return manifest, files


def describe(data, tokenizer):
    manifest, files = files_from_manifest(data)
    token = Path(tokenizer) / "tokenizer.json"
    token_hash = sha256(token)
    if token_hash != manifest["tokenizer_sha256"]:
        raise ValueError("Data and tokenizer do not match")
    return {"format": 1, "data_manifest_sha256": sha256(Path(data) / "manifest.json"),
            "tokenizer_sha256": token_hash}, files


def verify(root):
    root = Path(root)
    expected = read_json(root / "READY.json")
    actual, _ = describe(root / "data", root / "tokenizer")
    if expected != actual:
        raise ValueError("Asset bundle hashes do not match READY.json")
    print(f"Verified assets: {root}", flush=True)
    return root


def upload(data, tokenizer, repo, name):
    from huggingface_hub import HfApi
    safe_name(name)
    ready, files = describe(data, tokenizer)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(repo).private:
        raise ValueError("Prepared-data repository must be private")
    if f"{name}/READY.json" in api.list_repo_files(repo, repo_type="dataset"):
        raise ValueError("Bundle already complete. Use a NEW name for another corpus; never mutate a live bundle.")
    # Explicit allow lists prevent uploading credentials, logs or other local files.
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(data),
                      path_in_repo=f"{name}/data", allow_patterns=files)
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(Path(tokenizer) / "tokenizer.json"),
                    path_in_repo=f"{name}/tokenizer/tokenizer.json")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=json.dumps(ready).encode(),
                    path_in_repo=f"{name}/READY.json", commit_message=f"Complete prepared bundle {name}")
    print(f"Uploaded {repo}/{name}; revision={api.dataset_info(repo).sha}", flush=True)


def download(repo, name, out, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    safe_name(name)
    # Resolve once so different downloads cannot accidentally read different commits.
    revision = HfApi().dataset_info(repo, revision=revision).sha
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    if f"{name}/READY.json" not in files:
        raise ValueError("Bundle is absent or its upload has not completed")
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=str(out),
                      allow_patterns=[f"{name}/data/*", f"{name}/tokenizer/tokenizer.json", f"{name}/READY.json"])
    return verify(Path(out) / name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["upload", "download", "verify"])
    p.add_argument("--repo")
    p.add_argument("--name", default="train-v1")
    p.add_argument("--data")
    p.add_argument("--tokenizer")
    p.add_argument("--out", default="assets")
    p.add_argument("--revision")
    p.add_argument("--folder", help="Bundle folder for offline verification")
    args = p.parse_args()
    if args.command == "verify":
        if not args.folder:
            p.error("--folder required")
        verify(args.folder)
    elif not args.repo:
        p.error("--repo required")
    elif args.command == "upload":
        if not args.data or not args.tokenizer:
            p.error("--data and --tokenizer required")
        upload(args.data, args.tokenizer, args.repo, args.name)
    else:
        download(args.repo, args.name, args.out, args.revision)


if __name__ == "__main__":
    main()
