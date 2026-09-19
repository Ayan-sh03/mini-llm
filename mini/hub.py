"""Upload/download complete checkpoints. Native training format, not AutoModel export."""
import argparse
from pathlib import Path


def upload(folder, repo, prefix="pretrain"):
    from huggingface_hub import HfApi
    folder = Path(folder)
    if not prefix or Path(prefix).is_absolute() or ".." in Path(prefix).parts:
        raise ValueError("prefix must be a nonempty repo-relative path")
    if not (folder / "COMPLETE.json").exists():
        raise ValueError("Refusing to upload unfinished checkpoint")
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError("Checkpoint repo must be private; refusing public upload")
    api.upload_folder(repo_id=repo, folder_path=str(folder),
                      path_in_repo=f"{prefix}/{folder.name}",
                      commit_message=f"Complete {prefix} checkpoint {folder.name}")
    print(f"HF upload confirmed: {repo}/{prefix}/{folder.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["upload", "download"])
    p.add_argument("--repo", required=True)
    p.add_argument("--folder", help="Local complete checkpoint for upload")
    p.add_argument("--prefix", default="pretrain")
    p.add_argument("--checkpoint", help="e.g. pretrain/step-00000100")
    p.add_argument("--out", default="downloaded")
    p.add_argument("--revision", help="Optional HF commit ID")
    args = p.parse_args()
    if args.command == "upload":
        if not args.folder:
            p.error("--folder required")
        upload(args.folder, args.repo, args.prefix)
    else:
        if not args.checkpoint or Path(args.checkpoint).is_absolute() or ".." in Path(args.checkpoint).parts:
            p.error("--checkpoint must be an explicit repo-relative checkpoint folder")
        from huggingface_hub import snapshot_download
        snapshot_download(args.repo, revision=args.revision, local_dir=args.out,
                          allow_patterns=[f"{args.checkpoint}/*"])
        path = Path(args.out) / args.checkpoint
        if not (path / "COMPLETE.json").exists():
            raise ValueError("Remote checkpoint is incomplete or absent")
        print(path)


if __name__ == "__main__":
    main()
