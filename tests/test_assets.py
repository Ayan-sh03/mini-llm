import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini.assets import describe, download, safe_name, upload, verify
from mini.common import sha256, write_json
from mini.data import Blocks
from mini.prepare import Writer


def corpus(root):
    data, tok = root / "data", root / "tokenizer"
    data.mkdir(parents=True)
    tok.mkdir()
    (tok / "tokenizer.json").write_text('{}')
    parts = {}
    for split in ["train", "val"]:
        w = Writer(data, split)
        w.add(np.arange(512, dtype=np.uint16))
        w.flush()
        parts[split] = w.entries
    write_json(data / "manifest.json", {"kind": "pretrain", "splits": parts,
        "tokenizer_sha256": sha256(tok / "tokenizer.json")})
    return data, tok


def test_private_mock_transfer_preserves_reader(tmp_path, monkeypatch):
    import huggingface_hub
    data, tok = corpus(tmp_path / "source")
    (data / "secret.env").write_text('MUST NOT BE UPLOADED')
    remote = {}
    class API:
        def create_repo(self, **kw):
            assert kw["private"] is True and kw["repo_type"] == "dataset"
        def dataset_info(self, *args, **kw):
            return SimpleNamespace(private=True, sha="fixed-revision")
        def list_repo_files(self, *args, **kw):
            return list(remote)
        def upload_folder(self, **kw):
            for name in kw["allow_patterns"]:
                remote[f'{kw["path_in_repo"]}/{name}'] = (Path(kw["folder_path"]) / name).read_bytes()
        def upload_file(self, **kw):
            source = kw["path_or_fileobj"]
            remote[kw["path_in_repo"]] = source if isinstance(source, bytes) else Path(source).read_bytes()
    def snapshot(**kw):
        assert kw["revision"] == "fixed-revision"
        for name, content in remote.items():
            target = Path(kw["local_dir"]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    upload(data, tok, "test/assets", "train-v1")
    assert not any("secret.env" in name for name in remote)
    assert list(remote)[-1] == "train-v1/READY.json"
    with pytest.raises(ValueError, match="already complete"):
        upload(data, tok, "test/assets", "train-v1")
    bundle = download("test/assets", "train-v1", tmp_path / "download")
    before, after = Blocks(data, "train", 64), Blocks(bundle / "data", "train", 64)
    before.next_numpy(3)
    after.load_state_dict(before.state_dict())
    assert all(np.array_equal(a, b) for a, b in zip(before.next_numpy(2), after.next_numpy(2)))
    (bundle / "data/train-00000.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Corrupt"):
        verify(bundle)


def test_tokenizer_mismatch_and_names(tmp_path):
    data, tok = corpus(tmp_path)
    (tok / "tokenizer.json").write_text('changed')
    with pytest.raises(ValueError, match="do not match"):
        describe(data, tok)
    for name in ["", "..", "../escape", "/absolute", "has/slash", "a*b"]:
        with pytest.raises(ValueError):
            safe_name(name)
    assert safe_name("pilot-v1") == "pilot-v1"


def test_public_repository_rejected(tmp_path, monkeypatch):
    import huggingface_hub
    data, tok = corpus(tmp_path)
    class PublicAPI:
        def create_repo(self, **kw):
            pass
        def dataset_info(self, *args, **kw):
            return SimpleNamespace(private=False)
    monkeypatch.setattr(huggingface_hub, "HfApi", PublicAPI)
    with pytest.raises(ValueError, match="must be private"):
        upload(data, tok, "test/public", "train-v1")
