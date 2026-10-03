import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini.common import SPECIAL, holdout, read_json, write_json
from mini.data import Blocks
from mini.prepare import Writer, merge

import mini.prepare as prepare


def tiny_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["Hello world. A small bird sings a happy song in the garden."],
        trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def tokens_shard(root, tag, count, seq_len=8, masked=False):
    """Real shard files so byte-layout validation has something to check."""
    w = Writer(root, "train", limit=10_000, prefix=f"{tag}-")
    ids = np.arange(count, dtype=np.uint16)
    w.add(ids, np.ones(count, dtype="u1") if masked else None)
    w.flush()
    return {"train": w.entries}


def part(root, tag, splits, kind="pretrain", seq_len=8, vocab=400, tok_sha="tok", counts=None):
    write_json(root / f"manifest.{tag}.json", {
        "version": 1, "kind": kind, "seq_len": seq_len, "vocab_size": vocab,
        "tokenizer_sha256": tok_sha,
        "counts": counts or {"train": 32, "val": 0, "skipped": 0},
        "source": {"dataset": f"src-{tag}", "revision": f"rev-{tag}"},
        "splits": splits,
    })


def test_merge_rejects_mismatched_seq_len(tmp_path):
    splits = tokens_shard(tmp_path, "a", 64)
    part(tmp_path, "a", splits, seq_len=8)
    part(tmp_path, "b", splits, seq_len=4)
    with pytest.raises(ValueError, match="seq_len"):
        merge(SimpleNamespace(out=str(tmp_path)))


def test_merge_rejects_mismatched_kind_and_tokenizer(tmp_path):
    splits = tokens_shard(tmp_path, "a", 64)
    part(tmp_path, "a", splits, kind="pretrain")
    part(tmp_path, "b", splits, kind="sft")
    with pytest.raises(ValueError, match="kind"):
        merge(SimpleNamespace(out=str(tmp_path)))
    (tmp_path / "manifest.b.json").unlink()
    part(tmp_path, "b", splits, tok_sha="other")
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        merge(SimpleNamespace(out=str(tmp_path)))


def test_merge_rejects_bad_row_layout(tmp_path):
    # An SFT shard must be a whole number of (seq_len + 1)-token rows.
    splits = tokens_shard(tmp_path, "a", 10, masked=True)
    part(tmp_path, "a", splits, kind="sft", seq_len=8)
    with pytest.raises(ValueError, match="whole number of 9-token rows"):
        merge(SimpleNamespace(out=str(tmp_path)))


def test_merge_rejects_missing_shard(tmp_path):
    splits = tokens_shard(tmp_path, "a", 64)
    splits["train"][0]["file"] = "not-on-disk.bin"
    part(tmp_path, "a", splits)
    with pytest.raises(ValueError, match="absent"):
        merge(SimpleNamespace(out=str(tmp_path)))


def test_merge_keeps_provenance_and_cannot_overwrite(tmp_path):
    part(tmp_path, "a", tokens_shard(tmp_path, "a", 64), counts={"train": 32, "val": 0, "skipped": 1})
    part(tmp_path, "b", tokens_shard(tmp_path, "b", 32), counts={"train": 16, "val": 0, "skipped": 2})
    merge(SimpleNamespace(out=str(tmp_path)))
    meta = read_json(tmp_path / "manifest.json")
    assert meta["counts"] == {"train": 48, "val": 0, "skipped": 3}
    assert meta["source"]["merged"] == ["manifest.a.json", "manifest.b.json"]
    assert [p["source"]["revision"] for p in meta["source"]["provenance"]] == ["rev-a", "rev-b"]
    assert all(p["manifest_sha256"] for p in meta["source"]["provenance"])
    assert (tmp_path / "manifest.a.json").exists()  # parts retained, not unlinked
    Blocks(tmp_path, "train", 8)
    with pytest.raises(ValueError, match="already exists"):
        merge(SimpleNamespace(out=str(tmp_path)))


def test_local_pack_never_touches_the_hub(tmp_path, monkeypatch):
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    tiny_tokenizer().save(str(tok_dir / "tokenizer.json"))
    train, val, i = [], [], 0
    while len(val) < 3 or len(train) < 20:
        text = f"Document {i}. " + "A small bird sings a happy song in the garden. " * 6
        i += 1
        assert i < 100_000, "holdout never fired"
        (val if holdout(text) else train).append(text)
    source = tmp_path / "corpus.jsonl"
    with open(source, "w", encoding="utf-8") as f:
        for text in train + val:
            f.write(json.dumps({"content": text}) + "\n")

    def boom(*args, **kwargs):
        raise AssertionError("local packing must not resolve a Hub revision")

    monkeypatch.setattr(prepare, "resolve_revision", boom)
    monkeypatch.setattr(sys, "argv", [
        "prepare", "pack", "--local", str(source), "--out", str(tmp_path / "data"),
        "--tokenizer", str(tok_dir), "--max-tokens", "100000", "--val-tokens", "100000",
        "--seq-len", "64", "--max-docs", "1000"])
    prepare.main()
    meta = read_json(tmp_path / "data" / "manifest.json")
    assert meta["kind"] == "pretrain" and meta["seq_len"] == 64
    assert meta["counts"]["train"] > 0 and meta["counts"]["val"] > 0
    assert meta["source"]["local"] == str(source)
    Blocks(tmp_path / "data", "train", 64)
    Blocks(tmp_path / "data", "val", 64)
