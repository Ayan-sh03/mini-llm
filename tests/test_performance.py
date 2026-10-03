import copy
import json
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from mini.benchmark import legacy_update, main as benchmark
from mini.common import SPECIAL, write_json
from mini.data import Blocks
from mini.engine import BatchTransfer, build_model, make_loss, make_optimizer, seed_all, update
from mini.prepare import Writer, fast_pack

ROOT = Path(__file__).resolve().parents[1]


def test_optimized_update_matches_legacy_with_uneven_masks():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / "configs/tiny.json").read_text())
    device = torch.device("cpu")
    seed_all(42)
    model = build_model(cfg["model"], device)
    old = copy.deepcopy(model)
    opt, old_opt = [make_optimizer(m, cfg["train"], device) for m in (model, old)]
    x = np.tile(np.arange(1, 17, dtype=np.int64), (4, 1))  # int32 by default on Windows
    y = x + 1
    y[0, :] = -100
    y[1, 8:] = -100
    batches = [(x[:1], y[:1]), (x[1:2], y[1:2]), (x[2:], y[2:])]
    transfer = BatchTransfer(device)
    for _ in range(3):
        result = update(model, opt, batches, 0.001, device, 1.0,
                        loss_fn=make_loss(model), transfer=transfer)
        expected = legacy_update(old, old_opt, batches, 0.001, device, 1.0)
        assert result == pytest.approx(expected, abs=1e-7)
        for a, b in zip(model.parameters(), old.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_compile_captures_loss_and_matches_gradients():
    cfg = json.loads((ROOT / "configs/tiny.json").read_text())
    seed_all(7)
    model = build_model(cfg["model"], torch.device("cpu"))
    clone = copy.deepcopy(model)
    x = torch.arange(1, 17).reshape(2, 8)
    y = x + 1
    y[0, 4:] = -100
    eager = make_loss(model)(x, y)
    # AOT eager checks complete forward/backward capture without claiming CUDA
    # Inductor kernel validation on this CPU-only machine.
    compiled = torch.compile(make_loss(clone), backend="aot_eager", fullgraph=True)(x, y)
    torch.testing.assert_close(eager, compiled)
    eager.backward()
    compiled.backward()
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_reader_cache_preserves_original_permutation_across_epochs(tmp_path):
    writer = Writer(tmp_path, "train")
    writer.add(list(range(200)))
    writer.flush()
    write_json(tmp_path / "manifest.json", {"kind": "pretrain", "splits": {"train": writer.entries}})
    reader = Blocks(tmp_path, "train", 8, seed=13)
    # Non-monotonic indices exercise epoch-cache invalidation and resume.
    for index in [*range(3 * reader.n), 0, reader.n, 1]:
        epoch, offset = divmod(index, reader.n)
        rng = np.random.default_rng(13 + epoch)
        a = int(rng.integers(1, max(2, reader.n)))
        while math.gcd(a, reader.n) != 1:
            a = (a + 1) % reader.n or 1
        b = int(rng.integers(reader.n))
        assert reader.permuted(index) == (a * offset + b) % reader.n


def test_benchmark_exercises_full_updates_and_fresh_process_sweep(tmp_path):
    path = tmp_path / "results.json"
    summary = benchmark(["--config", str(ROOT / "configs/tiny.json"), "--device", "cpu",
                         "--micro-batches", "1,4", "--warmup", "1", "--steps", "2",
                         "--out", str(path)])
    assert len(summary["results"]) == 2
    assert all(r["tokens_per_second"] > 0 for r in summary["results"])
    assert all(r["supervised_tokens_per_second"] == r["tokens_per_second"]
               for r in summary["results"])
    assert json.loads(path.read_text())["best"] == summary["best"]


def test_local_pretrain_packing_never_queries_hub(tmp_path, monkeypatch):
    import mini.prepare as prepare
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(["Hello world. A small bird sings a happy song."],
        trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    def forbidden(*args):
        raise AssertionError("Local data must not resolve a remote dataset")
    monkeypatch.setattr(prepare, "resolve_revision", forbidden)
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    source = tmp_path / "local.jsonl"
    source.write_text("".join(json.dumps({"content": f"Story {i}: " + "Hello world. " * 20}) + "\n"
                              for i in range(500)))
    out = tmp_path / "packed"
    args = Namespace(local=str(source), out=str(out), tokenizer=str(tok_dir), tag=None,
                     shard_count=1, shard_tokens=100000, max_docs=500, field="content",
                     val_tokens=10000, max_tokens=500, seq_len=8)
    fast_pack(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "pretrain"
    assert all(manifest["splits"][s] for s in ("train", "val"))
