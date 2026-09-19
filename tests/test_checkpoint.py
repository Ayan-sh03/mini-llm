import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mini.common import write_json
from mini.data import Blocks
from mini.engine import (build_model, load_checkpoint, make_optimizer,
                         resume_signature, save_checkpoint, seed_all, update)
from mini.prepare import Writer


def test_full_checkpoint_and_checksum(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/tiny.json").read_text())
    writer = Writer(tmp_path, "train")
    writer.add(np.arange(512, dtype=np.uint16))
    writer.flush()
    write_json(tmp_path / "manifest.json", {"kind": "pretrain", "splits": {"train": writer.entries}})
    # Save helper copies opaque tokenizer bytes; tokenizer behavior is tested separately.
    token = tmp_path / "tokenizer.json"
    token.write_text('{}')
    reader = Blocks(tmp_path, "train", 64)
    device = torch.device("cpu")
    seed_all(10)
    model = build_model(cfg["model"], device)
    opt = make_optimizer(model, cfg["train"], device)
    update(model, opt, [reader.next_numpy(2)], 0.001, device, 1.0)
    progress = {"step": 1, "tokens": 128, "supervised_tokens": 128}
    folder = save_checkpoint(tmp_path / "out", model, opt, cfg, token,
                             {"train": reader}, progress, np.random.default_rng(42))
    loaded = load_checkpoint(folder)
    assert loaded["readers"]["train"] == reader.state_dict()
    assert loaded["progress"] == progress
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded["model"][name])
    with open(folder / "training.pt", "ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(folder)


def test_resume_signature_allows_microbatch_not_schedule():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/tiny.json").read_text())
    original = resume_signature(cfg)
    cfg["train"]["micro_batch"] = 1
    assert resume_signature(cfg) == original
    cfg["train"]["total_tokens"] *= 2
    assert resume_signature(cfg) != original
