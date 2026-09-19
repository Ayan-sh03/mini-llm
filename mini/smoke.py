"""Offline end-to-end test. Uses generated toy data; downloads no dataset/model."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .common import read_json, write_json
from .engine import load_checkpoint


def run(root):
    project = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_json(project / "configs/tiny.json")
    cfg["model"]["vocab_size"] = 384
    config_path = root / "tiny.json"
    write_json(config_path, cfg)
    source = root / "toy.jsonl"
    with open(source, "w") as f:
        for i in range(5000):
            text = (f"Story {i}: A child named Sam walked through a garden and saw a bird. "
                    "The bird sang a happy song. Sam smiled and went home to tell a friend. "
                    "They talked about the sky, the trees, their family, and tomorrow's plans.")
            f.write(json.dumps({"content": text}) + "\n")
    def call(module, *args):
        subprocess.run([sys.executable, "-m", f"mini.{module}", *map(str, args)], check=True)
    call("prepare", "tokenizer", "--local", source, "--out", root / "tok",
         "--vocab-size", 384, "--max-chars", 400000)
    call("prepare", "pack", "--local", source, "--out", root / "data",
         "--tokenizer", root / "tok", "--max-tokens", 40000, "--seq-len", 64)
    base = ["--config", config_path, "--data", root / "data",
            "--tokenizer", root / "tok", "--device", "cpu", "--max-seconds", 600]
    call("train", *base, "--out", root / "continuous", "--max-steps", 6)
    call("train", *base, "--out", root / "interrupted", "--max-steps", 3)
    checkpoint = root / "interrupted/step-00000003"
    call("train", *base, "--out", root / "resumed", "--resume", checkpoint, "--max-steps", 3)
    a = load_checkpoint(root / "continuous/step-00000006")
    b = load_checkpoint(root / "resumed/step-00000006")
    assert a["progress"] == b["progress"] and a["readers"] == b["readers"]
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
    for key, value in a["optimizer"]["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, b["optimizer"]["state"][key][field]), (key, field)
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    print("PASS: uninterrupted vs fresh-process resumed CPU training is bitwise equal.", flush=True)
    # Prepare complete short chats and run a fresh assistant-only SFT optimizer.
    chats = root / "chats.jsonl"
    with open(chats, "w") as f:
        for i in range(3000):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"Tell me a story {i}."},
                {"role": "assistant", "content": "Sam saw a bird. It sang a happy song."}
            ]}) + "\n")
    call("prepare", "sft", "--local", chats, "--field", "messages", "--out", root / "sft-data",
         "--tokenizer", root / "tok", "--seq-len", 64, "--max-examples", 100)
    call("train", "--config", config_path, "--data", root / "sft-data",
         "--tokenizer", root / "tok", "--device", "cpu", "--max-steps", 2,
         "--out", root / "sft", "--init", root / "continuous/step-00000006")
    call("chat", "--checkpoint", root / "sft/step-00000002", "--prompt", "Hello",
         "--chat", "--max-new-tokens", 5)
    print("PASS: tokenizer -> pack -> pretrain -> checkpoint -> resume -> SFT -> generation.", flush=True)
    print("Toy output is NOT a model-quality evaluation.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="Optional NEW directory to retain toy artifacts")
    args = p.parse_args()
    # Keep smoke test CPU-only, even on a GPU host, with bounded thread counts.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MINI_MODAL_VOLUME"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    if args.out:
        root = Path(args.out).resolve()
        root.mkdir(exist_ok=False, parents=True)
        run(root)
    else:
        with tempfile.TemporaryDirectory(prefix="mini-llm-smoke-") as tmp:
            run(Path(tmp))


if __name__ == "__main__":
    main()
