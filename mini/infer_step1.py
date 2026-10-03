"""Inference lab, step 1: prompt tokens -> next-token scores.

Run from mini_llm/ with a complete checkpoint directory:
    python -m mini.infer_step1 --checkpoint PATH --prompt "Once upon a time"
Add --chat when using an SFT checkpoint and asking it to answer as an assistant.
"""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .engine import amp, build_model, load_checkpoint
from .prepare import encode_plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    state = load_checkpoint(checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    device = torch.device(args.device)

    model = build_model(state["config"]["model"], device)
    model.load_state_dict(state["model"])
    model.eval()

    ids = encode_plain(tokenizer, args.prompt)
    if args.chat:
        ids = ([tokenizer.token_to_id("<|user|>")] + ids +
               [tokenizer.token_to_id("<|end|>"), tokenizer.token_to_id("<|assistant|>")])
    limit = state["config"]["train"]["seq_len"]
    if not ids or len(ids) >= limit:
        parser.error(f"Prompt must contain 1 to {limit - 1} tokens")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode(), amp(device):
        logits = model(input_ids)  # (batch=1, prompt length, vocabulary size)

    next_logits = logits[0, -1].float()
    next_id = int(next_logits.argmax())
    print(f"prompt tokens: {len(ids)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"highest-score token id: {next_id}")
    print(f"decoded token: {tokenizer.decode([next_id], skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
