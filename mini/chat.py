"""Plain completion before SFT; chat template after SFT. No inference server needed."""
import argparse

import torch
from tokenizers import Tokenizer

from .common import SPECIAL
from .engine import amp, build_model, load_checkpoint
from .prepare import encode_plain


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True, action="append",
                   help="Repeat the flag to run several prompts in one process")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()
    state = load_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = build_model(state["config"]["model"], device)
    model.load_state_dict(state["model"])
    model.eval()
    tok = Tokenizer.from_file(args.checkpoint + "/tokenizer.json")
    limit = state["config"]["train"]["seq_len"]
    stop = {tok.token_to_id("<|eos|>"), tok.token_to_id("<|end|>")}
    banned = [tok.token_to_id(s) for s in SPECIAL if tok.token_to_id(s) not in stop]
    many = len(args.prompt) > 1
    for prompt in args.prompt:
        ids = encode_plain(tok, prompt)
        if args.chat:
            ids = ([tok.token_to_id("<|user|>")] + ids +
                   [tok.token_to_id("<|end|>"), tok.token_to_id("<|assistant|>")])
        if not ids:
            p.error("Prompt must not be empty")
        if len(ids) >= limit:
            p.error("Prompt fills the context window; shorten it")
        start = len(ids)
        for _ in range(min(args.max_new_tokens, limit - len(ids))):
            with amp(device):
                logits = model(torch.tensor([ids], device=device))[:, -1, :].float()
            logits[:, banned] = -float("inf")
            if args.temperature <= 0:
                token = int(logits.argmax(-1))
            else:
                logits /= args.temperature
                if args.top_k > 0:
                    cut = logits.topk(min(args.top_k, logits.shape[-1])).values[:, -1:]
                    logits[logits < cut] = -float("inf")
                token = int(torch.multinomial(logits.softmax(-1), 1))
            if token in stop:
                break
            ids.append(token)
        if many:
            print(f"\n### {prompt}", flush=True)
            print(prompt + tok.decode(ids[start:]), flush=True)
        else:
            print(tok.decode(ids[start:]))


if __name__ == "__main__":
    main()
