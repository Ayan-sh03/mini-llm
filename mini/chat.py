"""Plain completion before SFT; chat template after SFT. No inference server needed."""
import argparse

import torch
from tokenizers import Tokenizer

from .common import SPECIAL
from .engine import amp, build_model, load_checkpoint
from .prepare import encode_plain


def sample_last(logits, args, banned):
    """Pick one token from the final position of a `(1, T, vocab)` logits tensor."""
    logits = logits[:, -1, :].float()
    logits[:, banned] = -float("inf")
    if args.temperature <= 0:
        return int(logits.argmax(-1))
    logits = logits / args.temperature
    if args.top_k > 0:
        cut = logits.topk(min(args.top_k, logits.shape[-1])).values[:, -1:]
        logits[logits < cut] = -float("inf")
    return int(torch.multinomial(logits.softmax(-1), 1))


@torch.no_grad()
def generate(model, prompt, limit, args, device, stop, banned):
    """Incremental decode through the LitGPT KV cache: one forward pass per new token.

    The prompt is prefilled once; every later step feeds a single token. Re-running the
    whole prefix each step made this path quadratic in the output length and unusable as
    a latency datapoint. Prefill uses positions `0..len(prompt)-1`, and each new token is
    written at position `len(ids) - 1` with `input_pos_maxp1` covering it.
    """
    ids = list(prompt)
    model.set_kv_cache(batch_size=1, max_seq_length=limit, device=device)
    chunk = torch.tensor([ids], device=device, dtype=torch.int64)
    input_pos = torch.arange(len(ids), device=device, dtype=torch.int64)
    input_pos_maxp1 = len(ids)
    while len(ids) < limit and len(ids) - len(prompt) < args.max_new_tokens:
        with amp(device):
            logits = model(chunk, input_pos, input_pos_maxp1=input_pos_maxp1)
        token = sample_last(logits, args, banned)
        if token in stop:
            break
        ids.append(token)
        chunk = torch.tensor([[token]], device=device, dtype=torch.int64)
        input_pos = torch.tensor([len(ids) - 1], device=device, dtype=torch.int64)
        input_pos_maxp1 = len(ids)
    return ids


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
        ids = generate(model, ids, limit, args, device, stop, banned)
        if many:
            print(f"\n### {prompt}", flush=True)
            print(prompt + tok.decode(ids[start:]), flush=True)
        else:
            print(tok.decode(ids[start:]))


if __name__ == "__main__":
    main()
