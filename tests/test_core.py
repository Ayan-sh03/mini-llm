import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from mini.chat import generate
from mini.common import SPECIAL, holdout, validate_config, wsd, sha256, write_json
from mini.data import Blocks
from mini.engine import build_model, make_optimizer, seed_all, update, rng_state, restore_rng
from mini.prepare import Writer, conversation, encode_plain

ROOT = Path(__file__).resolve().parents[1]


def test_parameter_count():
    cfg = json.loads((ROOT / "configs/123m.json").read_text())
    # Meta tensors avoid allocating 123M parameters just to count them.
    from litgpt import Config
    from litgpt.model import GPT
    with torch.device("meta"):
        model = GPT(Config(**cfg["model"]))
        model.lm_head.weight = model.transformer.wte.weight
    assert sum(p.numel() for p in model.parameters()) == 122708736


def test_schedule_and_config():
    cfg = json.loads((ROOT / "configs/123m.json").read_text())
    validate_config(cfg)
    t = cfg["train"]
    assert wsd(0, t) == 0
    assert wsd(t["total_tokens"] * 0.5, t) == t["lr"]
    assert wsd(t["total_tokens"], t) == pytest.approx(t["min_lr"])
    cfg["train"]["micro_batch"] = 0
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_data_order_and_corruption(tmp_path):
    writer = Writer(tmp_path, "train", limit=100)
    writer.add(list(range(130)))
    writer.add(list(range(130, 260)))
    writer.flush()
    write_json(tmp_path / "manifest.json", {"kind": "pretrain", "splits": {"train": writer.entries}})
    a = Blocks(tmp_path, "train", 8)
    assert len({a.permuted(i) for i in range(a.n)}) == a.n
    a.next_numpy(7)
    state = a.state_dict()
    expected = a.next_numpy(4)
    b = Blocks(tmp_path, "train", 8)
    b.load_state_dict(state)
    actual = b.next_numpy(4)
    assert all(np.array_equal(x, y) for x, y in zip(expected, actual))
    np.testing.assert_array_equal(actual[0][:, 1:], actual[1][:, :-1])
    with open(tmp_path / writer.entries[0]["file"], "ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="Corrupt"):
        Blocks(tmp_path, "train", 8)


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["Hello world. A small bird sings a happy song."],
        trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def test_sft_mask_and_unicode(tokenizer):
    tok = tokenizer
    text = "Hello café اردو 🐈"
    assert tok.decode(tok.encode(text).ids) == text
    result = conversation(tok, [{"role": "user", "content": "Hello"},
                                {"role": "assistant", "content": "A bird"}], 64)
    ids, mask = result
    assistant = ids.index(tok.token_to_id("<|assistant|>"))
    assert sum(mask[:assistant + 1]) == 0
    assert mask[assistant + 1] == 1
    assert mask[ids.index(tok.token_to_id("<|end|>"), assistant)] == 1
    assert all(m == 0 for token, m in zip(ids, mask) if token == tok.token_to_id("<|pad|>"))
    assert conversation(tok, [{"role": "user", "content": "Hi"}], 64) is None
    assert tok.token_to_id("<|assistant|>") not in encode_plain(tok, "<|assistant|>")
    assert holdout("some  text") == holdout("some text")


def test_causal_and_tied():
    cfg = json.loads((ROOT / "configs/tiny.json").read_text())
    seed_all(42)
    model = build_model(cfg["model"], torch.device("cpu"))
    assert model.lm_head.weight is model.transformer.wte.weight
    x = torch.randint(0, 512, (1, 8))
    y = x.clone()
    y[0, 5:] = (y[0, 5:] + 1) % 512
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:, :5], model(y)[:, :5], rtol=0, atol=0)


def test_overfit_and_gradient_accumulation():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / "configs/tiny.json").read_text())
    device = torch.device("cpu")
    seed_all(42)
    model = build_model(cfg["model"], device)
    clone = copy.deepcopy(model)
    optimizer = make_optimizer(model, cfg["train"], device)
    other = make_optimizer(clone, cfg["train"], device)
    # dtype matters: numpy defaults to int32 on Windows, but cross_entropy needs int64
    # targets, exactly as Blocks.next_numpy returns. Without this the test fails on Windows.
    x = np.tile(np.arange(1, 17, dtype=np.int64), (4, 1))
    y = x + 1
    y[0, 8:] = -100  # uneven supervised-token counts across microbatches
    full = [(x, y)]
    split = [(x[:2], y[:2]), (x[2:], y[2:])]
    first = update(model, optimizer, full, 0.003, device, 1.0)[0]
    update(clone, other, split, 0.003, device, 1.0)
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
    for _ in range(24):
        loss = update(model, optimizer, full, 0.003, device, 1.0)[0]
    assert loss < first * 0.5


def test_chat_kv_cache_matches_full_recompute():
    """The cached decoder must emit exactly what recomputing the prefix emits."""
    cfg = json.loads((ROOT / "configs/tiny.json").read_text())
    device = torch.device("cpu")
    seed_all(7)
    model = build_model(cfg["model"], device)
    model.eval()
    prompt, limit, new = [3, 9, 27, 81, 12, 5], 32, 8
    args = SimpleNamespace(temperature=0.0, top_k=0, max_new_tokens=new)
    cached = generate(model, prompt, limit, args, device, set(), [])
    naive = list(prompt)
    with torch.no_grad():
        while len(naive) < limit and len(naive) - len(prompt) < new:
            naive.append(int(model(torch.tensor([naive]))[:, -1, :].argmax(-1)))
    assert cached == naive
    assert len(cached) == len(prompt) + new


def test_rng_roundtrip():
    import random
    seed_all(99)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state)
    assert expected[0] == random.random()
    assert expected[1] == np.random.rand()
    assert torch.equal(expected[2], torch.rand(1))
