# Scratch 123M: a portable learning experiment

**Updated budget: $30 Modal + $51 Hyperbolic. Start with [HYPERBOLIC.md](HYPERBOLIC.md)**
for the revised allocation, H100/RTX PRO 6000 setup and data-transfer commands.
The older all-Modal walkthrough below remains useful as a reference; its $80 Modal
spending example is superseded by that guide.

This is runnable **starter code**, not a pretrained model or a validated quality recipe.
No paid compute launches on install/import. Cloud commands below do spend your credits.
Default training invocation stops after **30 minutes** of the training loop.
Data loading, compilation, checkpointing and upload overhead can extend wall time.

## What's included

- LitGPT **0.5.9 model**, with an explicit single-GPU PyTorch training loop.
- 122,708,736 parameters: 16 layers, width 768, SwiGLU 2048, 12 Q / 2 KV heads,
  RoPE, RMSNorm, no biases, tied 32,768-token embeddings.
- CPU byte-level BPE training, immutable uint16 token shards, document-level holdout.
- Token-based warmup/stable/linear-decay (WSD), BF16 autocast, FP32 master weights,
  AdamW, gradient accumulation, clipping and optional torch.compile.
- Full optimizer-boundary checkpoints, reader cursors, RNG, schedule progress,
  tokenizer, model config, dependency versions and checksums.
- Private HF uploads, download/resume on another provider.
- Assistant-only SFT with complete conversations; overlength conversations are skipped.
- CPU smoke tests and simple text generation. No pretrained weights are downloaded.

The custom loop is intentional: stock LitGPT's pretrain scheduler/checkpoint semantics
are not silently assumed to provide WSD and full RNG restoration. These `.pt` files are
**our trainer's format**, not directly loadable with `litgpt --resume` or HF `AutoModel`.
Use `python -m mini.train --resume ...` on every provider.

## 1. Install and prove the pipeline on CPU

Python 3.11/3.12; Linux/WSL is the easiest route. Use a NEW virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
# CPU smoke on your laptop: install the CPU wheel first to avoid CUDA downloads.
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
python -m mini.smoke
```

On a GPU provider, omit the CPU-wheel line and install requirements normally.
The smoke test creates a **disposable ~0.1M model**, trains a tokenizer on toy stories,
packs text, compares uninterrupted training with fresh-process checkpoint resume,
does two SFT updates, and generates text. It does not test conversational quality.
Use `python -m mini.smoke --out smoke-output` to retain its artifacts in a NEW folder.
Then, optionally, test overfitting the same batches:

```bash
python -m mini.train --config smoke-output/tiny.json --data smoke-output/data \
  --tokenizer smoke-output/tok --out overfit-check --device cpu \
  --overfit-batch --max-steps 100 --max-seconds 600
```

## 2. Modal setup

```bash
modal setup
```

For private HF uploads/gated datasets, create a Modal secret named `huggingface` in
the dashboard containing `HF_TOKEN`. Use a token with write access to your target
checkpoint repo and access to your chosen gated datasets. Never paste tokens into code.

```bash
export MINI_HF_AUTH=1
```

Omit that environment variable for public dataset preparation with local-only saves.
All cloud paths are relative to a persistent volume mounted at `/work`; project code
and configurations are at `/project`. Run **one job at a time**; do not share a run
output directory between writers. Automatic job retries are disabled.

You can run the same smoke test remotely on CPU, if wanted:

```bash
modal run modal_app.py --task smoke
```

## 3. Train the real tokenizer on CPU

```bash
modal run modal_app.py --task prepare --args 'tokenizer --out tokenizer --max-chars 100000000'
```

This uses the verified Ultra-FineWeb schema: dataset `openbmb/Ultra-FineWeb`, split
`en`, text field `content`. Streaming is bounded; it does not download the whole
trillion-token dataset. Its source revision is recorded. Byte fallback is provided
by the complete ByteLevel initial alphabet. Six reserved tokens are inside the 32k.

Freeze this tokenizer before your real run. **Do not retrain it during provider moves.**
Use the same tokenizer for pretraining, SFT, validation and generation.
If 100M sampled characters do not produce a full 32k vocabulary, the command fails
explicitly; increase the sample and use a NEW output directory.

## 4. Prepare a bounded pilot corpus on CPU

```bash
modal run modal_app.py --task prepare --args 'pack --tokenizer tokenizer --out pilot-data --max-tokens 100000000'
```

This writes about 200MB of training tokens plus validation and small metadata.
Preparation is not resumable yet: a failed partial output is never mistaken for a
complete corpus. Use a fresh output directory when retrying. Check the printed counts.
`--max-docs` also limits the scan; if hit first, fewer than requested tokens may exist.

## 5. Benchmark the ACTUAL 123M model for 20 minutes

```bash
modal run modal_app.py --task train --args '--config /project/configs/123m.json --data pilot-data --tokenizer tokenizer --out pilot --max-seconds 1200'
```

Watch `loss`, `grad_norm`, `val_loss`, `tokens_per_second` and `train_epoch`.
Ignore initial compilation/warm-up throughput. This is a performance/stability pilot,
not a useful pretrained model. Expect the small corpus to repeat if you let it run long.
`--micro-batch 8` or `16` can improve speed if memory permits; global batch stays 128.
Use `--no-compile` to diagnose compile errors; the first compiled steps can be slow.

Checkpoint folders are printed, e.g. `pilot/step-00000100`. Use the ACTUAL printed path.
Test a fresh-process restart of that same pilot:

```bash
modal run modal_app.py --task train --args '--config /project/configs/123m.json --data pilot-data --tokenizer tokenizer --out pilot-resume --resume pilot/step-00000100 --max-steps 5 --max-seconds 600'
```

The name above is an example, not a checkpoint guaranteed to exist.

## 6. Choose the real token budget, then train

`configs/123m.json` currently contains a **provisional 5B-token schedule**, NOT an
assertion that $80 buys 5B tokens. Choose the budget from measured throughput:

`affordable tokens ≈ sustained tokens/sec × training seconds × 0.85`

The 15% allowance is an estimate for evaluation/save overhead; watch actual billing.
Suggested spending envelopes: $5 prep/debug, $55 pretrain, $10 SFT, $10 reserve.
H100 GPU-only cost was $3.9492/hour when checked; CPU/RAM are additional.
The timer is NOT a dollar-spend cap. Configure Modal budget alerts and monitor costs.

Edit `train.total_tokens` in `configs/123m.json` BEFORE the full run. Prepare that
many tokens (e.g. 5B -> roughly 10GB uint16). Source scanning/tokenization costs CPU
time; do not do it while holding an H100.

```bash
modal run modal_app.py --task prepare --args 'pack --tokenizer tokenizer --out train-data --max-tokens 5000000000'
```

Then launch a bounded first segment:

```bash
modal run --detach modal_app.py --task train --args '--config /project/configs/123m.json --data train-data --tokenizer tokenizer --out pretrain --max-seconds 3600 --hub-repo YOUR_USERNAME/mini-123m-training'
```

Continue with `--resume pretrain/step-XXXXXXXX` for later time segments, preserving
the total-token schedule. Changing to a larger real corpus after the pilot means a
**fresh real run**: the strict resume check deliberately rejects a new data manifest.
If you prepared the final corpus before the pilot and keep the same recipe, its pilot
checkpoint can instead continue directly.

WSD here is 1% warmup, 89% stable, 10% linear decay. LR=6e-4 is a starting hypothesis.
Do not end the run abruptly halfway through the stable phase and expect a polished
base model. Budget for reaching the configured decay phase and for SFT afterward.
No muP/MiniCPM scaling, RL, FP8, MTP or sparse attention is implemented.

### Optional Ultra-FineWeb-L3

The source adapter accepts `--dataset`, `--subset`, `--split`, `--field`, `--revision`.
Prepare the English L3 slice separately using the same tokenizer after inspecting its
current schema. Pass `--decay-data l3-data` from the START of the real run, including
on every resume. It samples that corpus with `decay_mix=0.1` only in the last 10% of
training: **about 1% overall**, not 10% overall. This is an optional experimental
ablation, not a reproduction of OpenBMB's recipe. Baseline commands above use L2 only.

## 7. Chat SFT (start with 20k short conversations)

```bash
modal run modal_app.py --task prepare --args 'sft --dataset HuggingFaceH4/ultrachat_200k --split train_sft --field messages --tokenizer tokenizer --out sft-data --max-examples 20000'
```

This selects the first 20k accepted conversations from a deterministic shuffled
stream, deduplicates by first prompt, holds out about 1% by prompt, skips malformed
and overlength conversations, and masks user/system/pad targets. It is NOT a
semantic quality/language classifier; inspect samples. No truncated assistant
answers are used. Role headers themselves are masked; assistant end markers are trained.

```bash
modal run modal_app.py --task train --args '--config /project/configs/sft-123m.json --data sft-data --tokenizer tokenizer --out sft --init pretrain/step-XXXXXXXX --max-seconds 1800 --hub-repo YOUR_USERNAME/mini-123m-training --hub-prefix sft'
```

Replace `XXXXXXXX`. `--init` loads weights ONLY and starts a new SFT optimizer.
Use `--resume` for interrupted SFT. The SFT budget counts padded positions (20M),
approximately one pass over 20k examples at 1024 positions. `supervised_tokens`
separately reports actual assistant targets. This un-packed SFT baseline favors
correct masking and readability over maximum throughput. Judge short responses
at multiple checkpoints; larger SFT sets are not automatically better.

Optional UltraData-SFT `IF`/`Knowledge` `no_think` subsets are gated and are not
downloaded automatically. Accept terms, inspect schema/language/difficulty, then
adapt to `{"messages": [{"role": ..., "content": ...}, ...]}` JSONL and use
`--local your_filtered.jsonl --field messages` instead. Review all upstream licenses;
do not publicly mirror third-party source corpora without checking their terms.

## 8. Generate text

Download a complete checkpoint and run locally (CPU works for 123M, albeit slower):

```bash
python -m mini.chat --checkpoint downloaded/sft/step-XXXXXXXX --prompt 'How was your day?' --chat
```

Before SFT, omit `--chat` and use a completion prompt such as `Once upon a time`.
The included inference loop has no KV cache; it is for inspection, not deployment.
Validation loss + sample chats are included; formal lm-evaluation-harness integration,
HF Transformers/safetensors export and GGUF export are not included in this starter.

## Provider migration

Uploads happen after completed local saves, asynchronously on periodic intervals and
synchronously at normal exit. An unexpected hard kill loses work since the most
recent durable save/upload. Hourly uploads are NOT guaranteed if the process dies
before them. For a short pilot, explicitly upload its first checkpoint on CPU:

```bash
modal run modal_app.py --task hub --args 'upload --repo YOUR_USERNAME/mini-123m-training --folder pilot/step-XXXXXXXX --prefix pilot'
```

On a new provider, install the same project/dependencies and authenticate HF with a
secure environment variable or login. Download an explicit complete folder:

```bash
python -m mini.hub download --repo YOUR_USERNAME/mini-123m-training \
  --checkpoint pretrain/step-XXXXXXXX --out downloaded
python -m mini.train --config configs/123m.json --data train-data \
  --tokenizer downloaded/pretrain/step-XXXXXXXX --out continued \
  --resume downloaded/pretrain/step-XXXXXXXX --max-seconds 1800
```

**Also transfer the exact prepared `train-data` directory**, including `manifest.json`
and shards, and `l3-data` if used. They are not bundled into model checkpoint uploads.
For example, use `modal volume get ayan-mini-llm train-data ./train-data` from your
local machine, then copy the directory to the new provider. Don't regenerate it
from a moving remote dataset and assume identical order. Move this project's code
and unchanged config too. Different root paths are fine; different hashes are not.

Keep global batch, tokenizer, architecture and schedule fixed. `--micro-batch` may
change to fit a smaller BF16 GPU; floating-point round-off can then differ. GPU
hardware/versions may also change numerical results. CPU fresh-process parity is
tested; bitwise GPU equivalence is NOT promised. This is single-GPU only.

Checkpoint loading uses `torch.load(weights_only=False)` for Python/NumPy RNG state.
**Load only checkpoints you created/trust.** Checksums detect corruption, not malicious
pickle content. The HF repo must be private. No credentials go into checkpoints.
Native full checkpoints are ~1.5GB for 123M with FP32 weights and Adam moments.

Checkpoint files are intentionally not automatically deleted. Monitor disk/HF quota.
Deleting files in a Git-backed HF repo does not necessarily remove their historical
storage; retention/history cleanup is a separate explicit operation. Final HF upload
waits can consume GPU time; on poor networks, omit `--hub-repo` and use CPU upload
jobs after frequent committed saves, accepting less frequent off-provider backups.

## Code map

| File | Responsibility |
|---|---|
| `mini/prepare.py` | Tokenizer, bounded streaming, binary packing, SFT masks |
| `mini/data.py` | Checked shards, deterministic block order, saved cursor |
| `mini/engine.py` | LitGPT model, optimizer, loss, validation, full checkpoints |
| `mini/train.py` | WSD, accumulation, time limit, logs, resume and uploads |
| `mini/hub.py` | Explicit private checkpoint upload/download |
| `mini/chat.py` | Base completions and SFT chat |
| `mini/smoke.py` | Offline end-to-end CPU test |
| `modal_app.py` | CPU/GPU jobs and persistent volume |

## Known limits / evidence

This is an educational baseline. Sequential shards are reordered with an affine
bijection of blocks, not a full uniform random permutation. Pretraining packs across
documents separated by EOS; it does not use document-isolated attention. Exact-text
hash splitting is not near-duplicate decontamination. L3 rewrites may overlap L2
validation semantically; don't interpret that loss as independent evidence of gain.

Main dependencies are pinned; transitive versions can still change. Checkpoints
record core versions. For strict reproduction, retain your container image and a
full `pip freeze` from the successful pilot (never copy credentials).

References checked during implementation:
- LitGPT: https://github.com/Lightning-AI/litgpt/tree/v0.5.9
- Ultra-FineWeb schema: https://huggingface.co/datasets/openbmb/Ultra-FineWeb
- UltraChat splits: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- UltraData-SFT: https://huggingface.co/datasets/openbmb/UltraData-SFT-2605
- Modal volumes: https://modal.com/docs/guide/volumes
- Modal pricing: https://modal.com/pricing
- HF upload: https://huggingface.co/docs/huggingface_hub/guides/upload

See `VALIDATION.md` for checks actually run, versus cloud/GPU checks still required.
