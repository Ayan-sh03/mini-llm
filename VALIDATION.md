# Validation — 2026-09-20

## Review follow-up: code fixes

Applied against the 2026-09-20 external review, findings A/B/C/E and the public-repo guard:

- `mini/prepare.py` — `pack --local` no longer falls through to `fast_pack()`. The new
  `pack_local()` writes the same shard/manifest format without importing `datasets` or
  `huggingface_hub`, so the documented offline smoke path is offline again, and `--max-docs`
  is honored on that path.
- `mini/prepare.py` — `merge()` now requires every part to agree on `version`, `kind`,
  `seq_len`, `vocab_size` and `tokenizer_sha256`; validates each shard's byte length against
  the declared row layout (an SFT shard must be a whole number of `seq_len + 1` rows with a
  mask of the same length); refuses to overwrite an existing `manifest.json`; embeds each
  part's full metadata under `source.provenance`; and retains the part manifests instead of
  deleting them.
- `mini/train.py` — validation results are persisted to `metrics.jsonl` with `val_blocks`
  (the evaluated subset size) and the validation `data_fingerprint`, so a micro-batch change
  can no longer silently shift what a val curve means.
- `mini/chat.py` — generation uses the LitGPT KV cache (one prefill plus one forward pass per
  new token) instead of recomputing the entire prefix at every step.
- `mini/hub.py`, `mini/train.py` — uploading checkpoints to a public repo is now an explicit
  opt-in (`--allow-public`); the default still refuses, and a training run surfaces the
  refusal at launch rather than at the final upload.

Verification:

- `python -m pytest -q`: **19 passed** (was 12). Added 6 merge/local-pack tests and 1 test
  asserting that KV-cached decoding emits exactly the same tokens as full-prefix recompute.
- `python -m mini.smoke`: **passed** with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` set,
  so any accidental Hub access would have raised. It also asserts that a `val_loss` record
  reached `metrics.jsonl` with its subset size.
- Two **pre-existing, Windows-only** test failures were repaired in the tests, not the
  library: `np.arange` defaults to `int32` on Windows while `cross_entropy` requires `int64`
  targets, and `np.memmap` keeps a shard locked on Windows so the corruption test could not
  overwrite a live file. Both tests pass unmodified on Linux.
- Environment here: Windows, Python 3.12.8, PyTorch 2.7.1+cpu, LitGPT 0.5.9, NumPy 1.26.4.
  GPU paths (BF16, `torch.compile`, CUDA fused AdamW) remain unexercised in this pass.
  The throwaway test environment is the gitignored `.venv-test/` on the project drive:

  ```bash
  uv venv .venv-test --python 3.12.8
  uv pip install --python .venv-test/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-test/Scripts/python.exe litgpt==0.5.9 tokenizers==0.21.2 \
    numpy==1.26.4 huggingface-hub==0.33.4 pytest==8.4.1 datasets==3.6.0
  .venv-test/Scripts/python.exe -m pytest tests -q
  ```

Still open from the same review: microbatch-dependent decay-source sampling (D), SFT
padding/logit-cost utilization (F), cross-source and near-duplicate dedup, holdout vs
contamination checks, and the historical `train2-manifest.json`, whose source revisions were
destroyed by the old `merge` and cannot be restored by a code change.

The "offline smoke test passed" claim in the 2026-09-19 note was stale when it was written
against the later `prepare` dispatch; it is accurate again as of this date.

# Validation — 2026-09-19

## Hyperbolic update

- Clean isolated CPU environment installed successfully; `pip check` reports no broken requirements.
- `python -m pytest -q`: **12 passed** (original 9 plus 3 asset-transfer tests).
- Mock private HF upload/download preserves manifest hashes and reader resume order;
  unrelated local files are excluded, completed bundles cannot be overwritten, public
  upload destinations and corrupted shards are rejected. No live HF calls in tests.
- `python -m mini.preflight --device cpu`: passed actual forward/backward/AdamW update.
- `bash -n scripts/setup_hyperbolic.sh`: passed; GPU setup itself was not executed.
- Modal CLI help/import and Python compilation: passed.
- `mini/train.py`, `mini/engine.py`, and the 123M config are byte-identical to the
  previous package; the checkpoint format and model architecture did not change.
- CUDA 12.8 installation, Blackwell/H100 kernels, GPU compilation, rental billing and
  live transfers remain untested here. No rental was created and no credits spent.

The original CPU validation below remains applicable to the unchanged trainer.

## Executed locally

- `python -m pytest -q`: **9 passed**.
- `python -m mini.smoke`: **passed**, using generated toy text and no remote data.
- Actual 122,708,736-parameter model instantiated on CPU, forward and backward at
  sequence length 16: logits `[1, 16, 32768]`, all trainable gradients present/finite.
- Uninterrupted six-update training versus three updates + checkpoint + fresh process
  + three updates: **bitwise equal model weights and optimizer tensors on CPU**.
- Data reader order, cursor restoration and shard-corruption rejection tested.
- Tied embedding/head identity and causal attention tested.
- Assistant/user/padding loss masks, complete-response handling and Unicode round-trip tested.
- Unequal assistant-target counts across microbatches: accumulated gradients produce
  equivalent optimizer updates within FP32 tolerance.
- Tiny repeated-batch loss reduced by more than 50% in 25 updates.
- WSD boundary behavior and invalid batch configuration tested.
- Checkpoint checksum and recipe-change rejection tested.
- Full toy pipeline: tokenizer -> pretraining packing -> training -> save -> resume
  -> SFT packing -> SFT initialization -> generation.
- Modal app module imports without authentication or launching cloud compute.
- Python source compilation passed.

Environment: Linux, Python 3.12, PyTorch 2.7.1+cpu, LitGPT 0.5.9,
NumPy 1.26.4; core dependency pins are in requirements.txt.

## Not executed / still required before spending the main budget

- Modal remote image build or paid CPU/GPU function invocation.
- H100 BF16, CUDA fused AdamW, torch.compile throughput and memory checks.
- Authenticated HF upload/download round-trip or gated dataset access.
- Real Ultra-FineWeb/UltraChat ingestion through their live streaming services.
  Published schemas were checked; only local JSONL ingestion was executed here.
- Cross-provider GPU migration and GPU numerical reproducibility.
- Long-run stability, model benchmark scores or conversational quality.
- Dollar-budget enforcement: the timer is a training-loop limit, not a billing cap.

CPU toy throughput is deliberately NOT used to estimate H100 throughput or the
number of tokens your $80 can buy. Perform the 20-minute 123M pilot in README.md.
