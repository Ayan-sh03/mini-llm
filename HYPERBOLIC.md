# Modal $30 + Hyperbolic $51

No model/training-recipe changes are needed. Keep 123M, 32k vocab, BF16, WSD and
the same checkpoint format. Use Modal for CPU preparation and later SFT; use
Hyperbolic's rented GPU for the main pretraining. This guide supersedes the old
README's $80-on-Modal spending example.

## What changed

| File | Change |
|---|---|
| `scripts/setup_hyperbolic.sh` | Isolated environment, pinned PyTorch CUDA 12.8 wheel |
| `mini/preflight.py` | Reports actual GPU and runs a tiny forward/backward/AdamW check |
| `mini/assets.py` | Transfers identical prepared data + tokenizer through a private HF dataset repo |
| `modal_app.py` | Adds CPU `assets` task |
| `mini/train.py`, `mini/engine.py`, configs | Unchanged; existing checkpoints still work |

Assuming your RTX listing is **RTX PRO 6000 Blackwell**, use CUDA 12.8 PyTorch.
The setup script uses **torch 2.7.1+cu128**, which also works on H100 with a compatible
host driver. It doesn't install/change the host driver. A driver/kernel failure
must be resolved in the rental image before the main training run.
Sources: [PyTorch 2.7 Blackwell support](https://pytorch.org/blog/pytorch-2-7/) and
[official CUDA 12.8 wheel commands](https://pytorch.org/get-started/previous-versions/).

## Budget and GPU choice

Using the prices you supplied, not a separately verified live quote:

| Hyperbolic GPU | Hourly price | $51 theoretical maximum |
|---|---:|---:|
| H100 | $3.19 | 15.99 h |
| RTX PRO 6000 | $1.79 | 28.49 h |

These are upper bounds assuming those are the complete hourly charges. Setup,
downloads, compilation and uploads also occupy the rental. Reserve about $6 of
the $51 for overhead/pilots; $45 is about 14.1 H100 hours or 25.1 RTX hours.
Use a small slice of Modal's $30 for CPU preparation, retain the rest for SFT,
evaluation and recovery. Do not spend the entire credit balance on pretraining.

Choose using **measured tokens per dollar** after warm-up: RTX beats H100 if its
tokens/sec is greater than `1.79 / 3.19 = 56.1%` of H100 throughput. Same model,
context length, global batch and precision; tune only the microbatch for each GPU.
An H100-vs-RTX claim based only on advertised TFLOPs is not a reliable budget plan.

## 1. Prepare on Modal BEFORE renting

Follow the README to build the tokenizer and `pilot-data` on Modal. Use this updated
project locally for the new `assets` task. Keep your HF token in the Modal secret
named `huggingface`, and enable it in your local shell:

```bash
export MINI_HF_AUTH=1
modal run modal_app.py --task assets --args 'upload --repo YOUR_USERNAME/mini-123m-assets --name pilot-v1 --data pilot-data --tokenizer tokenizer'
```

This creates a PRIVATE **dataset** repository for your own prepared shards. Your
existing checkpoint repository remains a PRIVATE **model** repository. Only named
shards, masks, manifest and `tokenizer.json` are uploaded, not arbitrary local files.
An upload is ready only after `READY.json` is committed. An interrupted incomplete
upload may be retried with the same name; completed bundles require a NEW name.
One writer per bundle. Every downloaded shard is checksum-verified.

For the final corpus, after choosing a budget, prepare `train-data` and upload it
as `--name train-v1 --data train-data`. For optional L3, create a separate bundle.
If HF quota is insufficient, transfer the same directories with scp/rsync instead;
there is no requirement to use HF for data. Never regenerate the tokenizer to migrate.

## 2. Rent, SSH, copy this small code archive

Rent one GPU in your Hyperbolic account and use its displayed SSH connection details.
Copy/extract this code archive on the machine; use the exact port/username shown
in your account. No script here creates or terminates a rental or needs your
Hyperbolic API key.

Inside the extracted `mini_llm` directory:

```bash
bash scripts/setup_hyperbolic.sh
source .venv-hyperbolic/bin/activate
unset MINI_MODAL_VOLUME
huggingface-cli login
python -m mini.assets download --repo YOUR_USERNAME/mini-123m-assets --name pilot-v1 --out assets
```

The login is interactive; do not place access tokens in command arguments. Setup
requires Python 3.11/3.12 with venv and an NVIDIA driver already available. Set
`MINI_PYTHON=python3.11` if the default Python differs. If the machine exposes several
GPUs, set `CUDA_VISIBLE_DEVICES=0` before setup/training. Select a single-GPU rental
to avoid paying for GPUs this trainer won't use.

Preflight runs the tiny model, not 123M. Optionally exercise compiled kernels too:

```bash
python -m mini.preflight --compile
```

## 3. Run the 20-minute actual-model pilot

Use tmux if installed so an SSH disconnect doesn't kill training:

```bash
tmux new -s mini-llm
# Activate again inside tmux if the environment was not inherited:
source .venv-hyperbolic/bin/activate
python -m mini.train --config configs/123m.json \
  --data assets/pilot-v1/data --tokenizer assets/pilot-v1/tokenizer \
  --out pilot --max-seconds 1200
```

Detach with Ctrl-B then D; reconnect with `tmux attach -t mini-llm`. If tmux is not
available, keep the SSH session open for the pilot. Training defaults to CUDA.
Use `--no-compile` only if compile fails; benchmark the setting you will actually use.
The 20-minute loop timer includes initial lazy compilation, but not initial data
verification/model setup or final checkpoint/upload overhead.

Watch logged `tokens_per_second` after warm-up. With each rate measured, compute
`tokens_per_dollar = tokens_per_second * 3600 / hourly_price`.
`--micro-batch 8`/`16` may be faster if memory permits; global batch remains fixed.
No promised throughput is baked into the code.

## 4. Main run / provider resume

After the pilot, choose `train.total_tokens` before the real run. Prepare and transfer
the final corpus. The starter rejects a changed corpus on resume: a pilot with a small
corpus is disposable. A pilot on the FINAL corpus and recipe can be continued.

```bash
python -m mini.assets download --repo YOUR_USERNAME/mini-123m-assets --name train-v1 --out assets
python -m mini.train --config configs/123m.json \
  --data assets/train-v1/data --tokenizer assets/train-v1/tokenizer \
  --out pretrain --max-seconds 3600 \
  --hub-repo YOUR_USERNAME/mini-123m-training
```

Run in bounded segments. To resume an existing Modal/Hyperbolic checkpoint:

```bash
python -m mini.hub download --repo YOUR_USERNAME/mini-123m-training \
  --checkpoint pretrain/step-XXXXXXXX --out downloaded
python -m mini.train --config configs/123m.json \
  --data assets/train-v1/data --tokenizer assets/train-v1/tokenizer \
  --resume downloaded/pretrain/step-XXXXXXXX --out continued \
  --max-seconds 3600 --hub-repo YOUR_USERNAME/mini-123m-training
```

Replace `XXXXXXXX` with a checkpoint that actually exists. Choose a distinct
`--hub-prefix` if deliberately branching a run, so branches don't overwrite each
other's remote step paths. Preserve the same config/global batch/schedule; only
microbatch and runtime logging/compilation options may change. Preserve optional
decay-data if the original run used it.

## 5. Stop billing, then use Modal for SFT

Normal training exit saves a checkpoint and, with `--hub-repo`, waits for upload.
Confirm the completed checkpoint is in HF (or copy it off the rental) BEFORE
terminating the rental. A disk on a rented machine is not an off-provider backup.

**Exiting Python, detaching SSH/tmux or reaching `--max-seconds` does NOT terminate
the rental. Explicitly terminate it in Hyperbolic's console to end the rental.**
The script does not enforce a dollar cap or auto-delete an instance. Check any
separately billed storage too. Keep enough credit to finish the final upload.

Back on Modal, download the pretraining checkpoint into its persistent volume:

```bash
modal run modal_app.py --task hub --args 'download --repo YOUR_USERNAME/mini-123m-training --checkpoint pretrain/step-XXXXXXXX --out downloaded'
```

Then follow the README's SFT command with
`--init downloaded/pretrain/step-XXXXXXXX`. Keep the original frozen tokenizer.
The remaining Modal credit pays for SFT/evaluation; it is not transferable to Hyperbolic.

## Verified here

Local CPU tests, shell syntax and mock transfer tests are covered in VALIDATION.md.
No GPU rental, CUDA execution or authenticated HF transfer has been performed here.
Run preflight and the bounded pilot before committing the main budget.
