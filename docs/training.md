# Fine-tuning v1

LoRA fine-tuning of Qwen3-VL on the HIT-UAV QA train split. It runs on CUDA (Colab) with
the `train` extra; the Mac is used for code, tests and MLX inference.

- Code: `src/thermground/data/sft.py` (examples, targets, label masking),
  `src/thermground/models/lora.py` (training loop), `src/thermground/models/hf_vlm.py`
  (transformers inference)
- Configs: `configs/train/*.yaml`, base models in `configs/models/*_hf.yaml`
- Run: `notebooks/colab_train.ipynb`, or directly
  `python scripts/train_lora.py --config configs/train/qwen3_vl_4b_lora_v1.yaml --out runs/<name>`

## Examples

Each QA item becomes one chat turn: the image, then the question with the same
answer-format instruction as in evaluation (`configs/eval/hituav_eval_v1.yaml`). Training
and evaluation therefore ask exactly the same way.

| Family | Target |
|---|---|
| presence, region | `yes` or `no` |
| count | a bucket label: `0`, `1`, `2`, `3`, `4-6`, `7+` |
| point | one `[x1, y1, x2, y2]` per line in the model's coordinate format (0-1000 for Qwen3-VL), top to bottom; `none` if nobody is there |

Point targets are checked to parse back to the original boxes (`tests/test_sft.py`). The
loss is computed on the answer and the end of the turn only.

Only train-split items are used, and the wording templates held out for test never appear
in train, so the held-out wording check in evaluation stays valid.

## Family mix

The train split is dominated by region questions (6,258 of 9,885). Each family is
resampled with a weight from the config:

| Family | Items | Weight | Examples |
|---|---|---|---|
| presence | 1,856 | 1.0 | 1,856 |
| region | 6,258 | 0.3 | 1,877 |
| count | 696 | 2.0 | 1,392 |
| point | 1,075 | 1.0 | 1,075 |

Presence and region are balanced between yes and no by construction, which directly
targets the "no" bias of the zero-shot models (`results/week3/analysis.md`).

## Model and training settings (v1)

| Setting | Value |
|---|---|
| Base | Qwen3-VL-4B-Instruct, bf16, pinned revision (2B for pipeline checks) |
| Adapters | LoRA r=16, alpha=32, dropout 0.05 on the language model's attention and MLP projections |
| Frozen | vision encoder, merger, embeddings |
| Optimiser | AdamW, lr 1e-4, cosine schedule with 3% warm-up, gradient clipping at 1.0 |
| Batch | 4 x 4 gradient accumulation = 16; one epoch (~390 steps) |
| Memory | gradient checkpointing |
| Seed | 0 (Python, NumPy, torch, data order) |

A run writes `adapter/`, `train_log.jsonl`, `train_config.yaml` (the full config, base
model and step count) and `model.yaml`, which `scripts/evaluate.py --model` takes directly.

## Comparing with the zero-shot baseline

The week 3 numbers come from 4-bit MLX builds. Quantisation can change results, so the
fine-tuned model is compared with the same bf16 model run zero-shot through the same
transformers backend (`configs/models/qwen3_vl_4b_hf.yaml`).
