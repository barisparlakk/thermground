"""LoRA fine-tuning of a Hugging Face vision-language model on QA examples (`train` extra).

The vision encoder stays frozen; LoRA adapters go on the language model's attention and
MLP projections. The loss covers the answer tokens only (thermground.data.sft).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from thermground.data.sft import mask_labels, split_chat
from thermground.device import get_device
from thermground.models.hf_vlm import load_base, to_rgb


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Collator:
    """Batches examples into model inputs with answer-only labels (right padding)."""

    def __init__(self, processor, load_image: Callable[[str], np.ndarray]) -> None:
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.load_image = load_image

    def _template(self, chat: list[dict], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        texts, n_answer = [], []
        for ex in batch:
            full, tail = split_chat(self._template, ex["prompt"], ex["target"])
            texts.append(full)
            n_answer.append(len(self.processor.tokenizer(tail, add_special_tokens=False).input_ids))
        images = [to_rgb(self.load_image(ex["image"])) for ex in batch]
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        enc["labels"] = torch.tensor(mask_labels(enc["input_ids"].tolist(), lengths, n_answer))
        return enc


def train(
    cfg: dict,
    base: dict,
    examples: list[dict],
    load_image: Callable[[str], np.ndarray],
    out_dir: Path,
    max_steps: int | None = None,
) -> Path:
    """Fine-tune `base` (a backend: hf model config) and save the adapter to out_dir.

    Writes adapter/, train_log.jsonl, train_config.yaml and model.yaml (an evaluation
    config for scripts/evaluate.py). Returns the path of model.yaml.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    device = get_device()
    opt_cfg, lora_cfg = cfg["optim"], cfg["lora"]

    processor, model = load_base(
        base["model_id"], base.get("revision"), base.get("dtype", "bfloat16")
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        ),
    )
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M on {device}")

    loader = DataLoader(
        examples,
        batch_size=opt_cfg["batch_size"],
        shuffle=True,
        collate_fn=Collator(processor, load_image),
        generator=torch.Generator().manual_seed(cfg["seed"]),
        num_workers=cfg.get("num_workers", 2),
    )
    accum = opt_cfg["grad_accum"]
    steps = math.ceil(len(loader) * opt_cfg["epochs"] / accum)
    if max_steps:
        steps = min(steps, max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(opt_cfg["warmup_ratio"] * steps), steps
    )

    (out_dir / "train_config.yaml").write_text(
        yaml.safe_dump({"train": cfg, "base": base, "examples": len(examples), "steps": steps})
    )
    log = (out_dir / "train_log.jsonl").open("w")
    model.train()
    step, running, start = 0, [], time.perf_counter()
    while step < steps:
        for k, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            (loss / accum).backward()
            running.append(loss.item())
            if (k + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, opt_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 10) == 0 or step == steps:
                row = {
                    "step": step,
                    "loss": round(float(np.mean(running)), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "minutes": round((time.perf_counter() - start) / 60, 2),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(row)
                running = []
            if step == steps:
                break
    log.close()

    model.save_pretrained(out_dir / "adapter")
    model_cfg = {
        **{k: v for k, v in base.items() if k != "name"},
        "name": cfg["name"],
        "backend": "hf",
        "adapter": str(out_dir / "adapter"),
    }
    path = out_dir / "model.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False))
    return path
