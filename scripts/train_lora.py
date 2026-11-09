"""LoRA fine-tuning on the HIT-UAV QA train split (CUDA; needs the `train` extra).

Prompts come from the evaluation config, so training and evaluation ask the same way.
Writes the adapter, train_log.jsonl, train_config.yaml and model.yaml to --out; evaluate
the result with:
    python scripts/evaluate.py --model <out>/model.yaml --set val-mini

Usage:
    python scripts/train_lora.py --config configs/train/qwen3_vl_4b_lora_v1.yaml --out runs/x
    python scripts/train_lora.py --config configs/train/qwen3_vl_2b_lora_v1.yaml \\
        --out runs/smoke --max-steps 5      # pipeline check
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import yaml

from thermground.data.hituav import HitUav
from thermground.data.sft import make_examples, mix_items
from thermground.eval.run import load_items
from thermground.models.lora import train


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on HIT-UAV QA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hituav"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="stop early (pipeline checks)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load(Path(cfg["base"]).read_text())
    if base["backend"] != "hf":
        raise SystemExit(f"{cfg['base']} is not a backend: hf model config")
    ev = yaml.safe_load(Path(cfg["eval_config"]).read_text())
    qa = yaml.safe_load(Path(ev["qa_config"]).read_text())

    split = cfg["data"]["split"]
    qa_file = Path(qa["out_dir"]) / f"{split}.jsonl"
    if not qa_file.exists():
        raise SystemExit(f"{qa_file} not found; run scripts/make_hituav_qa.py first")
    items = mix_items(load_items(qa_file), cfg["data"]["mix"], cfg["seed"])
    examples = make_examples(items, ev["instructions"], ev["coords"], base["coord_format"])
    print(f"{len(examples)} training examples from {qa_file}")

    ds = HitUav(args.data_dir, split=split, split_file=qa["split_file"])
    index = {name: i for i, name in enumerate(ds.names)}

    @functools.cache
    def load_image(name: str):
        return ds[index[name]].image

    path = train(cfg, base, examples, load_image, args.out, args.max_steps)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
