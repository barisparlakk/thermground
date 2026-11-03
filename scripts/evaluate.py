"""Evaluate a model on the HIT-UAV QA set under the real / blank / shuffled conditions.

Writes predictions_{condition}.jsonl (gitignored), metrics.json and summary.md to --out.
Interrupted runs resume where they stopped.

Usage:
    uv run python scripts/evaluate.py --model oracle --set val-mini
    uv run python scripts/evaluate.py --model always_yes --set test-mini \\
        --out results/week2/eval/always_yes/test-mini
    uv run python scripts/evaluate.py --model random --set val-mini --limit 40   # quick check
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import yaml

from thermground.data.hituav import HitUav
from thermground.eval.run import CONDITIONS, evaluate, load_items, select_subset
from thermground.models.reference import REFERENCE_MODELS, reference_model


def spread(items: list[dict], limit: int) -> list[dict]:
    """Every k-th item, so a short run still covers every family."""
    step = max(1, len(items) // limit)
    return items[::step][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on HIT-UAV QA")
    parser.add_argument("--model", required=True, help=f"one of {', '.join(REFERENCE_MODELS)}")
    parser.add_argument("--set", default="val-mini", help="val-mini, test-mini or test")
    parser.add_argument("--config", type=Path, default=Path("configs/eval/hituav_eval_v1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/hituav"))
    parser.add_argument("--out", type=Path, help="default: results/eval/{model}/{set}")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--limit", type=int, help="evaluate only this many items")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    qa = yaml.safe_load(Path(cfg["qa_config"]).read_text())
    set_cfg = cfg["sets"][args.set]
    split = set_cfg["split"]

    qa_file = Path(qa["out_dir"]) / f"{split}.jsonl"
    if not qa_file.exists():
        raise SystemExit(f"{qa_file} not found; run scripts/make_hituav_qa.py first")
    items = select_subset(load_items(qa_file), set_cfg["sizes"], cfg["seed"])
    if args.limit:
        items = spread(items, args.limit)

    ds = HitUav(args.data_dir, split=split, split_file=qa["split_file"])
    index = {name: i for i, name in enumerate(ds.names)}
    samples = {name: ds.annotation(i) for name, i in index.items()}

    @functools.cache
    def load_image(name: str):
        return ds[index[name]].image

    buckets = qa["count"]["buckets"]
    if args.model not in REFERENCE_MODELS:
        raise SystemExit(f"unknown model {args.model!r}; available: {', '.join(REFERENCE_MODELS)}")
    model = reference_model(args.model, samples, load_image, buckets)
    held_out = {t["id"] for ts in qa["templates"].values() for t in ts if t.get("held_out")}

    out = args.out or Path("results/eval") / args.model / args.set
    evaluate(
        model,
        items,
        samples,
        load_image,
        out,
        instructions=cfg["instructions"],
        coords_text=cfg["coords"],
        buckets=buckets,
        held_out=held_out,
        conditions=tuple(args.conditions or cfg["conditions"]),
        seed=cfg["seed"],
        enlarge=cfg["point_enlarge"],
        blank_value=cfg["blank_value"],
        info={
            "set": args.set,
            "eval_config": str(args.config),
            "qa_config": cfg["qa_config"],
            "limit": args.limit,
        },
    )
    print((out / "summary.md").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
