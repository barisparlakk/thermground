"""Generate the HIT-UAV QA set from a QA config.

Writes {out_dir}/{train,val,test}.jsonl (gitignored, under data/) and statistics to
--stats (default results/week2/qa/):
    summary.json   items per family and split, answer distributions, prior baselines
    answers.png    test answer distribution per family

Prior baselines answer without looking at the image, using only answers seen in train:
    majority       the most common train answer of the family
    question       the most common train answer for the same template, class and region
A model has to beat these to show it uses the image.

Usage:
    uv sync --extra viz
    uv run python scripts/make_hituav_qa.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from thermground.data.hituav import SPLITS, HitUav
from thermground.data.qa import FAMILIES, QAItem, generate

SCORED = ("presence", "count", "region")  # point is scored with boxes, not answer accuracy


def question_key(item: QAItem) -> tuple:
    return item.family, item.template_id, item.target_class, item.region


def prior_baselines(train: list[QAItem], test: list[QAItem]) -> dict:
    by_family = defaultdict(Counter)
    by_question = defaultdict(Counter)
    for item in train:
        by_family[item.family][item.answer] += 1
        by_question[question_key(item)][item.answer] += 1

    out = {}
    for family in SCORED:
        items = [i for i in test if i.family == family]
        majority = by_family[family].most_common(1)[0][0]
        correct_majority = correct_question = 0
        for item in items:
            seen = by_question.get(question_key(item))
            guess = seen.most_common(1)[0][0] if seen else majority
            correct_question += guess == item.answer
            correct_majority += majority == item.answer
        out[family] = {
            "chance": round(1 / len(items[0].options), 4),
            "majority": round(correct_majority / len(items), 4),
            "question": round(correct_question / len(items), 4),
        }
    return out


def summarize(items: dict[str, list[QAItem]]) -> dict:
    out = {}
    for split, split_items in items.items():
        out[split] = {"items": len(split_items), "images": len({i.image for i in split_items})}
        for family in FAMILIES:
            fam = [i for i in split_items if i.family == family]
            out[split][family] = {
                "items": len(fam),
                "answers": dict(sorted(Counter(i.answer for i in fam).items())),
            }
            if family == "presence":
                out[split][family]["classes"] = dict(Counter(i.target_class for i in fam))
            if family == "point":
                boxes = [len(i.evidence) for i in fam if i.answer == "boxes"]
                out[split][family]["mean_boxes"] = round(sum(boxes) / len(boxes), 2)
        templates = Counter(i.template_id for i in split_items)
        out[split]["templates"] = dict(sorted(templates.items()))
    return out


def plot_answers(test: list[QAItem], path: Path) -> None:
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(4 * len(FAMILIES), 3.2))
    for ax, family in zip(axes, FAMILIES):
        counts = Counter(i.answer for i in test if i.family == family)
        labels = sorted(counts, key=lambda a: (len(a) > 3, a))
        ax.bar(labels, [counts[a] for a in labels], color="#4c72b0")
        ax.set_title(f"{family} (n={sum(counts.values())})")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("test items")
    fig.suptitle("HIT-UAV QA v1: test answer distribution per family")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HIT-UAV QA set")
    parser.add_argument("--config", type=Path, default=Path("configs/qa/hituav_qa_v1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/hituav"))
    parser.add_argument("--stats", type=Path, default=Path("results/week2/qa"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {}
    for split in SPLITS:
        ds = HitUav(args.data_dir, split=split, split_file=cfg["split_file"])
        samples = [ds.annotation(i) for i in range(len(ds))]
        items[split] = generate(samples, split, cfg, cfg["seed"])
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for item in items[split]:
                f.write(json.dumps(item.to_dict()) + "\n")

    args.stats.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(args.config),
        "seed": cfg["seed"],
        "split_file": cfg["split_file"],
        "prior_baselines_test": prior_baselines(items["train"], items["test"]),
        "splits": summarize(items),
    }
    (args.stats / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_answers(items["test"], args.stats / "answers.png")

    for split in SPLITS:
        print(split, summary["splits"][split]["items"], "items")
    print("prior baselines on test:", json.dumps(summary["prior_baselines_test"], indent=1))
    print(f"wrote {out_dir} and {args.stats}")


if __name__ == "__main__":
    main()
