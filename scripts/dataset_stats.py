"""Basic HIT-UAV statistics and figures.

Writes to --out (default results/week1/stats/):
    counts.csv                 images and objects per class, per split
    summary.json               the same numbers, Person box size percentiles, run config
    person_box_sizes.png       Person box width/height histograms (pixels)
    altitude_distribution.png  images per flight altitude, stacked by split
    samples_person.jpg         9 random images with Person boxes drawn

Usage:
    uv sync --extra viz
    uv run python scripts/dataset_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from thermground.data.hituav import CLASSES, SPLITS, HitUav

PERSON = CLASSES.index("Person")
PERCENTILES = (5, 25, 50, 75, 95)


def count_table(datasets: dict[str, HitUav]) -> list[dict]:
    rows = []
    for split, ds in datasets.items():
        labels = Counter(int(c) for i in range(len(ds)) for c in ds.annotation(i).labels)
        rows.append(
            {"split": split, "images": len(ds), **{c: labels[i] for i, c in enumerate(CLASSES)}}
        )
    total = {"split": "total"}
    for key in rows[0]:
        if key != "split":
            total[key] = sum(r[key] for r in rows)
    return rows + [total]


def person_sizes(datasets: dict[str, HitUav]) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.concatenate(
        [
            s.boxes[s.labels == PERSON]
            for ds in datasets.values()
            for s in map(ds.annotation, range(len(ds)))
        ]
    )
    return boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]


def describe(x: np.ndarray) -> dict:
    stats = {f"p{p}": float(np.percentile(x, p)) for p in PERCENTILES}
    return {"count": len(x), "min": float(x.min()), "max": float(x.max()), **stats}


def plot_person_sizes(widths: np.ndarray, heights: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.arange(0, max(widths.max(), heights.max()) + 2, 2)
    for ax, values, label in zip(axes, (widths, heights), ("width", "height")):
        ax.hist(values, bins=bins, color="#4c72b0")
        median = np.median(values)
        ax.axvline(median, color="#c44e52", linestyle="--", label=f"median {median:.0f} px")
        ax.set_xlabel(f"box {label} (px)")
        ax.legend()
    axes[0].set_ylabel("Person boxes")
    fig.suptitle(f"HIT-UAV Person box size, all splits (n={len(widths)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def altitude_counts(datasets: dict[str, HitUav]) -> dict[str, dict[int, int]]:
    return {
        split: dict(
            sorted(Counter(ds.annotation(i).meta.altitude_m for i in range(len(ds))).items())
        )
        for split, ds in datasets.items()
    }


def plot_altitudes(counts: dict[str, dict[int, int]], path: Path) -> None:
    altitudes = sorted({a for c in counts.values() for a in c})
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(altitudes))
    for split, c in counts.items():
        values = np.array([c.get(a, 0) for a in altitudes])
        ax.bar([str(a) for a in altitudes], values, bottom=bottom, label=split)
        bottom += values
    ax.set_xlabel("flight altitude (m)")
    ax.set_ylabel("images")
    ax.set_title("HIT-UAV images per altitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_samples(datasets: dict[str, HitUav], n: int, seed: int, path: Path) -> list[str]:
    candidates = [
        (split, i)
        for split, ds in datasets.items()
        for i in range(len(ds))
        if (ds.annotation(i).labels == PERSON).any()
    ]
    rng = np.random.default_rng(seed)
    picks = [candidates[j] for j in rng.choice(len(candidates), size=n, replace=False)]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, (split, i) in zip(axes.flat, picks):
        s = datasets[split][i]
        ax.imshow(s.image, cmap="gray", vmin=0, vmax=255)
        for x1, y1, x2, y2 in s.boxes[s.labels == PERSON]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", lw=1))
        m = s.meta
        ax.set_title(
            f"{s.name} ({split})\n{m.scenario}, {m.altitude_m} m, {m.angle_deg} deg", fontsize=8
        )
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 90})
    plt.close(fig)
    return [datasets[split].names[i] for split, i in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description="HIT-UAV dataset statistics")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hituav"))
    parser.add_argument("--out", type=Path, default=Path("results/week1/stats"))
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = {split: HitUav(args.data_dir, split=split) for split in SPLITS}

    table = count_table(datasets)
    with (args.out / "counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    widths, heights = person_sizes(datasets)
    plot_person_sizes(widths, heights, args.out / "person_box_sizes.png")

    altitudes = altitude_counts(datasets)
    plot_altitudes(altitudes, args.out / "altitude_distribution.png")

    samples = plot_samples(datasets, args.samples, args.seed, args.out / "samples_person.jpg")

    summary = {
        "config": {"data_dir": str(args.data_dir), "seed": args.seed, "samples": args.samples},
        "counts": table,
        "person_box_px": {"width": describe(widths), "height": describe(heights)},
        "altitude_images": altitudes,
        "sample_images": samples,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for row in table:
        print(row)
    print(f"Person width  p50 {np.median(widths):.0f} px, height p50 {np.median(heights):.0f} px")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
