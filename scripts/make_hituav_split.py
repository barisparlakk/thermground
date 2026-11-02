"""Make a leakage-aware train/val/test split for HIT-UAV.

The provided split is random per frame, so near-identical neighbouring video frames
end up in both train and test. This script assigns whole groups of frames instead:

1. Segments: sort images by frame id and start a new segment whenever the flight
   condition (day/night, altitude, camera angle) changes or the frame gap exceeds --gap.
2. Merge segments that share a near-duplicate image pair (thumbnail correlation above
   --sim-threshold), so the same spot filmed in an angle sweep stays together.
3. Assign groups, largest first, to the split that keeps images, Person/Car/Bicycle
   boxes, night images and images per altitude closest to the target ratios. Groups
   are often class-homogeneous (a bike park, a crowd), so balancing on images alone
   leaves val or test with a skewed class mix.

Writes a YAML file with the parameters, summary statistics and the name lists.

Usage:
    uv run python scripts/make_hituav_split.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from thermground.data.hituav import CLASSES, SPLITS, HitUav, Meta

THUMB = (64, 51)  # (w, h), keeps the 640x512 aspect ratio
NEAR_FRAMES = 5


def load_all(data_dir: Path) -> tuple[list[str], dict[str, Meta], dict[str, Path], dict, dict]:
    """All images sorted by frame id, with metadata, paths, provided split and class counts."""
    meta, paths, provided, classes = {}, {}, {}, {}
    for split in SPLITS:
        ds = HitUav(data_dir, split=split)
        for i in range(len(ds)):
            a = ds.annotation(i)
            meta[a.name], paths[a.name], provided[a.name] = a.meta, ds.image_path(i), split
            classes[a.name] = Counter(CLASSES[c] for c in a.labels)
    names = sorted(meta, key=lambda n: meta[n].frame)
    return names, meta, paths, provided, classes


def condition(m: Meta) -> tuple[str, int, int]:
    return m.scenario, m.altitude_m, m.angle_deg


def segment_ids(names: list[str], meta: dict[str, Meta], gap: int) -> np.ndarray:
    ids = np.zeros(len(names), dtype=int)
    for i in range(1, len(names)):
        prev, cur = meta[names[i - 1]], meta[names[i]]
        new = condition(prev) != condition(cur) or cur.frame - prev.frame > gap
        ids[i] = ids[i - 1] + new
    return ids


def similarity(names: list[str], paths: dict[str, Path]) -> np.ndarray:
    """Pearson correlation between downsampled images, (N, N) with -1 on the diagonal."""
    x = np.stack(
        [
            np.asarray(
                Image.open(paths[n]).convert("L").resize(THUMB, Image.BILINEAR), np.float32
            ).ravel()
            for n in names
        ]
    )
    x -= x.mean(axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)
    return sim


def merge_near_duplicates(seg: np.ndarray, sim: np.ndarray, threshold: float) -> np.ndarray:
    """Union segments that share an image pair above threshold; returns group ids."""
    parent = list(range(seg.max() + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(sim > threshold, k=1))):
        if seg[i] != seg[j]:
            parent[find(seg[i])] = find(seg[j])
    roots = np.array([find(s) for s in seg])
    return np.unique(roots, return_inverse=True)[1]


def balance_features(names: list[str], meta: dict[str, Meta], classes: dict) -> np.ndarray:
    """Per-image quantities to balance: image, Person/Car/Bicycle boxes, night, altitude one-hot."""
    altitudes = sorted({m.altitude_m for m in meta.values()})
    rows = []
    for n in names:
        m = meta[n]
        counts = [classes[n][c] for c in ("Person", "Car", "Bicycle")]
        rows.append([1, *counts, m.scenario == "night", *(m.altitude_m == a for a in altitudes)])
    return np.array(rows, dtype=float)


def assign(
    names: list[str],
    groups: np.ndarray,
    features: np.ndarray,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Greedy: largest group first, to the split minimising squared relative deviation."""
    splits = list(ratios)
    group_feats = np.zeros((groups.max() + 1, features.shape[1]))
    np.add.at(group_feats, groups, features)
    target = np.array([ratios[s] for s in splits])[:, None] * features.sum(axis=0)
    scale = target.clip(min=1)

    rng = np.random.default_rng(seed)  # only breaks ties between equal-sized groups
    order = sorted(rng.permutation(len(group_feats)), key=lambda g: -group_feats[g, 0])
    filled = np.zeros_like(target)
    group_split = {}
    for g in order:
        costs = []
        for k in range(len(splits)):
            trial = filled.copy()
            trial[k] += group_feats[g]
            costs.append((((trial - target) / scale) ** 2).sum())
        k = int(np.argmin(costs))
        filled[k] += group_feats[g]
        group_split[g] = splits[k]
    return {n: group_split[g] for n, g in zip(names, groups)}


def leakage(
    names: list[str],
    meta: dict[str, Meta],
    sim: np.ndarray,
    split_of: dict[str, str],
    threshold: float,
) -> dict[str, float]:
    """Share of test images with a near-duplicate or a nearby frame in train/val."""
    split = np.array([split_of[n] for n in names])
    test, rest = split == "test", split != "test"
    near_dup = (sim[np.ix_(test, rest)] > threshold).any(axis=1).mean()

    by_cond = defaultdict(list)
    for n in names:
        by_cond[condition(meta[n])].append((meta[n].frame, split_of[n]))
    close = [
        any(
            s2 != "test" and abs(f2 - meta[n].frame) <= NEAR_FRAMES
            for f2, s2 in by_cond[condition(meta[n])]
        )
        for n in names
        if split_of[n] == "test"
    ]
    return {
        f"test_with_near_duplicate_in_trainval(>{threshold})": round(float(near_dup), 4),
        f"test_with_trainval_frame_within_{NEAR_FRAMES}": round(float(np.mean(close)), 4),
    }


def summarize(names: list[str], meta: dict[str, Meta], classes: dict, split_of: dict) -> dict:
    out = {}
    for split in SPLITS:
        ns = [n for n in names if split_of[n] == split]
        objects = sum((classes[n] for n in ns), Counter())
        out[split] = {
            "images": len(ns),
            "share": round(len(ns) / len(names), 4),
            "night_share": round(sum(meta[n].scenario == "night" for n in ns) / len(ns), 4),
            "altitudes": sorted({meta[n].altitude_m for n in ns}),
            "objects": {c: objects[c] for c in CLASSES},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware HIT-UAV split")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hituav"))
    parser.add_argument("--out", type=Path, default=Path("configs/splits/hituav_grouped_v1.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap", type=int, default=50, help="frame gap that starts a new segment")
    parser.add_argument("--sim-threshold", type=float, default=0.95)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    args = parser.parse_args()
    ratios = dict(zip(SPLITS, args.ratios))

    names, meta, paths, provided, classes = load_all(args.data_dir)
    seg = segment_ids(names, meta, args.gap)
    sim = similarity(names, paths)
    groups = merge_near_duplicates(seg, sim, args.sim_threshold)
    features = balance_features(names, meta, classes)
    split_of = assign(names, groups, features, ratios, args.seed)

    result = {
        "method": {
            "script": "scripts/make_hituav_split.py",
            "seed": args.seed,
            "gap": args.gap,
            "sim_threshold": args.sim_threshold,
            "thumbnail": list(THUMB),
            "ratios": ratios,
            "segments": int(seg.max() + 1),
            "groups_after_merge": int(groups.max() + 1),
        },
        "leakage": {
            "provided_split": leakage(names, meta, sim, provided, args.sim_threshold),
            "this_split": leakage(names, meta, sim, split_of, args.sim_threshold),
        },
        "summary": summarize(names, meta, classes, split_of),
    }
    for split in SPLITS:
        result[split] = sorted(
            (n for n in names if split_of[n] == split), key=lambda n: meta[n].frame
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by scripts/make_hituav_split.py; do not edit by hand.\n"
    args.out.write_text(header + yaml.safe_dump(result, sort_keys=False))

    print(yaml.safe_dump({k: result[k] for k in ("method", "leakage", "summary")}, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
