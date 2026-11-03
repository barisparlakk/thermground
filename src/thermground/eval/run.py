"""Run a model over QA items under three image conditions and score it.

- real      the item's own image
- blank     a uniform grey image (shows the model's prior; the correct answer is "absent")
- shuffled  another image from the same set (does the answer follow what is shown?)

Predictions are appended to predictions_{condition}.jsonl one line at a time, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from thermground.data.hituav import Sample
from thermground.data.qa import FAMILIES, HEIGHT, WIDTH, answer_for
from thermground.eval.metrics import score
from thermground.eval.parse import parse_answer
from thermground.models.base import Query, VQAModel

CONDITIONS = ("real", "blank", "shuffled")


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def select_subset(items: list[dict], sizes: dict[str, int] | None, seed: int) -> list[dict]:
    """Fixed subset stratified by answer; yes/no pairs stay together. None keeps all."""
    if sizes is None:
        return items
    chosen = set()
    for family, n in sizes.items():
        rng = np.random.default_rng([seed, FAMILIES.index(family)])
        fam = [i for i in items if i["family"] == family]
        if fam and fam[0]["pair_id"]:
            pairs = sorted({i["pair_id"] for i in fam})
            keep = set(rng.choice(pairs, size=min(n // 2, len(pairs)), replace=False).tolist())
            chosen |= {i["id"] for i in fam if i["pair_id"] in keep}
            continue
        by_answer = defaultdict(list)
        for i in fam:
            by_answer[i["answer"]].append(i["id"])
        for answer in sorted(by_answer):
            ids = by_answer[answer]
            k = min(round(n * len(ids) / len(fam)), len(ids))
            chosen |= set(rng.choice(ids, size=k, replace=False).tolist())
    return [i for i in items if i["id"] in chosen]


def shuffle_map(names: list[str], seed: int) -> dict[str, str]:
    """Map every image to a different image of the same set (a seeded cycle)."""
    order = [names[k] for k in np.random.default_rng(seed).permutation(len(names))]
    return {a: b for a, b in zip(order, order[1:] + order[:1])}


def build_prompt(item: dict, instructions: dict[str, str], coords: str) -> str:
    options = ", ".join(item["options"] or [])
    return (
        f"{item['question']} {instructions[item['family']].format(options=options, coords=coords)}"
    )


def _read_predictions(path: Path) -> list[dict]:
    """Valid lines of a predictions file; a line cut off by an interrupted run is dropped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def predict(
    model: VQAModel,
    items: list[dict],
    condition: str,
    load_image: Callable[[str], np.ndarray],
    shown_of: dict[str, str],
    prompt_of: Callable[[dict], str],
    path: Path,
    blank_value: int = 128,
) -> list[dict]:
    rows = _read_predictions(path)
    done = {r["id"] for r in rows}
    with path.open("a") as f:
        for item in items:
            if item["id"] in done:
                continue
            if condition == "real":
                shown = item["image"]
            elif condition == "blank":
                shown = None
            else:
                shown = shown_of[item["image"]]
            if shown is None:
                image = np.full((HEIGHT, WIDTH), blank_value, dtype=np.uint8)
            else:
                image = load_image(shown)
            prompt = prompt_of(item)
            query = Query(
                image, prompt, item["family"], item["options"], item["target_class"], item["region"]
            )
            start = time.perf_counter()
            output = model.answer(query)
            row = {
                "id": item["id"],
                "condition": condition,
                "shown_image": shown,
                "prompt": prompt,
                "output": output,
                "seconds": round(time.perf_counter() - start, 3),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    return rows


def to_records(
    items: list[dict],
    rows: list[dict],
    samples: dict[str, Sample],
    buckets: list[str],
    coord_format: str,
    held_out: set[str],
) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    records = []
    for item in items:
        row = by_id.get(item["id"])
        if row is None:
            continue
        truth = {"answer": item["answer"], "evidence": item["evidence"]}
        if row["condition"] == "real":
            shown = truth
        else:
            sample = samples[row["shown_image"]] if row["shown_image"] else None
            answer, evidence = answer_for(
                sample, item["family"], item["target_class"], item["region"], buckets
            )
            shown = {"answer": answer, "evidence": evidence}
        records.append(
            {
                "id": item["id"],
                "family": item["family"],
                "condition": row["condition"],
                "pair_id": item["pair_id"],
                "meta": item["meta"],
                "held_out": item["template_id"] in held_out,
                "pred": parse_answer(row["output"], item["family"], buckets, coord_format),
                "truth": truth,
                "shown": shown,
            }
        )
    return records


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def summary_table(metrics: dict) -> str:
    cols = [
        ("real", "real"),
        ("blank", "blank"),
        ("grounding_gap", "gap"),
        ("pair_accuracy_real", "pair acc"),
        ("absent_on_blank", "absent on blank"),
        ("follows_shuffled", "follows shuffled"),
        ("invalid_real", "invalid"),
    ]
    lines = [
        "| family | " + " | ".join(c[1] for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for family, h in metrics["headline"].items():
        lines.append(f"| {family} | " + " | ".join(_pct(h.get(k)) for k, _ in cols) + " |")
    note = "Scores in %. point uses F1; the other families use balanced accuracy."
    return "\n".join(lines) + f"\n\n{note}\n"


def evaluate(
    model: VQAModel,
    items: list[dict],
    samples: dict[str, Sample],
    load_image: Callable[[str], np.ndarray],
    out_dir: Path,
    *,
    instructions: dict[str, str],
    coords_text: dict[str, str],
    buckets: list[str],
    held_out: set[str],
    conditions: tuple[str, ...] = CONDITIONS,
    seed: int = 0,
    enlarge: float = 0.25,
    blank_value: int = 128,
    info: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shown_of = shuffle_map(sorted({i["image"] for i in items}), seed)
    coords = coords_text[model.coord_format]

    def prompt_of(item: dict) -> str:
        return build_prompt(item, instructions, coords)

    records, seconds = [], {}
    for condition in conditions:
        path = out_dir / f"predictions_{condition}.jsonl"
        rows = predict(model, items, condition, load_image, shown_of, prompt_of, path, blank_value)
        seconds[condition] = round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 4)
        records += to_records(items, rows, samples, buckets, model.coord_format, held_out)

    ids = "\n".join(sorted(i["id"] for i in items))
    metrics = {
        "model": model.name,
        "coord_format": model.coord_format,
        **(info or {}),
        "items": len(items),
        "items_per_family": {f: sum(i["family"] == f for i in items) for f in FAMILIES},
        "item_ids_sha1": hashlib.sha1(ids.encode()).hexdigest(),
        "seconds_per_item": seconds,
        **score(records, buckets, enlarge),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "summary.md").write_text(
        f"# {model.name} on {metrics.get('set', 'QA')}\n\n" + summary_table(metrics)
    )
    return metrics
