"""Template QA generation from HIT-UAV annotations.

Every answer is computed from the boxes, and answers are balanced per family so a model
that ignores the image cannot beat chance:

- presence  "Is there a {object} ...?"           yes/no, paired yes/no per (class, day/night, altitude)
- count     "How many people ...?"                count bucket, equal items per bucket
- region    "Is there a person in the {region}?"  yes/no, paired yes/no per (region, day/night)
- point     "Mark every person."                  boxes in `evidence`, or "none"

Images with a DontCare box are skipped, since that region may hide the answer. Regions
are a 3x3 grid in image coordinates; a box belongs to the cell holding its centre.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from thermground.data.hituav import CLASSES, DONTCARE_ID, Sample

REGIONS = (
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
)  # fmt: skip
FAMILIES = ("presence", "count", "region", "point")
PERSON = CLASSES.index("Person")
WIDTH, HEIGHT = 640, 512  # every HIT-UAV image (docs/dataset_notes.md)


@dataclass
class QAItem:
    id: str
    image: str
    split: str
    family: str
    question: str
    answer: str
    options: list[str] | None  # valid answers, None for point
    target_class: str
    region: str | None
    evidence: list[list[float]]  # x1 y1 x2 y2 pixel boxes that support the answer
    template_id: str
    pair_id: str | None  # yes/no items built as a matched pair share an id
    meta: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Draft:
    """An answer derived from one image, before a question template is chosen."""

    sample: Sample
    family: str
    answer: str
    target_class: str = "Person"
    region: str | None = None
    evidence: list[list[float]] = dataclasses.field(default_factory=list)
    pair: int | None = None


def region_of(box: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    col = min(int(3 * cx / width), 2)
    row = min(int(3 * cy / height), 2)
    return REGIONS[3 * row + col]


def parse_bucket(label: str) -> tuple[int, int | None]:
    """'3' -> (3, 3), '4-6' -> (4, 6), '7+' -> (7, None)."""
    if label.endswith("+"):
        return int(label[:-1]), None
    lo, _, hi = label.partition("-")
    return int(lo), int(hi or lo)


def count_bucket(n: int, labels: list[str]) -> str:
    for label in labels:
        lo, hi = parse_bucket(label)
        if n >= lo and (hi is None or n <= hi):
            return label
    raise ValueError(f"no bucket for count {n}")


def _boxes(sample: Sample, class_id: int) -> np.ndarray:
    return sample.boxes[sample.labels == class_id]


def _as_lists(boxes: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 1) for v in box] for box in boxes]


def _condition(d: Draft) -> tuple[str, int]:
    return d.sample.meta.scenario, d.sample.meta.altitude_m


def _shuffled(items: list, rng: np.random.Generator) -> list:
    return [items[i] for i in rng.permutation(len(items))]


def _pairs(drafts: list[Draft], key: Callable[[Draft], tuple], rng: np.random.Generator) -> list:
    """Match yes and no drafts with the same key; unmatched drafts are dropped."""
    groups = defaultdict(lambda: ([], []))
    for d in drafts:
        groups[key(d)][d.answer == "no"].append(d)
    pairs = []
    for k in sorted(groups):
        yes, no = groups[k]
        pairs += zip(_shuffled(yes, rng), _shuffled(no, rng))
    return _shuffled(pairs, rng)


def _flatten(pairs: list, start: int = 0) -> list[Draft]:
    out = []
    for i, (yes, no) in enumerate(pairs, start):
        yes.pair = no.pair = i
        out += [yes, no]
    return out


def _balance(drafts: list[Draft], rng: np.random.Generator) -> list[Draft]:
    """Downsample so every answer has as many drafts as the rarest one."""
    groups = defaultdict(list)
    for d in drafts:
        groups[d.answer].append(d)
    n = min(len(g) for g in groups.values())
    return [d for answer in sorted(groups) for d in _shuffled(groups[answer], rng)[:n]]


def answer_for(
    sample: Sample | None,
    family: str,
    target_class: str = "Person",
    region: str | None = None,
    buckets: list[str] | None = None,
) -> tuple[str, list[list[float]]]:
    """Answer and evidence for a question about `sample`; None stands for an empty image."""
    if sample is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = _boxes(sample, CLASSES.index(target_class))
    if family == "region":
        boxes = boxes[np.array([region_of(b) == region for b in boxes], dtype=bool)]
    if family == "count":
        answer = count_bucket(len(boxes), buckets)
    elif family == "point":
        answer = "boxes" if len(boxes) else "none"
    else:
        answer = "yes" if len(boxes) else "no"
    return answer, _as_lists(boxes)


def presence(
    samples: list[Sample], shares: dict[str, float], rng: np.random.Generator
) -> list[Draft]:
    pairs = {}
    for name in shares:
        class_id = CLASSES.index(name)
        drafts = []
        for s in samples:
            boxes = _boxes(s, class_id)
            answer = "yes" if len(boxes) else "no"
            drafts.append(
                Draft(s, "presence", answer, target_class=name, evidence=_as_lists(boxes))
            )
        pairs[name] = _pairs(drafts, _condition, rng)
    # keep the class mix at the configured shares, limited by the scarcest class
    unit = min(len(pairs[name]) / share for name, share in shares.items())
    out, start = [], 0
    for name, share in shares.items():
        chosen = pairs[name][: round(unit * share)]
        out += _flatten(chosen, start)
        start += len(chosen)
    return out


def count(samples: list[Sample], buckets: list[str], rng: np.random.Generator) -> list[Draft]:
    drafts = []
    for s in samples:
        boxes = _boxes(s, PERSON)
        drafts.append(
            Draft(s, "count", count_bucket(len(boxes), buckets), evidence=_as_lists(boxes))
        )
    return _balance(drafts, rng)


def region(samples: list[Sample], rng: np.random.Generator) -> list[Draft]:
    drafts = []
    for s in samples:
        boxes = _boxes(s, PERSON)
        where = np.array([region_of(b) for b in boxes], dtype=object)
        for r in REGIONS:
            inside = boxes[where == r]
            drafts.append(
                Draft(
                    s,
                    "region",
                    "yes" if len(inside) else "no",
                    region=r,
                    evidence=_as_lists(inside),
                )
            )
    return _flatten(_pairs(drafts, lambda d: (d.region, d.sample.meta.scenario), rng))


def point(
    samples: list[Sample], max_boxes: int, negative_share: float, rng: np.random.Generator
) -> list[Draft]:
    pos, neg = [], []
    for s in samples:
        boxes = _boxes(s, PERSON)
        if len(boxes) == 0:
            neg.append(Draft(s, "point", "none"))
        elif len(boxes) <= max_boxes:
            pos.append(Draft(s, "point", "boxes", evidence=_as_lists(boxes)))
    n_neg = round(len(pos) * negative_share / (1 - negative_share))
    return pos + _shuffled(neg, rng)[:n_neg]


def generate(samples: list[Sample], split: str, cfg: dict, seed: int) -> list[QAItem]:
    """All QA items for one split. cfg is the parsed QA config (see configs/qa/)."""
    samples = [s for s in samples if DONTCARE_ID not in s.labels]
    split_index = ("train", "val", "test").index(split)
    rngs = {f: np.random.default_rng([seed, split_index, i]) for i, f in enumerate(FAMILIES)}
    drafts = {
        "presence": presence(samples, cfg["presence"]["classes"], rngs["presence"]),
        "count": count(samples, cfg["count"]["buckets"], rngs["count"]),
        "region": region(samples, rngs["region"]),
        "point": point(
            samples, cfg["point"]["max_boxes"], cfg["point"]["negative_share"], rngs["point"]
        ),
    }
    items = []
    for family in FAMILIES:
        rng = np.random.default_rng([seed, split_index, len(FAMILIES) + FAMILIES.index(family)])
        templates = [
            t for t in cfg["templates"][family] if split == "test" or not t.get("held_out")
        ]
        options = {
            "presence": ["yes", "no"],
            "count": list(cfg["count"]["buckets"]),
            "region": ["yes", "no"],
            "point": None,
        }[family]
        pair_template = {}
        for d in drafts[family]:
            if d.pair in pair_template:
                template = pair_template[d.pair]  # both halves of a pair get the same wording
            else:
                template = templates[rng.integers(len(templates))]
                if d.pair is not None:
                    pair_template[d.pair] = template
            names = cfg["objects"][d.target_class]
            fields = {"object": names[0], "objects": names[1]}
            if d.region:
                fields["region"] = cfg["region_phrases"][d.region]
            items.append(
                QAItem(
                    id=f"{split}-{family}-{len(items):05d}",
                    image=d.sample.name,
                    split=split,
                    family=family,
                    question=template["text"].format(**fields),
                    answer=d.answer,
                    options=options,
                    target_class=d.target_class,
                    region=d.region,
                    evidence=d.evidence,
                    template_id=template["id"],
                    pair_id=None if d.pair is None else f"{split}-{family}-pair{d.pair:05d}",
                    meta=dataclasses.asdict(d.sample.meta),
                )
            )
    return items
