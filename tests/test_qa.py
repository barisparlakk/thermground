from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from thermground.data.hituav import CLASSES, DONTCARE_ID, SPLITS, HitUav, Meta, Sample
from thermground.data.qa import REGIONS, count_bucket, generate, parse_bucket, region_of

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "qa" / "hituav_qa_v1.yaml").read_text())
DATA_DIR = ROOT / "data" / "hituav"
needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="HIT-UAV not downloaded")


def test_region_of():
    assert region_of(np.array([0, 0, 10, 10])) == "top-left"
    assert region_of(np.array([310, 246, 330, 266])) == "center"
    assert region_of(np.array([630, 502, 640, 512])) == "bottom-right"


def test_count_buckets():
    assert parse_bucket("4-6") == (4, 6) and parse_bucket("7+") == (7, None)
    buckets = CFG["count"]["buckets"]
    assert [count_bucket(n, buckets) for n in (0, 3, 4, 6, 7, 40)] == [
        "0",
        "3",
        "4-6",
        "4-6",
        "7+",
        "7+",
    ]


def synthetic_samples(n: int = 120, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        k = int(rng.integers(0, 9))
        xy = rng.uniform([0, 0], [620, 490], size=(k, 2))
        boxes = np.hstack([xy, xy + 15]).astype(np.float32)
        labels = rng.choice([0, 0, 1, 2], size=k)  # mostly Person
        meta = Meta(("day", "night")[i % 2], 60 + 10 * (i % 3), 50, i)
        flags = np.zeros(k, dtype=bool)
        samples.append(
            Sample(
                f"{i % 2}_{meta.altitude_m}_50_0_{i:05d}", None, boxes, labels, flags, flags, meta
            )
        )
    return samples


def check_answers(items, samples_by_name):
    """Recompute every answer and its evidence from the annotations."""
    for item in items:
        s = samples_by_name[item.image]
        assert DONTCARE_ID not in s.labels
        boxes = s.boxes[s.labels == CLASSES.index(item.target_class)]
        as_lists = [[round(float(v), 1) for v in b] for b in boxes]
        if item.family == "presence":
            assert item.answer == ("yes" if len(boxes) else "no")
            assert item.evidence == as_lists
        elif item.family == "count":
            assert item.answer == count_bucket(len(boxes), CFG["count"]["buckets"])
            assert item.evidence == as_lists
        elif item.family == "region":
            inside = [b for b, box in zip(as_lists, boxes) if region_of(box) == item.region]
            assert item.answer == ("yes" if inside else "no")
            assert item.evidence == inside
        elif item.family == "point":
            assert item.answer == ("boxes" if len(boxes) else "none")
            assert item.evidence == as_lists


def check_balance(items):
    for family in ("presence", "count", "region"):
        answers = Counter(i.answer for i in items if i.family == family)
        assert len(set(answers.values())) == 1, (family, answers)
    pairs = defaultdict(list)
    for i in items:
        if i.pair_id:
            pairs[i.pair_id].append(i)
    for pair in pairs.values():
        a, b = pair
        assert {a.answer, b.answer} == {"yes", "no"}
        assert (a.template_id, a.target_class, a.region) == (
            b.template_id,
            b.target_class,
            b.region,
        )


def test_generate_synthetic():
    samples = synthetic_samples()
    by_name = {s.name: s for s in samples}
    for split in SPLITS:
        items = generate(samples, split, CFG, seed=0)
        assert items
        check_answers(items, by_name)
        check_balance(items)
        held_out = any(i.template_id.endswith(("_05", "_06")) for i in items)
        assert held_out == (split == "test")
        assert [i.to_dict() for i in items] == [
            i.to_dict() for i in generate(samples, split, CFG, 0)
        ]


@needs_data
def test_generate_hituav():
    split_file = ROOT / CFG["split_file"]
    for split in SPLITS:
        ds = HitUav(DATA_DIR, split=split, split_file=split_file)
        samples = [ds.annotation(i) for i in range(len(ds))]
        items = generate(samples, split, CFG, CFG["seed"])
        check_answers(items, {s.name: s for s in samples})
        check_balance(items)
        assert {i.image for i in items} <= set(ds.names)
        assert {i.region for i in items if i.family == "region"} == set(REGIONS)
