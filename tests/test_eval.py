from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from thermground.data.hituav import Meta, Sample
from thermground.data.qa import generate
from thermground.eval.metrics import hits, score
from thermground.eval.parse import parse_boxes, parse_count, parse_yes_no
from thermground.eval.run import evaluate, load_items, select_subset, shuffle_map
from thermground.models.reference import reference_model

ROOT = Path(__file__).resolve().parents[1]
QA = yaml.safe_load((ROOT / "configs" / "qa" / "hituav_qa_v1.yaml").read_text())
EVAL = yaml.safe_load((ROOT / "configs" / "eval" / "hituav_eval_v1.yaml").read_text())
BUCKETS = QA["count"]["buckets"]
DATA_DIR = ROOT / "data" / "hituav"
needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="HIT-UAV not downloaded")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Yes.", "yes"),
        ("no", "no"),
        ("No, there is no person.", "no"),
        ("There is a person near the car.", "yes"),
        ("I can't see anyone.", "no"),
        ("Nobody is visible.", "no"),
        ("A person is present.", "yes"),
        ("Maybe.", None),
        ("", None),
    ],
)
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3", "3"),
        ("I count 5 people.", "4-6"),
        ("seven", "7+"),
        ("4-6", "4-6"),
        ("7+", "7+"),
        ("There are no people.", "0"),
        ("hmm", None),
    ],
)
def test_parse_count(text, expected):
    assert parse_count(text, BUCKETS) == expected


def test_parse_boxes():
    assert parse_boxes("[10, 20, 30, 40], [50, 60, 70, 80]") == [[10, 20, 30, 40], [50, 60, 70, 80]]
    assert parse_boxes("[[500, 500, 250, 250]]", "norm1000") == [[160, 128, 320, 256]]
    assert parse_boxes("(0.5, 0.5)", "norm1") == [[320, 256, 320, 256]]
    assert parse_boxes("There are no people.") == []
    assert parse_boxes("I am not sure") is None


def test_hits():
    gt = [[100, 100, 112, 119]]
    assert hits([[105, 108, 107, 110]], gt) == 1
    assert hits([[112.5, 109, 113.5, 111]], gt) == 1  # inside the 12.5% margin
    assert hits([[115, 109, 117, 111]], gt) == 0
    assert hits([[105, 108, 107, 110]] * 2, gt) == 1  # one GT box, one hit


def test_constant_answer_has_no_pair_accuracy_or_gap():
    meta = {"altitude_m": 60, "scenario": "night"}
    records = []
    for i in range(10):
        for cond in ("real", "blank"):
            for truth in ("yes", "no"):
                shown = "no" if cond == "blank" else truth
                records.append(
                    {
                        "family": "region",
                        "condition": cond,
                        "pair_id": f"p{i}",
                        "meta": meta,
                        "held_out": False,
                        "pred": "yes",
                        "truth": {"answer": truth, "evidence": []},
                        "shown": {"answer": shown, "evidence": []},
                    }
                )
    h = score(records, BUCKETS)["headline"]["region"]
    assert h["real"] == 0.5 and h["pair_accuracy_real"] == 0.0
    assert h["grounding_gap"] == 0.0 and h["absent_on_blank"] == 0.0


def synthetic(n: int = 60, seed: int = 0):
    """Samples with random Person/Car/Bicycle boxes and a distinct random image each."""
    rng = np.random.default_rng(seed)
    samples, images = {}, {}
    for i in range(n):
        k = int(rng.integers(0, 9))
        xy = rng.uniform([0, 0], [620, 490], size=(k, 2))
        boxes = np.hstack([xy, xy + 15]).astype(np.float32)
        flags = np.zeros(k, dtype=bool)
        meta = Meta(("day", "night")[i % 2], 60 + 10 * (i % 3), 50, i)
        name = f"{i % 2}_{meta.altitude_m}_50_0_{i:05d}"
        labels = rng.choice([0, 0, 1, 2], size=k)  # mostly Person
        samples[name] = Sample(name, None, boxes, labels, flags, flags, meta)
        images[name] = rng.integers(0, 256, size=(512, 640), dtype=np.uint8)
    return samples, images


def run_reference(name: str, tmp_path: Path) -> dict:
    samples, images = synthetic()
    items = [i.to_dict() for i in generate(list(samples.values()), "test", QA, seed=0)]
    model = reference_model(name, samples, images.__getitem__, BUCKETS)
    return evaluate(
        model,
        items,
        samples,
        images.__getitem__,
        tmp_path / name,
        instructions=EVAL["instructions"],
        coords_text=EVAL["coords"],
        buckets=BUCKETS,
        held_out=set(),
    )["headline"]


def test_oracle_end_to_end(tmp_path):
    for family, h in run_reference("oracle", tmp_path).items():
        assert h["real"] == 1.0, family
        assert h["absent_on_blank"] == 1.0, family
        assert h["follows_shuffled"] == 1.0, family


def test_always_yes_end_to_end(tmp_path):
    headline = run_reference("always_yes", tmp_path)
    for family in ("presence", "region"):
        h = headline[family]
        assert h["real"] == 0.5 and h["pair_accuracy_real"] == 0.0
        assert h["grounding_gap"] == 0.0 and h["follows_shuffled"] == 0.5


def test_resume_after_cut_line(tmp_path):
    first = run_reference("oracle", tmp_path)
    path = tmp_path / "oracle" / "predictions_shuffled.jsonl"
    text = path.read_text()
    path.write_text(text[: len(text) // 2])  # simulate an interrupted run
    assert run_reference("oracle", tmp_path) == first


def test_select_subset_keeps_pairs_and_balance():
    samples, _ = synthetic(200)
    items = [i.to_dict() for i in generate(list(samples.values()), "test", QA, seed=0)]
    sizes = {"presence": 20, "count": 12, "region": 30}
    subset = select_subset(items, sizes, seed=0)
    assert subset == select_subset(items, sizes, seed=0)
    for family in ("presence", "region"):
        fam = [i for i in subset if i["family"] == family]
        assert len(fam) == sizes[family]
        pairs = {}
        for i in fam:
            pairs.setdefault(i["pair_id"], []).append(i["answer"])
        assert all(sorted(a) == ["no", "yes"] for a in pairs.values())
    counts = Counter(i["answer"] for i in subset if i["family"] == "count")
    assert len(set(counts.values())) == 1  # same number of items per bucket


def test_shuffle_map_never_maps_to_itself():
    names = [f"img{i}" for i in range(50)]
    mapping = shuffle_map(names, seed=0)
    assert sorted(mapping) == sorted(mapping.values()) == sorted(names)
    assert all(a != b for a, b in mapping.items())


@needs_data
def test_oracle_on_val_mini(tmp_path):
    from thermground.data.hituav import HitUav

    split_cfg = EVAL["sets"]["val-mini"]
    items = select_subset(
        load_items(ROOT / QA["out_dir"] / "val.jsonl"), split_cfg["sizes"], EVAL["seed"]
    )
    assert len(items) == sum(split_cfg["sizes"].values())
    ds = HitUav(DATA_DIR, split="val", split_file=ROOT / QA["split_file"])
    index = {n: i for i, n in enumerate(ds.names)}
    samples = {n: ds.annotation(i) for n, i in index.items()}
    images = {n: ds[i].image for n, i in index.items()}  # the oracle indexes every image
    model = reference_model("oracle", samples, images.__getitem__, BUCKETS)
    headline = evaluate(
        model,
        items,
        samples,
        images.__getitem__,
        tmp_path,
        instructions=EVAL["instructions"],
        coords_text=EVAL["coords"],
        buckets=BUCKETS,
        held_out=set(),
    )["headline"]
    assert all(h["real"] == 1.0 and h["follows_shuffled"] == 1.0 for h in headline.values())
