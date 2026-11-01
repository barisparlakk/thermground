from pathlib import Path

import numpy as np
import pytest

from thermground.data.hituav import CLASSES, HitUav, Meta, parse_name

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hituav"
needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="HIT-UAV not downloaded")

# Expected values from docs/dataset_notes.md
IMAGES = {"train": 2029, "val": 290, "test": 579}
BOXES = {"train": 17628, "val": 2460, "test": 4811}
CLASS_TOTALS = {"Person": 12312, "Car": 7311, "Bicycle": 4980, "OtherVehicle": 148, "DontCare": 148}
WIDTH, HEIGHT = 640, 512


def test_parse_name():
    assert parse_name("1_130_90_0_04077.jpg") == Meta("night", 130, 90, 4077)
    assert parse_name("0_60_30_0_00012") == Meta("day", 60, 30, 12)


def test_parse_name_rejects_bad_names():
    with pytest.raises(ValueError):
        parse_name("not_a_hituav_name")


@pytest.fixture(scope="module")
def splits():
    return {s: HitUav(DATA_DIR, split=s) for s in IMAGES}


@needs_data
def test_counts_match(splits):
    totals = dict.fromkeys(CLASSES, 0)
    for split, ds in splits.items():
        labels = np.concatenate([ds.annotation(i).labels for i in range(len(ds))])
        assert len(ds) == IMAGES[split]
        assert len(labels) == BOXES[split]
        for class_id, n in zip(*np.unique(labels, return_counts=True)):
            totals[CLASSES[class_id]] += int(n)
    assert totals == CLASS_TOTALS


@needs_data
def test_boxes_inside_image(splits):
    for ds in splits.values():
        boxes = np.concatenate([ds.annotation(i).boxes for i in range(len(ds))])
        x1, y1, x2, y2 = boxes.T
        assert (x1 >= 0).all() and (y1 >= 0).all()
        assert (x2 <= WIDTH).all() and (y2 <= HEIGHT).all()
        assert (x1 < x2).all() and (y1 < y2).all()


@needs_data
def test_sample_shapes(splits):
    sample = splits["val"][0]
    assert sample.image.shape == (HEIGHT, WIDTH) and sample.image.dtype == np.uint8
    n = len(sample.labels)
    assert sample.boxes.shape == (n, 4)
    assert len(sample.difficult) == len(sample.truncated) == n


@needs_data
def test_drop_dontcare():
    ds = HitUav(DATA_DIR, split="val", drop_dontcare=True)
    labels = np.concatenate([ds.annotation(i).labels for i in range(len(ds))])
    assert CLASSES.index("DontCare") not in labels
