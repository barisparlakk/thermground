from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from thermground.data.hituav import SPLITS, HitUav, parse_name

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "configs" / "splits" / "hituav_grouped_v1.yaml"
DATA_DIR = ROOT / "data" / "hituav"
needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="HIT-UAV not downloaded")

RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
TOLERANCE = 0.04  # whole groups move together, so ratios are approximate
ALTITUDES = set(range(60, 131, 10))
NEAR_FRAMES = 5


@pytest.fixture(scope="module")
def split():
    return yaml.safe_load(SPLIT_FILE.read_text())


def test_every_image_once(split):
    names = [n for s in SPLITS for n in split[s]]
    assert all(isinstance(n, str) for n in names)
    assert len(names) == len(set(names)) == 2898


def test_ratios(split):
    for s, ratio in RATIOS.items():
        assert abs(len(split[s]) / 2898 - ratio) <= TOLERANCE


def test_coverage(split):
    for s in SPLITS:
        metas = [parse_name(n) for n in split[s]]
        assert {m.scenario for m in metas} == {"day", "night"}
        assert {m.altitude_m for m in metas} == ALTITUDES


def test_no_nearby_frames_across_splits(split):
    """No test image has a train/val frame within a few frame ids under the same condition."""
    frames = defaultdict(list)
    for s in ("train", "val"):
        for n in split[s]:
            m = parse_name(n)
            frames[m.scenario, m.altitude_m, m.angle_deg].append(m.frame)
    for n in split["test"]:
        m = parse_name(n)
        nearby = frames[m.scenario, m.altitude_m, m.angle_deg]
        assert all(abs(f - m.frame) > NEAR_FRAMES for f in nearby), n


@needs_data
def test_matches_dataset(split):
    provided = [n for s in SPLITS for n in HitUav(DATA_DIR, split=s).names]
    assert sorted(provided) == sorted(n for s in SPLITS for n in split[s])


@needs_data
def test_loader_reads_split_file():
    boxes = 0
    for s in SPLITS:
        ds = HitUav(DATA_DIR, split=s, split_file=SPLIT_FILE)
        boxes += sum(len(ds.annotation(i).labels) for i in range(len(ds)))
    assert boxes == 24899
