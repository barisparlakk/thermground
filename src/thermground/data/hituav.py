"""HIT-UAV loader (axis-aligned boxes from the normal_xml family).

Returns plain numpy arrays so it can be wrapped by any framework. Oriented boxes
(rotate_xml / rotate_json) are not handled here. See docs/dataset_notes.md for the
format details this module relies on.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("Person", "Car", "Bicycle", "OtherVehicle", "DontCare")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
DONTCARE_ID = CLASS_TO_ID["DontCare"]
SPLITS = ("train", "val", "test")
SCENARIOS = ("day", "night")


@dataclass(frozen=True)
class Meta:
    scenario: str  # "day" or "night"
    altitude_m: int
    angle_deg: int
    frame: int


@dataclass
class Sample:
    name: str
    image: np.ndarray | None  # (H, W) uint8, None when not loaded
    boxes: np.ndarray  # (N, 4) float32, x1 y1 x2 y2 in pixels
    labels: np.ndarray  # (N,) int64, index into CLASSES
    difficult: np.ndarray  # (N,) bool
    truncated: np.ndarray  # (N,) bool
    meta: Meta


def parse_name(name: str) -> Meta:
    """Parse '{scenario}_{altitude}_{angle}_{weather}_{frame}', e.g. '1_130_90_0_04077'."""
    parts = Path(name).stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"unexpected HIT-UAV file name: {name!r}")
    scenario, altitude, angle, _weather, frame = parts  # weather is always 0 (sunny)
    return Meta(SCENARIOS[int(scenario)], int(altitude), int(angle), int(frame))


def find_root(data_dir: str | Path) -> Path:
    """Return the extracted dataset folder (the one holding normal_xml/) under data_dir."""
    data_dir = Path(data_dir)
    if (data_dir / "normal_xml").is_dir():
        return data_dir
    matches = sorted(p.parent for p in data_dir.glob("*/normal_xml") if p.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one HIT-UAV folder under {data_dir}, found {len(matches)}"
        )
    return matches[0]


def _parse_xml(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes, labels, difficult, truncated = [], [], [], []
    for obj in ET.parse(path).getroot().iter("object"):
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASS_TO_ID[obj.find("name").text])
        difficult.append(obj.findtext("difficult") == "1")
        truncated.append(obj.findtext("truncated") == "1")
    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(labels, dtype=np.int64),
        np.array(difficult, dtype=bool),
        np.array(truncated, dtype=bool),
    )


class HitUav:
    """One split of HIT-UAV. Annotations are parsed up front; images load on access."""

    def __init__(
        self,
        data_dir: str | Path = "data/hituav",
        split: str = "train",
        drop_dontcare: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = find_root(data_dir) / "normal_xml"
        self.split = split
        self.names = (self.root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
        self._ann = [_parse_xml(self.root / "Annotations" / f"{n}.xml") for n in self.names]
        if drop_dontcare:
            self._ann = [
                tuple(arr[labels != DONTCARE_ID] for arr in (boxes, labels, diff, trunc))
                for boxes, labels, diff, trunc in self._ann
            ]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Sample:
        sample = self.annotation(i)
        with Image.open(self.image_path(i)) as im:
            sample.image = np.asarray(im.convert("L"))
        return sample

    def annotation(self, i: int) -> Sample:
        """Sample without decoding the image (image is None)."""
        boxes, labels, difficult, truncated = self._ann[i]
        name = self.names[i]
        return Sample(name, None, boxes, labels, difficult, truncated, parse_name(name))

    def image_path(self, i: int) -> Path:
        return self.root / "JPEGImages" / f"{self.names[i]}.jpg"
