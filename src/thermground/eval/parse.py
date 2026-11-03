"""Turn free-text model output into canonical QA answers.

Parsers return None when the output cannot be read; the evaluation scores that as
wrong and reports it as invalid.
"""

from __future__ import annotations

import re

from thermground.data.qa import HEIGHT, WIDTH, count_bucket

NUMBER_WORDS = {
    w: i
    for i, w in enumerate((
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty",
    ))
}  # fmt: skip
YES = {"yes", "yeah", "yep", "true", "correct", "affirmative"}
NO = {"no", "nope", "false", "incorrect", "negative"}
NEGATION = re.compile(r"\b(no|not|none|nobody|nothing|cannot|neither)\b|\bno one\b|n't\b")
AFFIRM = re.compile(r"\bthere (is|are)\b|\bi (can )?see\b|\b(visible|present|contains?)\b")
ABSENT = re.compile(r"\b(none|nobody|nothing|zero)\b|\bno (one|people|persons?|humans?)\b")
RANGE = re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)|(\d+)\s*\+")
NUMBER = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\b")
TUPLE = re.compile(r"[\[(]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*[\])]")
COORD_SCALE = {"pixel": None, "norm1": 1.0, "norm1000": 1000.0}


def parse_yes_no(text: str) -> str | None:
    """Leading yes/no word wins; otherwise any negation means no, an affirmation yes."""
    low = text.lower()
    words = re.findall(r"[a-z']+", low)
    if not words:
        return None
    if words[0] in YES:
        return "yes"
    if words[0] in NO:
        return "no"
    if NEGATION.search(low):
        return "no"
    if AFFIRM.search(low):
        return "yes"
    return None


def parse_count(text: str, buckets: list[str]) -> str | None:
    """Bucket label ('4-6', '7+'), else the first number, else an absence phrase -> '0'."""
    low = text.lower()
    for match in RANGE.finditer(low):
        label = f"{match[1]}-{match[2]}" if match[1] else f"{match[3]}+"
        if label in buckets:
            return label
    if match := NUMBER.search(low):
        word = match[1]
        return count_bucket(int(word) if word.isdigit() else NUMBER_WORDS[word], buckets)
    if ABSENT.search(low):
        return count_bucket(0, buckets)
    return None


def parse_boxes(
    text: str, coord_format: str = "pixel", width: int = WIDTH, height: int = HEIGHT
) -> list[list[float]] | None:
    """[x1, y1, x2, y2] boxes or [x, y] points (as zero-size boxes), in pixels.

    Returns [] for an explicit "none" answer and None when nothing can be read.
    """
    scale = COORD_SCALE[coord_format]
    boxes = []
    for match in TUPLE.finditer(text):
        values = [float(v) for v in match[1].split(",")]
        if len(values) == 2:
            values *= 2
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = values
        if scale:
            x1, x2 = x1 * width / scale, x2 * width / scale
            y1, y2 = y1 * height / scale, y2 * height / scale
        x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
        y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
        boxes.append([x1, y1, x2, y2])
    if boxes:
        return boxes
    if ABSENT.search(text.lower()):
        return []
    return None


def parse_answer(
    text: str, family: str, buckets: list[str], coord_format: str = "pixel"
) -> str | list[list[float]] | None:
    if family in ("presence", "region"):
        return parse_yes_no(text)
    if family == "count":
        return parse_count(text, buckets)
    if family == "point":
        return parse_boxes(text, coord_format)
    raise ValueError(f"unknown family {family!r}")
