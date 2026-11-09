"""Fine-tuning example tests: targets, family mixing, chat split and label masking."""

from collections import Counter

import pytest

from thermground.data.sft import (
    IGNORE,
    make_examples,
    mask_labels,
    messages,
    mix_items,
    split_chat,
    target_text,
)
from thermground.eval.metrics import hits
from thermground.eval.parse import parse_answer

BUCKETS = ["0", "1", "2", "3", "4-6", "7+"]
BOXES = [[320.0, 256.0, 336.0, 280.0], [10.0, 40.0, 22.0, 60.0], [600.0, 40.0, 612.0, 58.0]]


def item(family, answer="yes", evidence=(), idx=0):
    return {
        "id": f"{family}-{idx}",
        "image": "img",
        "family": family,
        "question": "Q?",
        "answer": answer,
        "options": BUCKETS if family == "count" else None,
        "evidence": [list(b) for b in evidence],
    }


def test_target_text_short_answers():
    assert target_text(item("presence", "no")) == "no"
    assert target_text(item("count", "4-6")) == "4-6"
    assert target_text(item("point", "none")) == "none"


@pytest.mark.parametrize("coord_format", ["pixel", "norm1", "norm1000"])
def test_point_target_round_trips_through_parser(coord_format):
    text = target_text(item("point", "boxes", BOXES), coord_format)
    pred = parse_answer(text, "point", BUCKETS, coord_format)
    assert len(pred) == len(BOXES) and hits(pred, BOXES) == len(BOXES)


def test_point_target_is_sorted_top_to_bottom():
    lines = target_text(item("point", "boxes", BOXES)).splitlines()
    assert lines == ["[16, 78, 34, 117]", "[938, 78, 956, 113]", "[500, 500, 525, 547]"]


def test_mix_items_counts_and_seed():
    items = [item("presence", idx=i) for i in range(10)]
    items += [item("count", "1", idx=i) for i in range(4)]
    mixed = mix_items(items, {"presence": 0.5, "count": 2.0}, seed=0)
    assert Counter(i["family"] for i in mixed) == {"presence": 5, "count": 8}
    assert mixed == mix_items(items, {"presence": 0.5, "count": 2.0}, seed=0)
    assert Counter(i["family"] for i in mix_items(items, {"count": 1.0}, seed=0)) == {"count": 4}


def test_make_examples_uses_eval_prompt():
    instructions = {"presence": "Answer with yes or no.", "point": "Boxes {coords}."}
    coords = {"norm1000": "in 0-1000"}
    ex = make_examples([item("presence"), item("point", "none")], instructions, coords, "norm1000")
    assert ex[0]["prompt"] == "Q? Answer with yes or no." and ex[0]["target"] == "yes"
    assert ex[1]["prompt"] == "Q? Boxes in 0-1000." and ex[1]["target"] == "none"


def fake_template(chat, add_generation_prompt):
    text = "".join(f"<{m['role']}>{m['content'][-1]['text']}</>" for m in chat)
    return text + ("<assistant>" if add_generation_prompt else "")


def test_split_chat_and_messages():
    assert messages("Q?")[0]["content"][0] == {"type": "image"}
    full, tail = split_chat(fake_template, "Q?", "yes")
    assert full == "<user>Q?</><assistant>yes</>" and tail == "yes</>"
    with pytest.raises(ValueError):
        split_chat(lambda c, add_generation_prompt: "prefix-changes" * len(c), "Q?", "yes")


def test_mask_labels_keeps_answer_only():
    ids = [[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]]
    labels = mask_labels(ids, lengths=[4, 6], n_answer=[2, 3])
    assert labels[0] == [IGNORE, IGNORE, 3, 4, IGNORE, IGNORE]
    assert labels[1] == [IGNORE, IGNORE, IGNORE, 8, 9, 10]
    with pytest.raises(ValueError):
        mask_labels(ids, lengths=[4, 6], n_answer=[5, 1])
