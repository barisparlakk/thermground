# HIT-UAV QA v1 design

Template questions generated from the HIT-UAV boxes, used to test whether a model answers
from the thermal image rather than from language priors.

- Code: `src/thermground/data/qa.py`
- Config: `configs/qa/hituav_qa_v1.yaml` (templates, buckets, sampling, seed)
- Generate: `uv run python scripts/make_hituav_qa.py`, which writes `data/qa/hituav_qa_v1/{train,val,test}.jsonl`
  (gitignored, deterministic) and `results/week2/qa/`.
- Splits: the leakage-aware grouped split, `configs/splits/hituav_grouped_v1.yaml`.

## Principles

1. **Every answer is computed from the annotations.** There is no manual labelling, so every
   answer can be checked; `tests/test_qa.py` recomputes all of them.
2. **No family can be solved without the image.** Answers are balanced per family and split,
   and yes/no questions come in matched pairs, so a guess from the question text scores chance.
3. **Answers carry evidence.** Each item lists the ground-truth boxes that support its answer,
   so a correct answer can later be checked against where the model looked.

## Families

| Family | Example | Answer | Balancing |
|---|---|---|---|
| presence | "Is there a person in this image?" | yes / no | yes/no pairs within the same (class, day/night, altitude); 70% Person, 15% Car, 15% Bicycle |
| count | "How many people are in this image?" | `0`, `1`, `2`, `3`, `4-6`, `7+` | equal items per bucket |
| region | "Is there a person in the top-left part of the image?" | yes / no | yes/no pairs within the same (region, day/night) |
| point | "Mark every person in the image." | `boxes` or `none`; boxes in `evidence` | 25% `none`; images with more than 10 people are skipped |

- **Regions** are a 3x3 grid in image coordinates, and a box belongs to the cell holding its
  centre. The camera is oblique (30 to 90 degrees), so "top" usually means "further away".
- **Count** uses buckets above 3, because crowds of 10 px boxes are not reliably countable.
- **Region** is the main grounding test: the same image produces both yes and no answers for
  different regions, so an image-level impression cannot solve it.
- **Point** stores pixel boxes. Each model's coordinate format is produced by an adapter at
  evaluation time, not stored in the data.
- Vehicles appear only in presence, as distractors, so a model cannot assume every question is
  about people.

## Rules

- **DontCare:** images with any `DontCare` box are skipped entirely, since that region may hide
  the answer. This removes 26 train, 7 val and 70 test images (test holds most `DontCare` boxes).
- **Difficult and truncated** Person boxes count as present. There is no minimum box size.
- **Templates:** 6 per family. Templates `_05` and `_06` are held out and appear only in test,
  so results can be split into seen and unseen wording. Both halves of a pair use the same
  template.
- **Determinism:** each family and split has its own seeded random stream.

## Item schema

One JSON object per line:

```json
{"id": "test-presence-00000", "image": "1_80_50_0_04666", "split": "test", "family": "presence",
 "question": "Does this image contain a person?", "answer": "yes", "options": ["yes", "no"],
 "target_class": "Person", "region": null, "evidence": [[300.0, 317.0, 316.0, 338.0]],
 "template_id": "presence_03", "pair_id": "test-presence-pair00000",
 "meta": {"scenario": "night", "altitude_m": 80, "angle_deg": 50, "frame": 4666}}
```

- `options` lists the valid answers (null for point).
- `evidence` is x1 y1 x2 y2 in pixels; empty for "no" and "none".
- `meta` comes from the image file name and is used to break results down by altitude and day/night.

## Size

| Split | Items | Images | presence | count | region | point |
|---|---|---|---|---|---|---|
| train | 9,885 | 1,898 | 1,856 | 696 | 6,258 | 1,075 |
| val | 1,837 | 347 | 148 | 132 | 1,238 | 319 |
| test | 2,442 | 536 | 416 | 162 | 1,588 | 276 |

Point items with people have 4.2 to 5.0 boxes on average.

## Prior baselines (test)

Accuracy when answering from train answer frequencies only, without the image:

| Family | Chance | Family majority | Same-question majority |
|---|---|---|---|
| presence | 50% | 50% | 50% |
| count | 16.7% | 16.7% | 16.7% |
| region | 50% | 50% | 50% |

The yes/no families sit exactly at chance by construction, since both halves of a pair share
their question. A model's own priors (for example a tendency to answer "yes") are measured
separately by running it on blank or shuffled images.

## Decisions and limitations

- **No "where is the person?" family.** A 9-way locate question balanced over regions left only
  9 test items, because single-region images are rare outside the centre. Unbalanced, "center"
  would win on the prior alone. Region and point cover localisation instead.
- **Small count set:** 27 test items per bucket, about +/-8 points of sampling uncertainty.
- **Region dominates the item count** (63% of train). Training should weight families
  explicitly rather than sample items uniformly.
- **Templates are English only** and few, so the wording is narrow even with the held-out set.
