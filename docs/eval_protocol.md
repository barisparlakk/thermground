# Evaluation protocol v1

How every model is scored on the HIT-UAV QA set (`docs/qa_design.md`). The protocol was
fixed and validated with reference models before any real model was run.

- Code: `src/thermground/eval/` (parsing, metrics, runner), `src/thermground/models/base.py` (interface)
- Config: `configs/eval/hituav_eval_v1.yaml`
- Run: `uv run python scripts/evaluate.py --model <name> --set val-mini --out results/<week>/eval/<name>/val-mini`

## Evaluation sets

| Set | Split | Items | Use |
|---|---|---|---|
| `val-mini` | val | 520 | every prompt, model and setting decision |
| `test-mini` | test | 520 | the reported number at weekly milestones |
| `test` | test | 2,442 | final results only |

The mini sets hold 120 presence (60 yes/no pairs), 120 count (20 per bucket), 200 region
(100 pairs) and 80 point items, selected with a fixed seed so that pairs stay together and
answers stay balanced. Precision is roughly +/-7 to 9 points per family at 95%, enough to see
whether a model uses the image but not to separate models a few points apart.

Decisions are made on val only. Looking at test while choosing prompts or models would make
the final test numbers optimistic.

## Image conditions

Every item is asked three times, with the same question and a different image:

| Condition | Image shown | `vs_original` | `vs_shown` |
|---|---|---|---|
| real | the item's image | its labels | (same) |
| blank | uniform grey (value 128) | the item's labels | "absent": no / 0 / none |
| shuffled | another image of the same set | the item's labels | the labels of the shown image |

Because answers are computed from boxes, the correct answer is known for any image shown.

## Metrics

**Per family**

- presence, region, count: **balanced accuracy**, the mean of per-answer accuracy, so a
  constant answer scores chance even when the shown images are unbalanced. Plain accuracy,
  the yes rate and (for count) within-one-bucket accuracy are also stored.
- presence, region: **pair accuracy**, where a yes/no pair counts only if both halves are
  correct. A constant answer scores 0%; random guessing about 25%.
- point: predicted boxes or points are matched one-to-one to ground-truth Person boxes. A
  prediction hits when its centre falls inside a GT box enlarged by 25% (12.5% per side;
  about 1.5 px for a 12 x 19 px person). Reports precision, recall, F1, and for items with
  nobody to find, the share answered "none".
- Unparseable answers count as wrong and are reported as `invalid`.

**Headline** (in `metrics.json` and `summary.md`)

| Name | Definition | Reading |
|---|---|---|
| real | score on real images | plain performance |
| blank | score on blank images vs original labels | what the prior alone achieves |
| grounding gap | real - blank | how much of the score needs the image |
| pair acc | pair accuracy on real images | the strictest yes/no number |
| absent on blank | share of "nothing here" answers on blank images | does the model admit there is no evidence? |
| follows shuffled | score on shuffled images vs the shown image | does the answer follow the image? |

Points to keep in mind:

- The grounding gap has a ceiling. Blank-image answers are scored against balanced labels,
  so they score chance whatever the model says: the largest possible gap is 50 points for
  presence and region, 83.3 for count and 100 for point. The oracle reaches exactly these.
- "Absent on blank" is only meaningful next to real accuracy. A model that always answers
  "no" scores 100% there and is useless on real images.

## Answer parsing

- yes/no: a leading yes/no word wins; otherwise a negation ("can't", "nobody", "no one")
  means no and an affirmation ("there is", "I see", "visible") means yes.
- count: a bucket label (`4-6`, `7+`), else the first number (digits or words up to twenty),
  else an absence phrase, which maps to `0`.
- point: bracketed `[x1, y1, x2, y2]` boxes or `[x, y]` points, converted from the model's
  coordinate format (`pixel`, `norm1` or `norm1000`) to pixels and clipped to the image;
  an absence phrase means no boxes.

## Prompts

The question is followed by a fixed instruction per family, set in the config:
"Answer with yes or no.", "Answer with one of: 0, 1, 2, 3, 4-6, 7+." and, for point, a
request for one box per person in the model's coordinate format.

## Reference models (test-mini)

| Model | presence real / pair | region real / pair | count real | point F1 | follows shuffled (presence) |
|---|---|---|---|---|---|
| oracle | 100 / 100 | 100 / 100 | 100 | 100 | 100 |
| always_yes | 50 / 0 | 50 / 0 | 16.7 | 0 | 50 |
| always_no | 50 / 0 | 50 / 0 | 16.7 | 0 | 50 |
| random | 55 / 28.3 | 51.5 / 25 | 22.5 | 0 | 50.1 |

Full tables are in `results/week2/eval/<model>/<set>/summary.md`. The oracle answers in
sentences ("I count 5 people.", "Here they are: [..]"), so its 100% also checks the parser.

## Running a model

- A model implements `VQAModel` (`src/thermground/models/base.py`): a `name`, a
  `coord_format` and `answer(query) -> str`. Real models may use only `query.image` and
  `query.prompt`.
- Predictions are written to `predictions_{condition}.jsonl` one line at a time
  (gitignored). An interrupted run resumes where it stopped.
- `--limit N` evaluates N items spread over all families, for quick checks.

## Limitations

- A uniform grey image may be unusual input for a model. It measures the prior, but some
  models may behave oddly on it rather than simply "seeing nothing".
- The parser is rule-based. Each real model's invalid rate should be checked, and a sample
  of its raw outputs read, before trusting its scores.
- Shuffled images come from the same evaluation set, so on the mini sets a few images
  appear in several shuffled items.
