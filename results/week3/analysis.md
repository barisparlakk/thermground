# Week 3: zero-shot baseline on the Mac

Qwen3-VL-2B and Qwen3-VL-4B (Instruct, 4-bit MLX), zero-shot, evaluated under protocol v1
(`docs/eval_protocol.md`) on `val-mini` and `test-mini`. Greedy decoding, pinned model
revisions; configs in `configs/models/`. Full tables are in
`results/week3/eval/<model>/<set>/summary.md`.

Decisions below rest on `val-mini`; `test-mini` is the reported number. Each family has a
95% interval of roughly +/-7 to 9 points, so single differences of that size are not
conclusive on their own.

## Headline

Scores in %. Point uses F1; the other families use balanced accuracy. Gap = real - blank
(ceiling: 50 for presence and region, 83.3 for count, 100 for point).

**test-mini**

| Model | presence real / pair / gap | region real / pair / gap | count real / gap | point F1 | follows shuffled (presence / region) |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 62.5 / 28.3 / 12.5 | 68.0 / 40.0 / 18.0 | 21.7 / 5.0 | 37.0 | 62.4 / 69.3 |
| Qwen3-VL-4B | 69.2 / 41.7 / 19.2 | 74.0 / 51.0 / 24.0 | 30.0 / 13.3 | 38.2 | 73.1 / 77.2 |

**val-mini**

| Model | presence real / pair / gap | region real / pair / gap | count real / gap | point F1 | follows shuffled (presence / region) |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 59.2 / 21.7 / 9.2 | 66.0 / 38.0 / 16.0 | 23.3 / 6.7 | 39.5 | 67.7 / 55.1 |
| Qwen3-VL-4B | 68.3 / 40.0 / 18.3 | 71.5 / 49.0 / 21.5 | 30.8 / 14.2 | 46.1 | 83.9 / 69.5 |

Both models give a valid answer to every item (0% invalid), and both answer "no" / "0" /
"none" to 100% of blank images.

## Findings

### 1. The 4B model is better on every family, but both are far from the ceiling

The 4B model scores higher on real images for all four families in both sets, and has
higher pair accuracy for presence and region in both sets. The blank score is the same
chance level for both models, so the grounding gap rises by the same amount. Each single
difference is near the edge of the mini-set precision, but all 12 comparisons point the
same way, so the improvement is real. The smallest difference is point F1 on `test-mini`
(37.0 vs 38.2), which is a tie.

Both models are still well short of the ceiling. The best presence gap is 19 of 50 points,
and the best count gap is 14 of 83. Most of each answer still comes from something other
than the image.

### 2. A strong bias towards "no"

Half of the presence items have the answer "yes", but the models answer "yes" far less
often.

| | yes rate | recall Person | recall Car | recall Bicycle | correct on "no" items |
|---|---|---|---|---|---|
| 2B val / test | 0.16 / 0.23 | 0.14 / 0.30 | 1.00 / 0.86 | 0.00 / 0.20 | 0.93 / 0.90 |
| 4B val / test | 0.25 / 0.34 | 0.40 / 0.58 | 1.00 / 0.86 | 0.00 / 0.10 | 0.93 / 0.85 |

- Cars, which are large and warm, are recognised.
- Bicycles are almost never recognised.
- People, the class that matters for search and rescue, are missed most of the time.

The region family shows the same pattern (yes rate 0.29 to 0.43). Count does too: the most
common wrong answer is "0" for every true bucket.

### 3. The models can find people they say are not there

The point and presence questions are asked on different images drawn from the same
distribution, so their results can be compared:

| | presence: says "yes" to a person | point: finds at least one person |
|---|---|---|
| 2B val / test | 14% / 30% | 53% / 40% |
| 4B val / test | 40% / 58% | 58% / 45% |

On both sets the 2B model finds at least one person far more often when asked to point
than it admits a person is present when asked yes/no. For the 4B model this holds on val
and reverses on test.

The visual evidence is often available to the model, but its short answer doesn't use it.
This is the main motivation for the grounding methods planned for weeks 6–7: make the model
locate the evidence first, then answer from it.

### 4. Under-counting

- **2B:** it rarely goes beyond "1" or "2". Crowds of 7+ get "2" in 10 to 12 of 20 items,
  and it never answers "4-6" correctly.
- **4B:** it separates large groups much better ("4-6" or "7+" for 15 of 20 crowds on
  `test-mini`). It still answers "0" for most items with 1 to 3 people, which is the
  "no" bias again.

Within-one-bucket accuracy is stored in `metrics.json` but not used for decisions.

### 5. Pointing: night works, day does not

Point F1 by scenario, on real images:

| | day | night |
|---|---|---|
| 2B val / test | 0.13 / 0.07 | 0.40 / 0.46 |
| 4B val / test | 0.22 / 0.13 | 0.47 / 0.43 |

At night people are much warmer than the ground. By day the ground heats up and the
contrast drops. Presence and region don't show a consistent day/night difference.

### 6. Person size: clear on val, confounded on test

Point recall by person box width:

| | < 10 px | 10–15 px | >= 15 px |
|---|---|---|---|
| 2B val | 0.12 (58) | 0.24 (108) | 0.47 (124) |
| 4B val | 0.16 (58) | 0.40 (108) | 0.48 (124) |
| 2B test | 0.31 (109) | 0.37 (98) | 0.11 (44) |
| 4B test | 0.61 (109) | 0.49 (98) | 0.18 (44) |

On val, recall rises steadily with size. On test the trend reverses, but the 44 wide boxes
come from only 15 images, mostly low-altitude day scenes with groups of people. That is
exactly the setting where pointing fails (finding 5). Size and scenario can't be separated
on the mini sets. The full test set at the end, or an analysis that controls for scenario,
should settle it.

### 7. Point precision vs recall

| | precision | recall | "none" on empty items |
|---|---|---|---|
| 2B val / test | 0.58 / 0.51 | 0.30 / 0.29 | 0.90 / 0.85 |
| 4B val / test | 0.61 / 0.32 | 0.37 / 0.47 | 0.75 / 0.80 |

The 2B model is cautious: few boxes, about half of them correct. The 4B model returns more
boxes. On `test-mini` it gains recall, but its precision halves, so F1 doesn't change.

### 8. Other checks

- **Blank images:** 100% "absent" for both models. This fits the "no" bias, so it shouldn't
  be read as calibrated uncertainty. A blank grey image may also be too easy a probe;
  see the limitations in `docs/eval_protocol.md`.
- **Shuffled images:** answers follow the image actually shown, more so for 4B (73 to 84%
  on presence vs 62 to 68% for 2B).
- **Held-out wording (`test-mini`):** no drop on the unseen templates (presence 0.60 vs
  0.64 for 2B, 0.69 vs 0.69 for 4B). The models aren't sensitive to how the question is
  phrased.

## Cost on the M2 (16 GB)

| Model | seconds per call (real / blank / shuffled) | one mini set (1,560 calls) |
|---|---|---|
| Qwen3-VL-2B | 1.8 to 2.3 / 1.6 to 1.9 / 1.9 | about 50 min |
| Qwen3-VL-4B | 2.8 to 3.2 / 2.8 / 3.0 to 3.2 | about 78 min |

Point items take about twice as long as short answers, because a JSON box costs about 25
tokens. Long runs slow down from thermal throttling.

## Implications for weeks 4–5 (fine-tuning)

1. **Base model:** use Qwen3-VL-4B as the main model, since it is better on all families
   in `val-mini`. Use 2B for quick ablations where turnaround matters.
2. **What fine-tuning should fix:** the "no" bias and the miss on people are the first
   targets. Balanced QA training data should move the yes rate towards 0.5.
3. **What to check after training:** the grounding gap and "follows shuffled", not real
   accuracy alone. A model that only loses its "no" bias would also drop "absent on
   blank" below 100%, so all three headline numbers have to be read together.
4. **Small people:** with most people 10 to 15 px wide, input resolution and tiling are
   worth an ablation.
5. **Day scenes:** day images are the weakest condition for pointing. Report day and night
   separately in every later week.
