"""Scores for QA predictions under the real / blank / shuffled image conditions.

Each record is a dict with:
    family, condition, pair_id, meta, held_out
    pred    parsed answer (str, list of boxes for point, or None if unparseable)
    truth   {"answer", "evidence"} for the original image
    shown   {"answer", "evidence"} for the image the model actually saw

"vs_original" scores against truth, "vs_shown" against shown. On a blank image the shown
answer is "absent" (no / 0 / none); on a shuffled image it is the other image's answer.
"""

from __future__ import annotations

from collections import defaultdict

YES_NO = ("presence", "region")


def _mean(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _balanced(records: list[dict], key: str) -> float | None:
    """Mean per-answer accuracy, so a constant answer scores chance even on skewed data."""
    by_answer = defaultdict(list)
    for r in records:
        by_answer[r[key]["answer"]].append(r["pred"] == r[key]["answer"])
    return _mean(_mean(v) for v in by_answer.values())


def hits(pred: list[list[float]], gt: list[list[float]], enlarge: float = 0.25) -> int:
    """One-to-one matches: a prediction's centre must fall inside a GT box enlarged by
    `enlarge` (half on each side); each GT box is matched at most once, nearest first."""
    used, tp = set(), 0
    for px1, py1, px2, py2 in pred:
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        best = None
        for j, (x1, y1, x2, y2) in enumerate(gt):
            mx, my = enlarge / 2 * (x2 - x1), enlarge / 2 * (y2 - y1)
            if j in used or not (x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my):
                continue
            dist = (cx - (x1 + x2) / 2) ** 2 + (cy - (y1 + y2) / 2) ** 2
            if best is None or dist < best[0]:
                best = (dist, j)
        if best:
            used.add(best[1])
            tp += 1
    return tp


def yes_no_scores(records: list[dict], key: str) -> dict:
    valid = [r for r in records if r["pred"] is not None]
    out = {
        "n": len(records),
        "invalid": _ratio(len(records) - len(valid), len(records)),
        "accuracy": _mean(r["pred"] == r[key]["answer"] for r in records),
        "balanced_accuracy": _balanced(records, key),
        "yes_rate": _mean(r["pred"] == "yes" for r in valid),
    }
    if key == "truth":
        pairs = defaultdict(list)
        for r in records:
            if r["pair_id"]:
                pairs[r["pair_id"]].append(r["pred"] == r["truth"]["answer"])
        out["pair_accuracy"] = _mean(all(p) for p in pairs.values() if len(p) == 2)
    return out


def count_scores(records: list[dict], key: str, buckets: list[str]) -> dict:
    valid = [r for r in records if r["pred"] is not None]
    index = {label: i for i, label in enumerate(buckets)}
    return {
        "n": len(records),
        "invalid": _ratio(len(records) - len(valid), len(records)),
        "accuracy": _mean(r["pred"] == r[key]["answer"] for r in records),
        "balanced_accuracy": _balanced(records, key),
        "within_one": _mean(
            r["pred"] is not None and abs(index[r["pred"]] - index[r[key]["answer"]]) <= 1
            for r in records
        ),
    }


def point_scores(records: list[dict], key: str, enlarge: float) -> dict:
    tp = n_pred = n_gt = 0
    for r in records:
        pred, gt = r["pred"] or [], r[key]["evidence"]
        tp += hits(pred, gt, enlarge)
        n_pred += len(pred)
        n_gt += len(gt)
    precision, recall = _ratio(tp, n_pred), _ratio(tp, n_gt)
    f1 = None  # undefined only when there is nothing to find
    if n_gt:
        f1 = round(2 * precision * recall / (precision + recall), 4) if tp else 0.0
    empty = [r for r in records if not r[key]["evidence"]]
    return {
        "n": len(records),
        "invalid": _ratio(sum(r["pred"] is None for r in records), len(records)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "none_accuracy": _mean(r["pred"] == [] for r in empty),
    }


def family_scores(records: list[dict], family: str, key: str, buckets: list[str], enlarge: float):
    if family in YES_NO:
        return yes_no_scores(records, key)
    if family == "count":
        return count_scores(records, key, buckets)
    return point_scores(records, key, enlarge)


def primary(scores: dict, family: str) -> float | None:
    """Balanced accuracy, or F1 for point."""
    return scores["f1"] if family == "point" else scores["balanced_accuracy"]


def absent(scores: dict, family: str) -> float | None:
    """Share of answers that say "nothing here" (scores vs_shown on a blank image)."""
    return scores["none_accuracy"] if family == "point" else scores["accuracy"]


def score(records: list[dict], buckets: list[str], enlarge: float = 0.25) -> dict:
    by = defaultdict(list)
    for r in records:
        by[r["condition"], r["family"]].append(r)
    conditions = sorted({r["condition"] for r in records})
    families = sorted({r["family"] for r in records})

    out = {"conditions": {}, "headline": {}, "breakdown_real": {}}
    for cond in conditions:
        out["conditions"][cond] = {}
        for family in families:
            recs = by[cond, family]
            entry = {"vs_original": family_scores(recs, family, "truth", buckets, enlarge)}
            if cond != "real":
                entry["vs_shown"] = family_scores(recs, family, "shown", buckets, enlarge)
            out["conditions"][cond][family] = entry

    for family in families:
        c = out["conditions"]
        real = c.get("real", {}).get(family, {}).get("vs_original")
        blank = c.get("blank", {}).get(family)
        shuffled = c.get("shuffled", {}).get(family)
        h = {"real": primary(real, family) if real else None}
        if real:
            h["invalid_real"] = real["invalid"]
            if family in YES_NO:
                h["pair_accuracy_real"] = real["pair_accuracy"]
        if blank:
            h["blank"] = primary(blank["vs_original"], family)
            h["absent_on_blank"] = absent(blank["vs_shown"], family)
            if h["real"] is not None and h["blank"] is not None:
                h["grounding_gap"] = round(h["real"] - h["blank"], 4)
        if shuffled:
            h["follows_shuffled"] = primary(shuffled["vs_shown"], family)
        out["headline"][family] = h

        if real:
            recs = by["real", family]
            groups = {
                "altitude": lambda r: r["meta"]["altitude_m"],
                "scenario": lambda r: r["meta"]["scenario"],
                "wording": lambda r: "held_out" if r["held_out"] else "seen",
            }
            out["breakdown_real"][family] = {}
            for name, key in groups.items():
                split = defaultdict(list)
                for r in recs:
                    split[key(r)].append(r)
                out["breakdown_real"][family][name] = {
                    str(k): {
                        "n": len(v),
                        "score": primary(
                            family_scores(v, family, "truth", buckets, enlarge), family
                        ),
                    }
                    for k, v in sorted(split.items())
                }
    return out
