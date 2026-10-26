"""
Shared decision logic: the cost model, the image-level score, and the
pass / review / fail rule. train.py uses it to sweep thresholds, app.py uses
it to recompute everything live, so both always agree.

Decision: a camera at the site turnstile looks at each worker entering.
Positive class = "head" (a bare head, i.e. someone without a hard hat).
Image score = highest confidence of any "head" box in the image (0 if none).

    score <= t_low             -> PASS   (turnstile opens)
    t_low <  score <= t_high   -> REVIEW (security guard looks at the frame)
    score >  t_high            -> FAIL   (turnstile stays shut, worker told
                                          to put the helmet on)

Comparisons are strict so that an image with no 'head' box at all (score 0)
always passes, even at threshold 0.

With t_low == t_high there is no review band and the model decides alone.

Every figure below is argued in REPORT.md. Change them here, not in code.
"""

from dataclasses import dataclass, asdict

import numpy as np

POSITIVE_CLASS = "head"     # bare head = PPE violation
NEGATIVE_CLASS = "helmet"   # head with a hard hat


@dataclass
class CostModel:
    # Euros per entry, by outcome.
    c_fn: float = 70.0      # bare-headed worker let onto site
    c_fp: float = 2.0       # helmeted worker wrongly stopped (worker + queue time)
    c_tp: float = 0.5       # violator correctly stopped (1 min to fetch helmet)
    c_tn: float = 0.0       # helmeted worker walks through
    c_review: float = 0.25  # guard looks at one frame (~10 s guard + worker wait)
    review_miss: float = 0.05   # share of real violations a guard still misses
    base_rate: float = 0.03     # share of entries that really are violations
    entries_per_day: int = 1200     # 600 workers x 2 entries (start, after lunch)
    peak_share: float = 0.5         # share of entries in the busiest hour
    review_seconds: float = 10.0    # guard time per reviewed frame

    def to_dict(self) -> dict:
        return asdict(self)


def image_score(head_confs) -> float:
    """Image-level violation score from the 'head' box confidences."""
    head_confs = list(head_confs)
    return float(max(head_confs)) if head_confs else 0.0


def decide(score: float, t_low: float, t_high: float) -> str:
    if score > t_high:
        return "FAIL"
    if score > t_low:
        return "REVIEW"
    return "PASS"


def outcome_rates(scores, labels, t_low, t_high) -> dict:
    """Share of positives / negatives landing in each of the three outcomes.

    Rates are conditional on the true class, so they do not depend on how
    often violations appear in the test set. That lets us re-weight them by
    the production base rate instead of the dataset's.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    t_high = max(t_high, t_low)
    fail = scores > t_high
    review = (scores > t_low) & ~fail
    passed = ~fail & ~review
    pos, neg = labels, ~labels
    n_pos, n_neg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "pos_fail": (fail & pos).sum() / n_pos,      # caught by model (TP)
        "pos_review": (review & pos).sum() / n_pos,
        "pos_pass": (passed & pos).sum() / n_pos,    # missed by model (FN)
        "neg_fail": (fail & neg).sum() / n_neg,      # wrongly stopped (FP)
        "neg_review": (review & neg).sum() / n_neg,
        "neg_pass": (passed & neg).sum() / n_neg,    # TN
        "counts": {
            "pos_fail": int((fail & pos).sum()), "pos_review": int((review & pos).sum()),
            "pos_pass": int((passed & pos).sum()), "neg_fail": int((fail & neg).sum()),
            "neg_review": int((review & neg).sum()), "neg_pass": int((passed & neg).sum()),
        },
    }


def cost_per_1000(rates: dict, cm: CostModel) -> dict:
    """Expected euros per 1,000 entries at the production base rate."""
    p = cm.base_rate
    # Reviewed violations: the guard catches most, misses review_miss of them.
    pos_review_cost = cm.c_review + (1 - cm.review_miss) * cm.c_tp + cm.review_miss * cm.c_fn
    neg_review_cost = cm.c_review + cm.c_tn
    per_pos = (rates["pos_fail"] * cm.c_tp + rates["pos_pass"] * cm.c_fn
               + rates["pos_review"] * pos_review_cost)
    per_neg = (rates["neg_fail"] * cm.c_fp + rates["neg_pass"] * cm.c_tn
               + rates["neg_review"] * neg_review_cost)
    review_share = p * rates["pos_review"] + (1 - p) * rates["neg_review"]
    reviews_day = review_share * cm.entries_per_day
    peak_reviews = reviews_day * cm.peak_share
    return {
        "cost_per_1000": 1000 * (p * per_pos + (1 - p) * per_neg),
        "review_share": review_share,
        "reviews_per_day": reviews_day,
        "peak_hour_reviews": peak_reviews,
        # guards needed in the busiest hour, at review_seconds per frame
        "guards_needed_peak": peak_reviews * cm.review_seconds / 3600,
        "missed_violations_per_1000": 1000 * p * (rates["pos_pass"]
                                                  + rates["pos_review"] * cm.review_miss),
        "false_stops_per_1000": 1000 * (1 - p) * rates["neg_fail"],
    }


def evaluate(scores, labels, t_low, t_high, cm: CostModel) -> dict:
    rates = outcome_rates(scores, labels, t_low, t_high)
    return {**rates, **cost_per_1000(rates, cm), "t_low": t_low, "t_high": t_high}


THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def sweep_single(scores, labels, cm: CostModel, grid=THRESHOLDS) -> list[dict]:
    """Model decides alone (no review band): one threshold, swept."""
    return [evaluate(scores, labels, t, t, cm) for t in grid]


def best_band(scores, labels, cm: CostModel, grid=THRESHOLDS) -> dict:
    """Cheapest (t_low, t_high) pair over the grid, review band allowed."""
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            r = evaluate(scores, labels, lo, hi, cm)
            if best is None or r["cost_per_1000"] < best["cost_per_1000"]:
                best = r
    return best


def reference_policies(cm: CostModel) -> dict:
    """The three alternatives the model is priced against, per 1,000 entries."""
    p = cm.base_rate
    return {
        # No check at all: every violator walks in.
        "do_nothing": 1000 * (p * cm.c_fn + (1 - p) * cm.c_tn),
        # A guard checks every single entry.
        "human_checks_all": 1000 * (cm.c_review
                                    + p * ((1 - cm.review_miss) * cm.c_tp + cm.review_miss * cm.c_fn)
                                    + (1 - p) * cm.c_tn),
    }
