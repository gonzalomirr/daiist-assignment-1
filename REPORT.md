# Assignment 2 Report

- **Name**:
- **Student ID**:
- **Email**: gmir.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**Decision.** A construction contractor runs a site of ~600 workers. Spanish
law (RD 1627/1997, RD 773/1997) makes the contractor responsible for workers
wearing a hard hat on site. Today a security guard at the turnstile looks at
every worker and decides: *let in*, or *stop and tell them to put the helmet
on*. The model replaces that look. A camera above the turnstile takes one frame
per entry, the model scores it, and the turnstile either opens (**pass**), stays
shut (**fail**), or puts the frame on the guard's screen (**review**). The guard
acts on review and fail cases; the site safety technician owns the thresholds.

**Positive class = `head`** (a bare head, i.e. a violation). The image score is
the highest `head` confidence in the frame, so one bare head anywhere is enough
to stop the entry.

**The two mistakes are very different:**

| Outcome | What happens | € per entry | Where the figure comes from |
|---|---|---|---|
| **False negative** (bare head let in) | A worker without a helmet walks onto an active site | **70** | ≈ €40 expected injury cost (assume 1 serious head injury per 10,000 unprotected worker-shifts × ~€400k to the company: *recargo de prestaciones* of 30–50 %, civil liability, stoppage, investigation) + ≈ €12 expected fine (failing to enforce PPE is a *grave* infraction under LISOS art. 12.16, minimum €2,451, × ~1/200 chance an inspector sees it that day) + ≈ €18 for the safety technician to find and remove the worker later (20 min at €40/h + worker time) |
| **False positive** (helmeted worker stopped) | Worker waits ~1 min for the guard; people queue behind | **2** | Loaded construction labour ≈ €25/h = €0.42/min, × ~4 people delayed at peak (the worker + ~3 queuing) |
| **True positive** (violator stopped) | Worker fetches the helmet, ~1 min | **0.5** | One worker-minute. This is the cost of compliance, paid anyway |
| **True negative** | Turnstile opens | **0** | |
| **Human review** | Guard looks at the frame for ~10 s; worker waits | **0.25** | 10 s of guard (€18/h ≈ €0.05) + ~15 s of the worker and one person behind (≈ €0.20). The guard is assumed to miss 5 % of real violations in review |

A miss costs **35× a false alarm** (€70 vs €2). That ratio, not the absolute
figures, is what places the threshold. Most of €70 is a small probability times
a very large number, so it is the least certain figure in the report; the
sensitivity table below spans €5 to €300.

What this implies before looking at any data:

- **Threshold.** Without a review band, the model should stop an entry once
  P(violation | image) > (C_FP − C_TN) / ((C_FP − C_TN) + (C_FN − C_TP))
  = 2 / (2 + 69.5) ≈ **2.8 %**. So the operating point favours recall on
  `head`, and the headline metric is recall on `head`, not mAP or accuracy.
- **Review band.** Reviewing costs €0.25 + 0.95·€0.5 + 0.05·€70 per real
  violation. Comparing the three actions at posterior p: review beats *pass*
  when 70p > 0.25 + 3.98p, i.e. p > 0.4 %, and beats *fail* when
  0.25 + 3.98p < 0.5p + 2(1 − p), i.e. p < 32 %. A guard should see the
  uncertain frames between those two posteriors. The model's score is not a
  calibrated probability, so the thresholds themselves are found by sweeping,
  not from this formula.
- **Base rate.** On a site that already enforces helmets, assume **3 %** of
  entries are violations (mostly a helmet carried in the hand, or forgotten
  after lunch). There is no published figure; this is a stated assumption.
  The dataset's own rate is far higher (see below), which is why costs are
  computed from the per-class rates (TPR, FPR), which do not depend on
  prevalence, re-weighted by 3 %, not from the raw test-set counts.
- **Volume.** 600 workers × 2 entries (morning, after lunch) = **1,200
  entries/day**, about half of them in the busiest hour. Reviews have to fit
  in that hour.

## Dataset

- **Roboflow Universe:** workspace `robertatroc-7xoey`, project
  `hard-hat-workers-dataset-hnvfo`, **version 1**, used as published (not forked).
  <https://universe.roboflow.com/robertatroc-7xoey/hard-hat-workers-dataset-hnvfo/dataset/1>
- 7,035 distinct images of construction workers (a Roboflow copy of the public
  *Hard Hat Workers* dataset from Northeastern University, China). Classes:
  `head` (bare head), `helmet` (head with a hard hat), `person`. Licence CC BY 4.0.
- **What the export actually contains, and what I did about it.** Version 1
  has **no validation split**: only `train` (9,177 files) and `test` (1,766).
  The 9,177 train files are 5,269 source images, most of them exported twice
  (Roboflow's "3 versions per image, 50 % horizontal flip" augmentation).
  Two consequences:
  1. If I carved validation out of the train files at random, a flipped twin
     of a validation image would sit in train. The validation numbers would
     leak, and the thresholds are chosen on validation. So `train.py`
     (`prepare_splits`) holds out **15 % of the source images** (790, seed 0)
     with all their copies.
  2. I keep **one copy per source image**. Ultralytics already flips at random
     during training, so the second copy adds no information and only doubles
     epoch time (which matters on CPU, see below).
  Test is untouched: it has one copy per source and shares no source image
  with train.
- **Why it serves the problem:** the visual evidence the guard uses is
  exactly what is labelled, a head with or without a helmet, with both classes
  at instance level. `head` vs `helmet` is the decision; `person` is ignored by
  the decision rule.
- **Where it only resembles the problem:** the images are mostly wide scenes
  from the internet and site cameras, with many small, distant heads. A
  turnstile camera sees one or two people close up, at a fixed angle. The test
  split therefore does not look like production (see next section).

## What the dataset's composition implies

From `artifacts/results.md` (counts after the de-duplication above):

| split | images | `head` | `helmet` | `person` | images with a bare head |
|---|---|---|---|---|---|
| train | 4,479 | 4,237 | 12,601 | 393 | 18.4 % |
| valid | 790 | 637 | 2,283 | 80 | 18.1 % |
| test | 1,766 | 1,803 | 4,863 | 142 | 19.2 % |

| class | median box area (% of image) | 10th–90th pct | boxes < 0.25 % of image ("small") |
|---|---|---|---|
| head | 0.60 % | 0.22 – 1.96 % | 13.9 % |
| helmet | 0.83 % | 0.22 – 4.86 % | 12.6 % |
| person | 7.8 % | 1.5 – 33 % | 0.3 % |

(Histogram: `artifacts/object_sizes.png`.)

- **`person` is thin and incomplete.** 615 boxes in total, against ~26,000
  heads and helmets, and most visible people in the images are *not* boxed as
  `person`. The model is therefore punished for correct person detections, and
  its `person` AP is meaningless (see per-class results). It does not matter
  here: the decision rule ignores `person`.
- **`helmet` outnumbers `head` 3 : 1** (12,601 vs 4,237 in train). The model
  sees three compliant heads for every violation, so I expected `head` to be
  the weaker class. In the end it was not (AP50 0.967 vs 0.977), because
  4,000+ examples are enough for a two-way look-alike problem.
- **19 % of test images contain a bare head; production is assumed 3 %.** A
  raw test-set cost would overweight violations six-fold. That is why every
  euro figure uses the per-class rates re-weighted to 3 %.
- **Objects are small.** The median head is 0.6 % of the image, i.e. a box
  side of about 8 % of the image side (~37 px at the 480 px I trained at), and
  one in seven is below COCO's "small" cut-off. At a turnstile, heads would
  fill a large part of the frame. So production is probably *easier* for
  localisation but has a different viewpoint, distance and lighting. The test
  numbers are neither an upper nor a lower bound on production, and a pilot on
  real turnstile frames is required before shipping.
- **The image-level decision rests on 339 violating test images** (143 on
  validation). One test image moves recall by ~0.3 points, and the validation
  cost curve moves in steps, one per positive image (see operating point).

## Finetuning and baselines

- **Model / task:** object detection, `yolo11n` finetuned from the COCO
  checkpoint. Detection, not classification, because the evidence is a small
  region (a head) inside a scene, and boxes let the guard see *which* worker is
  flagged. `n` rather than `s` because the target runs on an edge box at a
  turnstile, and because of the training budget below.
- **Where and how:** not on Colab in the end. It ran on a cloud container with
  **4 CPU cores and no GPU**, via the same `python main.py train`. On that
  machine one epoch at 640 px would have taken about an hour, so I traded
  resolution for epochs:
  **20 epochs, imgsz 480, batch 16, seed 0, `close_mosaic` 5** (mosaic off for
  the last 5 epochs; the default of 10 is half of such a short run), Ultralytics
  default augmentation. Wall time **3.6 h** of training, plus ~10 min of
  evaluation. Command:
  `EPOCHS=20 IMGSZ=480 CLOSE_MOSAIC=5 WORKERS=4 python main.py train`.
- **The model is under-trained.** Validation mAP50 rose from 0.587 (epoch 1)
  to 0.645 (epoch 20) and mAP50-95 from 0.340 to 0.434, still climbing at the
  last epoch (`artifacts/training_curve.png`). Best weights = last epoch.
  A GPU run at 640 px for 40+ epochs would likely do better, especially on the
  small heads.
- **Thresholds are chosen on the validation split and reported on test**, so
  the test numbers are not tuned on.
- **Baselines on the same test split:**
  1. *COCO `yolo11n` zero-shot.* COCO has no `head` or `helmet` class, so its
     detection AP on these classes is 0 by construction. For the decision, the
     only thing it can do is "flag whenever a person is visible", which is
     what it is scored on.
  2. *Trivial:* predict nothing (= always pass = do nothing), and the
     majority image-level class from train (compliant, 82 %).

**Detection, test split:**

| model | mAP50 | mAP50-95 |
|---|---|---|
| finetuned yolo11n | **0.657** | **0.446** |
| COCO yolo11n zero-shot | 0 | 0 |
| predict nothing | 0 | 0 |

The overall mAP is pulled down by the broken `person` class. On the two
classes that matter, AP50 is 0.97–0.98 (next section).

**Image-level decision, test split** (each policy's single threshold picked on
validation; € at the 3 % base rate):

| policy | threshold | ROC AUC | recall on `head` | precision | TP | FN | FP | TN | € / 1,000 entries |
|---|---|---|---|---|---|---|---|---|---|
| finetuned yolo11n | 0.12 | **0.970** | **93.5 %** | 85.7 % | 317 | 22 | 53 | 1,374 | **222** |
| COCO zero-shot (person ⇒ flag) | 0.82 | 0.626 | 74.9 % | 23.8 % | 254 | 85 | 814 | 613 | 1,644 |
| majority class (compliant) | – | 0.5 | 0 % | – | 0 | 339 | 0 | 1,427 | 2,100 |
| predict nothing (always pass) | – | 0.5 | 0 % | – | 0 | 339 | 0 | 1,427 | 2,100 |

The finetune beats both baselines by a wide margin, as expected. "Predict
nothing" costs 1,000 × 3 % × €70 = €2,100 per 1,000 entries, and its 80.8 %
accuracy shows why accuracy is the wrong metric here. The COCO model is
barely better than doing nothing: "someone is in the frame" says almost
nothing about whether they wear a helmet, and it costs **four times** what a
guard checking every entry would (€369). The finetune alone costs €222, a
**89 % saving over doing nothing**.

## Per-class evaluation and failure modes

Detection on the test split (`artifacts/results.md`, "Per-class results" tab):

| class | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| **head** | 0.934 | **0.931** | 0.967 | 0.656 |
| helmet | 0.951 | 0.933 | 0.977 | 0.667 |
| person | 1.000 | 0.000 | 0.027 | 0.014 |

- **Headline: recall on `head` = 0.93 at box level**, and 93.5 % at image level
  at the cost-optimal threshold (94.3 % once the review band is added, see
  below). That is the number that drives the €70 errors.
- **`person` is not a model failure.** Most people are unlabelled, so correct
  person boxes count as false positives. The model learned to give `person`
  low confidence, and at Ultralytics' operating confidence it reports none.
  Ignored by the decision, as planned.
- **Confusion matrix** (`artifacts/test_confusion_matrix_normalized.png`; true
  columns): of 1,803 true bare heads, 1,761 are found as `head`, **34 (1.9 %)
  are called `helmet`**, the dangerous confusion, and 8 (0.4 %) are missed
  entirely. Ultralytics builds this matrix at a very low confidence cutoff, so
  it understates misses relative to the real operating point. The image-level
  table above is the one to price.
- **What the 22 missed violating images look like** (all test images with a
  bare head but a score ≤ t\* = 0.12; I looked at them with the boxes drawn):
  - *Tiny or distant heads:* a driver inside an excavator or loader cab, a
    worker far down a railway track. **A turnstile camera would not see these.**
  - *Headwear that is not a hard hat:* a red baseball-type cap, a dark
    cowboy-style hat. The model reads "something on the head" as `helmet`.
    **A turnstile would see these**, and it is the most important failure mode
    for production.
  - *Occlusion / partial faces:* a face half hidden behind a pipe or another
    worker. Possible at a crowded turnstile, less so in a single-file lane.
  - *Label noise:* at least one "head" box drawn around a foot.
- **What the 11 most confident false alarms look like** (compliant images
  scoring > 0.70): almost all contain **visible bare heads that the annotators
  did not box** (officials and visitors next to helmeted workers, a man holding
  his helmet in his hand). On those images the model is right and the label is
  wrong. The measured false-alarm rate is therefore pessimistic, and "helmet
  carried in the hand", the case I assumed is most common at a turnstile, is
  flagged correctly.

## Cost matrix and operating point

Costs and base rate: see the table in the first section (all in `decision.py`).

**Method.** For each image the score is the highest `head` confidence. For
every threshold t from 0 to 1 in steps of 0.01, compute the share of violating
and of compliant images on each side, weight by the 3 % base rate, and price
with the cost matrix. This gives € per 1,000 entries against t
(`artifacts/cost_curve.png`, and live in the app). The cheapest t on the
**validation** split is the operating point; it is then evaluated on **test**.

**Result.** The cheapest single threshold on validation is **t\* = 0.12**
(fail if the highest `head` confidence is above 0.12), far below 0.5, as the
35 : 1 cost ratio predicted. On test it costs **€222 per 1,000 entries**
against €2,100 for doing nothing. The validation and test curves have the same
shape: a broad minimum between ~0.05 and ~0.3 (test €222–253), then a steep
rise once t passes ~0.5, and €2,100 by 0.9, where nothing is flagged any more.

A caveat on the exact value: 73 % of images score exactly 0 (no `head` box
above 0.01), and validation has only 143 positive images. So the validation
cost only changes at the scores of individual images, and the optimiser
returns the lowest t of a flat stretch. That explains why the chosen
thresholds in the sensitivity table jump between a few values (0.04, 0.12,
0.24, …). The thresholds are not precise to ±0.01. The *region* is what is
reliable.

**Sensitivity** (operating point re-optimised on validation for each
assumption; full table in `artifacts/sensitivity.csv` and `results.md`). At
the 3 % base rate:

| C_FN (€) | C_FN / C_FP | best single t | band (t_low, t_high] | band € / 1,000 (val) | reviewed | guard checks all € / 1,000 |
|---|---|---|---|---|---|---|
| 5 | 2.5 | 0.56 | (0.24, 0.86] | 46 | 5.1 % | 272 |
| 10 | 5 | 0.28 | (0.12, 0.83] | 62 | 5.6 % | 279 |
| 20 | 10 | 0.24 | (0.12, 0.83] | 91 | 5.6 % | 294 |
| 35 | 17.5 | 0.24 | (0.04, 0.72] | 133 | 6.5 % | 317 |
| **70** | **35** | **0.12** | **(0.04, 0.70]** | **211** | **6.2 %** | **369** |
| 140 | 70 | 0.12 | (0.04, 0.28] | 351 | 4.7 % | 474 |
| 300 | 150 | 0.04 | (0.04, 0.24] | 664 | 4.2 % | 714 |

- **The assumption that would most change the answer is C_FN.** I assumed
  €70 because a miss is a small chance of a very expensive injury plus a fine
  plus the technician's time (first section). The threshold is fairly stable
  between €35 and €140 (t\* 0.12–0.24). **If C_FN drops below about €10
  (C_FN/C_FP below 5), t\* moves up to 0.3–0.6**, i.e. we would let far more
  doubtful frames through. The review band **never disappears** in this range:
  at every C_FN from €5 to €300 it sends 4–7 % of entries to the guard and is
  cheaper than the model alone. In the other direction, as C_FN grows the fail
  threshold drops (0.70 → 0.24), and **above roughly €300 per miss the band
  is only ~7 % cheaper than a guard checking everyone** (€664 vs €714).
- **Base rate.** At 1 % violations the thresholds move up (t\* 0.24, band
  (0.12, 0.83]); at 10 % they move down (t\* 0.04, band (0.04, 0.24]). At 10 %
  combined with C_FN ≥ €140, **a guard checking everyone becomes cheaper than
  the model** (€998 vs €1,015 per 1,000). The model plus band still misses
  ~5.7 % of violations against the guard's 5 %, and when misses are frequent
  and very expensive that gap outweighs the saved review time.

## What a person still has to check

From the "Operating point" tab and `results.md`:

- **Thresholds** (picked on validation by searching all pairs): pass if score
  ≤ **0.04**, review if 0.04 < score ≤ **0.70**, fail if score > 0.70.
- **Volume on test:** **5.6 %** of entries go to review → **67 reviews per
  day** out of 1,200 → **34 in the peak hour** (~600 entries) × 10 s ≈
  **6 minutes of guard time in the peak hour (0.09 of a guard)**.
- **Is it realistic?** Yes, comfortably. A guard is already posted at the
  turnstile, so 6 minutes per peak hour costs nothing extra in headcount. 34
  reviews spread over 600 entries is one stop every ~18 people, well below the
  ~10–15 % at which queues would form. There is room to widen the band
  (e.g. raise t_high) if the site wants fewer automatic stops.
- **Outcomes per 1,000 entries (test, 3 % base rate):** 1.7 violators still
  get through (model misses plus the 5 % the guard misses), and 7.5 compliant
  workers are wrongly stopped. Of all violations, 94.3 % are stopped, by the
  model or by the guard.
- **Priced against the alternatives** (per 1,000 entries, test):

  | policy | € / 1,000 entries | share reviewed | violations stopped |
  |---|---|---|---|
  | do nothing | 2,100 | 0 % | 0 % |
  | guard checks every entry | 369 | 100 % | 95 % |
  | model alone at t\* = 0.12 | 222 | 0 % | 93.5 % |
  | **model + review band (0.04, 0.70]** | **164** | 5.6 % | 94.3 % |

  **Model + review band wins**: €205 per 1,000 entries cheaper than a guard
  checking everyone (−56 %), at almost the same safety (94.3 % vs 95 % of
  violations stopped). It is also €58 cheaper than the model alone, because the
  guard turns most of the doubtful frames into correct decisions. At 1,200
  entries/day that is ≈ €250 per working day, ≈ €5,000 a month, plus a guard
  freed from looking at 1,200 faces.
- **Is the saving worth the deployment risk?** Not yet, on these numbers
  alone. The test set is wide internet scenes, not turnstile frames, and the
  single most important production failure mode (caps and hats that look like
  helmets) is exactly the one the dataset covers worst. The saving justifies a
  pilot, not a switch: run the model in shadow mode next to the guard first
  (next section).

## Limitations & next steps

- **Viewpoint mismatch.** Test images are wide scenes; production is a fixed
  turnstile camera. Next: collect two weeks of frames from the actual
  turnstile (guards are already deciding, so their decisions are labels),
  re-measure the per-class rates, re-run the sweep. Ship in "shadow mode"
  (model scores, guard still decides) until then.
- **Base rate is assumed.** The same shadow period measures it directly.
- **C_FN is a guess built from small probabilities.** The sensitivity table
  shows how far it can move before the decision changes; the safety
  technician should sign off the figure, not the data team.
- **Image score uses only the top `head` box.** Caps, hoods and hats that
  resemble helmets are the main risk (seen in the misses above). A dedicated
  "non-hard-hat headwear" class, or more such examples labelled as `head`,
  would help most.
- **Under-trained, low-resolution model.** 20 epochs at 480 px on CPU, with
  validation mAP still rising. Retraining on a GPU at 640 px for 40+ epochs is
  cheap and should mostly help the small heads (which matter less at a
  turnstile anyway).
- **Label noise in both directions.** Unlabelled bare heads make the measured
  false-alarm rate pessimistic; a few wrong `head` boxes make the measured
  misses slightly pessimistic too. The turnstile pilot should use carefully
  labelled frames.
- **Thresholds rest on 143 violating validation images**, so they are only
  reliable to a region, not to two decimals. Re-pick them on pilot data.
- **Reviewer accuracy (95 %) is assumed**, and review fatigue at peak is not
  modelled.

## Generative AI use disclosure

I used Claude Code (Anthropic) to write the code (`train.py`, `decision.py`,
`app.py`, `colab_train.ipynb`), to run the training and evaluation, and for a
first draft of this report's structure, cost reasoning and the reading of the
results. **[TODO (student): state what you changed, which assumptions you
checked or replaced, and confirm the analysis and conclusions are yours.]**
