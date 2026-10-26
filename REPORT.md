# Assignment 2 Report

> **DRAFT.** Sections marked **[TODO after training]** need the numbers from
> `artifacts/results.md`, which `main.py train` writes. Every figure and argument
> here has to be one you can defend without the code in front of you: rewrite it in
> your own words and change any assumption you don't agree with (the costs live in
> `decision.py`, `CostModel`).

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
- 7,035 images of construction workers (a Roboflow copy of the public *Hard
  Hat Workers* dataset from Northeastern University, China). Classes: `head`
  (bare head), `helmet` (head with a hard hat), `person`. Licence CC BY 4.0.
- **Why it serves the problem:** the visual evidence the guard uses is
  exactly what is labelled, a head with or without a helmet, with both classes
  at instance level. `head` vs `helmet` is the decision; `person` is ignored by
  the decision rule.
- **Where it only resembles the problem:** the images are mostly wide scenes
  from the internet and site cameras, with many small, distant heads. A
  turnstile camera sees one or two people close up, at a fixed angle. The test
  split therefore does not look like production (see next section).

## What the dataset's composition implies

**[TODO after training]** paste the composition tables from
`artifacts/results.md` (images and instances per class per split, share of
images with a bare head, object size per class; histogram in
`artifacts/object_sizes.png`) and interpret them. Points to check and argue:

- Is `person` thin? (The original release labelled `person` on only a small
  fraction of images.) If it is in the low hundreds, its AP cannot be trusted,
  and it does not matter here because the decision ignores it. Say so.
- `head` vs `helmet` balance. If `helmet` outnumbers `head` several times, the
  model sees fewer violations to learn from: expect lower `head` recall.
- Share of test images containing a bare head vs the assumed 3 % in
  production. This is why every euro figure is re-weighted to the base rate.
- Object size: most boxes are small (median well under 1 % of the image). At a
  turnstile heads are large, so production will probably be *easier* for
  localisation but has a different viewpoint and lighting. The test numbers
  are therefore neither an upper nor a lower bound on production; a pilot on
  real turnstile frames is required before shipping.
- How many test images there are, and so how many violating images decide the
  cost curve: with a few hundred positives, one image moves recall by
  ~0.3 points.

## Finetuning and baselines

- **Model / task:** object detection, `yolo11n` finetuned from the COCO
  checkpoint. Detection, not classification, because the evidence is a small
  region (a head) inside a scene, and boxes let the guard see *which* worker is
  flagged. `n` rather than `s` because the target runs on an edge box at a
  turnstile, and the gap on this task is small.
- **Where and how:** Google Colab, T4 GPU, via `colab_train.ipynb`, which only
  clones the repo and runs `python main.py train`. Settings: 40 epochs
  (early stop with patience 10), imgsz 640, batch 16, seed 0, Ultralytics
  default augmentation. **[TODO: actual epochs run and wall time]**
- **Thresholds are chosen on the validation split and reported on test**, so
  the test numbers are not tuned on.
- **Baselines on the same test split:**
  1. *COCO `yolo11n` zero-shot.* COCO has no `head` or `helmet` class, so its
     detection AP on these classes is 0 by construction. For the decision, the
     only thing it can do is "flag whenever a person is visible", which is
     what it is scored on.
  2. *Trivial:* predict nothing (= always pass = do nothing), and the
     majority image-level class from train.

**[TODO after training]** paste both baseline tables (detection mAP50 /
mAP50-95, and the image-level table: AUC, precision, recall, € per 1,000) and
comment: does the finetune beat both, by how much, and is that what you
expected? Note that "predict nothing" costs exactly 1,000 × 3 % × €70 = €2,100
per 1,000 entries; anything that does not beat this is worse than no camera.

## Per-class evaluation and failure modes

**[TODO after training]** from `artifacts/results.md` and the "Per-class
results" tab:

- Per-class precision, recall, AP50, mAP50-95 for `head`, `helmet`, `person`.
- **Headline = recall on `head`**, because that is the expensive error.
- The detection confusion matrix (`artifacts/test_confusion_matrix_normalized.png`):
  how often a bare head is predicted as `helmet` (the dangerous confusion),
  and how often it is missed entirely (background). Those are the FNs that cost €70.
- Look at a few missed `head` images in the app and name the pattern (tiny
  distant heads, caps or hoods that look like helmets, back of the head,
  occlusion). Say which of those a turnstile camera would also see.

## Cost matrix and operating point

Costs and base rate: see the table in the first section (all in `decision.py`).

**Method.** For each image the score is the highest `head` confidence. For
every threshold t from 0 to 1 in steps of 0.01, compute the share of violating
and of compliant images on each side, weight by the 3 % base rate, and price
with the cost matrix. This gives € per 1,000 entries against t
(`artifacts/cost_curve.png`, and live in the app). The cheapest t on the
**validation** split is the operating point; it is then evaluated on **test**.

**[TODO after training]** t* = …, cost on test = € … per 1,000 vs €2,100 for
doing nothing. Then the sensitivity table (`artifacts/sensitivity.csv`, also
in `results.md`):

- **The assumption that would most change the answer is C_FN** (and with it
  the ratio C_FN / C_FP). Read off the table: below which C_FN does the
  optimal threshold move up noticeably, and below which does the review band
  disappear (cost of reviewing larger than the errors it prevents)? Write it as
  "I assumed €70 because …, and the answer changes if it drops below €…".
- The base rate: at 1 % vs 10 %, how much do the thresholds move?

## What a person still has to check

**[TODO after training]** fill from the "Operating point" tab and
`results.md`:

- Thresholds: pass if score ≤ t_low, review if t_low < score ≤ t_high, fail
  if score > t_high (both picked on validation by searching all pairs).
- Share sent to review → reviews per day out of 1,200 → reviews in the peak
  hour (~600 entries) × 10 s each → guards needed in the peak hour.
- **Is it realistic?** A guard is already posted at the turnstile, so up to
  about half a guard-hour in the peak hour costs nothing extra in headcount.
  Each review also holds that turnstile for ~10–15 s; if more than ~10–15 %
  of entries are reviewed at peak, queues form. If the band sends more than
  that, either widen the fail side (accept more false stops, €2 each) or add
  a second turnstile lane.
- Priced against the alternatives (per 1,000 entries, from the app):
  *do nothing* = €2,100; *guard checks every entry* = €0.25 × 1,000 + review
  outcomes ≈ €369 (and 600 × 10 s ≈ 1.7 guards fully busy in the peak hour); *model
  alone at t\** = …; *model + review band* = …. **[TODO]** Which wins, and by how
  much? Is the saving over "guard checks everything" worth the deployment risk
  given the dataset mismatch?

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
- **Image score uses only the top `head` box.** Hats, hoods and caps that
  resemble helmets are the risk; a dedicated "non-hard-hat headwear" class
  would help.
- **Reviewer accuracy (95 %) is assumed**, and review fatigue at peak is not
  modelled.
- **[TODO]** anything specific the per-class results show.

## Generative AI use disclosure

I used Claude Code (Anthropic) to write the code (`train.py`, `decision.py`,
`app.py`, `colab_train.ipynb`) and a first draft of this report's structure
and cost reasoning. **[TODO: state what you changed, which assumptions you
checked or replaced, and confirm the analysis and conclusions are yours.]**
