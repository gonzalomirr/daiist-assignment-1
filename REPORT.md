# Assignment 1 Report

- **Name**: _TODO_
- **Student ID**: _TODO_
- **Email**: gmir.ieu2022@student.ie.edu
- **Group**: _TODO_ [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

All numbers below are produced by `uv run python main.py train` (printed to the
console and saved in `artifacts/metrics.json` and `artifacts/ablation.csv`).

## Dataset

**UCI Bank Marketing, `bank-additional-full.csv`** (Moro, Cortez & Rita, 2014).
It holds the telemarketing campaigns of a Portuguese bank selling term
deposits, May 2008 to November 2010. The file is committed as
`data/bank-additional-full.csv` (5.6 MB). The UCI site was not reachable from
my build environment, so I took the file from a public GitHub mirror and checked
it against the UCI documentation: 41,188 rows × 21 columns, `;`-separated, same
columns and same class balance.

- **One row = one client in one campaign**, described at their *last* contact:
  client profile (age, job, marital, education, default/housing/personal loan),
  contact details (phone type, month, weekday, call duration, number of attempts
  in this campaign), history from earlier campaigns (`pdays`, `previous`,
  `poutcome`) and five macro-economic indicators (Euribor 3m, employment
  variation rate, CPI, consumer confidence, number employed).
- **Target**: `y` = did the client subscribe a term deposit (11.3% yes overall).
- **Why this dataset**: the obvious features are traps (one leaks the target,
  one is a sentinel value, one is only known after the fact), and the data
  spans the 2008–2010 financial crisis. That makes the split strategy matter
  as much as the features do.

**Recovering time.** The file has `month` but no year. The authors state the
rows are in date order, so I rebuild the calendar month: each time the month
name changes we're in a new month, and each time the month number goes down
(e.g. dec → mar) we're in a new year (`pipeline.load_raw`). This gives exactly
26 month-blocks from May 2008 to Nov 2010, matching the documented range.

## Business / real-life framing

**Decision supported:** at the start of each monthly campaign, the call centre
manager builds the **call list**: which clients from the eligible pool get
called at all. Every client put on the list costs agent time. Every client left
off is a deposit we might have missed. The model scores each client with
P(subscribe), and we call the client when the expected margin beats the cost.

**Economics (hypothetical but explicit, `pipeline.py`):**

| Quantity | Value | Reasoning |
|---|---|---|
| Cost of putting one client on the list | €12 | ~2.6 call attempts per client (mean of `campaign` in training) × ~8 min incl. dialing/no-answers/wrap-up at ~€30/h ≈ €10, plus ~€2 for annoyance/complaint/opt-out risk |
| Value of one new deposit | €40 | ~€4,000 average deposit × ~1% net interest margin over a 1-year term |
| **Break-even probability** | **0.30** | Call iff p × 40 > 12 |

How the framing drives the pipeline:

1. **What is known when the decision is made.** The list is built *before*
   anyone is called, so a feature is allowed only if it is known at that moment:
   - `duration` (call length) is dropped. It is only known after the call, and
     `duration = 0` means `y = 0`. Ablation H shows what it would do: validation
     ROC-AUC jumps to 0.80, a result that could never be reproduced in production.
   - `campaign` (number of attempts) and `day_of_week`/`month` describe the
     *last* call, so they're only known after calling ends. Worse, clients
     who say yes stop being called, so a high attempt count partly *encodes*
     "no". I dropped them **even though they help on validation** (ablation G:
     AUC 0.706 vs 0.699). That gain comes from information the manager
     won't have when building the list.
2. **Split by time, not at random.** The model will be used on *future*
   campaigns, and the data has an extreme regime change (see the dashboard's
   monthly charts). In May–Nov 2008 the bank made 3,600–7,800 calls a month at
   3–6% conversion. From mid-2009 it made a few hundred carefully chosen calls a
   month at 40–60%. A random split would mix 2010 clients into training and
   hide the hardest part of the problem: extrapolating to a regime the
   model has barely seen. Split (`pipeline.time_split`):
   - **Train** May 2008 – Dec 2009: 39,118 clients, 9.1% subscribe
   - **Validation** Mar – Jul 2010: 1,190 clients, 54.8% subscribe
   - **Test** Aug – Nov 2010: 868 clients, 48.5% subscribe (the "next campaign")
3. **Metric for tuning = log-loss, metric for the decision = profit.** The
   threshold is a cost ratio, which only makes sense if the probabilities are
   *calibrated*, not just well-ranked. So C and the learning rates are tuned on
   validation **log-loss** (a proper scoring rule), not accuracy or AUC.
   Profit in € at the chosen threshold is the business metric. ROC-AUC/PR-AUC
   are reported as threshold-free ranking quality.
4. **Threshold = 0.30 from the economics, checked on validation.** The
   models' mean predicted probability on validation is 0.516 vs an actual
   0.548, and the reliability plot in the dashboard follows the diagonal, so
   using the break-even rule is reasonable. I also computed the threshold that
   maximizes validation profit. It is **not stable**: the validation profit
   curve is flat from 0.10 to 0.34 (every value within ~€75, i.e. about two deposits),
   so scikit-learn's argmax is 0.31 but manual PyTorch's is 0.23. The two
   models have practically identical probabilities, so this is noise, not
   signal. I therefore use the cost-derived 0.30 for all three models.
   *Honesty note:* my first version used the per-model validation argmax, and
   I had already seen test results when I switched. The reason for the switch
   is visible on validation alone (the instability above), but it is a mild
   reuse of the test set.

## Data preparation & feature engineering

All transforms are row-wise (`pipeline.engineer`). The one fitted step
(standardization + one-hot, `pipeline.make_preprocessor`) is **fit on the
training months only** and saved, so validation/test/app inputs go through
exactly the same transformation.

| Decision | Why |
|---|---|
| Drop 12 exact duplicate rows | Identical in every column *including call duration to the second*. Two real clients matching like that is implausible, so I treat them as logging duplicates |
| Drop `duration`, `campaign`, `day_of_week`, `month` | Not known when the call list is built (see framing) |
| `pdays` → `prev_contacted` (pdays ≠ 999) + `prev_recent_week` (≤ 7 days) | 999 is a sentinel for "never contacted" (96% of rows). Left raw, a linear model reads never-contacted clients as "contacted 999 days ago", which is a huge number that dominates the scale |
| `previous` clipped at 3 | Tiny counts with a thin tail (max 7). Beyond 3 there are almost no rows to estimate from |
| `age` → 6 bands (≤25, 26–30, 31–40 ref., 41–50, 51–60, 60+) | The effect is U-shaped: students and retirees convert most. A single linear age coefficient can't represent that; bands give one level per life stage |
| `default` → `default_unknown` (≠ "no") | Only 3 "yes" rows. The informative contrast is "declared no default" (13% convert) vs "wouldn't say" (5%) |
| `education`: illiterate → basic.4y; `marital`: unknown → single | 18 and 80 rows cannot support their own coefficient |
| `housing`, `loan` → yes-flags | "unknown" is the same 990 rows for both and carries no signal |
| Macro: keep `euribor3m`, `cons_price_idx`, `cons_conf_idx`; drop `emp_var_rate`, `nr_employed` | `euribor3m`, `emp_var_rate` and `nr_employed` are 0.91–0.97 correlated (all three track the crisis). Keep one rate plus the two indicators that carry different information |
| One-hot with an explicit reference level per category | Avoids the dummy-variable trap and keeps the intercept interpretable (a 31–40 married admin with high-school education, no previous contact) |
| Standardize numeric columns | L2 penalizes all coefficients equally, so they need a common scale. Gradient descent in the PyTorch versions also converges far faster on a well-conditioned problem |

Final design matrix: **36 columns**.

**Feature ablation** (scikit-learn, C = 1, trained on train, scored on
**validation only**, so test played no part in choosing features):

| Feature set | Cols | Val ROC-AUC | Val PR-AUC | Val log-loss | Val mean p (actual 0.548) |
|---|---|---|---|---|---|
| A. naive: raw columns one-hot/scaled, no macro | 41 | 0.672 | 0.705 | 0.879 | 0.282 |
| B. naive + all 5 macro indicators | 46 | 0.692 | 0.708 | 0.751 | 0.331 |
| C. engineered, no macro | 38 | 0.678 | 0.710 | 0.860 | 0.296 |
| **D. engineered + 3 macro (FINAL)** | 41* | **0.699** | **0.713** | **0.633** | **0.510** |
| E. engineered + all 5 macro | 43 | 0.693 | 0.709 | 0.758 | 0.324 |
| F. engineered + month one-hot | 51 | 0.629 | 0.657 | 0.723 | 0.450 |
| G. engineered + attempts + weekday (excluded: not known in advance) | 47 | 0.706 | 0.724 | 0.629 | 0.512 |
| H. LEAKY: + call duration | 47 | 0.797 | 0.804 | 0.787 | 0.270 |

\*The ablation keeps every one-hot level, while the final preprocessor drops
reference levels, giving 36 columns. The fitted model is the same up to
regularization.

What I learned from it:
- **The macro indicators are what make the time split work.** Without them
  (A, C) the model predicts ~0.29 on months where 0.55 actually convert,
  because it learnt 2008's 4% world. With them, a lower Euribor/later-crisis
  regime moves the intercept, and the mean prediction goes to 0.51.
  Log-loss improves from 0.86 to 0.63, the biggest single gain of any decision.
- **Fewer macro variables beat more** (D vs E). With all five collinear
  indicators the coefficients partly cancel each other out on the training
  range, and that doesn't extrapolate to 2010 (mean p falls back to 0.32).
- **Engineering on its own adds little ranking power** (A→C: +0.006 AUC). Its
  value is mostly correctness and interpretability: no sentinel distortion,
  a U-shaped age effect, no leaky inputs. I was hoping for more here.
- **Month dummies hurt under a time split** (F). In training, "March" or
  "December" means a tiny, hand-picked, high-converting batch of calls, not a
  seasonal effect. The model learns campaign intensity and mislabels it as
  seasonality.

Covariate shift I could not engineer away: 34–37% of 2010 clients had been
contacted in an earlier campaign vs 2% in training, and 81% are on cellular
vs 63%. The 2010 campaigns were mostly re-contacting warm leads.

## Modeling: three implementations, one model

**Model:** L2-regularized **logistic regression**. The target is binary, and
the decision rule needs probabilities. All three implementations minimize the
same objective on the same 39,118 × 36 training matrix:

  mean BCE(y, σ(w·x + b)) + (λ/2)·‖w‖², with the bias unpenalized.

scikit-learn writes it as ½‖w‖² + C·Σ BCE. Dividing by C·n gives the
form above with **λ = 1/(C·n)**, so the PyTorch versions use exactly
scikit-learn's regularization (`models.sklearn_C_to_lambda`).

| | How it's trained | Tuned (on val log-loss) |
|---|---|---|
| scikit-learn | `LogisticRegression` (L-BFGS, tol 1e-8) | C ∈ {0.01, 0.1, 1, 10} → **10** (0.6324; C=1 gives 0.6330, i.e. regularization barely matters with 39k rows and 36 columns) |
| Manual PyTorch | raw tensors `w, b` with `requires_grad`, BCE written by hand via softplus, `loss.backward()`, update `w -= lr·w.grad` under `no_grad()`, zero the grads; **full-batch** GD, 3,000 steps | lr ∈ {0.1, 0.5, 2.0} → **2.0** |
| Standard PyTorch | `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, `torch.optim.Adam` (weight decay = λ on weights only), `DataLoader` mini-batches of 512, 40 epochs | lr ∈ {0.001, 0.01, 0.1} → **0.1** |

**Test results (Aug–Nov 2010, 868 clients, 421 subscribed; threshold 0.30 for every model):**

| Model | ROC-AUC | PR-AUC | Log-loss | Brier | Mean p | Calls | Profit |
|---|---|---|---|---|---|---|---|
| Naive baseline (train rate 0.091 for everyone) | 0.500 | 0.485 | 1.211 | 0.405 | 0.091 | 0 | €0 |
| Policy: call everyone (status quo) | – | – | – | – | – | 868 | €6,424 |
| scikit-learn | 0.751 | 0.723 | 0.596 | 0.204 | 0.511 | 744 | **€7,272** |
| Manual PyTorch | 0.751 | 0.724 | 0.595 | 0.204 | 0.510 | 743 | **€7,284** |
| Standard PyTorch | 0.755 | 0.722 | 0.596 | 0.204 | 0.540 | 766 | €7,088 |

Business reading: on the test campaign, scikit-learn's call list skips 124
clients (14%), loses 16 of 421 deposits, and earns **€848 (+13%) more than
calling everyone**. The naive baseline is the instructive failure. Trained on
history, it says the average client converts at 9%, which is below break-even,
so its advice is to stop the campaign, and that would have given up €6,424.
On **validation** the model adds nothing: €11,776 vs €11,800 for calling
everyone. Even the lowest-scored validation clients converted at ~30%, right at
break-even. I report this rather than hide it: the model's advantage depends on
there being a group of genuinely poor prospects in the list.

**Do the three agree?**

| vs scikit-learn | Max \|Δ coef\| | Mean \|Δ p\| test | Max \|Δ p\| test | Same call/skip decision | Final train objective (sklearn 0.265655) |
|---|---|---|---|---|---|
| Manual PyTorch | 0.211 | 0.0046 | 0.048 | 99.9% | 0.265662 |
| Standard PyTorch | 0.327 | 0.054 | 0.256 | 92.2% | 0.271432 |
| Diagnostic: same `nn.Module` + Adam, full batch, 3,000 steps (not saved) | 0.026 | 0.0077 | – | – | – |

- **Manual PyTorch ≈ scikit-learn.** The training objective matches to the
  5th decimal, and 33 of 36 coefficients match to within 0.007. The three that
  don't (`prev_contacted` −0.21, `poutcome_success` +0.16, `prev_recent_week`
  +0.05) are **nearly collinear**: of the 790 previously contacted training
  clients, 725 had `poutcome = success`, so `prev_contacted` and
  `poutcome_success` differ on only 65 rows. Along that direction the loss is
  almost flat, so plain gradient descent moves along it very slowly while the
  overall prediction for a previously-successful client barely changes. With
  C = 10 the penalty is too weak to pin it down. Both reach the same optimum
  in prediction space, but not in coefficient space.
- **Standard PyTorch is close but visibly different**, and I looked into why
  rather than accepting it. Its objective is 2% higher, its mean prediction is
  0.03 higher (larger intercept), and 8% of test decisions differ. Mini-batch
  Adam with a fixed lr = 0.1 never settles: each batch of 512 pulls the
  weights in a slightly different direction, so the run ends at a point
  scattered around the optimum. To test that this is optimizer noise and not
  a bug in my model/loss/weight decay, `train.py` re-runs the **same**
  `train_standard` code with one full batch. It lands within 0.026 of
  scikit-learn's coefficients. So the gap comes from the stochastic optimizer,
  not from the code. A smaller lr wasn't the fix within 40 epochs: lr = 0.01
  had *worse* validation log-loss (0.641), because it hadn't converged yet.
  A learning-rate schedule would fix it, but that is one more hyperparameter
  than the brief allows.
- Its slightly higher test AUC (0.755) comes from that noise, not from a
  better model. It has the worst training objective of the three.

## Limitations & next steps

1. **The economics are assumptions.** €12 per client and €40 per deposit
   are my estimates. Profit and the 0.30 threshold move with them (the
   dashboard has inputs for both). *Next:* get the real agent cost per hour,
   attempts per client and deposit margin from the bank, and run a
   sensitivity table.
2. **Thin, shifted evaluation data.** Validation and test are 1,190 and 868
   clients, from a regime that is only ~5% of the training rows. €848 is a
   difference of a few dozen deposits, and I have no confidence interval.
   *Next:* bootstrap the test profit, and do rolling-origin evaluation (train
   up to month m, test on m+1, for every month in 2009–2010) instead of one
   cut.
3. **The model is trained mostly on a world that no longer exists.** 2008
   rows dominate the fit, and only the macro features carry it into 2010.
   Training on recent data only did *not* help (fewer rows, worse validation
   log-loss in my exploration). *Next:* recency-weighted training, or refit
   on train + validation before deploying for the next campaign, which adds the
   most relevant 1,190 rows.
4. **Selection bias.** We only see clients the bank chose to call, and in
   2010 it was clearly calling warm leads (37% previously contacted vs 2%).
   Predictions for "cold" clients outside that selection are untested.
   *Next:* a small random holdout of the eligible pool in each campaign.
5. **Macro indicators are used as of the call month.** In reality CPI and
   confidence are published with a lag, so the list builder would use the
   previous month's values. *Next:* lag those features by one month.
6. **Unused information I left out on purpose.** Within-campaign features
   (attempt count, weekday) do add a little (ablation G). They would be valid
   for a *different* decision, "should we make attempt k+1?", but only with
   one row per attempt, which this dataset doesn't have.
7. **Linear model limits.** Interactions (e.g. previous success × cellular)
   aren't modelled. I kept the brief's model fixed and only engineered features.

## Generative AI use disclosure

_TODO (student): review and adjust this section so it accurately reflects your
own process before submitting._

I used Claude Code (Anthropic) as a coding assistant for this assignment. It
wrote most of the code (`pipeline.py`, `models.py`, `train.py`, `app.py`),
ran the exploratory analyses and ablations, and drafted this report's text.
I reviewed every decision and number and can explain each one.
