"""
Assignment 1 -- training pipeline. Run with `uv run python main.py train`.

Bank telemarketing: decide which clients to call for a term-deposit campaign.
Logistic regression trained three ways (scikit-learn, manual PyTorch loop,
nn.Module + torch.optim) on the same time-based split, compared against a
naive baseline, and saved to artifacts/ for app.py. app.py never trains.

Steps:
 1. load + clean + engineer features (pipeline.py)
 2. time-based split: train 2008-05..2009-12, val 2010-03..07, test 2010-08..11
 3. feature-set ablation (scikit-learn only, validation metrics only)
 4. fit the preprocessor on train only
 5. train the three implementations, tuning only C / learning rate on val
 6. decision threshold = cost / value break-even, checked on validation
 7. evaluate on test, compare implementations, save everything
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import models as M
import pipeline as P

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
P.ARTIFACTS.mkdir(exist_ok=True)

C_GRID = [0.01, 0.1, 1.0, 10.0]
MANUAL_LR_GRID = [0.1, 0.5, 2.0]
MANUAL_EPOCHS = 3000
STANDARD_LR_GRID = [0.001, 0.01, 0.1]
STANDARD_EPOCHS = 40
THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.01), 2)


# ---------------------------------------------------------------------------
# 1-2. Data, features, split
# ---------------------------------------------------------------------------
raw = P.clean(P.load_raw())
features = P.engineer(raw)
y = raw["y"].to_numpy()
tr, va, te = P.time_split(raw)

print("Rows  train/val/test:", tr.sum(), va.sum(), te.sum())
print("Conversion rate     : "
      f"{y[tr].mean():.3f} / {y[va].mean():.3f} / {y[te].mean():.3f}")


def metrics(y_true, p, threshold):
    call = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, call, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
        "mean_pred": float(p.mean()),
        "threshold": float(threshold),
        "calls": int(call.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "profit": P.profit(y_true, call),
    }


def best_threshold(y_true, p):
    """Threshold that maximises campaign profit on the given data (diagnostic)."""
    profits = [P.profit(y_true, p >= t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(profits))])


# ---------------------------------------------------------------------------
# 3. Feature-set ablation: what did each feature-engineering decision buy?
#    Judged on validation only, so the test months stay untouched.
# ---------------------------------------------------------------------------
RAW_NUM = ["age", "pdays", "previous"]
RAW_CAT = ["job", "marital", "education", "default", "housing", "loan",
           "contact", "poutcome"]
MACRO_ALL = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]


def naive_matrix(num, cat, frame):
    ct = ColumnTransformer([
        ("n", StandardScaler(), num),
        ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    ct.fit(frame[tr])
    return ct.transform(frame)


def engineered_matrix(extra_num=(), extra_cat=(), drop_num=()):
    frame = features.copy()
    for c in list(extra_num) + list(extra_cat):
        frame[c] = np.log1p(raw[c]) if c == "campaign" else raw[c]
    ct = ColumnTransformer([
        ("n", StandardScaler(), [c for c in P.NUMERIC if c not in drop_num] + list(extra_num)),
        ("b", "passthrough", P.BINARY),
        ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         P.CATEGORICAL + list(extra_cat)),
    ])
    ct.fit(frame[tr])
    return ct.transform(frame)


ablation_sets = {
    "A. naive: raw columns one-hot/scaled, no macro": naive_matrix(RAW_NUM, RAW_CAT, raw),
    "B. naive + all 5 macro indicators": naive_matrix(RAW_NUM + MACRO_ALL, RAW_CAT, raw),
    "C. engineered, no macro": engineered_matrix(drop_num=["euribor3m", "cons_price_idx", "cons_conf_idx"]),
    "D. engineered + 3 macro  (FINAL)": engineered_matrix(),
    "E. engineered + all 5 macro": engineered_matrix(extra_num=["emp_var_rate", "nr_employed"]),
    "F. engineered + month one-hot": engineered_matrix(extra_cat=["month"]),
    # Known only after calling ends (see pipeline.engineer): measured, not used.
    "G. engineered + log(campaign attempts) + weekday": engineered_matrix(
        extra_num=["campaign"], extra_cat=["day_of_week"]),
    # Shown only to make the leakage visible -- never used for the real model.
    "H. LEAKY: naive + macro + call duration": naive_matrix(
        RAW_NUM + MACRO_ALL + ["duration"], RAW_CAT, P.clean(P.load_raw(), keep_duration=True)),
}
ablation_rows = []
for name, X_ab in ablation_sets.items():
    m = LogisticRegression(C=1.0, max_iter=5000).fit(X_ab[tr], y[tr])
    p_va = m.predict_proba(X_ab[va])[:, 1]
    ablation_rows.append({
        "feature_set": name, "n_columns": X_ab.shape[1],
        "val_roc_auc": roc_auc_score(y[va], p_va),
        "val_pr_auc": average_precision_score(y[va], p_va),
        "val_log_loss": log_loss(y[va], p_va),
        "val_mean_pred": p_va.mean(),
    })
ablation = pd.DataFrame(ablation_rows)
ablation.to_csv(P.ARTIFACTS / "ablation.csv", index=False)
print("\nFeature ablation (validation, sklearn C=1):")
print(ablation.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Final preprocessing: fit on train only, apply to everything.
# ---------------------------------------------------------------------------
pre = P.make_preprocessor().fit(features[tr])
X = pre.transform(features).astype(np.float32)
feature_names = list(pre.get_feature_names_out())
X_tr, X_va, X_te = X[tr], X[va], X[te]
y_tr, y_va, y_te = y[tr], y[va], y[te]
n_train = len(y_tr)
print(f"\nDesign matrix: {X.shape[1]} columns")

results = {}       # name -> {"val": metrics, "test": metrics, ...}
pred_va, pred_te = {}, {}


# ---------------------------------------------------------------------------
# Naive baseline: every client gets the training conversion rate (0.091). It
# cannot rank anyone; under the break-even rule it calls nobody. The other
# naive *policy*, the status quo "call everyone on the list", is reported as
# call_everyone below.
# ---------------------------------------------------------------------------
base_rate = float(y_tr.mean())
pred_va["Baseline"] = np.full(len(y_va), base_rate)
pred_te["Baseline"] = np.full(len(y_te), base_rate)


# ---------------------------------------------------------------------------
# 5a. scikit-learn. Only C is tuned, by validation log-loss (a proper scoring
#     rule: we need probabilities, not just a ranking, to set a cost threshold).
# ---------------------------------------------------------------------------
c_scores = {}
for C in C_GRID:
    m = LogisticRegression(C=C, max_iter=10_000, tol=1e-8).fit(X_tr, y_tr)
    c_scores[C] = log_loss(y_va, m.predict_proba(X_va)[:, 1])
best_C = min(c_scores, key=c_scores.get)
print("\nsklearn val log-loss by C:", {k: round(v, 4) for k, v in c_scores.items()},
      "-> C =", best_C)
sk = LogisticRegression(C=best_C, max_iter=10_000, tol=1e-8).fit(X_tr, y_tr)
pred_va["scikit-learn"] = sk.predict_proba(X_va)[:, 1]
pred_te["scikit-learn"] = sk.predict_proba(X_te)[:, 1]

# The same regularization strength, translated for the PyTorch objective.
lam = M.sklearn_C_to_lambda(best_C, n_train)
print(f"lambda for PyTorch = 1 / (C * n) = {lam:.3e}")


# ---------------------------------------------------------------------------
# 5b. Manual PyTorch loop. Only the learning rate is tuned.
# ---------------------------------------------------------------------------
manual_runs = {}
for lr in MANUAL_LR_GRID:
    w, b, losses = M.train_manual(X_tr, y_tr, lam, lr, MANUAL_EPOCHS, seed=SEED)
    params = {"w": w, "b": b}
    p = M.predict_manual(params, X_va)
    score = log_loss(y_va, p) if np.isfinite(losses[-1]) else np.inf
    manual_runs[lr] = (score, params, losses)
best_manual_lr = min(manual_runs, key=lambda k: manual_runs[k][0])
print("manual val log-loss by lr:", {k: round(v[0], 4) for k, v in manual_runs.items()},
      "-> lr =", best_manual_lr)
_, manual_params, manual_losses = manual_runs[best_manual_lr]
pred_va["Manual PyTorch"] = M.predict_manual(manual_params, X_va)
pred_te["Manual PyTorch"] = M.predict_manual(manual_params, X_te)


# ---------------------------------------------------------------------------
# 5c. Standard PyTorch (nn.Module + Adam + DataLoader). Only lr is tuned.
# ---------------------------------------------------------------------------
standard_runs = {}
for lr in STANDARD_LR_GRID:
    model, losses = M.train_standard(X_tr, y_tr, lam, lr, STANDARD_EPOCHS, seed=SEED)
    standard_runs[lr] = (log_loss(y_va, M.predict_standard(model, X_va)), model, losses)
best_std_lr = min(standard_runs, key=lambda k: standard_runs[k][0])
print("standard val log-loss by lr:", {k: round(v[0], 4) for k, v in standard_runs.items()},
      "-> lr =", best_std_lr)
_, std_model, std_losses = standard_runs[best_std_lr]
pred_va["Standard PyTorch"] = M.predict_standard(std_model, X_va)
pred_te["Standard PyTorch"] = M.predict_standard(std_model, X_te)

# Diagnostic only (not saved, not used by the app): if the mini-batch model
# disagrees with scikit-learn, is it a bug or just optimizer noise? Same
# nn.Module + Adam code, but one batch = the whole training set and more
# steps. If THIS lands on sklearn's coefficients, the gap is mini-batch noise.
diag_model, _ = M.train_standard(X_tr, y_tr, lam, lr=0.01, epochs=3000,
                                 batch_size=None, seed=SEED)
diag_coef_gap = float(np.abs(diag_model.linear.weight.detach().numpy().ravel()
                             - sk.coef_.ravel()).max())
diag_prob_gap = float(np.abs(M.predict_standard(diag_model, X_te)
                             - sk.predict_proba(X_te)[:, 1]).mean())
print(f"diagnostic full-batch Adam: max |coef - sklearn| = {diag_coef_gap:.4f}, "
      f"mean |p - sklearn| on test = {diag_prob_gap:.5f}")


# ---------------------------------------------------------------------------
# 6-7. Thresholds from validation profit, then evaluate on test.
# ---------------------------------------------------------------------------
# The decision threshold comes from the business case: call when
# p * DEPOSIT_VALUE > CALL_COST, i.e. p > BREAK_EVEN_P (0.30). That rule is
# only valid if the probabilities are calibrated, so it is checked on
# validation (mean prediction vs actual rate, reliability plot in the app).
# The "best" threshold on validation profit is also recorded, but the
# validation profit curve is flat between ~0.1 and ~0.34 (+/- one or two
# deposits), so its argmax is noise: two models with near-identical
# probabilities pick 0.31 and 0.23. Same rule for every model keeps the
# three-way comparison clean.
call_everyone = {"val": P.profit(y_va, np.ones(len(y_va), bool)),
                 "test": P.profit(y_te, np.ones(len(y_te), bool))}
for name in pred_va:
    results[name] = {
        "val": metrics(y_va, pred_va[name], P.BREAK_EVEN_P),
        "test": metrics(y_te, pred_te[name], P.BREAK_EVEN_P),
        "val_argmax_threshold": best_threshold(y_va, pred_va[name]),
    }
    t_arg = results[name]["val_argmax_threshold"]
    results[name]["test_at_val_argmax"] = metrics(y_te, pred_te[name], t_arg)
print(f"\nValidation calibration check: actual rate {y_va.mean():.3f}, mean predicted "
      + ", ".join(f"{n} {results[n]['val']['mean_pred']:.3f}" for n in pred_va))
print("Val-profit argmax thresholds (diagnostic only):",
      {n: results[n]["val_argmax_threshold"] for n in pred_va})
print("Call-everyone profit:", call_everyone)

# Agreement between the three implementations (all vs scikit-learn).
coef = pd.DataFrame({
    "feature": feature_names,
    "scikit-learn": sk.coef_.ravel(),
    "Manual PyTorch": manual_params["w"].numpy(),
    "Standard PyTorch": std_model.linear.weight.detach().numpy().ravel(),
})
intercepts = {
    "scikit-learn": float(sk.intercept_[0]),
    "Manual PyTorch": float(manual_params["b"].item()),
    "Standard PyTorch": float(std_model.linear.bias.item()),
}
coef.loc[len(coef)] = ["(intercept)", *intercepts.values()]
coef.to_csv(P.ARTIFACTS / "coefficients.csv", index=False)

agreement = {}
for name in ("Manual PyTorch", "Standard PyTorch"):
    t_sk = P.BREAK_EVEN_P
    agreement[name] = {
        "max_abs_coef_diff": float((coef[name] - coef["scikit-learn"]).abs().max()),
        "max_abs_test_prob_diff": float(np.abs(pred_te[name] - pred_te["scikit-learn"]).max()),
        "mean_abs_test_prob_diff": float(np.abs(pred_te[name] - pred_te["scikit-learn"]).mean()),
        # Same decision as sklearn when both use sklearn's threshold:
        "test_decision_agreement": float(((pred_te[name] >= t_sk)
                                          == (pred_te["scikit-learn"] >= t_sk)).mean()),
    }

# Final training objective of each implementation, evaluated identically.
def objective(p, w):
    return float(log_loss(y_tr, p) + 0.5 * lam * np.sum(np.asarray(w) ** 2))


final_objective = {
    "scikit-learn": objective(sk.predict_proba(X_tr)[:, 1], sk.coef_),
    "Manual PyTorch": objective(M.predict_manual(manual_params, X_tr), manual_params["w"].numpy()),
    "Standard PyTorch": objective(M.predict_standard(std_model, X_tr),
                                  std_model.linear.weight.detach().numpy()),
}

print(f"\nTest-set comparison (threshold = break-even {P.BREAK_EVEN_P:.2f}):")
table = pd.DataFrame({
    name: {**{k: r["test"][k] for k in ("roc_auc", "pr_auc", "log_loss", "brier",
                                       "mean_pred", "threshold", "calls", "profit")}}
    for name, r in results.items()
}).T
print(table.round(4).to_string())
print("\nFinal training objective:", {k: round(v, 6) for k, v in final_objective.items()})
print("Agreement vs scikit-learn:", json.dumps(agreement, indent=2))


# ---------------------------------------------------------------------------
# Save everything app.py needs. Models are saved in their native formats.
# ---------------------------------------------------------------------------
joblib.dump(pre, P.ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk, P.ARTIFACTS / "sklearn_logreg.joblib")
torch.save({"w": manual_params["w"], "b": manual_params["b"]},
           P.ARTIFACTS / "manual_torch.pt")
torch.save(std_model.state_dict(), P.ARTIFACTS / "standard_torch.pt")

# Evaluation data in model-ready form (the app scores it with the loaded
# models) plus the identifying columns needed for plots.
np.savez_compressed(P.ARTIFACTS / "eval_data.npz",
                    X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)

json.dump({
    "feature_names": feature_names,
    "base_rate": base_rate,
    "best_C": best_C, "lambda": lam,
    "best_manual_lr": best_manual_lr, "manual_epochs": MANUAL_EPOCHS,
    "best_standard_lr": best_std_lr, "standard_epochs": STANDARD_EPOCHS,
    "c_grid_val_log_loss": {str(k): v for k, v in c_scores.items()},
    "manual_lr_val_log_loss": {str(k): v[0] for k, v in manual_runs.items()},
    "standard_lr_val_log_loss": {str(k): v[0] for k, v in standard_runs.items()},
    "call_cost": P.CALL_COST, "deposit_value": P.DEPOSIT_VALUE,
    "break_even_p": P.BREAK_EVEN_P,
    "call_everyone_profit": call_everyone,
    "results": results,
    "agreement": agreement,
    "diagnostic_full_batch_adam": {"max_abs_coef_diff": diag_coef_gap,
                                   "mean_abs_test_prob_diff": diag_prob_gap},
    "final_objective": final_objective,
    "loss_curves": {"Manual PyTorch": manual_losses[::10],   # every 10th step
                    "Standard PyTorch": std_losses},
    "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
}, open(P.ARTIFACTS / "metrics.json", "w"), indent=2, default=float)

print(f"\nSaved artifacts to {P.ARTIFACTS}")
