"""
Assignment 1 -- Gradio dashboard. Run with `uv run python main.py app`.

Loads the artifacts written by train.py (preprocessor + the three saved
models + evaluation data + metrics) and only ever runs *inference*. Nothing
is fitted here: if artifacts/ is missing, run `main.py train` first.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import confusion_matrix

import models as M
import pipeline as P

# Categorical slots 1-3 (validated all-pairs for colour-blind safety) for the
# three implementations; neutral grey for the baseline.
COLORS = {"scikit-learn": "#2a78d6", "Manual PyTorch": "#eb6834",
          "Standard PyTorch": "#1baf7a", "Baseline": "#8a8984"}
MODEL_NAMES = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50),
              font=dict(size=13), hoverlabel=dict(font_size=13))


# ---------------------------------------------------------------------------
# Load artifacts (no training anywhere below)
# ---------------------------------------------------------------------------
A = P.ARTIFACTS
if not (A / "metrics.json").exists():
    raise SystemExit("artifacts/ not found -- run `uv run python main.py train` first.")

meta = json.load(open(A / "metrics.json"))
preprocessor = joblib.load(A / "preprocessor.joblib")
sk_model = joblib.load(A / "sklearn_logreg.joblib")
manual_params = torch.load(A / "manual_torch.pt")
std_model = M.LogisticRegressionNet(len(meta["feature_names"]))
std_model.load_state_dict(torch.load(A / "standard_torch.pt"))
std_model.eval()
coefs = pd.read_csv(A / "coefficients.csv")
ablation = pd.read_csv(A / "ablation.csv")
evald = np.load(A / "eval_data.npz")


def predict_all(X: np.ndarray) -> dict:
    return {
        "scikit-learn": sk_model.predict_proba(X)[:, 1],
        "Manual PyTorch": M.predict_manual(manual_params, X),
        "Standard PyTorch": M.predict_standard(std_model, X),
    }


SPLITS = {}
for split, label in (("val", "Validation (Mar-Jul 2010)"), ("test", "Test (Aug-Nov 2010)")):
    X, y = evald[f"X_{split}"], evald[f"y_{split}"]
    preds = predict_all(X)
    preds["Baseline"] = np.full(len(y), meta["base_rate"])
    SPLITS[label] = {"y": y, "p": preds, "key": split}
SPLIT_LABELS = list(SPLITS)

# Raw data, only for the distribution views.
raw = P.clean(P.load_raw())
tr, va, te = P.time_split(raw)
raw["split"] = np.select([tr, va, te], ["train", "validation", "test"], default="")
features = P.engineer(raw)


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def metrics_table(split_label):
    key = SPLITS[split_label]["key"]
    rows = []
    for name in ["Baseline", *MODEL_NAMES]:
        r = meta["results"][name][key]
        rows.append({
            "Model": name, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"],
            "Log-loss": r["log_loss"], "Brier": r["brier"],
            "Mean predicted p": r["mean_pred"], "Threshold": r["threshold"],
            "Clients called": r["calls"], "Profit EUR": r["profit"],
        })
    y = SPLITS[split_label]["y"]
    rows.append({"Model": "Policy: call everyone", "Clients called": len(y),
                 "Profit EUR": meta["call_everyone_profit"][key]})
    actual = y.mean()
    df = pd.DataFrame(rows).round(4)
    return df, f"Actual conversion rate in this split: **{actual:.3f}**"


def reliability_plot(split_label, n_bins):
    """Predicted vs actual for a classifier: bin clients by predicted
    probability and compare with the fraction that actually subscribed."""
    s = SPLITS[split_label]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#b5b4ae", dash="dash", width=1.5),
                             hoverinfo="skip"))
    for name in MODEL_NAMES:
        p = s["p"][name]
        bins = pd.qcut(p, int(n_bins), duplicates="drop")
        g = pd.DataFrame({"p": p, "y": s["y"]}).groupby(bins, observed=True)
        agg = g.agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(
            x=agg["pred"], y=agg["actual"], mode="lines+markers", name=name,
            line=dict(color=COLORS[name], width=2), marker=dict(size=9),
            customdata=agg["n"],
            hovertemplate=name + "<br>mean predicted %{x:.2f}<br>actual rate %{y:.2f}"
                                 "<br>%{customdata} clients<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Predicted probability vs actual conversion rate "
                                      "(equal-size bins)",
                      xaxis_title="Mean predicted probability in bin",
                      yaxis_title="Actual share that subscribed",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=460,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def score_by_outcome_plot(split_label):
    s = SPLITS[split_label]
    fig = go.Figure()
    for name in MODEL_NAMES:
        for outcome, label in ((0, "did not subscribe"), (1, "subscribed")):
            fig.add_trace(go.Box(
                y=s["p"][name][s["y"] == outcome], x=[label] * int((s["y"] == outcome).sum()),
                name=name, legendgroup=name, showlegend=outcome == 0,
                marker_color=COLORS[name], boxpoints=False))
    fig.update_layout(**LAYOUT, boxmode="group", height=420,
                      title="Predicted probability by actual outcome",
                      yaxis_title="Predicted probability", yaxis_range=[0, 1],
                      legend=dict(orientation="h", y=-0.15))
    return fig


def pairwise_plot(split_label):
    """Each PyTorch model's prediction against scikit-learn's, client by client."""
    s = SPLITS[split_label]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color="#b5b4ae", dash="dash", width=1.5),
                             hoverinfo="skip"))
    for name in ("Manual PyTorch", "Standard PyTorch"):
        fig.add_trace(go.Scatter(
            x=s["p"]["scikit-learn"], y=s["p"][name], mode="markers", name=name,
            marker=dict(color=COLORS[name], size=6, opacity=0.6),
            hovertemplate="sklearn %{x:.3f}<br>" + name + " %{y:.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=420, title="Do the implementations agree? "
                                                  "(one dot per client)",
                      xaxis_title="scikit-learn predicted probability",
                      yaxis_title="PyTorch predicted probability",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def coef_plot():
    df = coefs[coefs["feature"] != "(intercept)"].copy()
    df = df.reindex(df["scikit-learn"].abs().sort_values().index)
    fig = go.Figure()
    for name in MODEL_NAMES:
        fig.add_trace(go.Bar(y=df["feature"], x=df[name], name=name, orientation="h",
                             marker_color=COLORS[name],
                             hovertemplate="%{y}: %{x:.3f}<extra>" + name + "</extra>"))
    fig.update_layout(**LAYOUT, barmode="group", height=1100, bargap=0.25,
                      title="Coefficients (log-odds per unit; numeric features are standardized)",
                      xaxis_title="Coefficient", legend=dict(orientation="h", y=1.02, x=0))
    return fig


def loss_plot():
    curves = meta["loss_curves"]
    fig = go.Figure()
    man = curves["Manual PyTorch"]
    fig.add_trace(go.Scatter(x=np.arange(len(man)) * 10 + 1, y=man, name="Manual PyTorch (full batch, per step)",
                             line=dict(color=COLORS["Manual PyTorch"], width=2)))
    std = curves["Standard PyTorch"]
    n_batches = int(np.ceil(meta["split_sizes"]["train"] / 512))
    fig.add_trace(go.Scatter(x=(np.arange(len(std)) + 1) * n_batches, y=std,
                             name="Standard PyTorch (mini-batch, per epoch)",
                             line=dict(color=COLORS["Standard PyTorch"], width=2)))
    fig.add_hline(y=meta["final_objective"]["scikit-learn"], line_dash="dash",
                  line_color=COLORS["scikit-learn"],
                  annotation_text="scikit-learn optimum", annotation_position="top right")
    fig.update_layout(**LAYOUT, height=400, title="Training objective (BCE + L2) vs parameter updates",
                      xaxis_title="Parameter updates (gradient steps, log scale)",
                      yaxis_title="Objective", xaxis_type="log",
                      yaxis_range=[0.25, 0.45], legend=dict(orientation="h", y=-0.2))
    return fig


def agreement_table():
    rows = []
    for name, a in meta["agreement"].items():
        rows.append({"Implementation": name,
                     "Max |coef - sklearn|": a["max_abs_coef_diff"],
                     "Mean |p - sklearn| (test)": a["mean_abs_test_prob_diff"],
                     "Max |p - sklearn| (test)": a["max_abs_test_prob_diff"],
                     "Same decision as sklearn (test)": a["test_decision_agreement"],
                     "Final train objective": meta["final_objective"][name]})
    d = meta["diagnostic_full_batch_adam"]
    rows.append({"Implementation": "Diagnostic: nn.Module + Adam, full batch (not saved)",
                 "Max |coef - sklearn|": d["max_abs_coef_diff"],
                 "Mean |p - sklearn| (test)": d["mean_abs_test_prob_diff"]})
    rows.append({"Implementation": "scikit-learn (reference)",
                 "Final train objective": meta["final_objective"]["scikit-learn"]})
    return pd.DataFrame(rows).round(5)


# ---------------------------------------------------------------------------
# Tab 2: data distributions
# ---------------------------------------------------------------------------
RAW_VIEW = ["age", "job", "marital", "education", "default", "housing", "loan",
            "contact", "poutcome", "pdays", "previous", "campaign",
            "euribor3m", "cons_price_idx", "cons_conf_idx", "emp_var_rate",
            "nr_employed", "duration (dropped: leakage)"]
ENGINEERED_VIEW = list(features.columns)
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}


def feature_plots(col):
    if col == "duration (dropped: leakage)":
        series = P.clean(P.load_raw(), keep_duration=True)["duration"]
    elif col in features.columns and col not in raw.columns:
        series = features[col]
    else:
        series = raw[col]
    df = pd.DataFrame({"x": series, "y": raw["y"], "split": raw["split"]})
    numeric = pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > 12
    if numeric:
        # A value that dominates the column (e.g. the pdays = 999 sentinel)
        # gets its own bucket; the rest is cut into deciles.
        mode = df["x"].mode()[0]
        special = df["x"] == mode if (df["x"] == mode).mean() > 0.3 else np.zeros(len(df), bool)
        rest = df.loc[~special, "x"]
        df["bucket"] = ""
        df.loc[~special, "bucket"] = (pd.qcut(rest, 10, duplicates="drop").astype(str)
                                      if rest.nunique() > 12 else rest.astype(str))
        df.loc[special, "bucket"] = f"= {mode:g} (sentinel/mode)"
        order = list(df[~special].groupby("bucket")["x"].min().sort_values().index)
        if special.any():
            order.append(f"= {mode:g} (sentinel/mode)")
    else:
        df["bucket"] = df["x"].astype(str)
        order = df["bucket"].value_counts().index

    dist = go.Figure()
    rate = go.Figure()
    for split, c in SPLIT_COLORS.items():
        d = df[df["split"] == split]
        share = d["bucket"].value_counts(normalize=True).reindex(order).fillna(0)
        dist.add_trace(go.Bar(x=list(order), y=share.values, name=split, marker_color=c,
                              hovertemplate="%{x}: %{y:.1%}<extra>" + split + "</extra>"))
        g = d.groupby("bucket")["y"].agg(["mean", "size"]).reindex(order)
        rate.add_trace(go.Scatter(x=list(order), y=g["mean"], name=split, mode="lines+markers",
                                  line=dict(color=c, width=2), marker=dict(size=8),
                                  customdata=g["size"],
                                  hovertemplate="%{x}: %{y:.1%} (n=%{customdata})<extra>"
                                                + split + "</extra>"))
    dist.update_layout(**LAYOUT, barmode="group", height=380,
                       title=f"Distribution of '{col}' by split (share of rows)",
                       yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.3))
    rate.update_layout(**LAYOUT, height=380, title=f"Conversion rate by '{col}'",
                       yaxis_tickformat=".0%", yaxis_title="Subscribed",
                       legend=dict(orientation="h", y=-0.3))
    return dist, rate


def monthly_plots():
    g = raw.groupby("period").agg(calls=("y", "size"), rate=("y", "mean"),
                                  split=("split", "first")).reset_index()
    vol = go.Figure()
    rate = go.Figure()
    for split, c in SPLIT_COLORS.items():
        d = g[g["split"] == split]
        vol.add_trace(go.Bar(x=d["period"], y=d["calls"], name=split, marker_color=c,
                             hovertemplate="%{x|%b %Y}: %{y} clients<extra></extra>"))
        rate.add_trace(go.Scatter(x=d["period"], y=d["rate"], name=split, mode="lines+markers",
                                  line=dict(color=c, width=2), marker=dict(size=8),
                                  hovertemplate="%{x|%b %Y}: %{y:.1%}<extra></extra>"))
    vol.update_layout(**LAYOUT, height=330, title="Clients contacted per month (target volume)",
                      legend=dict(orientation="h", y=-0.25))
    rate.update_layout(**LAYOUT, height=330, title="Conversion rate per month (target mean)",
                       yaxis_tickformat=".0%", yaxis_range=[0, 1],
                       legend=dict(orientation="h", y=-0.25))
    return vol, rate


# ---------------------------------------------------------------------------
# Tab 3: decision threshold and business cost
# ---------------------------------------------------------------------------
def threshold_view(model_name, split_label, threshold, call_cost, deposit_value):
    s = SPLITS[split_label]
    y, p = s["y"], s["p"][model_name]
    call = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, call, labels=[0, 1]).ravel()

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["Don't call", "Call"],
        y=["Did not subscribe", "Subscribed"], colorscale=[[0, "#eef4fc"], [1, "#1c5aa6"]],
        showscale=False, text=[[f"TN<br>{tn}", f"FP<br>{fp}<br>-EUR {fp * call_cost:,.0f}"],
                                [f"FN<br>{fn}<br>missed EUR {fn * (deposit_value - call_cost):,.0f}",
                                 f"TP<br>{tp}<br>+EUR {tp * (deposit_value - call_cost):,.0f}"]],
        texttemplate="%{text}", textfont=dict(size=15), hoverinfo="skip"))
    cm.update_layout(**LAYOUT, height=380, title=f"Confusion matrix at threshold {threshold:.2f}",
                     xaxis_title="Decision", yaxis_title="Actual", yaxis_autorange="reversed")

    prof = P.profit(y, call, call_cost, deposit_value)
    all_prof = P.profit(y, np.ones(len(y)), call_cost, deposit_value)
    n = len(y)
    summary = (
        f"### Campaign profit: **EUR {prof:,.0f}**\n"
        f"- Calls made: **{call.sum()} / {n}** clients ({call.mean():.0%}); "
        f"subscriptions won: **{tp} / {tp + fn}** ({tp / max(tp + fn, 1):.0%} of all possible)\n"
        f"- Precision of the call list: **{tp / max(call.sum(), 1):.1%}** "
        f"(break-even precision = cost / value = {call_cost / deposit_value:.1%})\n"
        f"- *Call everyone*: EUR {all_prof:,.0f} &nbsp;|&nbsp; *Call nobody*: EUR 0 "
        f"&nbsp;|&nbsp; this threshold vs call-everyone: **EUR {prof - all_prof:+,.0f}**\n"
        f"- Cost per client called EUR {call_cost:.0f}, value per deposit EUR {deposit_value:.0f}: "
        f"the calibrated-probability rule would call when p > {call_cost / deposit_value:.2f}."
    )

    ts = np.round(np.arange(0, 1.0001, 0.01), 2)
    curve = go.Figure()
    for name in MODEL_NAMES:
        curve.add_trace(go.Scatter(
            x=ts, y=[P.profit(y, s["p"][name] >= t, call_cost, deposit_value) for t in ts],
            name=name, line=dict(color=COLORS[name], width=3 if name == model_name else 1.5),
            hovertemplate="t=%{x:.2f}: EUR %{y:,.0f}<extra>" + name + "</extra>"))
    curve.add_hline(y=all_prof, line_dash="dot", line_color=COLORS["Baseline"],
                    annotation_text="call everyone", annotation_position="bottom left")
    curve.add_vline(x=threshold, line_color="#52514e", line_width=1)
    curve.update_layout(**LAYOUT, height=380, title="Profit vs threshold (all three models)",
                        xaxis_title="Threshold (call if p >= t)", yaxis_title="Profit (EUR)",
                        legend=dict(orientation="h", y=-0.25))
    return cm, summary, curve


# ---------------------------------------------------------------------------
# Tab 4: score one client with the saved preprocessor + models
# ---------------------------------------------------------------------------
def score_client(age, job, marital, education, default, housing, loan, contact,
                 pdays, previous, poutcome, euribor3m, cons_price_idx, cons_conf_idx):
    row = pd.DataFrame([dict(
        age=age, job=job, marital=marital, education=education, default=default,
        housing=housing, loan=loan, contact=contact, pdays=999 if pdays < 0 else pdays,
        previous=previous, poutcome=poutcome, euribor3m=euribor3m,
        cons_price_idx=cons_price_idx, cons_conf_idx=cons_conf_idx)])
    X = preprocessor.transform(P.engineer(row)).astype(np.float32)
    preds = predict_all(X)
    thr = meta["break_even_p"]
    lines = ["| Model | P(subscribe) | Decision |", "|---|---|---|"]
    for name, p in preds.items():
        t = meta["results"][name]["val"]["threshold"]
        lines.append(f"| {name} | {p[0]:.3f} | {'CALL' if p[0] >= t else 'skip'} (t={t:.2f}) |")
    ev = preds["scikit-learn"][0] * P.DEPOSIT_VALUE - P.CALL_COST
    lines.append(f"\nExpected value of calling (sklearn p): **EUR {ev:+.2f}** "
                 f"(break-even threshold {thr:.2f}).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Bank telemarketing: who to call") as demo:
    gr.Markdown(
        "# Term-deposit telemarketing: who should the call centre call?\n"
        "Logistic regression trained three ways on May 2008 - Dec 2009 calls, "
        "decision threshold = cost / value break-even (checked on Mar - Jul 2010), "
        "tested on Aug - Nov 2010. "
        f"Everything shown is computed from the saved models in `artifacts/` "
        f"(sklearn C = {meta['best_C']}, manual lr = {meta['best_manual_lr']}, "
        f"standard lr = {meta['best_standard_lr']}). Nothing is retrained here."
    )

    with gr.Tab("1. Model comparison"):
        split_sel = gr.Radio(SPLIT_LABELS, value=SPLIT_LABELS[1], label="Evaluation split")
        metrics_md = gr.Markdown()
        metrics_df = gr.Dataframe(interactive=False)
        n_bins = gr.Slider(4, 20, value=10, step=1, label="Bins for predicted-vs-actual plot")
        with gr.Row():
            rel = gr.Plot()
            box = gr.Plot()
        with gr.Row():
            pair = gr.Plot()
            loss = gr.Plot(value=loss_plot())
        gr.Markdown("### Agreement between implementations")
        gr.Dataframe(value=agreement_table(), interactive=False)
        gr.Plot(value=coef_plot())

        def refresh_comparison(split_label, bins):
            df, md = metrics_table(split_label)
            return md, df, reliability_plot(split_label, bins), \
                score_by_outcome_plot(split_label), pairwise_plot(split_label)

        for ev in (split_sel.change, n_bins.change):
            ev(refresh_comparison, [split_sel, n_bins], [metrics_md, metrics_df, rel, box, pair])
        demo.load(refresh_comparison, [split_sel, n_bins], [metrics_md, metrics_df, rel, box, pair])

    with gr.Tab("2. Data & distributions"):
        vol_plot, rate_plot = monthly_plots()
        with gr.Row():
            gr.Plot(value=vol_plot)
            gr.Plot(value=rate_plot)
        feat = gr.Dropdown(RAW_VIEW + [c for c in ENGINEERED_VIEW if c not in RAW_VIEW],
                           value="pdays", label="Feature (raw column or engineered feature)")
        with gr.Row():
            dist_plot = gr.Plot()
            frate_plot = gr.Plot()
        feat.change(feature_plots, feat, [dist_plot, frate_plot])
        demo.load(feature_plots, feat, [dist_plot, frate_plot])

    with gr.Tab("3. Threshold & business cost"):
        with gr.Row():
            m_sel = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            s_sel = gr.Radio(SPLIT_LABELS, value=SPLIT_LABELS[1], label="Split")
        thr = gr.Slider(0, 1, value=meta["break_even_p"],
                        step=0.01, label="Decision threshold: call the client if P(subscribe) >= t")
        with gr.Row():
            cost = gr.Number(value=P.CALL_COST, label="Cost of calling one client (EUR)")
            value = gr.Number(value=P.DEPOSIT_VALUE, label="Value of one new deposit (EUR)")
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        inputs = [m_sel, s_sel, thr, cost, value]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, summary, curve_plot])
        demo.load(threshold_view, inputs, [cm_plot, summary, curve_plot])

    with gr.Tab("4. Score a client"):
        gr.Markdown("Runs the saved preprocessor and the three saved models on one client. "
                    "Macro indicators default to their August 2010 values.")
        aug = raw[raw["period"] == P.TEST_START].iloc[0]
        with gr.Row():
            c_age = gr.Slider(18, 95, value=40, step=1, label="Age")
            c_job = gr.Dropdown(sorted(raw["job"].unique()), value="admin.", label="Job")
            c_mar = gr.Dropdown(sorted(raw["marital"].unique()), value="married", label="Marital")
            c_edu = gr.Dropdown(sorted(raw["education"].unique()), value="university.degree",
                                label="Education")
        with gr.Row():
            c_def = gr.Dropdown(["no", "unknown", "yes"], value="no", label="Credit default")
            c_hou = gr.Dropdown(["no", "yes", "unknown"], value="no", label="Housing loan")
            c_loan = gr.Dropdown(["no", "yes", "unknown"], value="no", label="Personal loan")
            c_con = gr.Dropdown(["cellular", "telephone"], value="cellular", label="Phone type")
        with gr.Row():
            c_pd = gr.Number(value=-1, label="Days since last contact (-1 = never contacted)")
            c_prev = gr.Number(value=0, label="Contacts before this campaign")
            c_pout = gr.Dropdown(["nonexistent", "failure", "success"], value="nonexistent",
                                 label="Previous campaign outcome")
        with gr.Row():
            c_eur = gr.Number(value=float(aug["euribor3m"]), label="Euribor 3m")
            c_cpi = gr.Number(value=float(aug["cons_price_idx"]), label="Consumer price index")
            c_cci = gr.Number(value=float(aug["cons_conf_idx"]), label="Consumer confidence")
        btn = gr.Button("Score", variant="primary")
        out = gr.Markdown()
        btn.click(score_client, [c_age, c_job, c_mar, c_edu, c_def, c_hou, c_loan, c_con,
                                 c_pd, c_prev, c_pout, c_eur, c_cpi, c_cci], out)

    with gr.Tab("5. Feature ablation"):
        gr.Markdown("scikit-learn (C = 1) trained on each feature set, scored on the "
                    "**validation** months only (the test months were not used to choose features).")
        gr.Dataframe(value=ablation.round(4), interactive=False)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
