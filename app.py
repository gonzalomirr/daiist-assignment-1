"""
Assignment 2: Gradio dashboard for the hard-hat turnstile decision.

    uv run python main.py app

Loads only what train.py produced: weights/best.pt and artifacts/. Nothing is
trained here. Every number on the operating-point views is recomputed live
from the saved per-image test scores (artifacts/scores_test.csv) with the same
code train.py used (decision.py).
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import decision as D

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

SUMMARY = json.loads((ARTIFACTS / "summary.json").read_text())
TEST = pd.read_csv(ARTIFACTS / "scores_test.csv")
OP = SUMMARY["operating_point"]
DEFAULT_CM = D.CostModel(**SUMMARY["cost_model"])
POS = SUMMARY["positive_class"]

_model = None


def model():
    # Loaded on first use so the page comes up immediately.
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def cost_model(c_fn, c_fp, c_review, base_rate, entries_per_day) -> D.CostModel:
    return D.CostModel(**{**DEFAULT_CM.to_dict(), "c_fn": c_fn, "c_fp": c_fp,
                          "c_review": c_review, "base_rate": base_rate,
                          "entries_per_day": int(entries_per_day)})


# ----------------------------------------------------------------- image upload

def predict(image, t_low, t_high):
    if image is None:
        return None, "Upload an image."
    t_high = max(t_high, t_low)
    r = model().predict(image, conf=0.01, verbose=False)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    names = r.names
    is_pos = np.array([names[c].lower() == POS for c in cls], dtype=bool)
    score = D.image_score(conf[is_pos])
    verdict = D.decide(score, t_low, t_high)
    # Draw every box above the lower threshold: that is everything that
    # influences the decision at the current setting.
    shown = r[conf > t_low]
    annotated = shown.plot()[:, :, ::-1]
    counts = pd.Series([names[c] for c in cls[conf > t_low]]).value_counts().to_dict()
    action = {"PASS": "turnstile opens", "REVIEW": "sent to the guard's screen",
              "FAIL": "turnstile stays shut: put your helmet on"}[verdict]
    text = (f"## {verdict}: {action}\n"
            f"Violation score (highest '{POS}' confidence): **{score:.2f}**  \n"
            f"Band: pass ≤ {t_low:.2f} < review ≤ {t_high:.2f} < fail  \n"
            f"Boxes shown (conf > {t_low:.2f}): {counts or 'none'}")
    return annotated, text


# ------------------------------------------------------------ operating point

def operating_views(t_low, t_high, c_fn, c_fp, c_review, base_rate, entries):
    t_high = max(t_high, t_low)
    cm = cost_model(c_fn, c_fp, c_review, base_rate, entries)
    r = D.evaluate(TEST["score"], TEST["label"], t_low, t_high, cm)
    c = r["counts"]

    z = [[c["pos_pass"], c["pos_review"], c["pos_fail"]],
         [c["neg_pass"], c["neg_review"], c["neg_fail"]]]
    labels = [["FN (missed)", "review", "TP (caught)"], ["TN", "review", "FP (wrongly stopped)"]]
    heat = go.Figure(go.Heatmap(
        z=z, x=["PASS", "REVIEW", "FAIL"], y=[f"true: {POS} (violation)", "true: compliant"],
        text=[[f"{l}<br>{v}" for l, v in zip(lr, zr)] for lr, zr in zip(labels, z)],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    heat.update_layout(title=f"Test split, {len(TEST)} images (counts)", height=320,
                       margin=dict(l=10, r=10, t=40, b=10))

    single = D.evaluate(TEST["score"], TEST["label"], t_low, t_low, cm)
    refs = D.reference_policies(cm)
    policies = pd.DataFrame([
        ["do nothing (no check)", refs["do_nothing"], 0.0],
        ["guard checks every entry", refs["human_checks_all"], 1.0],
        [f"model alone at t = {t_low:.2f}", single["cost_per_1000"], 0.0],
        ["model + review band (current)", r["cost_per_1000"], r["review_share"]],
    ], columns=["policy", "EUR per 1,000 entries", "share reviewed"]).round(3)

    kpi = (f"### Expected cost: **€{r['cost_per_1000']:,.0f} per 1,000 entries** "
           f"(base rate {base_rate:.1%})\n"
           f"- Sent for review: **{r['review_share']:.1%}** → {r['reviews_per_day']:.0f} per day, "
           f"{r['peak_hour_reviews']:.0f} in the peak hour → "
           f"**{r['guards_needed_peak']:.2f} guard(s)** busy at {cm.review_seconds:.0f} s each\n"
           f"- Missed violations: {r['missed_violations_per_1000']:.2f} per 1,000 entries; "
           f"false stops: {r['false_stops_per_1000']:.1f} per 1,000\n"
           f"- Recall on violations (caught by model or guard): "
           f"{r['pos_fail'] + r['pos_review'] * (1 - cm.review_miss):.1%}")
    return heat, kpi, policies, cost_curve(t_low, t_high, cm)


def cost_curve(t_low, t_high, cm):
    sweep = D.sweep_single(TEST["score"], TEST["label"], cm)
    xs = [s["t_low"] for s in sweep]
    ys = [s["cost_per_1000"] for s in sweep]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="model alone (test)"))
    t_star = OP["single_threshold"]
    fig.add_vline(x=t_star, line_dash="dash", line_color="black",
                  annotation_text=f"chosen t* = {t_star:.2f} (picked on val)",
                  annotation_position="top right")
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low), fillcolor="orange", opacity=0.2,
                  line_width=0, annotation_text="current review band",
                  annotation_position="bottom left")
    refs = D.reference_policies(cm)
    fig.add_hline(y=refs["do_nothing"], line_dash="dot", line_color="red",
                  annotation_text="do nothing")
    fig.add_hline(y=refs["human_checks_all"], line_dash="dot", line_color="green",
                  annotation_text="guard checks all")
    fig.update_layout(xaxis_title="confidence threshold on 'head'",
                      yaxis_title="EUR per 1,000 entries", height=420,
                      title="Cost against threshold (recomputed with the current assumptions)",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ----------------------------------------------------------------- static views

def per_class_tables():
    det = SUMMARY["detection"]
    per_class = pd.DataFrame([{"class": k, **v} for k, v in det["per_class"].items()]).round(3)
    det_base = pd.DataFrame(SUMMARY["detection_baselines"]).round(3)
    img_base = pd.DataFrame(SUMMARY["image_level_baselines"]).round(3)
    lab = det["confusion_labels"]
    conf = pd.DataFrame(det["confusion_matrix"], columns=[f"true {l}" for l in lab])
    conf.insert(0, "predicted", lab)
    ds = SUMMARY["dataset"]
    split_tbl = pd.DataFrame([{"split": s, "images": ds[s]["images"], **ds[s]["instances"],
                               "images with a bare head": round(ds[s]["image_level_violation_rate"], 3)}
                              for s in ("train", "valid", "test")])
    return per_class, det_base, img_base, conf, split_tbl


def build() -> gr.Blocks:
    cfg = SUMMARY["config"]
    per_class, det_base, img_base, conf, split_tbl = per_class_tables()
    with gr.Blocks(title="Hard-hat turnstile") as demo:
        gr.Markdown(
            f"# Hard-hat check at the site turnstile\n"
            f"Finetuned `{cfg['model']}` on `{cfg['workspace']}/{cfg['project']}` v{cfg['version']}. "
            f"Positive class: **{POS}** (bare head = violation). "
            f"Saved operating point: pass ≤ {OP['t_low']:.2f} < review ≤ {OP['t_high']:.2f} < fail "
            f"(model alone: t* = {OP['single_threshold']:.2f}).")
        with gr.Row():
            t_low = gr.Slider(0, 1, value=OP["t_low"], step=0.01, label="t_low: at or below → PASS")
            t_high = gr.Slider(0, 1, value=OP["t_high"], step=0.01, label="t_high: above → FAIL")
        with gr.Accordion("Cost assumptions (defaults argued in REPORT.md)", open=False):
            with gr.Row():
                c_fn = gr.Number(DEFAULT_CM.c_fn, label="€ false negative (violator enters)")
                c_fp = gr.Number(DEFAULT_CM.c_fp, label="€ false positive (compliant worker stopped)")
                c_review = gr.Number(DEFAULT_CM.c_review, label="€ per human review")
                base_rate = gr.Number(DEFAULT_CM.base_rate, label="base rate of violations")
                entries = gr.Number(DEFAULT_CM.entries_per_day, label="entries per day")

        with gr.Tab("Try an image"):
            with gr.Row():
                inp = gr.Image(type="numpy", label="upload")
                out = gr.Image(label="predictions above t_low")
            verdict = gr.Markdown()
            for ev in (inp.change, t_low.release, t_high.release):
                ev(predict, [inp, t_low, t_high], [out, verdict])

        with gr.Tab("Operating point"):
            kpi = gr.Markdown()
            with gr.Row():
                heat = gr.Plot()
                policies = gr.Dataframe(label="Priced against the alternatives")
        with gr.Tab("Cost curve"):
            curve = gr.Plot()
            gr.Image(str(ARTIFACTS / "cost_curve.png"), label="saved by train.py (validation vs test)")

        with gr.Tab("Per-class results & baselines"):
            gr.Markdown("### Detection on the test split, per class")
            gr.Dataframe(per_class)
            gr.Markdown("### Detection baselines (same test split)")
            gr.Dataframe(det_base)
            gr.Markdown("### Image-level decision vs baselines (threshold picked on validation)")
            gr.Dataframe(img_base)
            gr.Markdown("### Detection confusion matrix (rows predicted, columns true)")
            gr.Dataframe(conf)
            cm_png = ARTIFACTS / "test_confusion_matrix_normalized.png"
            if cm_png.exists():
                gr.Image(str(cm_png), label="normalised confusion matrix")
        with gr.Tab("Dataset"):
            gr.Dataframe(split_tbl)
            gr.Image(str(ARTIFACTS / "object_sizes.png"), label="object size relative to image")

        controls = [t_low, t_high, c_fn, c_fp, c_review, base_rate, entries]
        outputs = [heat, kpi, policies, curve]
        for ctl in controls:
            (ctl.release if isinstance(ctl, gr.Slider) else ctl.change)(operating_views, controls, outputs)
        demo.load(operating_views, controls, outputs)
    return demo


if __name__ == "__main__":
    build().launch()
