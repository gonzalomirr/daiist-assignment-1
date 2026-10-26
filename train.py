"""
Assignment 2: training pipeline (hard-hat compliance at a site turnstile).

    uv run python main.py train

Start to finish, no manual steps:

1. Download the dataset version from Roboflow Universe (key via api_key.py),
   and carve a validation split out of train by source image (v1 has none).
2. Describe it: images and instances per class per split, object size
   relative to the image.
3. Finetune yolo11n from the COCO checkpoint. Best weights -> weights/best.pt.
4. Evaluate on the test split: per-class P/R/AP50/mAP50-95 and the confusion
   matrix, against two baselines (COCO yolo11n zero-shot, and a trivial
   predict-nothing / majority-class rule).
5. Turn detections into one score per image, sweep the threshold on the
   validation split, price each point with decision.CostModel, pick the
   cheapest single threshold and the cheapest pass/review/fail band, then
   report both on the test split.
6. Save everything app.py needs into artifacts/ (small, committed).

Settings can be overridden with environment variables (see CONFIG below),
e.g. EPOCHS=5 for a quick run. REUSE_WEIGHTS=1 skips the finetune and
re-runs steps 4-6 on the existing weights/best.pt. DATASET_DIR points at an already-downloaded
YOLO-format export and skips the download.
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

import decision as D

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"
DATASETS = ROOT / "datasets"

CONFIG = {
    # Roboflow Universe dataset, used as published (not forked).
    "workspace": os.environ.get("RF_WORKSPACE", "robertatroc-7xoey"),
    "project": os.environ.get("RF_PROJECT", "hard-hat-workers-dataset-hnvfo"),
    "version": int(os.environ.get("RF_VERSION", "1")),
    # Training settings.
    "model": os.environ.get("BASE_MODEL", "yolo11n.pt"),
    "epochs": int(os.environ.get("EPOCHS", "40")),
    "imgsz": int(os.environ.get("IMGSZ", "640")),
    "batch": int(os.environ.get("BATCH", "16")),
    "patience": int(os.environ.get("PATIENCE", "10")),
    # Last epochs without mosaic (Ultralytics default 10, too many for a short run).
    "close_mosaic": int(os.environ.get("CLOSE_MOSAIC", "10")),
    "seed": 0,
    "workers": int(os.environ.get("WORKERS", "8")),
    "fraction": float(os.environ.get("FRACTION", "1.0")),
    # v1 has no validation split: this share of train source images becomes one.
    "val_share": 0.15,
    # Detections kept for the image-level score; thresholds are swept above it.
    "score_conf_floor": 0.01,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------- data

def download_dataset() -> Path:
    if os.environ.get("DATASET_DIR"):
        return Path(os.environ["DATASET_DIR"]).resolve()
    from roboflow import Roboflow
    from api_key import load_api_key

    target = DATASETS / f"{CONFIG['project']}-v{CONFIG['version']}"
    if (target / "data.yaml").exists():
        print(f"Dataset already downloaded at {target}")
        return target
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(CONFIG["workspace"]).project(CONFIG["project"]).version(CONFIG["version"])
    ds = version.download("yolov8", location=str(target), overwrite=True)
    return Path(ds.location).resolve()


def split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / split / "images", root / split / "labels"


def source_id(image: Path) -> str:
    """Roboflow names each copy '<source>_jpg.rf.<hash>.jpg'; copies of one
    source image share everything before '.rf.'."""
    return image.name.split(".rf.")[0]


def prepare_splits(raw: Path) -> Path:
    """Version 1 ships train and test only, and every train image comes in up
    to 2 horizontally flipped copies (Roboflow's augmentation). Build a working
    copy (symlinks) with:

    - valid: VAL_SHARE of the train *source* images, so no flipped twin of a
      validation image is in train (that would leak);
    - one copy per source image in train and valid, since Ultralytics already
      flips at random during training and twin copies only double epoch time;
    - test untouched (one copy per source already).
    """
    if (raw / "valid" / "images").exists():
        return raw
    out = DATASETS / f"{raw.name}-split"
    if (out / "data.yaml").exists():
        return out
    groups: dict[str, list[Path]] = {}
    for img in list_images(raw, "train"):
        groups.setdefault(source_id(img), []).append(img)
    ids = sorted(groups)
    rng = np.random.default_rng(CONFIG["seed"])
    val_ids = set(rng.choice(ids, size=round(CONFIG["val_share"] * len(ids)), replace=False))
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        for d in split_dirs(out, split):
            d.mkdir(parents=True)
    for sid in ids:
        img = sorted(groups[sid])[0]
        split = "valid" if sid in val_ids else "train"
        img_dir, lab_dir = split_dirs(out, split)
        (img_dir / img.name).symlink_to(img)
        lab = split_dirs(raw, "train")[1] / f"{img.stem}.txt"
        if lab.exists():
            (lab_dir / lab.name).symlink_to(lab)
    for src, dst in zip(split_dirs(raw, "test"), split_dirs(out, "test")):
        for f in src.iterdir():
            (dst / f.name).symlink_to(f)
    shutil.copy(raw / "data.yaml", out / "data.yaml")
    print(f"Built {out}: {len(ids) - len(val_ids)} train / {len(val_ids)} valid source images")
    return out


def write_data_yaml(root: Path) -> tuple[Path, list[str]]:
    """Roboflow's data.yaml uses relative paths that break depending on cwd.
    Rewrite it with absolute ones."""
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    for split, key in zip(SPLITS, ("train", "val", "test")):
        cfg[key] = str(split_dirs(root, split)[0])
    out = root / "data_abs.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out, list(names)


def list_images(root: Path, split: str) -> list[Path]:
    img_dir, _ = split_dirs(root, split)
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_labels(root: Path, split: str, image: Path) -> np.ndarray:
    """Rows of (class, x, y, w, h), normalised. Empty array if no objects."""
    label = split_dirs(root, split)[1] / f"{image.stem}.txt"
    if not label.exists() or label.stat().st_size == 0:
        return np.zeros((0, 5))
    rows = [list(map(float, ln.split()[:5])) for ln in label.read_text().splitlines() if ln.strip()]
    return np.array(rows).reshape(-1, 5)


def describe_dataset(root: Path, names: list[str], pos_idx: int) -> dict:
    stats, area_rows = {}, []
    for split in SPLITS:
        images = list_images(root, split)
        inst = {n: 0 for n in names}
        imgs_with = {n: 0 for n in names}
        empty, with_pos = 0, 0
        for img in images:
            lab = read_labels(root, split, img)
            if len(lab) == 0:
                empty += 1
            classes = lab[:, 0].astype(int)
            for c in np.unique(classes):
                imgs_with[names[c]] += 1
            for c in classes:
                inst[names[c]] += 1
            with_pos += int((classes == pos_idx).any())
            for c, w, h in zip(classes, lab[:, 3], lab[:, 4]):
                area_rows.append({"split": split, "class": names[c], "rel_area": w * h})
        stats[split] = {
            "images": len(images),
            "instances": inst,
            "images_containing": imgs_with,
            "images_without_labels": empty,
            "image_level_violation_rate": with_pos / max(len(images), 1),
        }
    areas = pd.DataFrame(area_rows)
    size = {}
    for cls, g in areas.groupby("class"):
        a = g["rel_area"]
        size[cls] = {
            "median_rel_area_pct": 100 * float(a.median()),
            "p10_rel_area_pct": 100 * float(a.quantile(0.10)),
            "p90_rel_area_pct": 100 * float(a.quantile(0.90)),
            # COCO's "small" is < 32x32 px; at 640x640 that is 0.25% of the image.
            "share_small_lt_0.25pct": float((a < 0.0025).mean()),
        }
    stats["object_size"] = size

    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, g in areas.groupby("class"):
        ax.hist(100 * np.sqrt(g["rel_area"]), bins=50, alpha=0.6, label=cls)
    ax.set_xlabel("box side relative to image side (%, sqrt of area share)")
    ax.set_ylabel("instances (all splits)")
    ax.set_title("Object size relative to the image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "object_sizes.png", dpi=110)
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------- training

def finetune(data_yaml: Path) -> Path:
    model = YOLO(CONFIG["model"])
    model.train(
        data=str(data_yaml), epochs=CONFIG["epochs"], imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"], patience=CONFIG["patience"], seed=CONFIG["seed"],
        workers=CONFIG["workers"], fraction=CONFIG["fraction"],
        close_mosaic=CONFIG["close_mosaic"],
        deterministic=True, project=str(RUNS), name="finetune", exist_ok=True,
        plots=True, verbose=False,
    )
    best = Path(model.trainer.best)
    WEIGHTS.mkdir(exist_ok=True)
    dest = WEIGHTS / "best.pt"
    shutil.copy(best, dest)
    print(f"Saved finetuned weights to {dest}")
    return dest


# --------------------------------------------------------------------- evaluation

def detection_eval(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics val on the TEST split: overall and per-class numbers."""
    m = YOLO(str(weights))
    res = m.val(data=str(data_yaml), split="test", imgsz=CONFIG["imgsz"], batch=CONFIG["batch"],
                project=str(RUNS), name="test_eval", exist_ok=True, plots=True, verbose=False)
    box = res.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
        }
    cm = res.confusion_matrix.matrix.astype(int).tolist()
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(res.save_dir) / f
        if src.exists():
            shutil.copy(src, ARTIFACTS / f"test_{f}")
    return {
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
        "per_class": per_class,
        # rows = predicted, cols = true; last index is "background"
        "confusion_matrix": cm, "confusion_labels": names + ["background"],
    }


def image_scores(weights: Path, root: Path, split: str, names: list[str], pos_idx: int,
                 coco: YOLO) -> pd.DataFrame:
    """One row per image: true label (any bare head?) and three scores."""
    model = YOLO(str(weights))
    images = list_images(root, split)
    coco_person = [k for k, v in coco.names.items() if v == "person"][0]
    rows = []
    kw = dict(conf=CONFIG["score_conf_floor"], imgsz=CONFIG["imgsz"], verbose=False)
    # One image per call: predicting on the whole list at once ran out of
    # memory on a 16 GB CPU machine.
    for img in images:
        r = model.predict(str(img), **kw)[0]
        rc = coco.predict(str(img), classes=[coco_person], **kw)[0]
        lab = read_labels(root, split, img)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        rows.append({
            "image": img.name,
            "label": int((lab[:, 0].astype(int) == pos_idx).any()),
            "n_true_heads": int((lab[:, 0].astype(int) == pos_idx).sum()),
            "score": D.image_score(conf[cls == pos_idx]),
            "n_pred_heads_25": int(((cls == pos_idx) & (conf >= 0.25)).sum()),
            # COCO zero-shot has no helmet/head class. The only thing it can do
            # for this decision is "flag whenever a person is visible".
            "coco_score": D.image_score(rc.boxes.conf.cpu().numpy()),
        })
    return pd.DataFrame(rows)


def roc_auc(scores, labels) -> float:
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def image_level_summary(name, val_scores, val_y, test_scores, test_y, cm) -> dict:
    """Pick the cheapest single threshold on VAL, report it on TEST."""
    sweep = D.sweep_single(val_scores, val_y, cm)
    t = min(sweep, key=lambda r: r["cost_per_1000"])["t_low"]
    r = D.evaluate(test_scores, test_y, t, t, cm)
    c = r["counts"]
    tp, fn, fp, tn = c["pos_fail"], c["pos_pass"], c["neg_fail"], c["neg_pass"]
    return {
        "policy": name, "threshold_from_val": float(t),
        "test_roc_auc": roc_auc(test_scores, test_y),
        "test_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "test_precision": tp / max(tp + fp, 1), "test_recall": tp / max(tp + fn, 1),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "test_cost_per_1000": r["cost_per_1000"],
    }


def sensitivity(val: pd.DataFrame, base: D.CostModel) -> pd.DataFrame:
    """How the chosen operating point moves when the two shakiest
    assumptions (cost of a miss, base rate) move."""
    rows = []
    for base_rate in (0.01, 0.03, 0.10):
        for c_fn in (5, 10, 20, 35, 70, 140, 300):
            cm = D.CostModel(**{**base.to_dict(), "c_fn": c_fn, "base_rate": base_rate})
            single = min(D.sweep_single(val["score"], val["label"], cm),
                         key=lambda r: r["cost_per_1000"])
            band = D.best_band(val["score"], val["label"], cm)
            rows.append({
                "base_rate": base_rate, "c_fn": c_fn, "c_fn_over_c_fp": c_fn / cm.c_fp,
                "best_single_t": single["t_low"], "single_cost": single["cost_per_1000"],
                "band_t_low": band["t_low"], "band_t_high": band["t_high"],
                "band_cost": band["cost_per_1000"], "band_review_share": band["review_share"],
                **{f"ref_{k}": v for k, v in D.reference_policies(cm).items()},
            })
    return pd.DataFrame(rows)


def plot_cost_curve(val_sweep, test_sweep, t_star, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["t_low"] for r in val_sweep], [r["cost_per_1000"] for r in val_sweep], label="validation")
    ax.plot([r["t_low"] for r in test_sweep], [r["cost_per_1000"] for r in test_sweep], label="test", alpha=0.7)
    ax.axvline(t_star, color="k", ls="--", label=f"chosen t = {t_star:.2f}")
    ax.set_xlabel("confidence threshold on 'head' (no review band)")
    ax.set_ylabel("expected cost per 1,000 entries (EUR)")
    ax.set_title("Cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------- main

def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    cm = D.CostModel()

    root = prepare_splits(download_dataset())
    data_yaml, names = write_data_yaml(root)
    lower = [n.lower() for n in names]
    if D.POSITIVE_CLASS not in lower:
        raise SystemExit(f"Positive class '{D.POSITIVE_CLASS}' not in dataset classes {names}")
    pos_idx = lower.index(D.POSITIVE_CLASS)
    print(f"Classes: {names}. Positive class: {names[pos_idx]} (index {pos_idx})")

    stats = describe_dataset(root, names, pos_idx)
    print(json.dumps(stats, indent=2))

    weights = WEIGHTS / "best.pt"
    if os.environ.get("REUSE_WEIGHTS") == "1" and weights.exists():
        print(f"REUSE_WEIGHTS=1: skipping the finetune, evaluating {weights}")
    else:
        weights = finetune(data_yaml)
    det = detection_eval(weights, data_yaml, names)

    coco = YOLO("yolo11n.pt")   # pretrained COCO, never finetuned
    val = image_scores(weights, root, "valid", names, pos_idx, coco)
    test = image_scores(weights, root, "test", names, pos_idx, coco)
    val.to_csv(ARTIFACTS / "scores_valid.csv", index=False)
    test.to_csv(ARTIFACTS / "scores_test.csv", index=False)

    # Trivial baselines: majority image-level class from TRAIN, and predict-nothing.
    train_rate = stats["train"]["image_level_violation_rate"]
    majority = 1.0 if train_rate >= 0.5 else 0.0
    baselines = [
        image_level_summary("finetuned yolo11n", val["score"], val["label"],
                            test["score"], test["label"], cm),
        image_level_summary("COCO yolo11n zero-shot (person => flag)", val["coco_score"],
                            val["label"], test["coco_score"], test["label"], cm),
        image_level_summary(f"majority class ({'violation' if majority else 'compliant'})",
                            np.full(len(val), majority), val["label"],
                            np.full(len(test), majority), test["label"], cm),
        image_level_summary("predict nothing (always pass)", np.zeros(len(val)), val["label"],
                            np.zeros(len(test)), test["label"], cm),
    ]
    detection_baselines = [
        {"model": "finetuned yolo11n", "mAP50": det["map50"], "mAP50-95": det["map50_95"]},
        # COCO has no 'head' or 'helmet' class, so it produces zero true
        # positives on them: AP is 0 by construction, not by measurement.
        {"model": "COCO yolo11n zero-shot", "mAP50": 0.0, "mAP50-95": 0.0},
        {"model": "predict nothing", "mAP50": 0.0, "mAP50-95": 0.0},
    ]

    # Operating point: chosen on VAL, reported on TEST.
    val_sweep = D.sweep_single(val["score"], val["label"], cm)
    test_sweep = D.sweep_single(test["score"], test["label"], cm)
    t_star = float(min(val_sweep, key=lambda r: r["cost_per_1000"])["t_low"])
    band_val = D.best_band(val["score"], val["label"], cm)
    t_low, t_high = float(band_val["t_low"]), float(band_val["t_high"])
    single_test = D.evaluate(test["score"], test["label"], t_star, t_star, cm)
    band_test = D.evaluate(test["score"], test["label"], t_low, t_high, cm)
    plot_cost_curve(val_sweep, test_sweep, t_star, ARTIFACTS / "cost_curve.png")
    pd.DataFrame([{k: v for k, v in r.items() if k != "counts"} | {"split": s}
                  for s, sw in (("valid", val_sweep), ("test", test_sweep)) for r in sw]
                 ).to_csv(ARTIFACTS / "cost_sweep.csv", index=False)

    refs = D.reference_policies(cm)
    policies = pd.DataFrame([
        {"policy": "do nothing (no check)", "cost_per_1000": refs["do_nothing"], "review_share": 0.0},
        {"policy": "guard checks every entry", "cost_per_1000": refs["human_checks_all"], "review_share": 1.0},
        {"policy": f"model alone (t={t_star:.2f})", "cost_per_1000": single_test["cost_per_1000"],
         "review_share": 0.0},
        {"policy": f"model + review band ({t_low:.2f}, {t_high:.2f}]",
         "cost_per_1000": band_test["cost_per_1000"], "review_share": band_test["review_share"]},
    ])
    sens = sensitivity(val, cm)
    sens.to_csv(ARTIFACTS / "sensitivity.csv", index=False)

    strip = lambda r: {k: v for k, v in r.items() if k != "counts"} | {"counts": r["counts"]}
    summary = {
        "config": CONFIG, "classes": names, "positive_class": names[pos_idx],
        "dataset": stats, "detection": det, "detection_baselines": detection_baselines,
        "image_level_baselines": baselines, "cost_model": cm.to_dict(),
        "operating_point": {"single_threshold": t_star, "t_low": t_low, "t_high": t_high,
                            "single_test": strip(single_test), "band_test": strip(band_test),
                            "band_valid": strip(band_val)},
        "reference_policies": refs,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Human-readable tables to quote from in REPORT.md.
    per_class = pd.DataFrame([{"class": k, **v} for k, v in det["per_class"].items()])
    split_tbl = pd.DataFrame([{"split": s, "images": stats[s]["images"],
                               **{f"{n} instances": stats[s]["instances"][n] for n in names},
                               "images with a bare head": f"{stats[s]['image_level_violation_rate']:.1%}"}
                              for s in SPLITS])
    size_tbl = pd.DataFrame([{"class": k, **v} for k, v in stats["object_size"].items()])
    md = [
        "# Results (generated by train.py, do not edit by hand)\n",
        f"Dataset: `{CONFIG['workspace']}/{CONFIG['project']}` v{CONFIG['version']}. "
        f"Positive class: `{names[pos_idx]}`.\n",
        "## Dataset composition\n", fmt_table(split_tbl), "",
        fmt_table(size_tbl), "",
        "## Detection, test split\n", fmt_table(pd.DataFrame(detection_baselines)), "",
        fmt_table(per_class), "",
        "Confusion matrix (rows = predicted, cols = true): " + json.dumps(det["confusion_labels"]),
        "```", "\n".join(" ".join(f"{v:6d}" for v in row) for row in det["confusion_matrix"]), "```", "",
        "## Image-level decision, test split (threshold picked on validation)\n",
        fmt_table(pd.DataFrame(baselines)), "",
        "## Operating point\n",
        f"Single threshold t* = {t_star:.2f} (fail if score > t*); band: pass <= {t_low:.2f} < review <= {t_high:.2f} < fail.\n",
        fmt_table(policies, "{:.2f}"), "",
        f"Band on test: review share {band_test['review_share']:.1%}, "
        f"{band_test['reviews_per_day']:.0f} reviews/day, {band_test['peak_hour_reviews']:.0f} in the peak hour, "
        f"{band_test['guards_needed_peak']:.2f} guards busy in the peak hour; "
        f"{band_test['missed_violations_per_1000']:.2f} missed violations and "
        f"{band_test['false_stops_per_1000']:.1f} false stops per 1,000 entries.\n",
        "## Sensitivity (operating point re-optimised on validation)\n",
        fmt_table(sens[["base_rate", "c_fn", "c_fn_over_c_fp", "best_single_t", "single_cost",
                        "band_t_low", "band_t_high", "band_cost", "band_review_share",
                        "ref_do_nothing", "ref_human_checks_all"]], "{:.3f}"),
    ]
    (ARTIFACTS / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
