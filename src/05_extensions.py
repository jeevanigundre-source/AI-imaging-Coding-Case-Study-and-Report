"""
Extensions for extra credit.

  A) ROBUSTNESS.  The dataset's own corrupted variants (heavy blur, low
     contrast) plus a self-generated additive-noise variant are pushed through
     the whole pipeline.  At every stage -- raw pixels, U-Net mask, Otsu mask,
     feature table, JSON record, narrative -- we measure how far the corrupted
     result has drifted from the clean one, so the *earliest stage at which the
     corruption is detectable* can be named rather than guessed.

  B) LOSS ABLATION.  The same U-Net is trained three times with BCE, soft Dice
     and BCE+Dice and compared on validation Dice/IoU.

Run:  python 05_extensions.py --part all       (trains 2 extra U-Nets)
      python 05_extensions.py --part robust    (robustness only, fast)

Outputs -> results/figures/fig5a_robustness.png, fig5b_loss_ablation.png
           results/json/task5_robustness.json, task5_loss_ablation.json
           results/task5_robustness.csv
"""
from __future__ import annotations

import argparse
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from common import (DATA, FIGURES, JSONDIR, RESULTS, banner, dice_iou, load_gray,
                    load_mask, save_json, set_seed, to_gray)
import features as F
import llm

_u = importlib.import_module("03_unet")
_h = importlib.import_module("04_hybrid")


# ============================================================ A) ROBUSTNESS
def add_noise(gray: np.ndarray, sigma: float = 0.10, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(gray + rng.normal(0, sigma, gray.shape), 0, 1).astype(np.float32)


def load_corrupted_gray(name: str) -> np.ndarray:
    im = Image.open(DATA / "test_corrupted" / "images" / f"{name}.png").convert("RGB")
    return to_gray(np.asarray(im, dtype=np.float32) / 255.0)


def stage_metrics(model, gray: np.ndarray, gt: np.ndarray, iid: str,
                  ref: dict | None = None) -> dict:
    """Push one image through every stage and record what each stage says."""
    unet = _h.predict_mask(model, gray)
    unet = F._drop_small_objects(unet, F.MIN_AREA)
    otsu, thr = F.otsu_mask(gray)
    tbl = F.feature_table(unet, gray)
    ev = F.summarise(tbl, unet, gray)
    ev["image_id"] = iid
    d_u, i_u = dice_iou(unet, gt)
    d_o, i_o = dice_iou(otsu, gt)

    prompt = llm.HYBRID_PROMPT.format(image_id=iid, summary=F.summary_text(ev))
    r = llm.ask(prompt, model=llm.TEXT_MODEL, temperature=0.2, seed=42,
                stub_kind="hybrid", stub_evidence=ev)
    parts = llm.split_sections(r["text"])
    rec = llm.extract_json(parts.get("json", r["text"])) or {}
    rec = llm._classify(ev) | {"image_id": iid, "n_objects": ev["n_objects"],
                               "mean_area": round(ev["mean_area"], 1)}

    # reference-free input QC: absolute rules that need no clean comparison image
    qc = []
    if float(gray.std()) < 0.020:
        qc.append("low_dynamic_range")
    if float(gray.mean()) > 0.15:
        qc.append("elevated_background")
    lap = ndi.laplace(gray)
    focus = float(lap.var())
    if focus < 2e-4:
        qc.append("out_of_focus")

    out = {
        "image_id": iid,
        "stage1_pixel_mean": round(float(gray.mean()), 4),
        "stage1_pixel_std": round(float(gray.std()), 4),
        "stage1_focus_laplacian_var": round(focus, 6),
        "stage1_qc_flags": ";".join(qc) or "none",
        "stage1_otsu_threshold": round(thr, 4),
        "stage2_unet_dice": round(d_u, 4), "stage2_unet_iou": round(i_u, 4),
        "stage2_otsu_dice": round(d_o, 4), "stage2_otsu_iou": round(i_o, 4),
        "stage3_n_objects": ev["n_objects"],
        "stage3_mean_area": ev["mean_area"],
        "stage3_foreground_fraction": ev["foreground_fraction"],
        "stage3_mean_solidity": ev["mean_solidity"],
        "stage3_mean_intensity": ev["mean_intensity"],
        "stage4_density_class": rec["density_class"],
        "stage4_quality_flag": rec["quality_flag"],
        "stage5_narrative": parts.get("narrative", "").strip(),
        "_mask": unet, "_gray": gray, "_otsu": otsu,
    }
    if ref is not None:                       # drift relative to the clean image
        out["drift_pixel_std_pct"] = round(
            100 * (out["stage1_pixel_std"] - ref["stage1_pixel_std"]) /
            max(ref["stage1_pixel_std"], 1e-9), 1)
        out["drift_unet_dice"] = round(out["stage2_unet_dice"] - ref["stage2_unet_dice"], 4)
        out["drift_n_objects"] = out["stage3_n_objects"] - ref["stage3_n_objects"]
        out["drift_mean_area_pct"] = round(
            100 * (out["stage3_mean_area"] - ref["stage3_mean_area"]) /
            max(ref["stage3_mean_area"], 1e-9), 1)
        out["drift_density_class"] = (out["stage4_density_class"] !=
                                      ref["stage4_density_class"])
        out["drift_quality_flag"] = (out["stage4_quality_flag"] !=
                                     ref["stage4_quality_flag"])
    return out


def robustness(base_id: str = "test_000") -> pd.DataFrame:
    banner("EXTENSION A -- robustness: tracing a corruption through the pipeline")
    model = _u.load_trained("bce_dice")
    gt = load_mask("test", base_id)

    variants = {
        "clean": load_gray("test", base_id),
        "blur": load_corrupted_gray(f"{base_id}_blur"),
        "low_contrast": load_corrupted_gray(f"{base_id}_lowcontrast"),
        "noise_sigma0.10": add_noise(load_gray("test", base_id), 0.10),
    }
    ref = stage_metrics(model, variants["clean"], gt, base_id)
    rows = [ref]
    for name, g in list(variants.items())[1:]:
        rows.append(stage_metrics(model, g, gt, f"{base_id}_{name}", ref=ref))

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df["variant"] = list(variants)
    df.to_csv(RESULTS / "task5_robustness.csv", index=False)
    show = [c for c in df.columns if c.startswith(("stage1_pixel", "stage1_qc", "stage1_focus", "stage2_unet", "stage3_n",
                                                   "stage3_mean_area", "stage4"))]
    print(df[["variant"] + show].to_string(index=False))

    # Earliest detectable stage, under two regimes:
    #   (a) reference-based -- we happen to hold the clean image and can diff
    #   (b) reference-free  -- what a deployed system could actually notice
    earliest = {}
    for r in rows[1:]:
        if abs(r["drift_pixel_std_pct"]) > 10:
            a = "stage 1 (raw pixel statistics)"
        elif abs(r["drift_unet_dice"]) > 0.02:
            a = "stage 2 (segmentation mask)"
        elif abs(r["drift_n_objects"]) > 0 or abs(r["drift_mean_area_pct"]) > 10:
            a = "stage 3 (feature table)"
        elif r["drift_density_class"] or r["drift_quality_flag"]:
            a = "stage 4 (structured JSON)"
        else:
            a = "not detected before the narrative (stage 5)"

        if r["stage1_qc_flags"] != "none":
            b = f"stage 1 (input QC: {r['stage1_qc_flags']})"
        elif r["stage4_quality_flag"] == "review":
            b = "stage 4 (quality_flag = review)"
        elif r["drift_density_class"]:
            b = ("stage 4 only as a changed density_class -- silently wrong, "
                 "no flag raised")
        else:
            b = "NOT DETECTED -- the record looks clean but the numbers are wrong"
        earliest[r["image_id"]] = {"reference_based": a, "reference_free": b}
        print(f"  {r['image_id']}:\n      with a clean reference : {a}"
              f"\n      reference-free (deployable): {b}")

    save_json({"base_image": base_id,
               "tolerances": {"pixel_std_pct": 10, "unet_dice": 0.02,
                              "mean_area_pct": 10},
               "reference_free_qc_rules": {"low_dynamic_range": "pixel std < 0.020",
                                           "elevated_background": "pixel mean > 0.15",
                                           "out_of_focus": "var(Laplacian) < 2e-4"},
               "earliest_detectable_stage": earliest,
               "table": df.to_dict("records")},
              JSONDIR / "task5_robustness.json")

    # --- figure
    fig, axes = plt.subplots(3, len(rows), figsize=(3.0 * len(rows), 8.4))
    for j, (r, name) in enumerate(zip(rows, variants)):
        axes[0, j].imshow(r["_gray"], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"{name}\nstd={r['stage1_pixel_std']:.3f}", fontsize=9)
        axes[1, j].imshow(r["_mask"], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"U-Net Dice {r['stage2_unet_dice']:.3f}\n"
                             f"n={r['stage3_n_objects']}", fontsize=9)
        axes[2, j].imshow(r["_otsu"], cmap="gray", vmin=0, vmax=1)
        axes[2, j].set_title(f"Otsu Dice {r['stage2_otsu_dice']:.3f}\n"
                             f"record: {r['stage4_density_class']}"
                             f" / {r['stage4_quality_flag']}", fontsize=9)
        for i in range(3):
            axes[i, j].axis("off")
    fig.suptitle("Fig. 5a  Corruption propagation (rows: input, U-Net mask, Otsu mask)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig5a_robustness.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    return df


# ========================================================= B) LOSS ABLATION
def loss_ablation(epochs: int = 40) -> pd.DataFrame:
    banner("EXTENSION B -- loss ablation: BCE vs Dice vs BCE+Dice")
    rows, hists = [], {}
    for name in ["bce", "dice", "bce_dice"]:
        hp = JSONDIR / f"task3_history_{name}.json"
        if hp.exists() and json.loads(hp.read_text())["epoch"][-1] == epochs:
            hist = json.loads(hp.read_text())
            model = _u.load_trained(name)
            va = _u.NucleiDataset("val", augment=False)
            print(f"[{name}] reusing cached run ({hp.name})")
        else:
            model, hist, va = _u.train(name, epochs=epochs, quiet=False)
        per = _u.evaluate(model, va)
        hists[name] = hist
        rows.append({"loss": name,
                     "mean_val_dice": round(float(per.dice.mean()), 4),
                     "mean_val_iou": round(float(per.iou.mean()), 4),
                     "worst_image_dice": round(float(per.dice.min()), 4),
                     "best_epoch": hist["best"]["epoch"],
                     "best_epoch_dice": round(hist["best"]["dice"], 4),
                     "train_seconds": hist["train_seconds"]})
    df = pd.DataFrame(rows).sort_values("mean_val_dice", ascending=False)
    print("\n", df.to_string(index=False))
    save_json({"epochs": epochs, "results": df.to_dict("records")},
              JSONDIR / "task5_loss_ablation.json")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, c in zip(["bce", "dice", "bce_dice"], ["#4c72b0", "#55a868", "#c44e52"]):
        h = hists[name]
        ax[0].plot(h["epoch"], h["val_dice"], color=c, label=name)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation Dice"); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("Validation Dice by loss", fontsize=10)
    ax[1].bar(df.loss, df.mean_val_dice, color="#4c72b0", label="mean Dice")
    ax[1].bar(df.loss, df.worst_image_dice, color="#dd8452", width=.45,
              label="worst image")
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("Final model, validation split", fontsize=10)
    for i, v in enumerate(df.mean_val_dice):
        ax[1].text(i, v + .02, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Fig. 5b  Loss ablation", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig5b_loss_ablation.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    return df


# ================================================ C) INTENSITY-ROBUST TRAINING
def robust_training(epochs: int = 40) -> pd.DataFrame:
    """
    The single change proposed in the report: train with per-image percentile
    contrast normalisation plus photometric augmentation (random gain, offset,
    blur and noise), so that the network stops keying on the absolute intensity
    scale of this one synthetic acquisition.

    Evaluated on the clean validation split AND on all four corruption variants.
    """
    banner("EXTENSION C -- intensity-robust training (normalisation + photometric aug)")
    tag = "bce_dice_norm"
    hp = JSONDIR / f"task3_history_{tag}.json"
    if hp.exists() and json.loads(hp.read_text())["epoch"][-1] == epochs:
        model = _u.load_trained(tag)
        print(f"[{tag}] reusing cached run")
    else:
        model, _, _ = _u.train("bce_dice", epochs=epochs, normalize=True,
                               photometric=True, tag=tag)
    base_model = _u.load_trained("bce_dice")

    va_plain = _u.NucleiDataset("val", augment=False)
    va_norm = _u.NucleiDataset("val", augment=False, normalize=True)
    clean_base = _u.evaluate(base_model, va_plain).dice.mean()
    clean_rob = _u.evaluate(model, va_norm).dice.mean()

    base_id = "test_000"
    gt = load_mask("test", base_id)
    variants = {
        "clean": load_gray("test", base_id),
        "blur": load_corrupted_gray(f"{base_id}_blur"),
        "low_contrast": load_corrupted_gray(f"{base_id}_lowcontrast"),
        "noise_sigma0.10": add_noise(load_gray("test", base_id), 0.10),
    }
    rows = [{"variant": "val split (20 images, mean)",
             "baseline_dice": round(float(clean_base), 4),
             "robust_dice": round(float(clean_rob), 4)}]
    for name, g in variants.items():
        mb = F._drop_small_objects(_h.predict_mask(base_model, g), F.MIN_AREA)
        mr = F._drop_small_objects(_h.predict_mask(model, _u.percentile_norm(g)),
                                   F.MIN_AREA)
        rows.append({"variant": name,
                     "baseline_dice": round(dice_iou(mb, gt)[0], 4),
                     "robust_dice": round(dice_iou(mr, gt)[0], 4)})
    df = pd.DataFrame(rows)
    df["delta"] = (df.robust_dice - df.baseline_dice).round(4)
    print("\n", df.to_string(index=False))
    df.to_csv(RESULTS / "task5_robust_training.csv", index=False)
    save_json(df.to_dict("records"), JSONDIR / "task5_robust_training.json")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(df))
    ax.bar(x - .2, df.baseline_dice, .4, label="baseline U-Net", color="#4c72b0")
    ax.bar(x + .2, df.robust_dice, .4, label="+ norm & photometric aug",
           color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" (20 images, mean)", "") for v in df.variant],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Fig. 5c  Intensity normalisation restores robustness", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "fig5c_robust_training.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "robust", "ablation", "robust_training"])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    set_seed()
    if a.part in ("all", "robust"):
        robustness()
    if a.part in ("all", "ablation"):
        loss_ablation(a.epochs)
    if a.part in ("all", "robust_training"):
        robust_training(a.epochs)
    print("\nExtensions complete.")
