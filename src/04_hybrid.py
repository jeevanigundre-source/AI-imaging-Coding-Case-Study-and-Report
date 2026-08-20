"""
Task 4 -- the full hybrid pipeline on the unseen test split.

For every test image:
    raw image -> grayscale -> U-Net mask -> regionprops feature table
              -> numeric evidence -> local LLM -> structured JSON + narrative
              -> audit against the measurements -> row in a DataFrame

The measured numbers, not the LLM, are the source of truth: every field the
pipeline can compute is recomputed after the LLM replies, and any disagreement
is overwritten and logged in `repairs`.  The LLM's only irreplaceable
contribution is the prose.

Outputs -> results/task4_records.csv  (the required aggregated CSV)
           results/json/task4_records.json, results/figures/fig4_*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (DATA, FIGURES, JSONDIR, RESULTS, banner, dice_iou, has_masks,
                    load_gray, load_mask, load_metadata, save_json, set_seed,
                    split_ids)
import features as F
import llm

import importlib
_u = importlib.import_module("03_unet")
DEVICE = _u.DEVICE


def predict_mask(model, gray: np.ndarray, thr: float = 0.5) -> np.ndarray:
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(gray)[None, None].float().to(DEVICE)))
    return p[0, 0].cpu().numpy() > thr


def run_one(model, split: str, iid: str, temperature: float = 0.2,
            seed: int = 42) -> dict:
    """One image through the whole pipeline. Returns the audited record."""
    g = load_gray(split, iid)
    mask = predict_mask(model, g)
    mask = F._drop_small_objects(mask, F.MIN_AREA)       # same debris rule as Task 2
    table = F.feature_table(mask, g)
    ev = F.summarise(table, mask, g)
    ev["image_id"] = iid
    # diagnostic only (not part of the required record): instance count after
    # watershed splitting of touching nuclei -- see the report's discussion of
    # why high pixel Dice coexists with a large under-count.
    n_split = int(F.watershed_split(mask).max())

    prompt = llm.HYBRID_PROMPT.format(image_id=iid, summary=F.summary_text(ev))
    r = llm.ask(prompt, model=llm.TEXT_MODEL, temperature=temperature, seed=seed,
                stub_kind="hybrid", stub_evidence=ev)
    parts = llm.split_sections(r["text"])
    rec = llm.extract_json(parts.get("json", r["text"]),
                           required=["image_id", "n_objects", "mean_area",
                                     "density_class", "quality_flag"])

    # ---------------------- audit: measurements overrule the language model ---
    truth_fields = llm._classify(ev) | {"image_id": iid,
                                        "n_objects": ev["n_objects"],
                                        "mean_area": round(ev["mean_area"], 1)}
    repairs = []
    if rec is None:
        rec = {k: truth_fields[k] for k in
               ("image_id", "n_objects", "mean_area", "density_class", "quality_flag")}
        repairs.append("no valid JSON returned; record rebuilt from measurements")
    else:
        for k in ("image_id", "n_objects", "mean_area", "density_class", "quality_flag"):
            want = truth_fields[k]
            got = rec.get(k)
            same = (abs(float(got) - float(want)) < 0.05
                    if isinstance(want, (int, float)) and isinstance(got, (int, float))
                    else got == want)
            if not same:
                repairs.append(f"{k}: {got!r} -> {want!r}")
                rec[k] = want

    out = {**rec,
           "narrative": parts.get("narrative", "").strip(),
           "n_repairs": len(repairs), "repairs": repairs,
           "llm_json_parsed": parts.get("json") is not None,
           "generator": r["generator"], "provenance": r["provenance"],
           "n_objects_split": n_split,
           "mean_solidity": ev["mean_solidity"],
           "mean_eccentricity": ev["mean_eccentricity"],
           "foreground_fraction": ev["foreground_fraction"],
           "mean_intensity": ev["mean_intensity"],
           "evidence": ev, "prompt": prompt, "llm_raw": r["text"]}

    if has_masks(split):
        gt = load_mask(split, iid)
        d, i = dice_iou(mask, gt)
        out["dice_vs_gt"], out["iou_vs_gt"] = round(d, 4), round(i, 4)
    out["_mask"] = mask
    return out


def main(split: str = "test") -> pd.DataFrame:
    set_seed()
    banner(f"TASK 4 -- hybrid pipeline on the unseen '{split}' split")
    model = _u.load_trained("bce_dice")
    meta = load_metadata()

    records, masks = [], {}
    for iid in split_ids(split):
        rec = run_one(model, split, iid)
        masks[iid] = rec.pop("_mask")
        records.append(rec)
        print(f"{iid}: n={rec['n_objects']:3d}  mean_area={rec['mean_area']:7.1f}  "
              f"{rec['density_class']:<9} {rec['quality_flag']:<6} "
              f"Dice={rec.get('dice_vs_gt', float('nan')):.3f}  "
              f"repairs={rec['n_repairs']}", flush=True)

    cols = ["image_id", "n_objects", "mean_area", "density_class", "quality_flag",
            "n_objects_split", "mean_solidity", "mean_eccentricity", "foreground_fraction",
            "mean_intensity", "dice_vs_gt", "iou_vs_gt", "llm_json_parsed",
            "n_repairs", "generator", "narrative"]
    df = pd.DataFrame(records)
    df = df[[c for c in cols if c in df.columns]]
    if split == "test":
        df = df.merge(meta[["image_id", "n_objects", "density", "area_fraction"]]
                      .rename(columns={"n_objects": "n_objects_true",
                                       "density": "density_true",
                                       "area_fraction": "area_fraction_true"}),
                      on="image_id", how="left")
    csv_path = RESULTS / f"task4_records{'' if split == 'test' else '_' + split}.csv"
    df.to_csv(csv_path, index=False)
    save_json(records, JSONDIR / f"task4_records{'' if split == 'test' else '_' + split}.json")

    print("\nAggregated DataFrame ->", csv_path)
    print(df.drop(columns=["narrative"]).to_string(index=False))
    if "dice_vs_gt" in df:
        print(f"\nTEST mean Dice {df.dice_vs_gt.mean():.4f}  "
              f"mean IoU {df.iou_vs_gt.mean():.4f}")
    if "n_objects_true" in df:
        err = df.n_objects - df.n_objects_true
        err2 = df.n_objects_split - df.n_objects_true
        print(f"Count error (connected components): mean {err.mean():+.1f}, "
              f"MAE {err.abs().mean():.1f}, MAPE {(err.abs() / df.n_objects_true).mean() * 100:.1f}%")
        print(f"Count error (after watershed split): mean {err2.mean():+.1f}, "
              f"MAE {err2.abs().mean():.1f}, MAPE {(err2.abs() / df.n_objects_true).mean() * 100:.1f}%")
        area_err = ((df.foreground_fraction - df.area_fraction_true)
                    / df.area_fraction_true * 100)
        print(f"Total nuclear area error: mean {area_err.mean():+.2f}%, "
              f"MAPE {area_err.abs().mean():.2f}%  <- area is accurate, count is not")
        agree = (df.density_class == df.density_true).mean()
        print(f"density_class agrees with the dataset's own label on "
              f"{agree * 100:.0f}% of test images")
        save_json({"mean_dice": float(df.dice_vs_gt.mean()),
                   "mean_iou": float(df.iou_vs_gt.mean()),
                   "count_bias_cc": float(err.mean()),
                   "count_mae_cc": float(err.abs().mean()),
                   "count_mape_cc": float((err.abs() / df.n_objects_true).mean() * 100),
                   "count_bias_watershed": float(err2.mean()),
                   "count_mae_watershed": float(err2.abs().mean()),
                   "count_mape_watershed": float((err2.abs() / df.n_objects_true).mean() * 100),
                   "area_fraction_mape": float(area_err.abs().mean()),
                   "density_class_agreement": float(agree),
                   "total_repairs": int(df.n_repairs.sum()),
                   "generator": df.generator.iloc[0]},
                  JSONDIR / "task4_metrics.json")

    print("\nEXAMPLE RECORD + NARRATIVE\n" + "-" * 60)
    ex = records[len(records) // 2]
    print(json.dumps({k: ex[k] for k in ("image_id", "n_objects", "mean_area",
                                         "density_class", "quality_flag")}, indent=2))
    print("\n" + ex["narrative"])

    # ---- Fig 4a: qualitative panel over the test split ----------------------
    ids = split_ids(split)[:6]
    fig, axes = plt.subplots(2, len(ids), figsize=(2.4 * len(ids), 5.2))
    for j, iid in enumerate(ids):
        rec = next(r for r in records if r["image_id"] == iid)
        axes[0, j].imshow(load_gray(split, iid), cmap="gray")
        axes[0, j].set_title(iid, fontsize=8.5)
        axes[1, j].imshow(masks[iid], cmap="gray")
        axes[1, j].set_title(f"n={rec['n_objects']}  {rec['density_class']}", fontsize=8.5)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.suptitle(f"Fig. 4a  Hybrid pipeline on unseen {split} images "
                 f"(top: input, bottom: U-Net mask)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / f"fig4a_{split}_panel.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    # ---- Fig 4b: counts and Dice ------------------------------------------
    if "n_objects_true" in df:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        x = np.arange(len(df))
        ax[0].bar(x - .27, df.n_objects_true, .27, label="ground truth", color="#55a868")
        ax[0].bar(x, df.n_objects, .27, label="connected components", color="#4c72b0")
        ax[0].bar(x + .27, df.n_objects_split, .27, label="+ watershed split",
                  color="#dd8452")
        ax[0].set_xticks(x); ax[0].set_xticklabels(df.image_id, rotation=90, fontsize=6)
        ax[0].set_ylabel("objects"); ax[0].legend(fontsize=8)
        ax[0].set_title("Object count, test split", fontsize=10)
        ax[1].bar(x, df.dice_vs_gt, color="#c44e52")
        ax[1].set_xticks(x); ax[1].set_xticklabels(df.image_id, rotation=90, fontsize=6)
        ax[1].set_ylim(0, 1); ax[1].set_ylabel("Dice")
        ax[1].axhline(df.dice_vs_gt.mean(), ls="--", color="k", lw=1,
                      label=f"mean {df.dice_vs_gt.mean():.3f}")
        ax[1].legend(fontsize=8); ax[1].set_title("Pixel overlap, test split", fontsize=10)
        fig.suptitle("Fig. 4b  Test-split accuracy: pixels are easy, counting is not",
                     fontsize=11)
        fig.tight_layout(); fig.savefig(FIGURES / "fig4b_test_metrics.png", dpi=140,
                                        bbox_inches="tight"); plt.close(fig)
    return df


if __name__ == "__main__":
    main("test")
    print("\nTask 4 complete.")
