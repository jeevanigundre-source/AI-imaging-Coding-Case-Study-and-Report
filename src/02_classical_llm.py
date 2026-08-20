"""
Task 2 -- classical features and numbers-first LLM interpretation.

  * Otsu threshold + morphological cleanup + connected-component labelling
  * per-object feature table via skimage.measure.regionprops_table
  * the table is collapsed to ~14 numbers, rendered as text, and passed to a
    local text LLM.  The model NEVER sees the image.
  * the model returns one paragraph + a JSON record
    (n_objects, density_class, shape_regularity, quality_flag)
  * quantitative check: Otsu Dice/IoU against ground truth on the full
    validation split, plus a grayscale-conversion ablation (luma vs blue channel)
  * counting check: detected n_objects vs the ground-truth count, per density

Outputs -> results/figures/fig2_*.png, results/json/task2_*.json,
           results/json/task2_feature_table_<id>.csv
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage.measure import label

from common import (FIGURES, JSONDIR, banner, dice_iou, load_gray, load_mask,
                    load_metadata, save_json, set_seed, split_ids)
import features as F
import llm


def otsu_benchmark(meta: pd.DataFrame) -> pd.DataFrame:
    """Otsu quality on the validation split, for two grayscale conversions."""
    banner("TASK 2a -- Otsu segmentation benchmark on the validation split")
    rows = []
    for iid in split_ids("val"):
        gt = load_mask("val", iid)
        truth = meta.loc[meta.image_id == iid].iloc[0]
        for mode in ("luma", "blue"):
            g = load_gray("val", iid, mode=mode)
            m, thr = F.otsu_mask(g)
            d, i = dice_iou(m, gt)
            n_det = int(label(m).max())
            rows.append(dict(image_id=iid, density=truth.density, gray=mode,
                             threshold=round(thr, 4), dice=d, iou=i,
                             n_true=int(truth.n_objects), n_detected=n_det,
                             count_err=n_det - int(truth.n_objects)))
    df = pd.DataFrame(rows)
    df.to_csv(JSONDIR / "task2_otsu_per_image.csv", index=False)
    agg = (df.groupby(["gray", "density"])[["dice", "iou", "count_err"]]
             .mean().round(3).reset_index())
    overall = df.groupby("gray")[["dice", "iou", "count_err"]].mean().round(3)
    print(agg.to_string(index=False))
    print("\nOverall:\n", overall.to_string())
    save_json({"per_density": agg.to_dict("records"),
               "overall": overall.reset_index().to_dict("records")},
              JSONDIR / "task2_otsu_metrics.json")
    return df


def analyse_image(split: str, iid: str, gray_mode: str = "luma") -> dict:
    """Full classical route for one image: mask -> table -> evidence."""
    g = load_gray(split, iid, mode=gray_mode)
    raw, thr = F.otsu_mask(g, clean=False)
    m, _ = F.otsu_mask(g, clean=True)
    df = F.feature_table(m, g)
    ev = F.summarise(df, m, g)
    return {"gray": g, "raw_mask": raw, "mask": m, "table": df,
            "evidence": ev, "threshold": thr}


def main() -> None:
    set_seed()
    meta = load_metadata()
    bench = otsu_benchmark(meta)

    # ---------------- representative image, same one used in Task 1 -----------
    sub = meta[meta.split == "val"]
    iid = sub.iloc[(sub.n_objects - meta.n_objects.median()).abs().argsort().iloc[0]].image_id
    banner(f"TASK 2b -- classical features for {iid}")
    a = analyse_image("val", iid)
    gt = load_mask("val", iid)
    truth = meta.loc[meta.image_id == iid].iloc[0].to_dict()
    d, i = dice_iou(a["mask"], gt)
    print(f"Otsu threshold {a['threshold']:.4f} | Dice {d:.3f} IoU {i:.3f} | "
          f"detected {a['evidence']['n_objects']} vs true {truth['n_objects']}")

    a["table"].round(3).to_csv(JSONDIR / f"task2_feature_table_{iid}.csv", index=False)
    print("\nHead of regionprops_table:\n",
          a["table"][["label", "area", "eccentricity", "solidity", "extent",
                      "mean_intensity", "circularity"]].head(8).round(3).to_string(index=False))

    # ---------------- figure: pipeline stages + feature distributions ---------
    lab = label(a["mask"])
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
    ax[0].imshow(a["gray"], cmap="gray"); ax[0].set_title(f"{iid} grayscale", fontsize=9)
    ax[1].imshow(a["raw_mask"], cmap="gray")
    ax[1].set_title(f"raw Otsu (t={a['threshold']:.3f})", fontsize=9)
    ax[2].imshow(a["mask"], cmap="gray"); ax[2].set_title("after morphology", fontsize=9)
    ax[3].imshow(label2rgb(lab, bg_label=0))
    ax[3].set_title(f"{lab.max()} labelled objects", fontsize=9)
    for k in range(4):
        ax[k].axis("off")
    ax[4].scatter(a["table"].area, a["table"].eccentricity,
                  c=a["table"].solidity, cmap="viridis", s=26, edgecolor="k", lw=.3)
    ax[4].set_xlabel("area (px)", fontsize=8); ax[4].set_ylabel("eccentricity", fontsize=8)
    ax[4].set_title("per-object features\n(colour = solidity)", fontsize=9)
    ax[4].tick_params(labelsize=7)
    fig.suptitle("Fig. 2a  Classical route: Otsu -> morphology -> labelling -> features",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig2a_classical.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    # ---------------- figure: Otsu accuracy vs density ------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    order = ["sparse", "normal", "dense", "clustered"]
    for mode, c in [("luma", "#4c72b0"), ("blue", "#c44e52")]:
        s = bench[bench.gray == mode].groupby("density").dice.mean().reindex(order)
        ax[0].plot(order, s.values, "o-", color=c, label=f"{mode} grayscale")
    ax[0].set_ylabel("mean Dice"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[0].set_title("Otsu Dice by density regime", fontsize=10); ax[0].grid(alpha=.3)
    lu = bench[bench.gray == "luma"]
    ax[1].scatter(lu.n_true, lu.n_detected, c="#4c72b0", s=26)
    lim = [0, max(lu.n_true.max(), lu.n_detected.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("true object count"); ax[1].set_ylabel("Otsu detected count")
    ax[1].set_title("Under-counting from merged nuclei", fontsize=10); ax[1].grid(alpha=.3)
    fig.suptitle("Fig. 2b  Where classical thresholding breaks down", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig2b_otsu_limits.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    # ---------------- numbers-first LLM --------------------------------------
    banner("TASK 2c -- numbers-first LLM interpretation (model never sees the image)")
    summary = F.summary_text(a["evidence"])
    prompt = llm.NUMBERS_FIRST_PROMPT.format(summary=summary)
    print("PROMPT SENT:\n" + prompt[:1400] + "\n...")
    r = llm.ask(prompt, model=llm.TEXT_MODEL, temperature=0.2, seed=11,
                stub_kind="numbers", stub_evidence=a["evidence"])
    parts = llm.split_sections(r["text"])
    rec = llm.extract_json(parts.get("json", r["text"]),
                           required=["n_objects", "density_class",
                                     "shape_regularity", "quality_flag"])

    # --- audit: the JSON must agree with the measurements, or we repair it ----
    audit = {"llm_json_parsed": rec is not None, "repairs": []}
    measured_n = a["evidence"]["n_objects"]
    if rec is None:
        rec = llm._classify(a["evidence"]) | {"n_objects": measured_n}
        audit["repairs"].append("no valid JSON returned; record rebuilt from measurements")
    elif int(rec.get("n_objects", -1)) != measured_n:
        audit["repairs"].append(
            f"n_objects {rec.get('n_objects')} -> {measured_n} (measured value wins)")
        rec["n_objects"] = measured_n
    print("\nNARRATIVE:\n", parts.get("paragraph", "(none)"))
    print("\nJSON RECORD:\n", json.dumps(rec, indent=2))
    print("AUDIT:", audit)

    out = {
        "image_id": iid,
        "ground_truth": truth,
        "prompt": prompt,
        "prompt_template": llm.NUMBERS_FIRST_PROMPT,
        "evidence_sent_to_llm": a["evidence"],
        "otsu": {"threshold": a["threshold"], "dice": d, "iou": i,
                 "n_detected": measured_n, "n_true": int(truth["n_objects"])},
        "llm_raw": r["text"],
        "paragraph": parts.get("paragraph"),
        "json_record": rec,
        "audit": audit,
        "generator": r["generator"],
        "provenance": r["provenance"],
    }
    save_json(out, JSONDIR / "task2_numbers_first.json")
    print("\nTask 2 complete -> results/figures/fig2*.png, results/json/task2_*")


if __name__ == "__main__":
    main()
