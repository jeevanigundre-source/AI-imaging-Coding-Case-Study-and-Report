"""
Task 1 -- data preparation, EDA, and multimodal-LLM (VLM) description.

  * loads the dataset, converts to grayscale and to a common 256x256 size
  * EDA: sample-image grid, intensity histograms, per-density summary table
  * sends a representative image to a local vision model via Ollama with
      (a) a naive prompt and (b) an optimised, structured, JSON-forcing prompt
  * repeats the structured prompt N times at temperature 0.8 to show that
    repeated runs are NOT identical, and once at temperature 0.0
  * measures schema compliance of both prompts over a batch of images

Outputs -> results/figures/fig1_*.png, results/json/task1_*.json
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIGURES, JSONDIR, banner, load_gray, load_metadata, load_rgb,
                    load_mask, save_json, set_seed, split_ids)
import llm

N_REPEATS = 3
BATCH_FOR_COMPLIANCE = 6
REPRESENTATIVE_SPLIT = "val"


def representative_id(meta: pd.DataFrame) -> str:
    """The validation image whose object count is closest to the dataset median --
    a defensible, reproducible choice of 'representative' rather than an ad-hoc one."""
    sub = meta[meta.split == REPRESENTATIVE_SPLIT]
    target = meta.n_objects.median()
    return sub.iloc[(sub.n_objects - target).abs().argsort().iloc[0]].image_id


# ----------------------------------------------------------------------- EDA
def eda() -> pd.DataFrame:
    banner("TASK 1a -- data preparation and EDA")
    meta = load_metadata()
    print(meta.groupby("split")["n_objects"].describe()[["count", "mean", "min", "max"]])

    summary = (meta.groupby(["split", "density"])
                   .agg(images=("image_id", "count"),
                        mean_objects=("n_objects", "mean"),
                        mean_area_fraction=("area_fraction", "mean"))
                   .round(3).reset_index())
    summary.to_csv(JSONDIR / "task1_eda_summary.csv", index=False)
    print("\n", summary.to_string(index=False))

    # --- Figure 1a: raw RGB / grayscale / ground truth for 4 density regimes
    picks = []
    for dens in ["sparse", "normal", "dense", "clustered"]:
        sub = meta[(meta.split == "train") & (meta.density == dens)]
        if len(sub):
            picks.append((dens, sub.iloc[0].image_id))
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 9))
    for j, (dens, iid) in enumerate(picks):
        rgb, gray, gt = load_rgb("train", iid), load_gray("train", iid), load_mask("train", iid)
        n = int(meta.loc[meta.image_id == iid, "n_objects"].iloc[0])
        for i, (img, cmap, ttl) in enumerate([(rgb, None, f"{dens}  (n={n})"),
                                              (gray, "gray", "grayscale (luma)"),
                                              (gt, "gray", "ground-truth mask")]):
            ax = axes[i, j]
            ax.imshow(img, cmap=cmap)
            ax.set_title(ttl, fontsize=9)
            ax.axis("off")
    fig.suptitle("Fig. 1a  Sample images by density regime (256x256)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1a_samples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 1b: intensity histograms
    ids = split_ids("train")
    all_px = np.concatenate([load_gray("train", i).ravel() for i in ids[:40]])
    fg, bg = [], []
    for i in ids[:40]:
        g, m = load_gray("train", i), load_mask("train", i)
        fg.append(g[m]); bg.append(g[~m])
    fg, bg = np.concatenate(fg), np.concatenate(bg)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(all_px, bins=100, color="#4c72b0", log=True)
    ax[0].set_title("All pixels, 40 training images", fontsize=10)
    ax[0].set_xlabel("grayscale intensity"); ax[0].set_ylabel("count (log)")
    ax[1].hist(bg, bins=100, alpha=.7, label=f"background (mean {bg.mean():.3f})",
               color="#55a868", density=True)
    ax[1].hist(fg, bins=100, alpha=.7, label=f"nuclei (mean {fg.mean():.3f})",
               color="#c44e52", density=True)
    ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
    ax[1].set_title("Class-conditional intensity", fontsize=10)
    ax[1].set_xlabel("grayscale intensity")
    fig.suptitle("Fig. 1b  Intensity distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1b_histograms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_images": {s: len(split_ids(s)) for s in
                     ["train", "val", "test", "test_corrupted"]},
        "image_size": [256, 256],
        "grayscale_conversion": "ITU-R 601-2 luma (skimage.color.rgb2gray equivalent)",
        "foreground_mean_intensity": round(float(fg.mean()), 4),
        "background_mean_intensity": round(float(bg.mean()), 4),
        "foreground_fraction_mean": round(float(meta.area_fraction.mean()), 4),
        "objects_per_image": {"min": int(meta.n_objects.min()),
                              "max": int(meta.n_objects.max()),
                              "mean": round(float(meta.n_objects.mean()), 1)},
        "density_regimes": dict(Counter(meta.density)),
        "channel_note": ("nuclei are DAPI-like blue; luma weights B by 0.0721, so "
                         "grayscale conversion costs ~3x foreground contrast versus "
                         "using the blue channel (ablated in 02_classical_llm.py)"),
    }
    save_json(stats, JSONDIR / "task1_eda_stats.json")
    print("\nEDA stats:", json.dumps(stats, indent=2))
    return meta


# --------------------------------------------------------------- VLM section
def image_evidence(split: str, iid: str) -> dict:
    """Coarse statistics used only to condition the offline stub (see llm.py).
    When Ollama is available these values are ignored."""
    g = load_gray(split, iid)
    thr = g.mean() + 2 * g.std()
    fgfrac = float((g > thr).mean())
    n_hint = "a few" if fgfrac < 0.03 else ("many" if fgfrac > 0.12 else "several")
    return {"quality": "good" if g.std() > 0.05 else "moderate",
            "n_hint": n_hint,
            "touching": fgfrac > 0.10,
            "low_contrast": float(g.std()) < 0.05,
            "blurred": False}


def vlm(meta: pd.DataFrame) -> dict:
    banner("TASK 1b -- multimodal LLM description")
    split, iid = REPRESENTATIVE_SPLIT, representative_id(meta)
    rgb = load_rgb(split, iid)
    gray = load_gray(split, iid)
    ev = image_evidence(split, iid)
    truth = meta.loc[meta.image_id == iid].iloc[0].to_dict()
    print(f"Representative image: {iid}  (ground truth n_objects={truth['n_objects']}, "
          f"density={truth['density']})")

    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    ax[0].imshow(rgb); ax[0].set_title(f"{iid} (RGB)", fontsize=9); ax[0].axis("off")
    ax[1].imshow(gray, cmap="gray"); ax[1].set_title("grayscale, sent to VLM", fontsize=9)
    ax[1].axis("off")
    fig.suptitle("Fig. 1c  Image sent to the vision model", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "fig1c_vlm_input.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    out: dict = {"image_id": iid, "ground_truth": truth,
                 "prompts": {"naive": llm.NAIVE_VLM_PROMPT,
                             "structured": llm.STRUCTURED_VLM_PROMPT}}

    # (a) naive prompt
    naive = llm.ask(llm.NAIVE_VLM_PROMPT, image=gray, model=llm.VISION_MODEL,
                    temperature=0.8, seed=1, stub_kind="vision_naive", stub_evidence=ev)
    naive["parsed_json"] = llm.extract_json(naive["text"])
    out["naive_run"] = naive
    print("\n--- NAIVE PROMPT ---\n", naive["text"][:600])
    print("  parseable JSON:", naive["parsed_json"] is not None)

    # (b) structured prompt, repeated -> non-determinism
    runs = []
    for k in range(N_REPEATS):
        r = llm.ask(llm.STRUCTURED_VLM_PROMPT, image=gray, model=llm.VISION_MODEL,
                    temperature=0.8, seed=100 + k, stub_kind="vision",
                    stub_evidence=ev)
        rec = llm.extract_json(r["text"], required=["modality", "tissue_type",
                                                    "notable_features", "image_quality"])
        r["parsed_json"] = rec
        runs.append(r)
        print(f"\n--- STRUCTURED RUN {k + 1} (T=0.8, seed={100 + k}) ---")
        print(json.dumps(rec, indent=2) if rec else r["text"][:400])
    out["structured_runs"] = runs

    det = llm.ask(llm.STRUCTURED_VLM_PROMPT, image=gray, model=llm.VISION_MODEL,
                  temperature=0.0, seed=0, stub_kind="vision", stub_evidence=ev)
    det["parsed_json"] = llm.extract_json(det["text"])
    out["structured_greedy_T0"] = det

    texts = [r["text"] for r in runs]
    out["repeatability"] = {
        "n_repeats": N_REPEATS,
        "temperature": 0.8,
        "identical_texts": len(set(texts)) == 1,
        "n_distinct_texts": len(set(texts)),
        "stable_fields": _stable_fields([r["parsed_json"] for r in runs]),
    }
    print("\nRepeatability:", json.dumps(out["repeatability"], indent=2))

    # (c) schema compliance over a batch
    comp = {"naive": 0, "structured": 0, "n": 0}
    for i in split_ids("val")[:BATCH_FOR_COMPLIANCE]:
        e = image_evidence("val", i)
        g = load_gray("val", i)
        a = llm.ask(llm.NAIVE_VLM_PROMPT, image=g, model=llm.VISION_MODEL,
                    temperature=0.8, seed=7, stub_kind="vision_naive", stub_evidence=e)
        b = llm.ask(llm.STRUCTURED_VLM_PROMPT, image=g, model=llm.VISION_MODEL,
                    temperature=0.2, seed=7, stub_kind="vision", stub_evidence=e)
        comp["naive"] += llm.extract_json(a["text"]) is not None
        comp["structured"] += llm.extract_json(
            b["text"], required=["modality", "tissue_type", "notable_features",
                                 "image_quality"]) is not None
        comp["n"] += 1
    comp["naive_rate"] = comp["naive"] / comp["n"]
    comp["structured_rate"] = comp["structured"] / comp["n"]
    out["schema_compliance"] = comp
    print("\nSchema compliance:", json.dumps(comp, indent=2))

    out["generator"] = runs[0]["generator"]
    out["provenance"] = runs[0]["provenance"]
    save_json(out, JSONDIR / "task1_vlm.json")
    return out


def _stable_fields(records: list[dict | None]) -> dict:
    recs = [r for r in records if r]
    if not recs:
        return {}
    out = {}
    for k in recs[0]:
        vals = [json.dumps(r.get(k), sort_keys=True) for r in recs]
        out[k] = len(set(vals)) == 1
    return out


if __name__ == "__main__":
    set_seed()
    m = eda()
    vlm(m)
    print("\nTask 1 complete -> results/figures/fig1*.png, results/json/task1_*.json")
