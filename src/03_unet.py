"""
Task 3 -- U-Net segmentation.

Trains the small U-Net on the 80-image training split and evaluates mean Dice
and IoU on the held-out 20-image validation split.  `--loss` selects the
objective, so the same script drives the loss ablation in 05_extensions.py.

  python 03_unet.py                      # BCE+Dice, 30 epochs (the reported model)
  python 03_unet.py --loss dice --tag d  # one arm of the ablation

Outputs -> results/models/unet_<loss>.pt,
           results/json/task3_history_<loss>.json, task3_val_metrics_<loss>.csv,
           results/figures/fig3a_curves.png, fig3b_panels.png
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from common import (FIGURES, JSONDIR, MODELS, banner, dice_iou, load_gray,
                    load_mask, load_metadata, save_json, set_seed, split_ids)
from unet_model import LOSSES, UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def percentile_norm(g: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    """Per-image contrast normalisation: map the 1st/99.5th intensity percentiles
    to 0/1. Makes the input scale-and-offset invariant, like Otsu."""
    lo, hi = np.percentile(g, [lo_p, hi_p])
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


class NucleiDataset(Dataset):
    """256x256 grayscale image + binary mask.

    Geometric augmentation (flips, 90-degree rotations) is on by default for
    training: nuclei have no canonical orientation, so these are exactly
    label-preserving.

    `photometric` and `normalize` are OFF for the reported model and are used by
    the intensity-robustness experiment in 05_extensions.py:
      photometric -- random gain/offset, blur and noise, i.e. simulate the
                     acquisition variation the training set does not contain
      normalize   -- per-image percentile contrast normalisation at input.
    """

    def __init__(self, split: str, augment: bool = False, gray_mode: str = "luma",
                 normalize: bool = False, photometric: bool = False):
        self.ids = split_ids(split)
        self.split, self.augment, self.gray_mode = split, augment, gray_mode
        self.normalize, self.photometric = normalize, photometric
        self.imgs = [load_gray(split, i, gray_mode) for i in self.ids]
        self.masks = [load_mask(split, i) for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x, y = self.imgs[idx], self.masks[idx].astype(np.float32)
        if self.photometric:                      # intensity-domain augmentation
            x = x * np.random.uniform(0.25, 1.75) + np.random.uniform(-0.05, 0.45)
            if np.random.rand() < 0.4:
                x = ndi.gaussian_filter(x, np.random.uniform(0.5, 2.5))
            if np.random.rand() < 0.4:
                x = x + np.random.normal(0, np.random.uniform(0.01, 0.08), x.shape)
            x = np.clip(x, 0, 1).astype(np.float32)
        if self.normalize:
            x = percentile_norm(x)
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = np.fliplr(x), np.fliplr(y)
            if np.random.rand() < 0.5:
                x, y = np.flipud(x), np.flipud(y)
        x = torch.from_numpy(np.ascontiguousarray(x))[None].float()
        y = torch.from_numpy(np.ascontiguousarray(y))[None].float()
        return x, y


@torch.no_grad()
def evaluate(model, ds, thr: float = 0.5):
    """Per-image Dice/IoU on a dataset (no augmentation)."""
    model.eval()
    rows = []
    for i in range(len(ds)):
        x, y = ds[i]
        p = torch.sigmoid(model(x[None].to(DEVICE)))[0, 0].cpu().numpy()
        d, iou = dice_iou(p > thr, y[0].numpy() > 0.5)
        rows.append({"image_id": ds.ids[i], "dice": d, "iou": iou})
    return pd.DataFrame(rows)


def train(loss_name: str = "bce_dice", epochs: int = 30, batch: int = 8,
          lr: float = 1e-3, base: int = 16, quiet: bool = False,
          normalize: bool = False, photometric: bool = False,
          tag: str | None = None):
    set_seed()
    tag = tag or loss_name
    tr = NucleiDataset("train", augment=True, normalize=normalize,
                       photometric=photometric)
    va = NucleiDataset("val", augment=False, normalize=normalize)
    dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

    model = UNet(base=base).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = LOSSES[loss_name]

    if not quiet:
        banner(f"TASK 3 -- training U-Net  (loss={loss_name}, {model.n_params():,} params, "
               f"device={DEVICE})")
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best = {"dice": -1.0, "epoch": -1}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = crit(model(x), y)
            l.backward()
            opt.step()
            tot += l.item() * x.size(0)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, ds_, is_ = 0.0, [], []
            for i in range(len(va)):
                x, y = va[i]
                x, y = x[None].to(DEVICE), y[None].to(DEVICE)
                logits = model(x)
                vl += crit(logits, y).item()
                p = torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5
                d, iou = dice_iou(p, y[0, 0].cpu().numpy() > 0.5)
                ds_.append(d); is_.append(iou)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tot / len(tr))
        hist["val_loss"].append(vl / len(va))
        hist["val_dice"].append(float(np.mean(ds_)))
        hist["val_iou"].append(float(np.mean(is_)))
        if hist["val_dice"][-1] > best["dice"]:
            best = {"dice": hist["val_dice"][-1], "epoch": ep}
            torch.save(model.state_dict(), MODELS / f"unet_{tag}.pt")
        if not quiet:
            print(f"epoch {ep:3d}/{epochs}  train {hist['train_loss'][-1]:.4f}  "
                  f"val {hist['val_loss'][-1]:.4f}  dice {hist['val_dice'][-1]:.4f}  "
                  f"iou {hist['val_iou'][-1]:.4f}", flush=True)
    hist["train_seconds"] = round(time.time() - t0, 1)
    hist["best"] = best
    hist["loss"] = loss_name
    hist["n_params"] = model.n_params()
    hist["tag"] = tag
    hist["normalize"] = normalize
    hist["photometric"] = photometric
    save_json(hist, JSONDIR / f"task3_history_{tag}.json")

    model.load_state_dict(torch.load(MODELS / f"unet_{tag}.pt", map_location=DEVICE))
    return model, hist, va


def load_trained(tag: str = "bce_dice", base: int = 16) -> UNet:
    m = UNet(base=base).to(DEVICE)
    m.load_state_dict(torch.load(MODELS / f"unet_{tag}.pt", map_location=DEVICE))
    m.eval()
    return m


def figures(model, hist, va, loss_name: str) -> pd.DataFrame:
    meta = load_metadata()
    per = evaluate(model, va)
    per = per.merge(meta[["image_id", "density", "n_objects"]], on="image_id", how="left")
    per.round(4).to_csv(JSONDIR / f"task3_val_metrics_{loss_name}.csv", index=False)
    print("\nValidation, per density regime:\n",
          per.groupby("density")[["dice", "iou"]].mean().round(4).to_string())
    print(f"\nMEAN VAL DICE {per.dice.mean():.4f}   MEAN VAL IoU {per.iou.mean():.4f}")

    # --- Fig 3a: curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(f"{loss_name} loss")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Loss", fontsize=10)
    ax[1].plot(hist["epoch"], hist["val_dice"], color="#c44e52", label="val Dice")
    ax[1].plot(hist["epoch"], hist["val_iou"], color="#4c72b0", label="val IoU")
    ax[1].axhline(hist["best"]["dice"], ls="--", lw=1, color="grey")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3); ax[1].set_title("Validation overlap", fontsize=10)
    fig.suptitle(f"Fig. 3a  U-Net training ({loss_name}, {hist['train_seconds']}s, "
                 f"best Dice {hist['best']['dice']:.4f} @ epoch {hist['best']['epoch']})",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "fig3a_curves.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    # --- Fig 3b: input / GT / prediction / error, for best, median and worst
    per_sorted = per.sort_values("dice")
    picks = [("worst", per_sorted.iloc[0]),
             ("median", per_sorted.iloc[len(per_sorted) // 2]),
             ("best", per_sorted.iloc[-1])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.8 * len(picks)))
    for r, (tag, row) in enumerate(picks):
        iid = row.image_id
        g, gt = load_gray("val", iid), load_mask("val", iid)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(g)[None, None].float().to(DEVICE)))
        pred = p[0, 0].cpu().numpy() > 0.5
        err = np.zeros((*gt.shape, 3))
        err[..., 0] = pred & ~gt          # red   = false positive
        err[..., 1] = gt & pred           # green = true positive
        err[..., 2] = gt & ~pred          # blue  = false negative
        # zoom on the densest error region: the errors are a thin boundary band
        # and are invisible at full field size
        wrong = (pred != gt).astype(float)
        heat = ndi.uniform_filter(wrong, size=24)
        cy, cx = np.unravel_index(np.argmax(heat), heat.shape)
        h = 16
        y0, x0 = np.clip(cy - h, 0, 256 - 2 * h), np.clip(cx - h, 0, 256 - 2 * h)
        crop = err[y0:y0 + 2 * h, x0:x0 + 2 * h]
        for c, (im, cmap, ttl) in enumerate([
                (g, "gray", f"{iid} ({tag}, {row.density}, n={int(row.n_objects)})"),
                (gt, "gray", "ground truth"),
                (pred, "gray", f"U-Net  Dice {row.dice:.3f}  IoU {row.iou:.3f}"),
                (crop, None, f"32x32 zoom on worst errors\ngreen TP / red FP / blue FN "
                             f"({int(wrong.sum())} px wrong)")]):
            axes[r, c].imshow(im, cmap=cmap, interpolation="nearest")
            axes[r, c].set_title(ttl, fontsize=8.5)
            axes[r, c].axis("off")
    fig.suptitle("Fig. 3b  Validation predictions: worst, median and best by Dice",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig3b_panels.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    compare_with_otsu(model, per)
    return per


def compare_with_otsu(model, per: pd.DataFrame) -> pd.DataFrame:
    """Q2: did the U-Net improve on classical Otsu? Per-image, on the val split,
    plus the one image where each approach wins by the largest margin."""
    import features as Fx
    rows = []
    for iid in per.image_id:
        g, gt = load_gray("val", iid), load_mask("val", iid)
        om, _ = Fx.otsu_mask(g)
        od, oi = dice_iou(om, gt)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(g)[None, None].float().to(DEVICE)))
        um = p[0, 0].cpu().numpy() > 0.5
        ud, ui = dice_iou(um, gt)
        rows.append({"image_id": iid, "otsu_dice": od, "otsu_iou": oi,
                     "unet_dice": ud, "unet_iou": ui, "delta": ud - od,
                     "n_otsu": int(Fx.feature_table(om, g).shape[0]),
                     "n_unet": int(Fx.feature_table(um, g).shape[0])})
    cmp = pd.DataFrame(rows).merge(per[["image_id", "density", "n_objects"]],
                                   on="image_id")
    cmp.round(4).to_csv(JSONDIR / "task3_otsu_vs_unet.csv", index=False)
    print("\nOtsu vs U-Net on the validation split:")
    print(f"  Otsu  mean Dice {cmp.otsu_dice.mean():.4f}  IoU {cmp.otsu_iou.mean():.4f}")
    print(f"  U-Net mean Dice {cmp.unet_dice.mean():.4f}  IoU {cmp.unet_iou.mean():.4f}")
    print(f"  U-Net wins on {int((cmp.delta > 0).sum())}/{len(cmp)} images")

    best_unet = cmp.loc[cmp.delta.idxmax()]
    best_otsu = cmp.loc[cmp.delta.idxmin()]
    save_json({"otsu_mean_dice": float(cmp.otsu_dice.mean()),
               "otsu_mean_iou": float(cmp.otsu_iou.mean()),
               "unet_mean_dice": float(cmp.unet_dice.mean()),
               "unet_mean_iou": float(cmp.unet_iou.mean()),
               "unet_wins": int((cmp.delta > 0).sum()), "n_images": len(cmp),
               "largest_unet_win": best_unet.to_dict(),
               "largest_otsu_win": best_otsu.to_dict()},
              JSONDIR / "task3_otsu_vs_unet.json")

    # Third panel: the one regime where Otsu beats the U-Net -- a low-contrast
    # acquisition. Otsu is invariant to intensity rescaling; the U-Net has only
    # ever seen one intensity regime.
    from PIL import Image as _Im
    from common import DATA, to_gray
    cim = _Im.open(DATA / "test_corrupted" / "images" / "test_000_lowcontrast.png").convert("RGB")
    lc_gray = to_gray(np.asarray(cim, dtype=np.float32) / 255.0)
    lc_gt = load_mask("test", "test_000")
    lc_otsu, _ = Fx.otsu_mask(lc_gray)
    with torch.no_grad():
        lc_p = torch.sigmoid(model(torch.from_numpy(lc_gray)[None, None].float().to(DEVICE)))
    lc_unet = lc_p[0, 0].cpu().numpy() > 0.5
    lc_row = {"image_id": "test_000_lowcontrast", "density": "sparse, corrupted",
              "n_objects": int(load_metadata().query("image_id=='test_000'").n_objects.iloc[0]),
              "otsu_dice": dice_iou(lc_otsu, lc_gt)[0],
              "unet_dice": dice_iou(lc_unet, lc_gt)[0],
              "n_otsu": int(Fx.feature_table(lc_otsu, lc_gray).shape[0]),
              "n_unet": int(Fx.feature_table(lc_unet, lc_gray).shape[0])}

    fig, axes = plt.subplots(3, 4, figsize=(11, 8.4))
    panels = [(best_unet.to_dict(), "U-Net wins by most", "val"),
              (best_otsu.to_dict(), "smallest U-Net margin", "val"),
              (lc_row, "low contrast: OTSU WINS", "corrupt")]
    for r, (row, tag, kind) in enumerate(panels):
        iid = row["image_id"]
        if kind == "val":
            g, gt = load_gray("val", iid), load_mask("val", iid)
            om, _ = Fx.otsu_mask(g)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(g)[None, None].float().to(DEVICE)))
            um = p[0, 0].cpu().numpy() > 0.5
        else:
            g, gt, om, um = lc_gray, lc_gt, lc_otsu, lc_unet
        for c, (im, ttl) in enumerate([
                (g, f"{iid} ({tag}, {row['density']})"),
                (gt, f"ground truth, n={int(row['n_objects'])}"),
                (om, f"Otsu  Dice {row['otsu_dice']:.3f}, n={int(row['n_otsu'])}"),
                (um, f"U-Net Dice {row['unet_dice']:.3f}, n={int(row['n_unet'])}")]):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            axes[r, c].set_title(ttl, fontsize=8.5)
    save_json(lc_row, JSONDIR / "task3_otsu_wins_lowcontrast.json")
    fig.suptitle("Fig. 3c  Classical Otsu vs U-Net", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fig3c_otsu_vs_unet.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)
    return cmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="bce_dice", choices=list(LOSSES))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--normalize", action="store_true",
                    help="per-image percentile contrast normalisation")
    ap.add_argument("--photometric", action="store_true",
                    help="random gain/offset/blur/noise augmentation")
    ap.add_argument("--tag", default=None, help="checkpoint name (default: loss name)")
    a = ap.parse_args()
    model, hist, va = train(a.loss, epochs=a.epochs, batch=a.batch,
                            normalize=a.normalize, photometric=a.photometric,
                            tag=a.tag)
    if not a.no_figures:
        per = figures(model, hist, va, hist["tag"])
        save_json({"loss": a.loss, "epochs": a.epochs,
                   "mean_val_dice": float(per.dice.mean()),
                   "mean_val_iou": float(per.iou.mean()),
                   "per_density": per.groupby("density")[["dice", "iou"]]
                                     .mean().round(4).reset_index().to_dict("records"),
                   "train_seconds": hist["train_seconds"],
                   "n_params": hist["n_params"]},
                  JSONDIR / f"task3_summary_{hist['tag']}.json")
    print("\nTask 3 complete.")
