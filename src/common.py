"""
common.py -- shared paths, I/O and preprocessing for the biomedical image-analysis
pipeline (fluorescence-microscopy nuclei).

Everything downstream (EDA, Otsu, U-Net, hybrid pipeline) imports from here so that
exactly one definition of "the preprocessed image" exists in the project.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nuclei_dataset"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
JSONDIR = RESULTS / "json"
MODELS = RESULTS / "models"
for _d in (RESULTS, FIGURES, JSONDIR, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256          # the assignment's common size; the source images are already 256x256
SEED = 20260820

# ------------------------------------------------------------------------ determinism
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ------------------------------------------------------------------------------ I/O
def split_ids(split: str) -> list[str]:
    """Sorted image ids for a split ('train' | 'val' | 'test' | 'test_corrupted')."""
    d = DATA / split / "images"
    return sorted(p.stem for p in d.glob("*.png"))


def load_rgb(split: str, image_id: str) -> np.ndarray:
    """Raw RGB image as float32 in [0, 1], resized to IMG_SIZE if needed."""
    im = Image.open(DATA / split / "images" / f"{image_id}.png").convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_gray(rgb: np.ndarray, mode: str = "luma") -> np.ndarray:
    """
    Grayscale conversion.

    mode='luma'  : ITU-R 601-2 luminance, i.e. skimage.color.rgb2gray. This is the
                   default the assignment asks for.
    mode='blue'  : the blue channel alone. DAPI-like nuclear stain lives almost
                   entirely in B, and luma weights B by only 0.0721, so 'blue'
                   preserves far more of the signal. Used for an ablation in
                   02_classical_llm.py.
    mode='max'   : per-pixel channel maximum (stain-agnostic).
    """
    if mode == "luma":
        g = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
    elif mode == "blue":
        g = rgb[..., 2]
    elif mode == "max":
        g = rgb.max(axis=2)
    else:
        raise ValueError(f"unknown grayscale mode {mode!r}")
    return g.astype(np.float32)


def load_gray(split: str, image_id: str, mode: str = "luma") -> np.ndarray:
    """Preprocessed image: RGB -> grayscale -> 256x256 float32 in [0, 1]."""
    return to_gray(load_rgb(split, image_id), mode=mode)


def load_mask(split: str, image_id: str) -> np.ndarray:
    """Binary ground-truth mask as bool (256x256). Raises if the split has no masks."""
    p = DATA / split / "masks" / f"{image_id}.png"
    m = Image.open(p).convert("L")
    if m.size != (IMG_SIZE, IMG_SIZE):
        m = m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.asarray(m) > 127


def load_labels(split: str, image_id: str) -> np.ndarray:
    """16-bit instance-label image (each nucleus a distinct integer)."""
    p = DATA / split / "labels" / f"{image_id}.png"
    return np.asarray(Image.open(p)).astype(np.int32)


def has_masks(split: str) -> bool:
    return (DATA / split / "masks").is_dir()


def load_metadata():
    import pandas as pd
    return pd.read_csv(DATA / "metadata.csv")


# ------------------------------------------------------------------- overlap metrics
def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Dice coefficient and IoU between two boolean masks (empty-empty -> 1.0)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    s = pred.sum() + gt.sum()
    dice = 1.0 if s == 0 else 2.0 * inter / s
    iou = 1.0 if union == 0 else inter / union
    return float(dice), float(iou)


# --------------------------------------------------------------------------- helpers
def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)
