"""
features.py -- classical segmentation and region features (scikit-image).

Shared by Task 2 (Otsu route), Task 4 (U-Net route) and the extensions, so that
the *same* measurement code is applied whatever produced the mask. This is what
makes the two routes comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, disk, opening
from skimage.segmentation import clear_border

MIN_AREA = 20          # px; below this an object is noise, not a nucleus
MAX_HOLE = 64          # px; holes smaller than this are stain drop-out, not structure
OPEN_RADIUS = 2
CLOSE_RADIUS = 2


def _drop_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Version-independent replacement for skimage.remove_small_objects."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= min_size
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill background components that are enclosed by foreground and small."""
    lab, n = ndi.label(~mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = mask.copy()
    for i in range(1, n + 1):
        if i not in border and counts[i] <= max_hole:
            out[lab == i] = True
    return out


def otsu_mask(gray: np.ndarray, clean: bool = True) -> tuple[np.ndarray, float]:
    """Otsu threshold + morphological cleanup. Returns (binary mask, threshold)."""
    thr = float(threshold_otsu(gray))
    m = gray > thr
    if clean:
        m = opening(m, disk(OPEN_RADIUS))          # kill speckle noise
        m = closing(m, disk(CLOSE_RADIUS))         # close small stain gaps
        m = _fill_small_holes(m, MAX_HOLE)         # fill nucleoli-like drop-out
        m = _drop_small_objects(m, MIN_AREA)       # discard sub-nuclear debris
    return m.astype(bool), thr


def watershed_split(mask: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """
    Distance-transform + watershed instance splitting.

    A binary mask cannot distinguish one large nucleus from two touching ones, so
    connected-component labelling systematically under-counts in dense/clustered
    fields. Seeding a watershed at the local maxima of the distance transform
    splits most touching pairs. Returned as an instance-label image.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)
    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=min_distance, labels=mask,
                            exclude_border=False)
    seeds = np.zeros(dist.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)
    lab = watershed(-dist, markers, mask=mask)
    # drop slivers created by over-splitting
    counts = np.bincount(lab.ravel())
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= MIN_AREA
    return np.where(keep[lab], lab, 0).astype(np.int32)


def feature_table(mask: np.ndarray, gray: np.ndarray,
                  drop_border: bool = False) -> pd.DataFrame:
    """
    Label connected components and measure them with regionprops_table.

    drop_border=False by default: border-touching nuclei are *kept* because
    removing them would bias the count downwards on dense images, which is
    exactly where the count matters most.
    """
    lab = label(mask)
    if drop_border:
        lab = clear_border(lab)
    if lab.max() == 0:
        return pd.DataFrame(columns=["label", "area", "perimeter", "eccentricity",
                                     "solidity", "extent", "mean_intensity",
                                     "max_intensity", "equivalent_diameter_area",
                                     "major_axis_length", "minor_axis_length",
                                     "circularity", "centroid-0", "centroid-1"])
    props = regionprops_table(
        lab, intensity_image=gray,
        properties=("label", "area", "perimeter", "eccentricity", "solidity",
                    "extent", "mean_intensity", "max_intensity",
                    "equivalent_diameter_area", "axis_major_length",
                    "axis_minor_length", "centroid"))
    df = pd.DataFrame(props).rename(columns={"axis_major_length": "major_axis_length",
                                             "axis_minor_length": "minor_axis_length"})
    with np.errstate(divide="ignore", invalid="ignore"):
        df["circularity"] = 4 * np.pi * df.area / np.square(df.perimeter.replace(0, np.nan))
    df["circularity"] = df.circularity.clip(upper=1.0).fillna(0.0)
    return df


def summarise(df: pd.DataFrame, mask: np.ndarray, gray: np.ndarray) -> dict:
    """Collapse the per-object table into the handful of numbers the LLM is given.
    This dict is the *only* thing the language model ever sees in Tasks 2 and 4."""
    n = int(len(df))
    fg = float(mask.mean())
    ev = {
        "n_objects": n,
        "foreground_fraction": round(fg, 4),
        "mean_area": round(float(df.area.mean()), 2) if n else 0.0,
        "std_area": round(float(df.area.std(ddof=0)), 2) if n else 0.0,
        "min_area": round(float(df.area.min()), 2) if n else 0.0,
        "max_area": round(float(df.area.max()), 2) if n else 0.0,
        "median_area": round(float(df.area.median()), 2) if n else 0.0,
        "mean_eccentricity": round(float(df.eccentricity.mean()), 3) if n else 0.0,
        "mean_solidity": round(float(df.solidity.mean()), 3) if n else 1.0,
        "mean_circularity": round(float(df.circularity.mean()), 3) if n else 0.0,
        "mean_extent": round(float(df.extent.mean()), 3) if n else 0.0,
        "mean_intensity": round(float(df.mean_intensity.mean()), 3) if n else 0.0,
        # if the mask covers the whole field there is no background left to measure
        "background_intensity": (round(float(gray[~mask].mean()), 4)
                                 if (~mask).any() else float("nan")),
        "mean_equiv_diameter": round(float(df.equivalent_diameter_area.mean()), 2) if n else 0.0,
    }
    return ev


def summary_text(ev: dict) -> str:
    """Render the evidence dict as the plain-text block embedded in the prompt.
    Numbers only -- no image, no colour, no stain, no diagnosis."""
    return "\n".join([
        f"- objects detected (connected components): {ev['n_objects']}",
        f"- foreground fraction of the 256x256 field: {ev['foreground_fraction']:.4f}",
        f"- object area px: mean {ev['mean_area']:.1f}, sd {ev['std_area']:.1f}, "
        f"median {ev['median_area']:.1f}, range {ev['min_area']:.0f}-{ev['max_area']:.0f}",
        f"- equivalent diameter px: mean {ev['mean_equiv_diameter']:.1f}",
        f"- mean eccentricity: {ev['mean_eccentricity']:.3f}  (0 = circle, 1 = line)",
        f"- mean solidity: {ev['mean_solidity']:.3f}  (1 = convex, lower = lobed or merged)",
        f"- mean circularity: {ev['mean_circularity']:.3f}",
        f"- mean extent (area / bounding-box area): {ev['mean_extent']:.3f}",
        f"- mean object intensity: {ev['mean_intensity']:.3f} "
        f"(background {ev['background_intensity']:.4f}), scale 0-1",
    ])
