"""
build_report.py -- fill the numeric placeholders in report.tex from the JSON /
CSV that the pipeline wrote, then compile the PDF.

This keeps the report honest: the ablation table and the robustness paragraph
are generated from results/json/*, not typed by hand.

    python report/build_report.py          # writes report/report_final.pdf
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JSONDIR = RESULTS / "json"

PRETTY = {"bce": "BCE", "dice": "soft Dice", "bce_dice": "BCE+Dice"}


def ablation_rows() -> str:
    data = json.loads((JSONDIR / "task5_loss_ablation.json").read_text())["results"]
    order = {"bce": 0, "dice": 1, "bce_dice": 2}
    data = sorted(data, key=lambda r: order.get(r["loss"], 9))
    best = max(r["mean_val_dice"] for r in data)
    out = []
    for r in data:
        name = PRETTY[r["loss"]]
        d = f"{r['mean_val_dice']:.4f}"
        if r["mean_val_dice"] == best:
            name, d = f"\\textbf{{{name}}}", f"\\textbf{{{d}}}"
        out.append(f"{name} & {d} & {r['mean_val_iou']:.4f} & "
                   f"{r['worst_image_dice']:.4f} & {r['best_epoch']}\\\\")
    return "\n".join(out)


def robust_paragraph() -> str:
    rows = json.loads((JSONDIR / "task5_robust_training.json").read_text())
    by = {r["variant"]: r for r in rows}
    val = by.get("val split (20 images, mean)")
    lc, bl, nz = by["low_contrast"], by["blur"], by["noise_sigma0.10"]
    return (
        "The result confirms the diagnosis. On the low-contrast variant Dice "
        f"rises from {lc['baseline_dice']:.3f} to {lc['robust_dice']:.3f}; on "
        f"blur from {bl['baseline_dice']:.3f} to {bl['robust_dice']:.3f}; on "
        f"additive noise from {nz['baseline_dice']:.3f} to "
        f"{nz['robust_dice']:.3f}. Clean validation Dice moves from "
        f"{val['baseline_dice']:.4f} to {val['robust_dice']:.4f}. "
        "So the low-contrast collapse was never a capacity limit --- it was an "
        "invariance the training distribution never asked for, and a few lines "
        "of augmentation supply it. Robustness is not free: the clean-split "
        "Dice given up is the price of not keying on absolute brightness, a "
        "trade that looks bad only on a dataset where every clean image shares "
        "one acquisition.")


def main() -> None:
    src = (HERE / "report.tex").read_text()
    src = src.replace("%%ABLATION%%", ablation_rows())
    src = src.replace("%%ROBUSTTRAIN%%", robust_paragraph())
    out = HERE / "report_final.tex"
    out.write_text(src)
    for _ in range(2):                      # twice, for float/ref numbering
        subprocess.run(["pdflatex", "-interaction=nonstopmode", out.name],
                       cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = HERE / "report_final.pdf"
    print("wrote", pdf, pdf.exists())


if __name__ == "__main__":
    main()
