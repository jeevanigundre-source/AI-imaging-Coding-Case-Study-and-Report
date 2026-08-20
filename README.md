# Hybrid biomedical image-analysis pipeline — fluorescence-microscopy nuclei

Local VLM description → classical (Otsu + `regionprops`) features → U-Net segmentation →
auditable structured JSON records + narratives.

Modality: **fluorescence microscopy, DAPI-like stained nuclei** (synthetic 256×256 dataset,
80 train / 20 val / 12 unseen test / 4 corrupted).

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only is fine: the U-Net is ~0.48 M parameters and 40 epochs take about 15 minutes
on two CPU cores (a few seconds on a GPU).

## 2. Data

The dataset is already in `data/nuclei_dataset/`. To fetch it again:

```bash
git clone https://github.com/Nickolay-K/Assingnment-3-dataset data/tmp
mv data/tmp/nuclei_dataset data/nuclei_dataset
```

Expected layout: `data/nuclei_dataset/{train,val,test}/{images,masks,labels}/*.png`,
`data/nuclei_dataset/test_corrupted/images/*.png`, `data/nuclei_dataset/metadata.csv`.

## 3. Local LLMs (Ollama)

```bash
ollama serve &
ollama pull llama3.2-vision      # Task 1: image → description
ollama pull llama3.1             # Tasks 2 and 4: numbers → description
python src/llm.py --check        # prints which back-end is active
```

**If Ollama is not running, everything still runs.** `src/llm.py` falls back to a
rule-based *offline stub* that returns schema-valid records built from the same
numeric evidence. The stub is **not** a language model. Every artefact it produces is
tagged `"generator": "offline_stub"`; real Ollama output is tagged with the model name
(e.g. `"generator": "llama3.2-vision"`). Check any result file for that field before
quoting it. Re-running the scripts with Ollama up overwrites the stub outputs with
genuine model outputs; no other code changes are needed.

## 4. Run

```bash
cd src
python 01_eda_vlm.py             # Task 1: EDA + VLM, naive vs structured prompt
python 02_classical_llm.py       # Task 2: Otsu + regionprops + numbers-first LLM
python 03_unet.py --epochs 40    # Task 3: train + evaluate the U-Net (BCE+Dice)
python 04_hybrid.py              # Task 4: full pipeline on the unseen test split
python 05_extensions.py          # Extensions: robustness, loss ablation, robust training
cd ../report && python build_report.py   # rebuild the 4-page PDF from the results
```

`05_extensions.py` trains three extra U-Nets (two for the loss ablation, one with
normalisation + photometric augmentation), roughly 35 min on CPU; use
`--part robust` for the fast robustness analysis only. `run_all.sh` runs everything
in order.

## 5. What each file does

| file | role |
|---|---|
| `src/common.py` | paths, loading, grayscale + resize, Dice/IoU, seeding |
| `src/features.py` | Otsu + morphology, `regionprops_table`, numeric summary → prompt text |
| `src/llm.py` | Ollama client, **all optimised prompts**, offline stub, JSON extraction/repair |
| `src/unet_model.py` | small U-Net (16→128 ch, depth 3) and the three losses |
| `src/01_eda_vlm.py` … `05_extensions.py` | Tasks 1–4 and the extensions |

## 6. Outputs

Every number and figure in the report is written by these scripts.

```
results/
  task4_records.csv                  <- required aggregated DataFrame (Task 4)
  task5_robustness.csv
  task5_robust_training.csv
  figures/  fig1a fig1b fig1c fig2a fig2b fig3a fig3b fig3c fig4a fig4b fig5a fig5b fig5c
  json/     task1_* task2_* task3_* task4_* task5_*  (+ per-image CSVs)
  models/   unet_bce.pt  unet_dice.pt  unet_bce_dice.pt  unet_bce_dice_norm.pt
report/
  report.tex  build_report.py  report_final.pdf   <- the submitted 4-page report
```

`report/build_report.py` fills the report's ablation table and robustness paragraph
from `results/json/` and compiles the PDF, so the report cannot drift from the code.

## 7. Headline results (from the committed `results/`)

| what | value |
|---|---|
| Otsu, validation (20 images) | Dice **0.9733**, IoU 0.9480 |
| U-Net (BCE+Dice, 40 ep), validation | Dice **0.9946**, IoU 0.9893 — wins 20/20 images |
| U-Net, unseen test (12 images) | Dice **0.9942**, IoU 0.9885 |
| Total nuclear area, test | 0.7 % error |
| Object **count**, test | −11.9 per image, MAPE **25.3 %** (13.95 % after watershed split) |
| `density_class` vs dataset label | 9/12 (75 %) |
| Loss ablation | BCE+Dice 0.9946 > BCE 0.9940 > soft Dice 0.9926 |
| Low-contrast corruption | U-Net 0.046 vs Otsu 0.980 |
| After normalisation + photometric aug | low contrast 0.046 → **0.987**, noise 0.820 → 0.932, clean 0.9946 → 0.9846 |
| Prompt schema compliance | naive 0/6, structured 6/6 |
| Structured prompt, 3 runs @ T=0.8 | 3 distinct responses; constrained fields stable |

The headline finding: near-perfect pixel overlap coexists with a 25 % under-count,
because a binary mask cannot separate touching nuclei. Dice is the wrong metric for
a counting task.

## 8. Reproducibility

Global seed `20260820` (`common.set_seed`) covers NumPy, Python `random` and torch.
CPU training is deterministic to ~1e-3 in Dice; GPU runs may differ slightly through
cuDNN kernel selection. LLM calls pass an explicit `seed` and `temperature` to Ollama,
and `results/json/*.json` records both alongside the raw response text.

## 9. Limits

Outputs are for **educational use only**. No model here is validated or cleared for
clinical use, the dataset is synthetic, and the LLM stages are constrained to
description — never diagnosis.
