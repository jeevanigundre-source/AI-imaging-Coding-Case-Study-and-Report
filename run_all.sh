#!/usr/bin/env bash
# Reproduce every number and figure in the report, in order.
set -e
cd "$(dirname "$0")/src"
python llm.py --check
python 01_eda_vlm.py
python 02_classical_llm.py
python 03_unet.py --loss bce_dice --epochs 40
python 04_hybrid.py
python 05_extensions.py --part all --epochs 40
cd ../report && python build_report.py
echo
echo "Done. See ../results/ (task4_records.csv, figures/, json/)."
