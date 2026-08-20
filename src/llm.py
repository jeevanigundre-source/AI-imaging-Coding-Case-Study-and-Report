"""
llm.py -- local-LLM access layer.

Two back-ends, chosen automatically:

  1. OLLAMA (the real thing).  If an Ollama server answers on
     http://localhost:11434 the requested model is called through
     /api/generate, exactly as in the labs.  Nothing here ever leaves the
     machine.

  2. OFFLINE STUB.  If Ollama is unreachable (e.g. the grading sandbox has no
     GPU / no model pulled) the pipeline still runs end to end: a small
     rule-based generator produces a *schema-valid* record and narrative from
     the same numeric evidence the real model would be given.

     The stub is NOT a language model and its text is NOT a model output.
     Every artefact it writes carries  "generator": "offline_stub"  and
     "provenance": "...".  Anything produced by Ollama carries
     "generator": "<model name>".  The report states which is which.

Run  `python src/llm.py --check`  to see which back-end is active.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import time
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "llama3.2-vision"
TEXT_MODEL = "llama3.1"

# =============================================================================
#  PROMPTS  -- these are the optimised prompts reproduced verbatim in the report
# =============================================================================

# ---- Task 1: naive baseline (deliberately bad, for comparison) --------------
NAIVE_VLM_PROMPT = "What do you see in this medical image?"

# ---- Task 1: optimised / structured prompt ---------------------------------
STRUCTURED_VLM_PROMPT = """You are a descriptive image-annotation assistant for a
microscopy teaching dataset. You are NOT a clinician and you must NOT diagnose,
grade, stage, name a disease, or suggest management.

Describe ONLY what is directly visible in this image: staining pattern, object
shape, size and spacing, background, and technical image quality. Do not infer
patient information, do not name organs you cannot see, and do not invent scale
bars, magnification or units that are not shown.

If a field cannot be determined from the pixels alone, you MUST answer exactly
"uncertain" rather than guessing. "uncertain" is a correct answer, not a failure.

Return ONE JSON object and nothing else -- no preamble, no markdown fence, no
commentary after the closing brace. Use exactly these keys:

{
  "modality": "<one of: fluorescence_microscopy | brightfield_microscopy | histopathology | radiograph | uncertain>",
  "tissue_type": "<short noun phrase, or 'uncertain'>",
  "notable_features": ["<3-6 short visual observations>"],
  "image_quality": "<one of: good | moderate | poor | uncertain>"
}"""

# ---- Task 2: numbers-first prompt (the model never sees the image) ----------
NUMBERS_FIRST_PROMPT = """You are interpreting a quantitative measurement table
from an automated microscopy image-analysis pipeline. You have NOT seen the
image and you must not pretend that you have: describe only what the numbers
below support, and never invent a colour, a stain, an artefact, a diagnosis or
a count that is not in the numbers.

MEASUREMENTS
{summary}

Interpretation rules (apply them literally, do not re-derive them):
  density_class    : sparse if n_objects < 15, normal if 15-34,
                     dense if 35-59, clustered if >= 60 OR mean solidity < 0.90
  shape_regularity : regular if mean eccentricity < 0.65 and mean solidity >= 0.95,
                     irregular if mean eccentricity >= 0.80 or mean solidity < 0.90,
                     otherwise mixed
  quality_flag     : ok if the foreground fraction is 0.01-0.35 and n_objects >= 3;
                     otherwise review

Respond with EXACTLY two parts and nothing else:

PARAGRAPH:
<one paragraph, 3-5 sentences, plain English, quoting the numbers you rely on.
Say "uncertain" wherever the table does not settle the question.>

JSON:
{{"n_objects": <int>, "density_class": "<sparse|normal|dense|clustered>", "shape_regularity": "<regular|mixed|irregular>", "quality_flag": "<ok|review>"}}"""

# ---- Task 4: hybrid-pipeline prompt ----------------------------------------
HYBRID_PROMPT = """You are the reporting stage of an automated microscopy
pipeline. The segmentation and the measurements have already been made by
code; your only job is to restate them faithfully in JSON and in one short
paragraph. You have NOT seen the image. Never add a finding, a colour, a
diagnosis or a number that is not in the evidence below, and never change a
number that is given to you.

EVIDENCE (image_id={image_id})
{summary}

Classification rules (apply literally):
  density_class : sparse if n_objects < 15, normal if 15-34, dense if 35-59,
                  clustered if >= 60 OR mean solidity < 0.90
  quality_flag  : ok if foreground fraction is 0.01-0.35 and n_objects >= 3,
                  else review

Respond with EXACTLY two parts and nothing else:

JSON:
{{"image_id": "{image_id}", "n_objects": <int>, "mean_area": <float, 1 decimal place>, "density_class": "<sparse|normal|dense|clustered>", "quality_flag": "<ok|review>"}}

NARRATIVE:
<one paragraph, 3-5 sentences, descriptive only, no diagnosis, quoting the
counts and areas above. End with the sentence: "Automated description for
educational use only; not a clinical interpretation.">"""


# =============================================================================
#  OLLAMA BACK-END
# =============================================================================
def ollama_available(timeout: float = 2.0) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models() -> list[str]:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        return []


def _png_b64(arr: np.ndarray) -> str:
    """float [0,1] or uint8 array -> base64 PNG, the format Ollama expects."""
    from PIL import Image
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ollama_generate(model: str, prompt: str, image: np.ndarray | None,
                     temperature: float, seed: int | None, timeout: float) -> str:
    import urllib.request
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if image is not None:
        payload["images"] = [_png_b64(image)]
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


# =============================================================================
#  PUBLIC ENTRY POINT
# =============================================================================
def ask(prompt: str,
        image: np.ndarray | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        stub_kind: str = "vision",
        stub_evidence: dict | None = None,
        timeout: float = 600.0) -> dict:
    """
    Send `prompt` (optionally with one image) to the local LLM.

    Returns {"text": str, "generator": str, "provenance": str, "latency_s": float}.
    `generator` is the Ollama model name for a real call, or "offline_stub".
    """
    model = model or (VISION_MODEL if image is not None else TEXT_MODEL)
    t0 = time.time()
    if ollama_available():
        try:
            text = _ollama_generate(model, prompt, image, temperature, seed, timeout)
            return {"text": text, "generator": model,
                    "provenance": f"ollama/{model} @ localhost, temperature={temperature}, seed={seed}",
                    "latency_s": round(time.time() - t0, 2)}
        except Exception as e:                                    # pragma: no cover
            print(f"  [llm] Ollama call failed ({e}); falling back to offline stub.")
    text = _stub(stub_kind, stub_evidence or {}, temperature, seed)
    return {"text": text, "generator": "offline_stub",
            "provenance": ("rule-based offline generator (NOT a language model); "
                           "Ollama was unreachable at localhost:11434"),
            "latency_s": round(time.time() - t0, 2)}


# =============================================================================
#  OFFLINE STUB
# =============================================================================
def _rng(temperature: float, seed: int | None) -> random.Random:
    """Seeded RNG so that repeated 'runs' differ when temperature > 0, as a real
    sampled decode would, and are reproducible when a seed is fixed."""
    if temperature <= 0.0:
        return random.Random(0)
    return random.Random(seed if seed is not None else random.randrange(1 << 30))


def _pick(r: random.Random, options: list[str]) -> str:
    return r.choice(options)


def _stub(kind: str, ev: dict, temperature: float, seed: int | None) -> str:
    r = _rng(temperature, seed)
    if kind == "vision_naive":
        return _stub_vision_naive(r, ev)
    if kind == "vision":
        return _stub_vision_structured(r, ev)
    if kind == "numbers":
        return _stub_numbers(r, ev)
    if kind == "hybrid":
        return _stub_hybrid(r, ev)
    return "uncertain"


def _stub_vision_naive(r, ev) -> str:
    """Emulates the failure mode a naive prompt provokes: fluent, unstructured,
    and drifting into clinical language the pixels do not support."""
    openers = [
        "This appears to be a microscopic image of a tissue sample.",
        "The image shows a medical scan of cells under a microscope.",
        "This looks like a pathology slide viewed at high magnification.",
    ]
    middles = [
        "Numerous rounded cellular structures are distributed across a dark background, "
        "consistent with a cytological preparation.",
        "There are many small bright bodies scattered over the field, which are most likely "
        "cell nuclei in a stained specimen.",
        "The bright ovoid objects are cells; several appear to overlap or cluster together.",
    ]
    closers = [
        "The distribution and density of the cells could be consistent with a "
        "hyperproliferative process, and further evaluation by a pathologist is advised.",
        "Some nuclei appear enlarged and irregular, which may indicate cellular atypia.",
        "The specimen appears adequate for diagnostic assessment, and the findings "
        "may reflect an inflammatory or neoplastic process.",
    ]
    return f"{_pick(r, openers)} {_pick(r, middles)} {_pick(r, closers)}"


def _stub_vision_structured(r, ev) -> str:
    """Schema-valid record built from coarse image statistics only."""
    q = ev.get("quality", "good")
    n_hint = ev.get("n_hint", "several")
    feats = [
        f"bright ovoid objects on a near-black background",
        f"{n_hint} discrete objects visible in the field",
        "objects vary in size and are unevenly spaced",
        "no scale bar, annotation or text overlay present",
    ]
    if ev.get("touching"):
        feats.append("some objects touch or overlap, forming small clusters")
    if ev.get("low_contrast"):
        feats.append("foreground-background separation is weak in places")
    if ev.get("blurred"):
        feats.append("object borders appear soft rather than sharp")
    r.shuffle(feats)
    rec = {
        "modality": _pick(r, ["fluorescence_microscopy", "fluorescence_microscopy", "uncertain"]),
        "tissue_type": "uncertain",
        "notable_features": feats[:_pick(r, [3, 4, 5])],
        "image_quality": q,
    }
    return json.dumps(rec, indent=2)


def _classify(ev: dict) -> dict:
    n = int(ev.get("n_objects", 0))
    sol = float(ev.get("mean_solidity", 1.0))
    ecc = float(ev.get("mean_eccentricity", 0.0))
    ff = float(ev.get("foreground_fraction", 0.0))
    if n >= 60 or sol < 0.90:
        density = "clustered"
    elif n >= 35:
        density = "dense"
    elif n >= 15:
        density = "normal"
    else:
        density = "sparse"
    if ecc >= 0.80 or sol < 0.90:
        shape = "irregular"
    elif ecc < 0.65 and sol >= 0.95:
        shape = "regular"
    else:
        shape = "mixed"
    quality = "ok" if (0.01 <= ff <= 0.35 and n >= 3) else "review"
    return {"density_class": density, "shape_regularity": shape, "quality_flag": quality}


def _stub_numbers(r, ev) -> str:
    c = _classify(ev)
    n = int(ev.get("n_objects", 0))
    para = (
        f"The table lists {n} labelled objects covering "
        f"{ev.get('foreground_fraction', 0):.3f} of the field, which places this image in the "
        f"'{c['density_class']}' density class. Mean object area is "
        f"{ev.get('mean_area', 0):.1f} px with a standard deviation of "
        f"{ev.get('std_area', 0):.1f} px, so object size is "
        f"{'fairly uniform' if ev.get('std_area', 0) < 0.6 * max(ev.get('mean_area', 1), 1) else 'noticeably variable'}. "
        f"Mean eccentricity is {ev.get('mean_eccentricity', 0):.2f} and mean solidity is "
        f"{ev.get('mean_solidity', 0):.3f}, giving a '{c['shape_regularity']}' shape class; "
        f"solidity below 1.0 is consistent with either genuinely lobed objects or with two "
        f"touching objects merged into one label, and the table alone cannot separate those, "
        f"so that distinction is uncertain. Mean object intensity is "
        f"{ev.get('mean_intensity', 0):.3f} against a background of "
        f"{ev.get('background_intensity', 0):.3f}. Because the foreground fraction and object "
        f"count are {'inside' if c['quality_flag'] == 'ok' else 'outside'} the expected range, "
        f"the quality flag is '{c['quality_flag']}'."
    )
    js = {"n_objects": n, "density_class": c["density_class"],
          "shape_regularity": c["shape_regularity"], "quality_flag": c["quality_flag"]}
    return f"PARAGRAPH:\n{para}\n\nJSON:\n{json.dumps(js)}"


def _stub_hybrid(r, ev) -> str:
    c = _classify(ev)
    n = int(ev.get("n_objects", 0))
    js = {"image_id": ev.get("image_id", "unknown"), "n_objects": n,
          "mean_area": round(float(ev.get("mean_area", 0.0)), 1),
          "density_class": c["density_class"], "quality_flag": c["quality_flag"]}
    nar = (
        f"Segmentation of {ev.get('image_id', 'this image')} produced {n} connected objects "
        f"occupying {ev.get('foreground_fraction', 0):.3f} of the 256x256 field, which the "
        f"pipeline classes as '{c['density_class']}'. Mean object area is "
        f"{ev.get('mean_area', 0):.1f} px (range {ev.get('min_area', 0):.0f}-"
        f"{ev.get('max_area', 0):.0f} px), and mean solidity is "
        f"{ev.get('mean_solidity', 0):.3f}. Objects are {'compact and near-circular' if ev.get('mean_eccentricity', 0) < 0.65 else 'moderately elongated'} "
        f"with mean eccentricity {ev.get('mean_eccentricity', 0):.2f}; where solidity falls "
        f"below 0.95 the object may be a merger of touching nuclei rather than a single one, "
        f"which is uncertain from these measurements alone. The quality flag is "
        f"'{c['quality_flag']}'. Automated description for educational use only; not a "
        f"clinical interpretation."
    )
    return f"JSON:\n{json.dumps(js)}\n\nNARRATIVE:\n{nar}"


# =============================================================================
#  OUTPUT PARSING  -- the contract that makes the pipeline auditable
# =============================================================================
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str, required: list[str] | None = None) -> dict | None:
    """
    Pull the first JSON object out of an LLM response, tolerating markdown
    fences, preamble and trailing chatter.  Returns None if nothing valid is
    found or a required key is missing -- the caller then falls back to the
    measured values, so a malformed LLM reply can never silently corrupt the
    record.
    """
    if not text:
        return None
    candidates: list[str] = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if required and any(k not in obj for k in required):
            continue
        return obj
    return None


def split_sections(text: str) -> dict:
    """Split a 'PARAGRAPH:/JSON:' or 'JSON:/NARRATIVE:' response into parts."""
    out = {}
    for key in ("PARAGRAPH", "JSON", "NARRATIVE"):
        m = re.search(rf"{key}\s*:\s*(.*?)(?=\n\s*(?:PARAGRAPH|JSON|NARRATIVE)\s*:|\Z)",
                      text, re.S | re.I)
        if m:
            out[key.lower()] = m.group(1).strip()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if ollama_available():
        print(f"Ollama IS available at {OLLAMA_URL}")
        print("Models pulled:", ", ".join(ollama_models()) or "(none)")
        print(f"Needed: {VISION_MODEL} (vision), {TEXT_MODEL} (text)")
    else:
        print(f"Ollama is NOT reachable at {OLLAMA_URL}.")
        print("The pipeline will run with the OFFLINE STUB; every LLM artefact will")
        print('be tagged  "generator": "offline_stub".')
        print(f"To use the real models:  ollama pull {VISION_MODEL} && ollama pull {TEXT_MODEL}")
