"""
metrics.py — Evaluation script for the EmoSpeech TTS pipeline
==============================================================

Goal
----
The only model trained in this project is the TTS (FastSpeech2 + iSTFTNet
vocoder + emotion conditioning, based on EmoSpeech).

STT and SER are used off-the-shelf as independent judges to measure
whether the TTS correctly preserves:
    - the textual content     -> WER / CER  (between reference text and STT output)
    - the target emotion      -> accuracy / macro F1  (between target emotion and SER output)

This script takes the results produced by running the pipeline
(TTS -> audio -> STT + SER) and produces the evaluation report.

CSV format expected
-------------------
Required columns:
    sample_id        : unique id  (e.g. "0020_001494")
    text_input       : reference text (grapheme form, for WER)
    emotion_input    : target emotion passed to the TTS
                       (label like "Happy" or id 0-4, see ESD_EMOTION_ID_TO_NAME)
    text_output      : text transcribed by STT from the generated audio
    emotion_output   : emotion predicted by SER from the generated audio

Optional columns:
    condition        : baseline_clean / baseline_noisy / emotion_clean / emotion_noisy
                       (if present, the report is computed per condition)

Metrics produced
----------------
Text side (STT vs reference):
    - WER, CER (mean + std)
Emotion side (SER vs target):
    - Accuracy
    - Macro F1, per-class precision/recall/F1
    - Confusion matrix
Aggregate:
    - Fidelity Score = (1 - WER) * emotion_accuracy  in [0, 1]

How to tell if the TTS is well trained
--------------------------------------
We compare the Fidelity Score against two reference points:
    - A lower bound: WER on noisy/silence baseline -> if the trained model is
      close to that, training failed.
    - An upper bound: the STT/SER ceiling measured on the ORIGINAL audio of
      the test set (use --reference-csv to pass it). If the model is close
      to this ceiling, the TTS reproduces the reference nearly perfectly.
See `interpret_training_quality()` for the heuristic rules.

Usage
-----
python -m src.scripts.metrics \\
    --results results/pipeline_output.csv \\
    --output  results/report.json

python -m src.scripts.metrics \\
    --results results/generated.csv \\
    --reference-csv results/original_audio.csv \\
    --output  results/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pretty table (optional)
# ---------------------------------------------------------------------------
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ---------------------------------------------------------------------------
# Column names — edit here if the CSV uses different headers
# ---------------------------------------------------------------------------
COL_ID         = "sample_id"
COL_TEXT_IN    = "text_input"
COL_EMO_IN     = "emotion_input"
COL_TEXT_OUT   = "text_output"
COL_EMO_OUT    = "emotion_output"
COL_CONDITION  = "condition"      # optional

# ---------------------------------------------------------------------------
# Emotion label space (ESD + EmoSpeech inference.py: -emo 0..4)
# ---------------------------------------------------------------------------
ESD_EMOTION_ID_TO_NAME = {
    0: "Neutral",
    1: "Angry",
    2: "Happy",
    3: "Sad",
    4: "Surprise",
}
ESD_EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]

# Common aliases returned by public SER models, normalized to ESD names
SER_LABEL_ALIASES = {
    "neutral":  "Neutral", "neu": "Neutral", "calm": "Neutral",
    "angry":    "Angry",   "anger": "Angry",  "ang": "Angry",
    "happy":    "Happy",   "happiness": "Happy", "joy": "Happy", "hap": "Happy",
    "sad":      "Sad",     "sadness": "Sad",  "sa": "Sad",
    "surprise": "Surprise","surprised": "Surprise", "sur": "Surprise",
}


def normalize_emotion(label) -> str:
    """Map a SER / TTS emotion (id, alias, case variant) to an ESD label."""
    if pd.isna(label):
        return ""
    if isinstance(label, (int, np.integer)):
        return ESD_EMOTION_ID_TO_NAME.get(int(label), str(label))
    s = str(label).strip().lower()
    if s.isdigit():
        return ESD_EMOTION_ID_TO_NAME.get(int(s), s)
    return SER_LABEL_ALIASES.get(s, str(label).strip().capitalize())


# ---------------------------------------------------------------------------
# Text normalization for WER / CER
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    """
    Lowercase, strip accents, remove punctuation, collapse whitespace.
    Keeps apostrophes (useful for English contractions: don't, I'm).
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===========================================================================
# Text metrics
# ===========================================================================

def _edit_distance(a: list, b: list) -> int:
    """Standard DP edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 1.0
    return _edit_distance(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference))
    hyp = list(normalize_text(hypothesis))
    if not ref:
        return 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ===========================================================================
# Emotion metrics
# ===========================================================================

def emotion_accuracy(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def emotion_f1_macro(y_true, y_pred, labels) -> dict:
    stats, f1s = {}, []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        stats[label] = {"precision": round(precision, 4),
                        "recall":    round(recall,    4),
                        "f1":        round(f1,        4)}
        f1s.append(f1)
    stats["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    return stats


def confusion_matrix(y_true, y_pred, labels) -> dict:
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


# ===========================================================================
# Aggregate score
# ===========================================================================

def fidelity_score(mean_wer: float, emo_accuracy: float) -> float:
    """
    Single score in [0, 1], higher is better.
    Formula: (1 - clamp(WER, 0, 1)) * emotion_accuracy
    """
    text_fidelity = max(0.0, 1.0 - min(mean_wer, 1.0))
    return round(text_fidelity * emo_accuracy, 4)


def interpret_training_quality(score: float,
                               reference_score: float | None = None) -> str:
    """
    Rule-of-thumb verdict on the TTS training, based on the fidelity score.
    If a reference_score is provided (STT+SER ceiling on original audio),
    the verdict is relative to that ceiling.
    """
    if reference_score is not None and reference_score > 0:
        ratio = score / reference_score
        if ratio >= 0.90:
            return (f"TTS matches the STT/SER ceiling ({ratio:.0%} of it). "
                    "Training is excellent.")
        if ratio >= 0.70:
            return (f"TTS reaches {ratio:.0%} of the STT/SER ceiling. "
                    "Training is decent; emotion fidelity or intelligibility "
                    "can still improve.")
        if ratio >= 0.40:
            return (f"TTS reaches only {ratio:.0%} of the STT/SER ceiling. "
                    "Training is underfit or an axis (text or emotion) is weak.")
        return (f"TTS reaches {ratio:.0%} of the STT/SER ceiling. "
                "Training likely failed — check data, loss curves, checkpoint.")
    # No reference: absolute thresholds
    if score >= 0.75:
        return "High fidelity. TTS appears well trained."
    if score >= 0.50:
        return "Moderate fidelity. Training is working but perfectible."
    if score >= 0.25:
        return "Low fidelity. Training is insufficient or one axis is broken."
    return "Very low fidelity. Training likely failed."


# ===========================================================================
# Report builder
# ===========================================================================

def compute_report(df: pd.DataFrame,
                   labels: list | None = None) -> dict:
    """Build the full evaluation report from a DataFrame of results."""
    required = [COL_TEXT_IN, COL_EMO_IN, COL_TEXT_OUT, COL_EMO_OUT]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'. Required: {required}. "
                f"Got: {list(df.columns)}"
            )

    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[warn] Dropped {dropped} rows containing NaN in required columns.")

    n = len(df)
    if n == 0:
        raise ValueError("No valid rows to evaluate.")

    # Normalize emotions
    y_true = [normalize_emotion(e) for e in df[COL_EMO_IN]]
    y_pred = [normalize_emotion(e) for e in df[COL_EMO_OUT]]

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Text metrics
    wers = [word_error_rate(r, h) for r, h in zip(df[COL_TEXT_IN], df[COL_TEXT_OUT])]
    cers = [char_error_rate(r, h) for r, h in zip(df[COL_TEXT_IN], df[COL_TEXT_OUT])]

    mean_wer = round(float(np.mean(wers)), 4)
    std_wer  = round(float(np.std(wers)),  4)
    mean_cer = round(float(np.mean(cers)), 4)
    std_cer  = round(float(np.std(cers)),  4)

    # Emotion metrics
    acc     = round(emotion_accuracy(y_true, y_pred), 4)
    f1_info = emotion_f1_macro(y_true, y_pred, labels)
    cm      = confusion_matrix(y_true, y_pred, labels)

    score = fidelity_score(mean_wer, acc)

    return {
        "n_samples": n,
        "text_metrics": {
            "mean_WER": mean_wer, "std_WER": std_wer,
            "mean_CER": mean_cer, "std_CER": std_cer,
        },
        "emotion_metrics": {
            "accuracy":  acc,
            "macro_f1":  f1_info["macro_f1"],
            "per_class": {k: v for k, v in f1_info.items() if k != "macro_f1"},
        },
        "confusion_matrix": cm,
        "fidelity_score":   score,
    }


# ===========================================================================
# Pretty printing
# ===========================================================================

def print_report(report: dict,
                 title: str = "EVALUATION REPORT",
                 verdict: str | None = None) -> None:
    n  = report["n_samples"]
    tm = report["text_metrics"]
    em = report["emotion_metrics"]
    cm = report["confusion_matrix"]

    bar = "=" * 60
    print(f"\n{bar}\n  {title}  ({n} samples)\n{bar}")

    print("\n-- Text --")
    print(f"  WER  : {tm['mean_WER']:.4f}  (std {tm['std_WER']:.4f})")
    print(f"  CER  : {tm['mean_CER']:.4f}  (std {tm['std_CER']:.4f})")

    print("\n-- Emotion --")
    print(f"  Accuracy : {em['accuracy']:.4f}")
    print(f"  Macro F1 : {em['macro_f1']:.4f}\n")

    rows = [[lbl, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}"]
            for lbl, v in em["per_class"].items()]
    headers = ["Emotion", "Precision", "Recall", "F1"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
                 for i, h in enumerate(headers)]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  " + "  ".join("-" * w for w in col_w))
        for r in rows:
            print(fmt.format(*r))

    print("\n-- Confusion Matrix (rows = true, cols = predicted) --")
    labels = list(cm.keys())
    if HAS_TABULATE:
        cm_rows = [[lbl] + [cm[lbl][p] for p in labels] for lbl in labels]
        print(tabulate(cm_rows, headers=["true \\ pred"] + labels,
                       tablefmt="rounded_outline"))
    else:
        col_w = max((len(l) for l in labels), default=6)
        header = "  " + " " * col_w + "  " + "  ".join(f"{l:>{col_w}}" for l in labels)
        print(header)
        for lbl in labels:
            row = "  ".join(f"{cm[lbl][p]:>{col_w}}" for p in labels)
            print(f"  {lbl:>{col_w}}  {row}")

    print("\n-- Fidelity Score --")
    print(f"  Score = (1 - WER) * Accuracy = {report['fidelity_score']:.4f}")
    if verdict:
        print(f"  Verdict: {verdict}")
    print(f"{bar}\n")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the EmoSpeech TTS via its STT/SER reconstructions."
    )
    p.add_argument("--results", "-r", required=True, type=Path,
                   help="CSV of pipeline outputs (generated audio judged by STT+SER).")
    p.add_argument("--reference-csv", type=Path, default=None,
                   help="Optional CSV of STT+SER run on ORIGINAL audio, used as ceiling.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Where to save the JSON report.")
    p.add_argument("--emotions", nargs="+", default=None,
                   help="Emotion labels (default: ESD set).")
    p.add_argument("--sep", default=",",
                   help="CSV separator (default ','). Use '\\t' for TSV.")
    p.add_argument("--per-condition", action="store_true",
                   help=f"Split report per '{COL_CONDITION}' column if present.")
    return p.parse_args()


def _load(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, sep=sep)


def main():
    args = parse_args()
    sep = "\t" if args.sep == "\\t" else args.sep
    labels = args.emotions or ESD_EMOTIONS

    print(f"[info] Loading results from: {args.results}")
    df = _load(args.results, sep)
    print(f"[info] {len(df)} rows. Columns: {list(df.columns)}")

    ref_score = None
    if args.reference_csv is not None:
        print(f"[info] Loading reference (ceiling) from: {args.reference_csv}")
        df_ref = _load(args.reference_csv, sep)
        ref_report = compute_report(df_ref, labels=labels)
        ref_score = ref_report["fidelity_score"]
        print_report(ref_report,
                     title="CEILING (original audio judged by STT+SER)")

    full_report = {}

    if args.per_condition and COL_CONDITION in df.columns:
        for cond, sub in df.groupby(COL_CONDITION):
            print(f"\n>>> Condition: {cond}")
            rep = compute_report(sub, labels=labels)
            verdict = interpret_training_quality(rep["fidelity_score"], ref_score)
            print_report(rep, title=f"REPORT — {cond}", verdict=verdict)
            full_report[str(cond)] = {**rep, "verdict": verdict}
    else:
        rep = compute_report(df, labels=labels)
        verdict = interpret_training_quality(rep["fidelity_score"], ref_score)
        print_report(rep, verdict=verdict)
        full_report = {**rep, "verdict": verdict}

    if ref_score is not None:
        full_report["ceiling_fidelity_score"] = ref_score

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"[info] Report saved to: {args.output}")


if __name__ == "__main__":
    main()
