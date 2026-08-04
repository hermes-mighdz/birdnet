#!/usr/bin/env python3
"""
Threshold-tuning harness for the redaction gate.

Scans a directory of labeled WAV clips: filename convention
``speech_*.wav`` (positive) vs ``nospeech_*.wav`` (negative): runs each
through the LiteRT YAMNet speech-scoring path (the same one
``run_redaction_on_capture.py`` uses), then sweeps ``enter_threshold`` over
0.05..0.50 (0.05 steps) and reports, per threshold:

  - recall          = (# speech clips flagged) / (# speech clips)
  - false-positive  = (# no-speech clips flagged) / (# no-speech clips)

A clip is "flagged" if ``RedactionGate.get_redaction_windows`` returns any
non-empty window list for that clip's per-frame scores.

Other gate parameters are held at the notes-ref design defaults
(``exit_threshold`` is pinned to ``min(enter, 0.15)`` so it never exceeds
``enter_threshold``, which RedactionGate rejects; everything else: pre_roll,
hangover, post_roll: is the notes' default).

If matplotlib is importable in the running venv, a recall-vs-FPR-per-threshold
scatter is also written to ``--plot PATH`` (default: recall_vs_threshold.png
in the CWD). If matplotlib is NOT available, the script prints a one-line skip
and still exits 0: the table is the primary output.

Does NOT modify any redaction module. ``redaction_gate`` is imported UNMODIFIED
from notes-ref (same import pattern as ``run_redaction_on_capture.py``).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the LiteRT scoring front-end + WAV loader from the sibling script
# (speech_scores_lithert: same YAMNet .tflite path + _prepare_waveform logic).
# The sibling script's NOTES_REDACTION sys.path.insert also re-exports
# `redaction_gate` and `speech_classes`: so importing it as a module buys us
# all of those, plus keeps a single source of truth for the scoring path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_redaction_on_capture import (  # noqa: E402
    speech_scores_lithert, load_wav_mono, YAMNET_TFLITE,
)

# notes-ref RedactionGate (imported unchanged): already on sys.path via the
# sibling script's NOTES_REDACTION insert. Import lazily so `--help` works
# even when the .tflite / notes-ref path is missing.
def _gate_cls():
    from redaction_gate import RedactionGate  # noqa: E402
    return RedactionGate


# ── the sweep ─────────────────────────────────────────────────────────
ENTER_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 11)]  # 0.05..0.50
EXIT_DEFAULT = 0.15   # notes-ref design default; clamped <= enter per gate's rule


def gate_flags_clip(scores, enter_threshold, exit_threshold):
    """Return True if RedactionGate flags ANY frame under this (enter, exit).

    Uses the notes-ref RedactionGate with the *other* parameters at notes'
    defaults (pre_roll, hangover, post_roll). ``fail_closed`` is left at True
    (the production default) but for the tuning harness a zero-score buffer
    raising RedactionGateFailure is treated as "not flagged": a no-speech clip
    that produces no YAMNet scores at all should count as a TN, not a
    fail-closed full-redaction. (The production path in app.py treats the
    same failure as fail-closed; here we're measuring the *gate's binary
    decision*, not its safety posture.)
    """
    RedactionGate = _gate_cls()
    try:
        gate = RedactionGate(
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            # other args default per notes-ref design:
            # pre_roll_seconds=1.5, hangover_seconds=0.75,
            # post_roll_seconds=0.75, frame_hop=0.48, frame_duration=0.96
        )
        windows = gate.get_redaction_windows(scores)
    except Exception:
        # Gate raised (empty scores with fail_closed, etc.). For tuning
        # purposes: a clip with no YAMNet score frame counts as not-flagged.
        return False
    return len(windows) > 0


def iter_labeled_clips(wav_dir):
    """Yield (path, label) for speech_*.wav / nospeech_*.wav in wav_dir.

    Case-sensitive on the prefix (matches the requested convention). Files
    not matching either prefix are silently skipped (with a stderr note if
    any are found, so the user knows their dir has stragglers).
    """
    wav_dir = Path(wav_dir)
    speech, nospeech, skipped = [], [], []
    for p in sorted(wav_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        name = p.name
        if name.startswith("speech_"):
            speech.append(p)
        elif name.startswith("nospeech_"):
            nospeech.append(p)
        else:
            skipped.append(name)
    if skipped:
        print(f"NOTE: skipping {len(skipped)} non-conforming file(s) in {wav_dir}:",
              file=sys.stderr)
        for n in skipped[:5]:
            print(f"  {n}", file=sys.stderr)
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more", file=sys.stderr)
    return speech, nospeech


def score_all_clips(clips, tflite_path):
    """Run YAMNet scoring once per clip; cache the score list for the sweep.

    Scoring is the expensive step and is threshold-independent, so we run it
    once and reuse the score vectors across all (enter, exit) pairs in the sweep.
    Returns list of (path, scores) tuples. On a YAMNet failure for an individual
    clip, we record scores=[] and let the gate treat it as not-flagged (see
    gate_flags_clip). Two cases are distinguished so the user isn't spammed:

      - clip shorter than YAMNet's 0.96 s frame minimum → silent (expected edge
        case); scores=[] records it as a true-negative at every threshold.
      - any other YAMNet failure → loud WARN with the exception (real bug).
    """
    YAMNET_FRAME_SECONDS = 0.96  # YAMNet window length; 0.48 s hop. <1 frame → 0 scores.
    out = []
    for p in clips:
        try:
            audio, sr = load_wav_mono(str(p))
            if audio.size / sr < YAMNET_FRAME_SECONDS:
                # Too short for a single YAMNet frame. Silent: scores=[] →
                # gate_flags_clip's except branch counts it as not-flagged (TN).
                # Counts against both labels symmetrically, so it doesn't bias
                # recall or FPR: it just dilutes the effective N.
                out.append((p, []))
                continue
            scores = speech_scores_lithert(audio, sr, tflite_path=tflite_path)
            if not isinstance(scores, list):
                # Defensive: the sibling adapter returns a list; if a future
                # change lets a scalar through, coerce rather than crash later.
                scores = list(scores) if hasattr(scores, "__iter__") else []
        except Exception as e:
            print(f"  WARN: scoring failed for {p.name}: {e}", file=sys.stderr)
            scores = []
        out.append((p, scores))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Tune redaction-gate enter_threshold over a labeled WAV set.")
    p.add_argument("wav_dir",
                   help="directory of labeled WAVs (speech_*.wav / nospeech_*.wav)")
    p.add_argument("--yamnet-tflite", default=YAMNET_TFLITE,
                   help=f"YAMNet .tflite path (default: %(default)s)")
    p.add_argument("--plot", default="recall_vs_threshold.png",
                   help="output PNG for the recall-vs-FPR plot (default: %(default)s; "
                        "skipped silently if matplotlib is unavailable)")
    p.add_argument("--exit-threshold", type=float, default=EXIT_DEFAULT,
                   help=f"exit_threshold pinned across the sweep "
                        f"(default: {EXIT_DEFAULT}; clamped <= enter_threshold)")
    args = p.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        print(f"ERROR: {wav_dir} is not a directory", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.yamnet_tflite):
        print(f"ERROR: {args.yamnet_tflite} not found: build it first "
              f"(see REDACTION-INTEGRATION-NOTES.md §3)", file=sys.stderr)
        sys.exit(2)

    speech_paths, nospeech_paths = iter_labeled_clips(wav_dir)
    n_speech, n_nospeech = len(speech_paths), len(nospeech_paths)
    if n_speech == 0 or n_nospeech == 0:
        print(f"ERROR: need at least 1 speech_*.wav AND 1 nospeech_*.wav in {wav_dir}",
              file=sys.stderr)
        print(f"  found: {n_speech} speech, {n_nospeech} nospeech", file=sys.stderr)
        sys.exit(2)
    print(f"Scoring {n_speech} speech + {n_nospeech} nospeech clips "
          f"({n_speech + n_nospeech} total) via LiteRT YAMNet ...")
    speech_scores = score_all_clips(speech_paths, args.yamnet_tflite)
    nospeech_scores = score_all_clips(nospeech_paths, args.yamnet_tflite)

    # ── sweep ────────────────────────────────────────────────────────
    rows = []  # (enter, exit, TP, FN, FP, TN, recall, fpr)
    print(f"\nSweeping enter_threshold over {len(ENTER_THRESHOLDS)} values; "
          f"exit_threshold pinned at {args.exit_threshold} (clamped <= enter):\n")
    print(f"{'enter':>6} {'exit':>6} | {'recall':>7} {'TP/N':>9} | "
          f"{'FPR':>6} {'FP/N':>9} | notes-default")
    print("-" * 70)
    for enter in ENTER_THRESHOLDS:
        exit_t = min(args.exit_threshold, enter)  # gate rejects exit > enter
        tp = sum(gate_flags_clip(s, enter, exit_t) for _, s in speech_scores)
        fp = sum(gate_flags_clip(s, enter, exit_t) for _, s in nospeech_scores)
        recall = tp / n_speech if n_speech else 0.0
        fpr = fp / n_nospeech if n_nospeech else 0.0
        rows.append((enter, exit_t, tp, n_speech - tp, fp, n_nospeech - fp, recall, fpr))
        flag = "  <-- notes default" if abs(enter - 0.25) < 1e-9 else ""
        print(f"{enter:6.2f} {exit_t:6.2f} | {recall:7.3f} {tp:>3}/{n_speech:<3}     | "
              f"{fpr:6.3f} {fp:>3}/{n_nospeech:<3}     {flag}")

    # ── plot (graceful: skip if matplotlib is missing) ───────────────
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available in this venv: skipping plot "
              f"({args.plot} not written). The table above is the primary output.",
              file=sys.stderr)
        return  # exit 0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r[0] for r in rows]
    recalls = [r[6] for r in rows]
    fprs = [r[7] for r in rows]
    ax.plot(xs, recalls, "o-", label="recall (speech clip caught)", color="tab:blue")
    ax.plot(xs, fprs, "s--", label="FPR (no-speech clip flagged)", color="tab:red")
    ax.axvline(0.25, color="grey", linestyle=":", alpha=0.6, label="notes default (0.25)")
    ax.set_xlabel("enter_threshold")
    ax.set_ylabel("rate")
    ax.set_title(f"Redaction gate threshold sweep "
                 f"(n_speech={n_speech}, n_nospeech={n_nospeech})")
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"\nPlot written to {args.plot}")


if __name__ == "__main__":
    main()
