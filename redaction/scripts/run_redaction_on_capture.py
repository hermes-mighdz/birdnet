#!/usr/bin/env python3
"""
End-to-end test of the redaction pipeline against a real camera-audio capture.

Inputs:
  /tmp/cam1_audio.wav   10 s mono 16 kHz PCM captured from Reolink Camera 1

Pipeline:
  WAV -> numpy array -> [YAMNet .tflite -> per-frame speech scores] ->
  [RedactionGate -> redaction windows] -> print

NOTE on the architecture:
  This script reads the capture from a WAV FILE. That is the *test* harness.
  The production mic AND camera paths in app.py must redact the in-memory
  array BEFORE persistence (see REDACTION-INTEGRATION-NOTES.md §1 and
  notes-ref/docs/04-audio-redaction.md §1). This script deliberately writes
  nothing redacted or unredacted back to disk; it just prints the windows.

Why this uses a LiteRT adapter instead of notes-ref yamnet_speech.py directly:
  The notes' yamnet_speech.py loads YAMNet via `tensorflow_hub`, which is NOT
  available on this Thor and is not a birdnet dependency. The verified aarch64
  path (REDACTION-INTEGRATION-NOTES.md §2-3) is the YAMNet .tflite run via
  `ai_edge_litert.interpreter.Interpreter`. This script provides a drop-in
  `speech_scores()` that mirrors the notes' public API but uses LiteRT.
  `RedactionGate` and `speech_classes` are imported UNMODIFIED from notes-ref.
"""
import argparse
import os
import sys

import numpy as np

# notes-ref redaction modules: imported unchanged
NOTES_REDACTION = "/home/mighdz/AI-Projects/notes-ref/code/redaction"
sys.path.insert(0, NOTES_REDACTION)
from redaction_gate import RedactionGate, RedactionGateFailure  # noqa: E402
from speech_classes import speech_score, CORE_SPEECH  # noqa: E402  (verification only)

# LiteRT YAMNet front-end (verified working on this Thor earlier this session)
from ai_edge_litert.interpreter import Interpreter  # noqa: E402

# Resolve the .tflite path with the same fallback chain yamnet_speech.py uses
# (env override → plugin container → persistent dev → volatile dev), BUT with
# the persistent dev location ahead of /tmp so the script survives reboots:
# /tmp/yamnet.tflite is walked off by a reboot or tmpfiles.d clean, which would
# silently break this demo. The persistent path is the one recommended in
# REDACTION-INTEGRATION-NOTES.md §3 "Reboot persistence of the .tflite".
#
# NOTE: literal absolute path, not ~ expansion. On a Sage plugin container AND
# in this dev sandbox, HOME is NOT /home/mighdz: os.path.expanduser("~") would
# resolve to the wrong home and the file would never be found. This is a dev-
# Thor-specific path; for any other dev machine, export BIRDNET_YAMNET_TFLITE.
# Order:
#   1. BIRDNET_YAMNET_TFLITE env override (highest precedence)
#   2. /app/models/yamnet.tflite                  (plugin container; Dockerfile COPY)
#   3. /home/mighdz/AI-Projects/models/yamnet.tflite (persistent dev: survives reboots)
#   4. /tmp/yamnet.tflite                          (volatile dev scratch: last resort)
_YAMNET_TFLITE_PATHS = [
    os.environ.get("BIRDNET_YAMNET_TFLITE"),
    "/app/models/yamnet.tflite",
    "/home/mighdz/AI-Projects/models/yamnet.tflite",
    "/tmp/yamnet.tflite",
]
YAMNET_TFLITE = next((p for p in _YAMNET_TFLITE_PATHS if p and os.path.exists(p)), None)

YAMNET_SAMPLE_RATE = 16000


def load_wav_mono(path):
    """Read a WAV into a 1-D float32 array in [-1, 1] + its sample rate."""
    import soundfile as sf
    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32), int(sr)


def speech_scores_lithert(audio_1d, samplerate, tflite_path=YAMNET_TFLITE,
                          include_ambiguous=False):
    """LiteRT-backed equivalent of notes-ref yamnet_speech.speech_scores.

    Mirrors the notes' public API exactly. Uses the verified YAMNet .tflite
    via ai_edge_litert; no tensorflow_hub, no network.
    """
    from speech_classes import AMBIGUOUS  # local import; lives in notes-ref
    # Reproduce the notes' _prepare_waveform normalization (downmix + int->float
    # + resample to 16 kHz). The notes use linear interpolation as a first pass.
    audio = np.asarray(audio_1d)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("empty audio buffer")
    if samplerate <= 0:
        raise ValueError(f"invalid samplerate {samplerate}")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    audio = audio.astype(np.float32)
    if samplerate != YAMNET_SAMPLE_RATE:
        n_out = int(round(audio.size * YAMNET_SAMPLE_RATE / samplerate))
        t_out = np.arange(n_out) * (samplerate / YAMNET_SAMPLE_RATE)
        audio = np.interp(t_out, np.arange(audio.size), audio).astype(np.float32)

    # Inference: the converter froze the input dim to [1]; resize per waveform.
    interp = Interpreter(model_path=tflite_path)
    in_det = interp.get_input_details()[0]
    interp.resize_tensor_input(in_det["index"], [audio.size])
    interp.allocate_tensors()
    interp.set_tensor(in_det["index"], audio)
    interp.invoke()

    # Find the scores tensor by shape (..., 521). Converter-mangled output names
    # like 'StatefulPartitionedCall:0' are unreliable; shape is stable.
    scores_arr = None
    for t in interp.get_output_details():
        arr = interp.get_tensor(t["index"])
        if arr.shape[-1] == 521:
            scores_arr = arr
            break
    if scores_arr is None:
        raise RuntimeError("YAMNet .tflite produced no (N,521) scores tensor")

    if scores_arr.ndim == 2 and scores_arr.shape[0] == 1:
        scores_arr = scores_arr[0]  # squeeze leading batch dim if present
    return [speech_score(frame, include_ambiguous=include_ambiguous)
            for frame in scores_arr]


def zero_windows_in_place(audio_1d, windows, samplerate):
    """Zero redaction windows into a COPY of audio_1d. Mirrors redaction/apply.py.

    Returns a new array; the caller's original buffer is untouched. Windows are
    (start_s, end_s) float pairs (seconds), converted to sample indices at the
    ORIGINAL samplerate: same index math as redaction/apply.py:80-84.
    """
    out = audio_1d.copy()
    n = out.size
    for start_s, end_s in windows:
        i0 = max(0, int(start_s * samplerate))
        i1 = min(n, int(end_s * samplerate))
        if i0 < i1:
            out[i0:i1] = 0.0
    return out


def main():
    # Positional WAV path (defaults to the Camera 1 capture) + optional
    # --write-redacted PATH (demo flag). When --write-redacted is absent the
    # script's behavior is byte-identical to before this flag existed.
    p = argparse.ArgumentParser(
        description="Run the redaction pipeline on a WAV capture and print windows.")
    p.add_argument("wav_path", nargs="?", default="/tmp/cam1_audio.wav",
                   help="input WAV (default: %(default)s)")
    p.add_argument("--write-redacted", metavar="PATH", default=None,
                   help="also write the redacted audio as a 16 kHz mono WAV to PATH "
                        "(the input file is never modified)")
    args = p.parse_args()

    wav_path = args.wav_path
    write_redacted = args.write_redacted
    if not os.path.exists(wav_path):
        print(f"ERROR: {wav_path} not found: run the ffmpeg capture first", file=sys.stderr)
        sys.exit(1)
    if not YAMNET_TFLITE or not os.path.exists(YAMNET_TFLITE):
        print(f"ERROR: YAMNet .tflite not found. Checked: "
              f"{', '.join(p or '<env unset>' for p in _YAMNET_TFLITE_PATHS)}. "
              f"Set BIRDNET_YAMNET_TFLITE or bake the model into the image "
              f"(see REDACTION-INTEGRATION-NOTES.md §3).", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {wav_path} ...")
    audio, sr = load_wav_mono(wav_path)
    duration_s = audio.size / sr
    print(f"  loaded {audio.size} samples @ {sr} Hz  ({duration_s:.2f} s, dtype={audio.dtype})")
    print(f"  amplitude: min={audio.min():.4f} max={audio.max():.4f} rms={np.sqrt((audio**2).mean()):.4f}")

    print("Running YAMNet .tflite (LiteRT, aarch64 CPU) ...")
    scores = speech_scores_lithert(audio, sr)
    n_frames = len(scores)
    print(f"  {n_frames} YAMNet frames (0.96 s window / 0.48 s hop)")
    print(f"  per-frame speech scores: " + ", ".join(f"{s:.3f}" for s in scores))

    # Also show the top scoring frames for sanity (which time ranges lit up).
    if n_frames:
        threshold_show = 0.10
        flagged = [(i, s) for i, s in enumerate(scores) if s >= threshold_show]
        if flagged:
            print(f"  frames >= {threshold_show:.2f}:")
            for i, s in flagged:
                t0 = i * 0.48
                t1 = t0 + 0.96
                print(f"    frame {i:>2}  [{t0:5.2f}-{t1:5.2f}s]  score={s:.4f}")
        else:
            print(f"  no frames >= {threshold_show:.2f} (max={max(scores):.4f})")

    print("\nRunning RedactionGate (notes-ref defaults: enter=0.25 exit=0.15 "
          "pre_roll=1.5 hangover=0.75 post_roll=0.75) ...")
    gate = RedactionGate()  # notes' defaults; fail_closed=True by default
    try:
        windows = gate.get_redaction_windows(scores)
    except RedactionGateFailure as e:
        print(f"  RedactionGate failed closed: {e}")
        print("  -> Correct privacy posture: redact the ENTIRE capture [0, duration].")
        windows = [(0.0, duration_s)]

    # --- summary (printed in all cases; reorganized so it's usable with/without --write-redacted) ---
    total = float(sum(end - start for start, end in windows))
    pct = 100.0 * total / duration_s if duration_s else 0.0

    print("\n=== REDACTION SUMMARY ===")
    print(f"  input: {os.path.basename(wav_path)}  ({duration_s:.2f}s @ {sr} Hz)")
    print(f"  windows: {len(windows)}")
    if windows:
        for (start, end) in windows:
            w = end - start
            tag = "  <-- clamped to full buffer (fail-closed)" if start == 0.0 and end >= duration_s - 0.01 else ""
            print(f"    [{start:6.3f}, {end:6.3f}]  width={w:5.3f}s{tag}")
    else:
        print("  (none: no speech detected, no redaction needed)")
    print(f"  redacted: {total:.3f}s / {duration_s:.2f}s capture ({pct:.1f}%)")

    # --- optional before/after demo: write the redacted copy out as 16 kHz mono WAV ---
    #! The INPUT file is never modified: zero_windows_in_place() works on a copy,
    #! and soundfile.write() opens a fresh file at write_redacted.
    if write_redacted:
        import soundfile as sf
        redacted = zero_windows_in_place(audio, windows, sr)
        # Resample to 16 kHz mono if needed (YAMNet's rate; same linear-interp as
        # speech_scores_lithert's _prepare path, so the redacted WAV matches what
        # the model saw). For an input already at 16 kHz mono this is a no-op copy.
        out = redacted
        if sr != YAMNET_SAMPLE_RATE:
            n_out = int(round(out.size * YAMNET_SAMPLE_RATE / sr))
            t_out = np.arange(n_out) * (sr / YAMNET_SAMPLE_RATE)
            out = np.interp(t_out, np.arange(out.size), out).astype(np.float32)
        sf.write(write_redacted, out, YAMNET_SAMPLE_RATE, subtype="FLOAT")
        print(f"  output: {write_redacted}  ({out.size} samples @ {YAMNET_SAMPLE_RATE} Hz mono FLOAT)")


if __name__ == "__main__":
    main()
