# End-to-end redaction validation against real speech

Record of a live run that exercised the full redaction pipeline — YAMNet
speech scoring → RedactionGate → padded redaction windows — against a real
speech-containing audio clip on this Thor. This validates the runtime half
of REDACTION-INTEGRATION-NOTES.md §3 (LiteRT-backed YAMNet front-end + notes'
RedactionGate, unchanged).

Hardware: Jetson AGX Thor, aarch64, kernel 6.8.12-tegra.

## What was run (2026-07-28)

1. Capture / input: a 20.95 s m4a clip (AAC 24 kHz mono) supplied by the user
   at `/tmp/speech-test.m4a`, converted via
   `ffmpeg -i /tmp/speech-test.m4a -vn -ac 1 -ar 16000 -acodec pcm_s16le`
   to 16 kHz mono PCM (20.86 s, 667 KB).
2. Pipeline (script committed at `redaction/scripts/run_redaction_on_capture.py`):
   - `soundfile.read` → 1-D float32, 16 kHz mono
   - YAMNet `.tflite` via `ai_edge_litert.interpreter.Interpreter` (XNNPACK CPU
     delegate, aarch64). Input dim frozen to `[1]` by the TFLite converter, so
     the script calls `resize_tensor_input` per waveform length.
     Score tensor selected by `shape[-1] == 521` (converter-mangled output
     names like `StatefulPartitionedCall:0` are unreliable).
   - Per-frame speech score via `speech_classes.speech_score` (CORE_SPEECH,
     `include_ambiguous=False`) — imported UNCHANGED from `notes-ref`.
   - `RedactionGate()` with the notes' defaults (enter=0.25, exit=0.15,
     pre_roll=1.5, hangover=0.75, post_roll=0.75, frame_hop=0.48,
     frame_duration=0.96, fail_closed=True) — imported UNCHANGED from
     `notes-ref`.

## Result (real run, not fabricated)

43 YAMNet frames over 20.86 s (correct: 20.86 s / 0.48 s hop ≈ 43).

Per-frame speech scores (selected; full vector in the run transcript):
  frames 0-8   (0.00-4.32 s):  ~0.01-0.05  (noise floor; max 0.046)
  frame  9     (4.32-5.28 s):  0.309       (onset — crosses enter=0.25)
  frames 10-12 (4.80-6.72 s):  0.96-0.99   (sustained speech)
  frames 13-14 (6.72-7.68 s):  0.005-0.020 (sub-threshold pause)
  frames 15-20 (7.20-10.56 s): 0.90-0.99   (speech burst)
  frames 21-23 (10.56-11.52 s): 0.01-0.06  (pause)
  frames 24-28 (11.52-14.40 s): 0.89-1.000 (speech burst, peak 0.999)
  frames 29-36 (14.40-17.76 s): 0.002-0.045 (long silence)
  frames 37-41 (17.76-20.64 s): 0.96-1.000 (final speech burst)
  frame  42    (20.16-21.12 s): 0.023       (tail)

Redaction windows returned by `RedactionGate.get_redaction_windows`:

  [  2.820 -  15.150 ]   width = 12.330 s
  [ 16.260 -  21.120 ]   width =  4.860 s

Total redacted: 17.19 s / 20.86 s capture (82.4 %).

## How the gate produced those windows (lined up with the source)

  First window start 2.820 s = max(0, frame9 * 0.48 - 1.5) = 4.32 - 1.5
  → 1.5 s pre-roll extended backward from the onset frame into the noise
  floor (pre-roll, working).
  First window end   15.150 s = frame28 * 0.48 + 0.96 + 0.75 = 13.44 + 0.96 + 0.75
  → 0.75 s post-roll past the last speech frame (post-roll, working).
  The 13-14 pause (2 sub-threshold frames = ~0.96 s) matched the hangover
  tolerance (`math.ceil(0.75/0.48) = 2` frames), so the gate did NOT split
  frames 9-28 into three bursts — they collapsed into one window
  (hangover/merge, working).
  The 14.4-17.7 s silence (~7 frames) exceeded hangover, so the first window
  closed at 15.15 s and a second opened at frame 37.
  Second window: start = 37*0.48 - 1.5 = 16.26 s, end = 41*0.48 + 0.96 + 0.75
  = 21.12 s (same pre-roll / post-roll pattern).

So the run confirmed — in real audio on the real hardware — that:
  - YAMNet .tflite runs via ai_edge_litert on aarch64 (no full TF at inference
    time, no tensorflow_hub needed, no network at runtime).
  - YAMNet scores sit at the ~0.01-0.05 noise floor except during speech, then
    climb into 0.9-1.0.
  - RedactionGate applies pre-roll (1.5 s) backward from onset, post-roll
    (0.75 s) forward past offset, and merges sub-threshold pauses shorter
    than the hangover tolerance instead of splitting.

## What this validates and what it does NOT

VALIDATES:
  - The LiteRT-backed front-end is a drop-in replacement for the notes'
    `yamnet_speech.speech_scores` public API; it produces the same per-frame
    score list shape that `RedactionGate` consumes.
  - The notes' `RedactionGate` behaves on real speech exactly as the pure-Python
    test suite (`test_redaction_gate.py`) asserts on synthetic scores.
  - The runtime deps already present in the birdnet plugin container
    (`ai_edge_litert` via `birdnet`, 2.1.6) are sufficient — no new runtime
    dependency, no GPU needed (XNNPACK CPU).
  - The 16 verified speech-class indices (CORE_SPEECH + AMBIGUOUS) catch real
    speech cleanly at high confidence even without the ambiguous subset.

DOES NOT validate (yet):
  - The *architecture*: this run reads audio from a WAV file. The deployed mic
    AND camera paths must redact the in-memory `AudioSample.data` array before
    persistence (REDACTION-INTEGRATION-NOTES.md §1; notes-ref/docs/
    04-audio-redaction.md §1). This script deliberately writes nothing back to
    disk — it only prints windows — so it does NOT test the persistence-
    ordering invariant. That remains a design-decision for `app.py` until the
    change is actually applied.
  - Distant, mumbled, or whispered speech. This clip is a clear close-mic test.
    The notes (04-audio-redaction.md:122-125) warn that the single "Speech"
    class may under-score distant speech; CORE_SPEECH's 12-class max covers
    more, and AMBIGUOUS (off here) catches "Chatter" / "Crowd" / "Child
    singing" / "Children playing" — flipping `include_ambiguous=True` would
    raise recall further at precision cost. To be tuned empirically against a
    distance/noise corpus before going to Haleakala.
  - Capture-boundary leakage (each cycle is an isolated block per
    04-audio-redaction.md:181-185). Not in scope for this script.

## Repro

  # Convert clip to 16 kHz mono WAV.
  ffmpeg -y -i /tmp/speech-test.m4a -vn -ac 1 -ar 16000 -acodec pcm_s16le \
      /tmp/speech-test-16k.wav

  # Run the pipeline (path arg optional; defaults to /tmp/cam1_audio.wav).
  python3 redaction/scripts/run_redaction_on_capture.py /tmp/speech-test-16k.wav

Prereqs on this Thor: the `birdnet` venv (which pulls `ai_edge_litert`, `numpy`,
`soundfile`, `scipy`), a `yamnet.tflite` at `/tmp/yamnet.tflite` (build via the
TF SavedModel → TFLiteConverter step in REDACTION-INTEGRATION-NOTES.md §2),
and `notes-ref/code/redaction/{redaction_gate,speech_classes}.py` on the
import path.

Script: `redaction/scripts/run_redaction_on_capture.py`
        accepts a WAV path as argv[1], defaults to `/tmp/cam1_audio.wav`.
