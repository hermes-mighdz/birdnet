# Redaction system overview: how the gate, scores, and fail-closed contract fit together

Reference explanation of the speech-redaction subsystem in the birdnet
plugin: the microphone path integration, the confidence/score mechanisms,
and the fail-closed invariants that keep raw audio off disk on every path.

This document is the *explanation* layer. For the integration walkthrough
and history, see `REDACTION-INTEGRATION-NOTES.md` (especially §4 Step 4,
which records the actually-landed caller pattern in `record_from_microphone`).
For the design proposal to extend redaction to the camera path, see
`redaction/CAMERA-PATH-DESIGN.md`.

---

## 1. What the mic redaction path does

`record_from_microphone` (app.py:164-222) records audio from the node's USB
microphone via pywaggle, runs a **speech-redaction gate** on the in-memory
PCM array *before* persistence, then writes the (already-redacted) array
to a temp FLAC for downstream BirdNET inference. The raw array:
specifically the speech segments of it: never reaches disk.

Concretely, between `mic.record(duration_s)` and `sample.save(flac_path)`:

```python
try:
    _redacted, _events, _reason = redaction_apply.redact_speech(
        sample.data, int(sample.samplerate)
    )
    if _reason is not None:
        logger.warning("Speech redaction failed closed (%s) ...", _reason, duration_s)
    else:
        logger.info("Speech redaction applied: %d window(s) ...", len(_events), duration_s)
except (YAMNetRedactionFailure, RedactionGateFailure) as e:
    sample.data.fill(0.0); logger.error(...)
except Exception:
    sample.data.fill(0.0); logger.exception(...)
```

`redact_speech` (`redaction/apply.py:34-86`) takes a 1-D float32 array and a
samplerate, runs YAMNet per-frame speech scoring (`redaction/yamnet_speech.
speech_scores`), passes those scores to a `RedactionGate` to compute speech
windows, and **zeros those windows in place on the same array object**. It
returns `(audio, windows, reason)` where `reason` is `None` on the normal
path or a string on the fail-closed path.

The downstream pipeline is unchanged: `classify_file(audio_path)`,
`publish_detections`, and the `--save-match` upload logic are all
file-path-shaped and operate on the already-redacted FLAC. The privacy gate
is invisible to BirdNET.

---

## 2. The fail-closed contract

Every path from `mic.record()` to `sample.save()` either:

- **Successfully redacts**: `redact_speech` ran YAMNet + the gate, zeroed speech
  windows in place on `sample.data`, returned `_reason is None`. The FLAC
  contains silence at speech windows and the non-speech audio elsewhere.
- **Fails closed inside `redact_speech`**: `RedactionFailure` or
  `RedactionGateFailure` was raised inside the function (model missing,
  inference blew up, RedactionGate got zero frames with `fail_closed=True`).
  `redact_speech` catches those itself at `apply.py:69-74`, runs
  `audio_1d.fill(0.0)`, and returns the all-zero buffer with
  `windows=[(0.0, duration)]` and `_reason = str(e)`. Caller logs WARNING and
  falls through to `sample.save()`: which writes silence. Raw audio is
  already gone (overwritten at apply.py:73).
- **Fails closed in the caller's typed-except**: defensive, currently
  unreachable given apply.py:69's catch, but guarantees fail-closed
  semantics if `redact_speech`'s internal `except` clause ever narrows.
  Force-zero the buffer, log ERROR, save silence.
- **Fails closed in the caller's broad `except Exception`**: for anything
  that escapes `redact_speech` unwrapped (`MemoryError`, a LiteRT
  `RuntimeError` that slipped past the bare `except Exception` in
  `speech_scores` at yamnet_speech.py:132). Force-zero, log full traceback,
  save silence.
- **The `BaseException` case (`KeyboardInterrupt`, `SystemExit`)**: `except
  Exception` does not catch these. A Ctrl-C between `mic.record()` and
  `sample.save()` unwinds straight out of `record_from_microphone`.
  `sample.save()` is never reached (no raw FLAC is written), and the raw
  `sample.data` array dies with the stack frame when GC reclaims it. The
  "raw audio never hits disk" invariant holds here too, by a different
  mechanism: the cycle fails entirely rather than saving zeros.

Privacy posture: if we cannot classify speech, assume all of it is speech.
Under-redacting is the catastrophic-direction failure (a leaked syllable is
a privacy violation), so the gate err toward over-redaction at every branch.

---

## 3. Confidence scores and the two-track score semantics

"Confidence" means two distinct things in this codebase, and they don't
share a metric or a calibration. Worth pulling apart because they're
frequently conflated.

### 3a. BirdNET species confidence: what gets *published*

`classify_file` (`app.py:122-156`) returns one detection dict per
recognized species, with `det['confidence']` as BirdNET V2.4's sigmoid-style
probability for that species over its 6,522-class label set. When
`--lat/--lon--week` are provided, this is geo-filtered via the eBird range
model so out-of-range species are suppressed; `--sf-thresh` (default 0.03)
sets the species-filter threshold for the geo model independently.

Two downstream decisions ride on BirdNET confidence:

- **Publication.** Detections with confidence >= `--min-confidence`
  (default 0.03) are published to Waggle under their soundscape-ecology
  category (`env.detection.biophony.<taxon>` /
  `env.detection.anthrophony.<name>` / `env.detection.geophony.<name>`).
- **Selective audio upload.** The `--save-match` rules
  (`Name:confidence` or `*:confidence`) decide which captured audio clips
  get uploaded to Beehive *after* BirdNET inference. This is the mechanism
  that decides whether a clip leaves the node at all.

BirdNET confidence is **not** calibrated to YAMNet's speech score: they
are different models with different output semantics, different label sets,
and different calibration. Don't compare the numbers directly.

### 3b. YAMNet speech score: what feeds the redaction gate

`speech_scores()` (`redaction/yamnet_speech.py:99-153`) returns one float
per 0.96 s YAMNet frame over 0.48 s hops, in [0, 1]. Each frame's score is
`max` over the 12 CORE_SPEECH YAMNet class indices (Speech, Child speech,
Conversation, Narration, Babbling, Speech synthesizer, Shout, Yell,
Children shouting, Screaming, Whispering, Hubbub): or, when
`include_ambiguous=True`, also over the 4 AMBIGUOUS classes
(Child singing, Chatter, Crowd, Children playing).

The score is **not** a "probability of speech": it's the model's raw
softmax output for the speech-family class indices, taken as a max-of-12
(or max-of-16) per frame. We verified all 16 indices against the canonical
`yamnet_class_map.csv` (REDACTION-INTEGRATION-NOTES.md §2: all 16 match
exactly).

### 3c. The RedactionGate mechanism: where the decision actually lives

YAMNet's per-frame score is **not** the redaction decision. The decision is
made by `RedactionGate` (`redaction/redaction_gate.py:9-95`), which applies
**hysteresis** plus padding to convert per-frame scores into contiguous
redaction windows. Three parameters matter:

- **`enter_threshold=0.25`** (default): a frame must reach this score to
  *start* a speech segment.
- **`exit_threshold=0.15`** (default): once a segment is active, a frame
  must drop below this (for longer than `hangover_seconds`) to *end* the
  segment. The `enter > exit` gap is the hysteresis: it takes a strong
  signal to start redacting, and redaction persists through dips until the
  signal has been weak for ~0.75 s.
- **`pre_roll_seconds=1.5`, `hangover_seconds=0.75`, `post_roll_seconds=0.75`**
 : pad each detected segment in both directions. Pre/post-roll is
  conservative: better to over-redact 1.5 s of audio around a real speech
  burst than to clip the burst's leading/trailing edges and leak a
  syllable.

The hysteresis is the **recall bias** of the system. Under-redacting is
catastrophic (a leaked syllable is a privacy violation), so the gate
deliberately prefers a 0.10 gap between enter and exit plus 1.5 s of
pre-roll plus 0.75 s of hangover. This trades precision (more non-speech
audio gets zeroed) for recall (less speech leaks through) on purpose.

The redaction windows are in seconds (YAMNet's wall-clock framing),
converted back to sample indices at the *original* samplerate (48 kHz or
whatever the mic was set to) for the in-place zeroing in `apply.py:80-84`:
`speech_scores` internally resamples 48 k → 16 k for YAMNet, then `apply.py`
indices back at 48 k. The windows themselves are always in wall-clock time,
so resampler choice doesn't shift where redaction lands, only how well
YAMNet classifies the audio it sees.

---

## 4. Mechanisms that interact with the gate but aren't "confidence"

- **In-place mutation contract.** `redact_speech` returns the *same* array
  object it was passed, with windows zeroed in place (apply.py:80-84). No
  copy on the normal path; no `_replace` on the `AudioSample`; `sample.save()`
  reads the same backing buffer that `.fill(0.0)` / the per-window zeroing
  touched. We verified earlier that pywaggle's `AudioSample` is a plain
  `NamedTuple(data: np.ndarray, samplerate: int, timestamp=...)`, so the
  mutation-through-reference is airtight. **If pywaggle ever switched
  `AudioSample.data` to a `@property` returning a copy**, `.fill` would zero
  only the copy and `sample.save()` would still see raw audio: that would be
  a silent leak. Documented in REDACTION-INTEGRATION-NOTES.md §4 Step 4 as a
  future-watching concern; switched at that point to
  `sample._replace(data=...)` rebinds in both the normal and except paths.
- **`--save-match` upload.** A *separate* mechanism from redaction. By the
  time `--save-match` evaluates an upload decision, the audio has already
  been redacted in the mic path: so even if a clip is uploaded because it
  contains a Northern Cardinal detection at confidence 0.71, the bytes
  leaving the node have speech windows zeroed. Redaction runs *before*
  persistence; the upload decision runs after inference; they don't
  interact, but the gate guarantees the upload safety independent of what
  `--save-match` decides.
- **Soundscape-ecology category routing.** Detections are published under
  `biophony` / `anthrophony` / `geophony` topics per their BirdNET class
  (app.py:296-300). This is a taxonomic routing decision, not a confidence
  decision: but it's the *visible* layer of the system to downstream
  consumers, so a leaked speech segment would surface as an
  `anthrophony.*` detection if BirdNET misclassified it. The redaction gate
  prevents that from being a visible failure mode even when BirdNET would
  have flagged the speech.
- **Tuning harness.** `redaction/scripts/tune_thresholds.py` is the tool
  for an actual operating-point decision. It treats the *gate* as the
  binary classifier (clip is "flagged" if RedactionGate returns any
  non-empty redaction window over the clip's per-frame scores), not the raw
  frame score: so hysteresis, pre/post-roll, and hangover all factor into
  the per-clip decision the harness measures. Recall = speech clips
  flagged / total speech; FPR = no-speech clips flagged / total no-speech.
  Currently verified-correct-but-not-informative on a 3-clip clean test
  set (saturates recall=1.000 / FPR=0.000 at every threshold in [0.05,
  0.50]); needs a richer labeled corpus to actually discriminate
  thresholds.

---

## 5. Model path resolution (why reboots don't break the gate)

The YAMNet `.tflite` model is loaded by `redaction/yamnet_speech.py:31-58`.
The path is resolved by an **existence-filtered fallback chain** (not a
Python `or` chain, which would return the first truthy string regardless of
file existence):

1. `BIRDNET_YAMNET_TFLITE` env override (highest precedence)
2. `/app/models/yamnet.tflite`: plugin container (Dockerfile COPY)
3. `/home/mighdz/AI-Projects/models/yamnet.tflite`: persistent dev
   (survives reboots)
4. `/tmp/yamnet.tflite`: volatile dev scratch (last resort)

If every path misses, `_load_model` raises `FileNotFoundError` listing every
path the chain checked; `speech_scores` catches that and raises
`RedactionFailure`; `redact_speech` catches *that* and zeroes the entire
buffer (the fail-closed path). So a missing model is *loud on the metrics*
(WARNING log with the reason) but *silent on the audio* (silence-only FLAC).

This is why **a reboot-fragile path resolution is a privacy concern in
disguise**: if the chain resolves to a non-existent path, every cycle
fail-closes silent: the node appears scientifically dead (no BirdNET
detections) with no indication the redaction gate is the cause. The
persistent dev path at `~/AI-Projects/models/` (literally absolute, not
`~`-expanded: `~` is unreliable on Sage plugin containers and dev
sandboxes where HOME is not `/home/mighdz`) eliminates that fragility on
a Thor. See REDACTION-INTEGRATION-NOTES.md §3 "Reboot persistence of the
.tflite" for the recommendation to export `BIRDNET_YAMNET_TFLITE` on dev
machines where the model lives elsewhere.

---

## 6. Still uncalibrated / open

The 0.25 / 0.15 gate defaults are from the notes-ref design, biased hard
toward recall: and `tune_thresholds.py` is the intended tool for an actual
empirical operating point once a richer labeled corpus exists (borderline
distant/mumbled speech, ambient that trips speech classes: exactly what
the 3-clip test-wavs set does NOT have). Until then the defaults are a
*design choice*, not a measured operating point; the mic-path redaction is
shipping on defaults because under-redacting is the catastrophic-direction
failure and the defaults err on the safe side.

Three concrete TODOs that fall out of this overview, if a deeper
confidence story becomes wanted:

1. **A richer `redaction/test-wavs/` set.** Borderline cases at the
   0.20-0.40 score band that would let `tune_thresholds.py` actually
   discriminate thresholds. The current 3-clip clean set saturates at
   every threshold in [0.05, 0.50] and tells us nothing about where the
   gate sits on the recall/FPR curve.
2. **A redaction-event metric publish** (REDACTION-INTEGRATION-NOTES.md §5,
   originally deferred). If gate failures go silent on a deployed node, you
   can't tell a quiet forest from a broken YAMNet. Publishing
   `redaction.event` per cycle with the windows + reason makes the gate
   observable. Named as load-bearing (Q6) in `CAMERA-PATH-DESIGN.md`
   before camera-path ship.
3. **Anti-aliased resampling** in `yamnet_speech._prepare_waveform`.
   Currently linear interpolation 48 k → 16 k with no anti-alias filter;
   for a privacy-critical gate, high-frequency speech content above 8 kHz
   could alias into the YAMNet input band and potentially trip the model
   less. One-line swap to `scipy.signal.resample_poly` (scipy already a
   birdnet dep) is the noted follow-up.

---

## See also

- `REDACTION-INTEGRATION-NOTES.md`: Step-by-step integration record and
  design analysis (the "what we built and verified" doc).
- `redaction/CAMERA-PATH-DESIGN.md`: proposed extension of the same gate
  to the camera-audio path + Pete's audio/video timeline use case.
  PROPOSED: NOT BUILT.
- `redaction/apply.py`: `redact_speech` and the in-place zeroing contract.
- `redaction/yamnet_speech.py`: LiteRT YAMNet front-end and the
  existence-filtered model-path fallback chain.
- `redaction/redaction_gate.py`: the hysteresis gate (enter/exit/hangover
  with pre/post-roll), imported unchanged from notes-ref.
- `redaction/speech_classes.py`: the verified CORE_SPEECH and AMBIGUOUS
  YAMNet class indices.
- `redaction/scripts/run_redaction_on_capture.py`: demo script with
  `--write-redacted PATH` before/after artifact output.
- `redaction/scripts/tune_thresholds.py`: enter_threshold sweep harness
  over labeled clips.
- `app.py:164-222`: the integration site in `record_from_microphone`.
- `tests/test_redaction.py`: the 14-pass unit suite (monkeypatches
  `speech_scores`, so it verifies the gate and fail-closed contract, NOT
  the LiteRT model-load path).
