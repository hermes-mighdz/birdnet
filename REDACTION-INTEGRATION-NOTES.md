# Microphone-path redaction integration — scratch analysis

Scratch analysis for inserting a speech-redaction step into the microphone
path of `app.py`, so the unredacted PCM array never touches disk. This file
records what I verified on the actual hardware/contents in this session; the
integration itself is NOT yet applied to `app.py` (per instruction: plan only,
await user go-ahead).

Repo: https://github.com/<fork>/birdnet (the hermes-mighdz fork).
Context lives in `~/AI-Projects/notes-ref/` (design doc, VAD-research notes,
and the working/tested redaction modules). This scratch file is committed to
the birdnet repo, not to notes-ref.

Hardware verified this session: Jetson AGX Thor, `aarch64`, kernel
`6.8.12-tegra-aarch64`.

---

## 1. The exact microphone path and where to insert

### What `record_from_microphone()` does today (`app.py:160-178`)

```python
def record_from_microphone(duration_s: float, sample_rate: int = 48000) -> str:
    from waggle.data.audio import Microphone
    mic = Microphone(samplerate=sample_rate)
    sample = mic.record(duration_s)          # <-- line 170: the array is in memory HERE
    tmpdir = tempfile.mkdtemp(prefix="birdnet_")
    flac_path = os.path.join(tmpdir, "recording.flac")
    sample.save(flac_path)                   # <-- line 176: array -> temp FLAC on disk
    return flac_path                         # only the path is returned
```

`sample` is a pywaggle `AudioSample` (a NamedTuple). Confirmed earlier from
`/home/mighdz/work/.venv/lib/python3.12/site-packages/waggle/data/audio.py:12-15`:
`AudioSample.data` is `np.ndarray` (1-D float32 mono, channels=1 default) and
`AudioSample.samplerate` is the int rate. `sample.save()` is an *optional*
separate step that just calls `soundfile.write`. The unredacted array is in
Python memory at line 170, before any disk write.

### How the caller consumes it (`_get_audio` and `run_cycle`)

```python
# app.py:577-584
def _get_audio(args) -> tuple[str, bool]:
    if args.input:   return args.input, False
    elif args.camera: return record_from_camera(...), True
    else:            return record_from_microphone(...), True   # <-- line 584

# app.py:676-695 (run_cycle)
audio_path, cleanup = _get_audio(args)
...
detections = classifier.classify_file(audio_path)   # <-- line 693/695: BirdNET reads the path
```

`classify_file` at `app.py:122-156` takes a *file path* and calls
`self.model.predict(audio_path, ...)`. The whole pipeline downstream of
`record_from_microphone` is file-path-shaped.

### Where to insert the redaction (decision)

Two viable insertion sites, drawn tightly:

**Option A — inside `record_from_microphone()`, before `sample.save()`**
  Insertion point: between `app.py:170` (`sample = mic.record(duration_s)`)
  and `app.py:176` (`sample.save(flac_path)`).
  Array variable at that point: `sample.data` (1-D float32, mono),
  with `sample.samplerate` available (48000 by default) and
  `sample.timestamp` available for the redaction-event log.
  What happens: `sample.data` is fed through YAMNet → RedactionGate →
  zero-out speech windows in place, THEN `sample.save(flac_path)` writes the
  already-redacted array. The temp FLAC on disk contains only silence at
  speech windows. Simplest change; everything downstream of `record_from_microphone`
  stays file-path-shaped and unchanged. Classify proceeds with `classify_file`
  exactly as today.

**Option B — array-everywhere: skip the temp FLAC entirely**
  Replace `record_from_microphone` to return the array + samplerate, add a
  `classify_array` method on `BirdNETClassifier` (already verified earlier this
  session: `model.predict_arrays((arr, sr), ...)` exists on the birdnet model
  and accepts the same kwargs as `model.predict`), and call it from `run_cycle`
  when the source is the mic. No temp file is ever created for the mic path.
  More invasive; touches `_get_audio`, `run_cycle`, and `BirdNETClassifier`,
  and changes the `_get_audio` return contract. But it eliminates the temp
  file (and the small window where the redacted FLAC is still momentarily on
  disk before the `finally`'s `shutil.rmtree`).

**Recommendation:** Option A is the right first step. The redaction gate is
the security-critical boundary; building it *before* persistence in
`record_from_microphone` keeps the rest of the pipeline untouched and lets
this be reviewed as a single-function change. Option B is a clean follow-up
once the redaction logic is validated end-to-end, and it also gains the
CPU/IO savings of skipping a temp-file round trip.

In both options, the file path returned is then consumed by `classify_file`
(app.py:122) → `model.predict(audio_path, ...)` (app.py:124). Nothing inside
the classifier needs to change for Option A.

---

## 2. Can YAMNet run on this Thor? — verified end-to-end YES

The notes' `yamnet_speech.py` uses `tensorflow_hub`, which is NOT available on
this host and is not a `birdnet` dependency. Loading YAMNet from TF Hub would
also require network at runtime (a non-starter on a deployed node). The right
path on aarch64 is the **TFLite YAMNet** model run via `ai_edge_litert`
(Google's LiteRT, the blessed TFLite runtime), which is *already* a transitive
dependency of `birdnet>=0.2.16` (the plugin's `requirements.txt` pins it).

I verified the full chain on this Thor this session:

### Runtime deps present (or pulled transitively by birdnet)

- `ai_edge_litert` 2.1.6 is already installed in the work venv (birdnet pulls it).
  System Python has nothing (no tflite_runtime, no tensorflow, no tensorflow_hub),
  but the **plugin container** (Dockerfile:14-15 `pip install -r requirements.txt`)
  installs birdnet and therefore also `ai-edge-litert` and `tensorflow`.
- `tensorflow` 2.21.0 also imports on aarch64 (birdnet pulls it). LiteRT
  subclasses/uses it; for *inference only* with a pre-converted `.tflite`, full
  TF is NOT needed at runtime — only `ai_edge_litert`.

### Model acquisition (verified working, no auth)

- Canonical Kaggle slug: `google/yamnet/tensorFlow2/yamnet` (the TF Hub
  `google/yamnet/1` model is hosted there now). Downloaded via
  `kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")` —
  `kagglehub` is *already* a `birdnet` dependency. No Kaggle auth needed for
  public models.
- Comes down as a TF2 SavedModel (~18 MB: saved_model.pb + variables). NOT a
  `.tflite` out of the box.

### Conversion (done once, at build time)

`tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR).convert()` produced a
15,034,264-byte (15 MB) `yamnet.tflite`. This conversion needs full TF — but
only at image-build time, not at runtime. Mirror it into the image the same
way the BirdNET models are pre-downloaded in `Dockerfile:18-22`.

I/O signature, captured by introspecting the SavedModel (`m.signatures`):
- Input: `waveform`, shape `[None]`, `float32` (1-D mono PCM in `[-1, 1]`)
- Outputs:
  - `output_0` scores,  shape `(N, 521)`  ← this is what `speech_classes.speech_score` consumes
  - `output_1` embeddings, shape `(N, 1024)`
  - `output_2` log-mel spectrogram, shape `(spectrogram_frames, 64)`
- `N` (frame count) is determined by YAMNet's internal framing: 0.96 s window,
  0.48 s hop. 3 s of audio → 6 frames — matches `RedactionGate` defaults
  (`frame_hop=0.48, frame_duration=0.96`).

### Runtime inference on aarch64 via LiteRT (done this session)

```
from ai_edge_litert.interpreter import Interpreter
interp = Interpreter(model_path="yamnet.tflite")
# YAMNet's TFLite input dim is frozen to [1] by the converter (variable length);
# use resize_tensor_input() per waveform length, e.g. 16000*3 for 3 s.
interp.resize_tensor_input(input_index, [n_samples])
interp.allocate_tensors(); interp.set_tensor(input_index, wav); interp.invoke()
```

- 3 s of 16 kHz silence → 6 × (521) score rows, correct frame count.
- Top scoring class on silence: idx 494 "Silence" at 1.0; idx 0 "Speech" at 0.0 —
  sanity check passes.
- XNNPACK CPU delegate created automatically. Runs on the ARM cores, no GPU.
- Inference time was sub-second for 3 s of audio in my probe; on a deployed
  cycle (default `--duration 15`), the YAMNet pass runs once over ~15 s of audio
  → frame budget ~30 frames; cheap.

### Class index verification (per 04-audio-redaction.md:127-128)

The notes' `speech_classes.py` had CORE_SPEECH indices (0,1,2,3,4,5,6,9,10,11,12,65)
and AMBIGUOUS (29,63,64,66). I pulled the canonical `yamnet_class_map.csv` from
`https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv`
and verified every index against it:

  CORE_SPEECH (12 correct):
    0  Speech                              65  Hubbub, speech noise, speech babble
    1  Child speech, kid speaking          9   Yell
    2  Conversation                        10  Children shouting
    3  Narration, monologue                11  Screaming
    4  Babbling                             12  Whispering
    5  Speech synthesizer                   6   Shout

  AMBIGUOUS (4 correct):
    29  Child singing    63  Chatter    64  Crowd    66  Children playing

All 16 indices match the canonical CSV exactly. The notes' `speech_classes.py`
is correct — these are the speech-family class indices to aggregate via max.

---

## 3. One adaptation needed: from tfhub SavedModel to LiteRT .tflite

The notes' `yamnet_speech.py` is written against TF Hub (`hub.load(YAMNET_HUB_URL)`
returns a callable that accepts a numpy waveform and returns a 3-tuple of
(score, embeddings, spectrogram) Tensors). The LiteRT interface is different:
create an `Interpreter`, `resize_tensor_input` per waveform length,
`set_tensor`, `invoke`, `get_tensor` by output index.

The change to `yamnet_speech.py` is mechanical and contained to `_load_model`
and the `model(...)` call inside `speech_scores`. The public API
`speech_scores(audio_1d, samplerate, include_ambiguous=...)` can stay identical,
so `redaction_gate.py`, `speech_classes.py`, and all three test files are
unaffected. Concretely:

- Replace `tensorflow_hub` import with `ai_edge_litert.interpreter.Interpreter`.
- Resolve and load `yamnet.tflite` from a path resolved at startup (env var or
  plugin data dir), NOT via `hub.load` at runtime.
- Inside `speech_scores`:
    1. `interp.resize_tensor_input(in_idx, [len(waveform)])`
    2. `interp.allocate_tensors()`
    3. `interp.set_tensor(in_idx, waveform)`
    4. `interp.invoke()`
    5. Loop over output tensors, find the one whose `shape[-1] == 521` (that's
       `scores`); the converter-mangled names are unreliable (`StatefulPartitionedCall:0`
       in my probe), so select by shape, not by name.
    6. Per frame, call `speech_score(frame, include_ambiguous=...)` exactly as today.

This is the only code change forced by the platform difference (tfhub on a
dev laptop vs LiteRT on aarch64). The test suite
(`test_yamnet_speech.py`) uses a `FakeYamnet` stub, so it does NOT exercise
the LiteRT path — recommend adding one test that loads the real `.tflite`
and asserts the output shape `(N, 521)` on a short zero waveform, skipped
if the model file is absent.

---

## 4. Proposed integration plan (not yet applied)

Goal: when the audio source is the microphone (the camera path is out of scope
per instruction), run speech redaction on the in-memory `AudioSample.data`
array BEFORE persistence, so the raw array never hits disk.

### Step 1 — vendor the redaction modules into the birdnet repo

Copy from `~/AI-Projects/notes-ref/code/redaction/` into a package in this
repo, e.g. `redaction/`:

  redaction/__init__.py
  redaction/redaction_gate.py     (unchanged — pure Python, well-tested)
  redaction/speech_classes.py     (unchanged — verified index-correct)
  redaction/yamnet_speech.py      (adapted: tfhub → LiteRT — §3 above)
  tests/test_redaction_gate.py    (unchanged — pure-Python, runs anywhere)
  tests/test_speech_classes.py    (unchanged)
  tests/test_yamnet_speech.py     (unchanged — uses FakeYamnet stub)
  tests/test_yamnet_litet_live.py (NEW — live .tflite smoke test, skipped if file missing)

Vendoring into the birdnet repo keeps the build self-contained and lets CI run
the pure-Python tests in any environment.

### Step 2 — adapt `yamnet_speech.py` to LiteRT (per §3)

Single-file mechanical change to `_load_model` and the `speech_scores` body.
Public signature preserved.

### Step 3 — bake the .tflite model into the Docker image

The birdnet Dockerfile pre-downloads BirdNET models at build time
(`Dockerfile:18-22`). Add a parallel step for YAMNet:

  RUN python3 -c "import kagglehub; kagglehub.model_download('google/yamnet/tensorFlow2/yamnet')"
  RUN python3 -c "<TF SavedModel -> .tflite conversion; copy to /app/models/yamnet.tflite>"

OR (cleaner, smaller image, no TF at build-time-after-install either):
pre-convert the .tflite locally, commit the 15 MB `models/yamnet.tflite` into
the repo, and COPY it in `Dockerfile`. The 15 MB binary is small enough to
vendor; it avoids a build-time TF dependency for the YAMNet step and survives
kagglehub outages at build time.

YAMNet class map: copy `yamnet_class_map.csv` (14 KB, canonical source verified
this session) into `models/` too, for verification/debugging — though the
indices are baked into `speech_classes.py` and the CSV isn't needed at runtime.

### Step 4 — wire the redaction gate into `record_from_microphone`

Add a small helper and call it in the microphone path only. Pseudocode for the
Option-A change at `app.py:170-176`:

  sample = mic.record(duration_s)
  audio = sample.data                            # 1-D float32 mono, 48 kHz
  sr   = int(sample.samplerate)                 # 48000
  try:
      redacted, events = redact_speech(audio, sr)   # in-place zeroing of speech windows
  except RedactionGateFailure:
      # fail closed: redact the ENTIRE buffer if YAMNet produced no scores
      redacted, events = np.zeros_like(audio), [(0.0, duration_s)]
  # publish events as a measurement here (timestamp = sample.timestamp),
  # OR pass events up to run_cycle and publish alongside detections.
  # Replace the array in the AudioSample (NamedTuple is immutable → _replace):
  sample = sample._replace(data=redacted)
  tmpdir = tempfile.mkdtemp(prefix="birdnet_")
  flac_path = os.path.join(tmpdir, "recording.flac")
  sample.save(flac_path)                        # only the redacted array is written
  return flac_path

Where `redact_speech` is (sketch, in a new `redaction/apply.py` or in app.py):

  def redact_speech(audio_1d, sr, gate=None):
      gate = gate or RedactionGate(enter_threshold=0.25, exit_threshold=0.15,
                                   pre_roll_seconds=1.5, hangover_seconds=0.75,
                                   post_roll_seconds=0.75)   # notes' defaults
      scores = yamnet_speech.speech_scores(audio_1d, sr)     # downmixes+resamples to 16k internally
      windows = gate.get_redaction_windows(scores)           # raises RedactionGateFailure on empty
      out = audio_1d.copy()
      for (start_s, end_s) in windows:
          i0 = int(start_s * sr); i1 = int(end_s * sr)
          out[i0:i1] = 0.0      # silence the speech window
      return out, windows

Key invariants:
- YAMNet resamples to 16 kHz internally (the notes' `_prepare_waveform` linearly
  interpolates 48 kHz → 16 kHz; for first-pass this is fine — a follow-up can swap
  to `scipy.signal.resample_poly` for properly anti-aliased downsampling).
- `RedactionGate` selected on YAMNet's default per-frame 0.96 s window / 0.48 s
  hop. Its `fail_closed=True` default means no-scores → raise, and the caller
  (above) handles it by redacting the entire buffer. That is the correct
  privacy posture: if the speech gate cannot run, assume ALL of it is speech.
- The redacted array is what `sample.save()` writes. The unredacted 48 kHz
  array is overwritten by the `out = audio_1d.copy()` then zeroing loop; the
  *original* reference `audio_1d` is still referenced by `sample.data` until
  the `_replace` — to be airtight, do `out = audio_1d.copy()` and pass `out`
  forward, but also `del audio` / rebind so the unredacted buffer is freed
  sooner.

### Step 5 — redaction event measurement (small, optional-but-recommended)

The notes propose publishing each redaction as its own data product. Add to
`run_cycle` after redaction: `plugin.publish("redaction.event", json.dumps({
"timestamp": ..., "duration_s": ..., "windows": [...] }))`. This gives an
auditable log of when speech was suppressed, and free human-presence stats.
Out of scope for the first patch — can land separately.

### Step 6 — tests

- Re-run the pure-Python redaction tests in CI (no TF/model deps needed).
- Add a mic-path integration test (mock the pywaggle Microphone to return a
  fixed `AudioSample` containing a synthetic "speech-like" YAMNet-positive
  segment — e.g. band-limited noise in the 300–3000 Hz vocal band — and assert
  that the saved FLAC has a zeroed segment with the expected time bounds).
- The `notes-ref` test files (redaction_gate, speech_classes, yamnet_speech
  with FakeYamnet) are portable as-is.

### Step 7 — make the redaction step a runtime opt-in

Ship redaction on for the Haleakala deployment node (H032) but leave a CLI
flag (`--no-redact` for testing / `--redact-threshold` to tune). Default ON
matches the design principle in 04-audio-redaction.md:67-70 — "the safe
default state of the system is redacting." A dev machine without the YAMNet
.tflite available should fail loudly if redaction is requested and the model
is missing (don't silently disable).

---

## 5. Open issues to resolve before applying

- **Camera path is explicitly out of scope here** (awaiting user confirmation
  the RLC-81MA exposes audio over RTSP, per 04-audio-redaction.md:206-213).
  The plan above is microphone-only. If/when the camera path is opened, the
  same redaction logic applies to the array you'd get from an ffmpeg-stdout
  pipe — but that's a separate change to `record_from_camera`.
- **Capture-boundary leakage** (04-audio-redaction.md:180-185): each cycle is a
  fixed-length isolated block. Speech crossing a cycle boundary gets less
  padding than speech mid-buffer. First patch accepts this limitation;
  follow-up can carry gate state across cycles or unconditionally redact the
  leading/trailing N seconds when speech is near an edge.
- **Resample quality**: the notes' linear-interpolation downsampler
  (`yamnet_speech._prepare_waveform`) has no anti-aliasing filter. For a
  privacy-critical gate this could let high-frequency speech content alias
  into the 0–8 kHz band and trip the model less. First patch keeps the
  notes' code as-is; a one-line swap to `scipy.signal.resample_poly(48k→16k)`
  (scipy already a birdnet dep) is a cheap follow-up.
- **No streaming / chunking**: 15 s capture → 30 frames YAMNet inference on
  the whole 15 s array. Plenty fast on Thor (my 3 s probe was sub-second), but
  if the cycle `--duration` is ever raised to minutes, prefer a streaming
  YAMNet run.

---

## 6. Hardware/deps summary for the plan

- aarch64 Jetson AGX Thor, kernel 6.8.12-tegra.
- `ai_edge_litert` 2.1.6 ✓ (birdnet transitive dep; in the container via requirements.txt).
- `tensorflow` 2.21.0 ✓ (birdnet transitive dep; needed only for the one-time
  SavedModel→.tflite conversion at build time, NOT at inference time).
- `kagglehub` ✓ (birdnet dep; fetches YAMNet SavedModel from Kaggle at build).
- YAMNet `.tflite` end-to-end+LiteRT inference: VERIFIED on this Thor this session.
- `pywaggle[audio]>=0.56` ✓ (already in requirements.txt; supplies the
  <->`Microphone` / `AudioSample` API used at the insertion point).
- BirdNET `model.predict_arrays` exists (verified previously in this repo's
  session log) if we later want Option B (array→BirdNET, no temp file).

No new heavyweight runtime dependency is added to the plugin container —
LiteRT and TF are already pulled in by `birdnet>=0.2.16`. The only new artifact
is the ~15 MB `yamnet.tflite` binary, which I propose vendoring in-repo (same
pattern as the existing model/ls in the Dockerfile build step).

---

This is a plan and verification record; `app.py` is unchanged. When you confirm
the camera's audio path, the above applies unchanged to the microphone side,
and a parallel change for `record_from_camera` (ffmpeg stdout pipe → array →
same `redact_speech` → BirdNET) can be designed from this template.
