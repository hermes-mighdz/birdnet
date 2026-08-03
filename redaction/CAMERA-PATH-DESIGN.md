# Camera-path redaction — design proposal

**Status: PROPOSED — NOT BUILT.** No code changes have been made to
`record_from_camera`. This document exists for review and discussion. It
proposes extending the speech-redaction gate (currently landed on the
microphone path at `app.py:164-222`, commit `cbcb2fa`) to the network-camera
audio path, AND introducing a parallel audio/video timeline capability Pete
raised that the current camera path doesn't support at all.

Cross-reference: `REDACTION-INTEGRATION-NOTES.md` §1, §5 ("Camera path is
explicitly out of scope"); this document is the follow-up promised there.

---

## 1. Why the current ffmpeg-to-disk flow can't redact before persistence

Today's `record_from_camera` (app.py:225-281) is functionally:

```python
def record_from_camera(url, duration_s, sample_rate=48000) -> str:
    tmpdir = tempfile.mkdtemp(prefix="birdnet_")
    flac_path = os.path.join(tmpdir, "camera_audio.flac")
    input_args = (["-f","mxg"] if "MxPEG" in url
                  else ["-rtsp_transport","tcp"] if url.startswith("rtsp://")
                  else [])
    cmd = (["ffmpeg","-y"] + input_args
           + ["-i", url, "-vn", "-acodec","flac",
              "-ar", str(sample_rate), "-ac","1", "-t", str(duration_s),
              flac_path])
    result = subprocess.run(cmd, capture_output=True, timeout=int(duration_s)+30)
    ...
    return flac_path
```

`ffmpeg` is invoked with a **file output** (`flac_path`) — not a pipe, not a
stdout consumer. `subprocess.run(..., capture_output=True)` blocks until ffmpeg
exits. By the time the function returns, **the unredacted FLAC is on disk**.
There is no point in this flow at which Python holds the raw PCM as an
in-memory array, so there is no point at which the redaction gate could run
*before* the bytes are written. The mic path is different precisely because
`Microphone().record()` returns an `AudioSample.data` numpy array (already in
Python memory, not yet persisted) and `sample.save()` is a separate explicit
step — `redact_speech` slots between them (app.py:182-214).

Three reasons the current flow is structurally unsafe for a privacy-critical
deployment:

1. **No redaction insertion point.** The function returns a path, not an
   array. Anything wanting to redact after this point must read the FLAC back
   into memory, redact, and re-write — but the unredacted original has already
   been observed by the filesystem (and, on a deployed node, possibly by
   Beehive if an upload happened between ffmpeg's exit and the redaction). The
   mic path's invariant was "raw array never touches disk"; the camera path
   violates that invariant by construction.

2. **The temp file is the artifact.** The `finally` block higher up in
   `run_cycle` does `shutil.rmtree(tmpdir)` for cleanup, but there's a real
   window — the entire inference + publish duration — where the raw FLAC lives
   on disk. If the plugin crashes, gets OOM-killed, or the cleanup doesn't run,
   raw audio is left behind. The mic path's in-place zeroing guarantees the
   persisted bytes are *already redacted* before that window opens.

3. **`-vn` means there's no video sidecar to leak either**, but the proposal
   below introduces one — so this concern gets bigger, not smaller, with the
   audio/video timeline feature.

The fix must move the PCM into Python memory *before* any disk write, the same
way the mic path keeps it in memory. The next section describes how.

---

## 2. Proposed: ffmpeg stdout pipe → in-process PCM array

Replace the file-output ffmpeg invocation with a stdout pipe, demuxed into a
numpy array in Python, then run the same `redact_speech` gate the mic path
uses before any persistence.

### Pipeline shape

```
                       (no -o file; output to stdout)
ffmpeg ─y─i URL ─vn ─f f32le ─acodec pcm_f32le ─ar 48000 ─ac 1 ─t DUR pipe:1
                                                                        │
                                                                        ▼
                                       subprocess.Popen(stdout=PIPE)
                                                                        │
                                              read stdout in 4-byte chunks
                                                                        │
                                                                        ▼
                np.frombuffer(stdout_bytes, dtype=np.float32)  →  1-D array
                                                                        │
                                                                        ▼
                                   redact_speech(arr, 48000)   ──zeros speech──┘
                                                                        │
                                                                        ▼
                                       soundfile.write(flac_path, redacted, 48000)
```

Key ffmpeg changes vs. the current invocation:

- Drop `-acodec flac` and the trailing `flac_path`; replace with `-f f32le`
  (raw container) + `-acodec pcm_f32le` (32-bit little-endian float PCM) +
  `-` (stdout) — or `pipe:1`, same thing. `pcm_f32le` matches the dtype the
  mic path already uses (`AudioSample.data` is `np.float32`), so the array
  type that flows into `redact_speech` is identical.
- `-ar 48000 -ac 1 -t DUR` stay — they pin the sampling rate / channel count
  / duration, exactly as today.
- `-vn` stays for now (the time-windowing proposal in §3 introduces a parallel
  video stream, but the audio *redaction* path needs to remain audio-only so
  `redact_speech` input shape is unchanged).

### Why `Popen(stdout=PIPE)`, not `subprocess.run(capture_output=True)`

`subprocess.run` with `capture_output=True` buffers the entire stdout in
memory until the process exits — that's fine for a 15 s clip (≈960 KB at
48 kHz mono f32le), but it also blocks until ffmpeg exits. For a 15 s capture
that's ~15 s of unavoidable wall-clock (the camera is real-time streaming).
`subprocess.Popen + read()` lets us stream-accumulate the bytes if we ever
want to (a) start YAMNet scoring before the capture finishes, or (b) bound
peak memory on long captures — neither is needed for the first cut, but
Popen keeps the option open. For the first pass, `Popen.stdout.read()` in one
go is the simplest correct shape and is what's proposed here.

### Memory bound

`4 bytes/sample × 48000 samples/s × duration_s` = 192 KB/s of audio. A 15 s
default cycle = 2.88 MB; a 60 s long cycle = 11.5 MB. Comfortable on the Thor
(32 GB). The mic path holds the same-sized array for the same duration with
no issue, and `redact_speech` mutates it in place (no doubling), so the
camera path's memory profile matches the mic path's after this change.

### Where `redact_speech` slots in (same contract as the mic path)

```
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, ...)
audio_bytes = proc.stdout.read()                 # raw PCM bytes from ffmpeg
proc.wait(timeout=...)
if proc.returncode != 0 or len(audio_bytes) < MIN_BYTES:
    raise RuntimeError("ffmpeg camera capture failed: ...")
audio = np.frombuffer(audio_bytes, dtype=np.float32)

try:
    redaction_apply.redact_speech(audio, int(sample_rate))   # IN-PLACE; same call as app.py:183
    logger.info("Camera audio redacted: %d window(s) over %.2fs", len(_events), duration_s)
except (YAMNetRedactionFailure, RedactionGateFailure, Exception):
    audio.fill(0.0)                                          # same fail-closed posture
    logger.exception(...)
    # see REDACTION-INTEGRATION-NOTES.md §4 Step 4 for the three-arm rationale

# only now persist -- the array is already redacted (or force-zeroed)
soundfile.write(flac_path, audio, sample_rate, subtype="FLOAT")
return flac_path
```

The redaction call site is structurally identical to the mic path
(app.py:182-214). This is deliberate: one redaction contract, two acquisition
front-ends (pywaggle Microphone / ffmpeg pipe) that both produce a 1-D float32
array + samplerate.

### Re-export through `soundfile`, not pywaggle

The mic path uses `AudioSample.save(flac_path)` because pywaggle already has
the array wrapped. The camera pipe produces a bare numpy array, so it calls
`soundfile.write` directly (soundfile is already a dependency — both pywaggle
audio and BirdNET's `librosa` stack pull it in). Same FLAC container, same
lossless posture, same portal-inline behavior. No new dependency.

### What `redact_speech` does NOT change

- The gate's fail-closed posture, the three-arm `try/except` in the caller,
  the in-place mutation contract — all unchanged from the mic path's
  implementation (see the corrected sketch at
  `REDACTION-INTEGRATION-NOTES.md` §4 Step 4, commit `8d8de8f`). The camera
  proposal reuses that exact caller pattern; the only difference is how the
  raw array enters Python (ffmpeg pipe vs. pywaggle Microphone).
- The downstream pipeline (`classify_file`, `publish_detections`,
  `--save-match` upload) is still file-path-shaped and unchanged — the
  returned `flac_path` is consumed by `run_cycle` exactly as today.

---

## 3. The audio/video timeline question (Pete's windowing use case)

Pete's use case: *detect an audio event at wall-clock time t (a BirdNET
detection or a YAMNet-speech event), then snip a video clip around that time*
— e.g. ±N seconds around t, to publish a short video snippet showing what was
on camera when the sound happened.

### What the current path gives us (and doesn't)

The current camera capture uses `-vn` (no video) and produces audio only.
There is **no video retained at all**, and the only timestamp in the system is
the `int(time.time_ns())` taken at the top of `run_cycle` (app.py:722),
**before** `_get_audio` runs. That `timestamp` is the *cycle start*, not the
wall-clock offset of any particular sample within the captured audio — there
is no per-sample or per-frame timecode flowing through the pipeline. BirdNET
detections carry `start_time` / `end_time` relative to the *file* (zero-based,
seconds from the start of the FLAC), not wall-clock. So today there is no way
to answer "what wall-clock time did the bird at file-offset 4.2 s correspond
to," which is what the windowing use case needs.

### The timeline invariant a windowing feature needs

To snip video around an audio event, the system needs a **shared wall-clock
timebase** between:
- each audio sample (or at least the first sample of the capture), and
- each video frame.

Wall-clock here means a UTC instant, not a relative offset — Pete's "time t"
needs to be a nameable moment that both the audio array and the video frames
can be indexed against.

### Proposed shape (audio + video in one ffmpeg invocation)

One ffmpeg invocation captures audio AND writes a parallel video file, with a
synthesized wall-clock anchor:

```
ffmpeg -y -rtsp_transport tcp -i rtsp://... \
    -map 0:a:0 -acodec pcm_f32le -ar 48000 -ac 1 -t DUR -f f32le pipe:1 \
    -map 0:v:0 -vcodec copy -t DUR /tmp/<cycle>_video.<ext>
```

This is one ffmpeg with **two outputs from the same input**:
1. Audio to stdout as `pcm_f32le` (consumed by Python as before).
2. Video to a file (codec copied, no re-encode — preserves the camera's
   native H.264/MxPEG stream byte-for-byte).

`subprocess.Popen(stdout=PIPE)` reads only the audio pipe; the video file is
written by ffmpeg directly. ffmpeg muxes the two outputs concurrently from the
same demuxed input, so they share a single wall-clock capture window.

### Establishing the wall-clock anchor

ffmpeg itself doesn't emit a wall-clock anchor usable by Python unless:
- the camera's RTSP/HTTP stream carries an embedded timecode (Reolink RTSP
  has RTCP SR packets with NTP timestamps; MxPEG has Mobotix's own frame
  metadata), **or**
- we use ffmpeg's `-use_wallclock_as_timestamps 1` to stamp each packet with
  the local system clock at demux time, **or**
- we anchor to `time.time_ns()` at `Popen` start and accept that the first
  audio sample and the first video frame both correspond to "shortly after
  Popen started."

The third is the naive but always-available anchor. The first two depend on
camera-firmware specifics and need verification per camera model — tracked as
open question §4-Q2.

### Indexing video frames against the audio timeline

A BirdNET or YAMNet event arrives with `start_time` and `end_time` as
file-relative seconds (0 to `duration_s`). To snip video around that event at
file-offset `t_event`:

1. Convert file-offset to wall-clock: `t_wall = cycle_start_wall + t_event`.
   (Assumes the first audio sample is at `cycle_start_wall` — naive anchor.)
2. Snip video around `t_wall`: extract frames from the video file in
   `[t_wall - N, t_wall + N]` via a second ffmpeg pass over the saved video
   file (`-ss`/`-to` for seek; `-c copy` to avoid re-encoding).

The video file lives alongside the audio file in the same temp dir. The
redaction gate runs only on the audio array — video is not speech-redacted
(see §3-Open Issues below).

### What redacts, what doesn't (this is a privacy-relevant design choice)

The current `redact_speech` gate zeros audio samples in *speech windows*. The
video file is an independent artifact. Two privacy postures are possible:

- **Audio-only redaction (proposed first phase).** Speech windows zero the
  audio; the video is left alone. Pete's windowing use case then snips
  video around the BirdNET *bird* detection, not around speech — speech is
  silenced in the audio file but the video frames at that same wall-clock
  remain. This means a human face visible in the video at the moment of the
  speech event is NOT redacted. That's an explicit posture to document and
  get sign-off on, not a default to slip in.
- **Audio+video redaction (proposed later phase, OUT OF SCOPE here).**
  Extends redaction to video frames inside speech windows (blur faces, drop
  frames, replace with black). Requires a different image-processing pipeline
  and a different privacy review. Not designed in this document.

The first-phase proposal redacts audio only and is explicit about that.
Visual privacy (faces, license plates, identifiable clothing) is a separate
privacy domain that the audio redaction gate does not address; it belongs in
a follow-up design.

---

## 4. Open questions

These need answers or empirical verification before this proposal is worth
implementing. None of them block the *audio* redaction path (§2 is
self-contained and unblocked), but the §3 timeline feature has several
unverified assumptions.

### Q1. Does the Reolink RLC-81MA expose audio over RTSP at all?

`REDACTION-INTEGRATION-NOTES.md` §5 raised this and it remains unverified. The
current `record_from_camera` is camera-agnostic (any ffmpeg URL works), but
the proposed pipe approach assumes there IS an audio stream in the RTSP URL
to extract via `-map 0:a:0`. If the RLC-81MA's RTSP URL is video-only, the
camera path produces no audio to redact — the privacy gate is moot for that
camera model and §3's audio-event windowing can't work either. **Need**: a
live RTSP URL + creds from the instructor, run `ffprobe -show_streams URL` on
the Thor, confirm an audio stream exists and its codec.

### Q2. What wall-clock anchor does each camera model actually give us?

Three candidate sources (§3):
- (a) Embedded camera-side timecode (RTCP SR NTP, MxPEG metadata). Per-camera
  and possibly absent; needs a per-model probe.
- (b) ffmpeg `-use_wallclock_as_timestamps 1` — system clock at demux. Always
  available but ~tens of ms drift from the camera's actual sensor capture.
- (c) `time.time_ns()` at Popen start in Python. Always available, naive,
  sub-second accuracy under no load but unbounded drift under CPU contention.

**Need**: decide the accuracy Pete's windowing actually requires. "Snip video
±2 s around the bird call" tolerates ±0.5 s of anchor slop easily; "lip-sync
speech to video for a forensic record" needs sub-frame alignment and is
beyond this proposal. Until that tolerance is specified, the naive (c) anchor
is the safe default.

### Q3. Does ffmpeg actually tear down cleanly when stdout is a pipe and -vn is
dropped (i.e. video copies while audio pipes)?

The single-ffmpeg-two-output pattern in §3 is standard and well-supported,
but specific RTSP server implementations (Mobotix MxPEG especially) have
quirks. **Need**: a smoke test on the actual production camera (whichever
model is chosen) — `ffmpeg -i URL -map 0:a:0 -f f32le - -map 0:v:0 -c copy
/tmp/v.mkv -t 5`, eyeball both outputs.

### Q4. What's the cycle-duration-vs-memory tradeoff for the video file?

A 15 s capture of `-c copy` H.264 from a 1080p Reolink is ~2-4 MB; trivial.
But the proposal in §3 keeps the video file in the temp dir for the whole
cycle so it's available for the windowing snip. If the snip never runs (no
detection passes `--save-match`), the full video file is written to disk and
deleted in the `finally` — wasted I/O. **Need**: decide whether to capture
video always-and-discard (simple, more I/O) or capture video only after an
audio event is detected (more complex — requires a second ffmpeg capture
pass on a transient event, which loses the wall-clock alignment the first
pass would have had). The first is operationally simpler; the second is
cheaper but breaks the timeline feature's main invariant. Current lean: capture always, discard if unused.

### Q5. What gets published to Beehive, and is that a privacy problem?

The mic-path integration (commit `cbcb2fa`) redacts audio before
`sample.save()`, so the only thing persisted is silence-zeroed FLAC. In the
camera path with §3's audio+video capture, the audio leaves Python
post-redaction (safe), but the **video file is written unredacted by ffmpeg
itself in parallel** — Python never touches it. If §3's first phase uploads
only audio (as the current `--save-match` does), video never leaves the node
and the privacy question is local-only (the temp dir is cleaned in `finally`).
If video clips ARE later uploaded (Pete's windowing publish step), then
identifiable imagery leaves the node without any visual-redaction gate. **Need**: a privacy review before any video-upload feature ships. This document
proposes audio-only upload; video clips staying local-and-deleted is the
default safe posture.

### Q6. Should the redaction failure posture differ for camera vs. mic?

Today both paths share the same fail-closed contract: redaction fails → zero
the entire buffer → save silence. For the mic path this is uncontroversial
(silence FLAC is a fine degraded artifact). For the camera path, an entire
silence FLAC means a missed BirdNET cycle — and BirdNET's purpose is
detection. Failing closed means "if we can't redact, we'd rather miss the bird
than risk a speech leak." That's the correct privacy posture, but it shifts
the operational consequence critically: a chronic YAMNet model failure on the
camera path = **silence on every cycle = no detections ever** = the node is
scientifically dead. **Need**: decide if a per-cycle redaction metric + alert
on sustained fail-closed is required before shipping this to a deployment
node. Probably yes — the redaction-event measurement (§5 of the integration
notes, originally deferred) becomes load-bearing.

### Q7. Should IIUC (multi-image ultrawide) / multichannel audio be passed to
`redact_speech`?

`redact_speech` accepts a 1-D array (`yamnet_speech._prepare_waveform` does
downmix if 2-D). Cameras with stereo audio would arrive as 2-D from a
`-ac 2` ffmpeg pipe. The current mic path is mono by construction
(`AudioSample.data` from pywaggle is mono), so the camera path is the first
place multichannel could surface. **Need**: decide whether to force `-ac 1`
in the ffmpeg pipe (simplest, drops stereo info that's useless for BirdNET
anyway) or pass 2-D through and let `_prepare_waveform` downmix (matches the
notes' design intent). Lean: `-ac 1` for now, revisit only if a use case for
stereo camera audio emerges.

---

## Scope boundary (what this proposal commits to vs. doesn't)

This proposal:
- Designs how to get camera audio into an in-process array before persistence (§2).
- Specifies the same `redact_speech` caller pattern the mic path uses (§2, reuse).
- Sketches an audio+video timeline capability for Pete's windowing use case (§3).
- Names the open questions that block or scope the implementation (§4).

This proposal does NOT:
- Touch `record_from_camera` in `app.py`. The function is unchanged.
- Introduce any video processing beyond `-c copy` in a parallel file output.
- Address visual redaction of video frames (open; §3-Open Issues; Q5).
- Commit to any particular wall-clock anchor source (Q2).
- Specify which camera model is targeted (Q1, Q3).

Reviewers: please resolve Q1 (camera exposes audio?) first — the entire
proposal is moot if the deployed camera's RTSP URL is video-only. Until then
this is a design document, not a work item.
