# Speech Redaction at the Edge

**Miguel Hernandez**, Northwestern University
Sage Grande: Summer of AI 2026

> Keep a Sage node listening for birds without ever recording the people
> walking past it.

## The Problem

Sage is deploying a node at Haleakalā National Park in Hawaii. The park runs
a BirdNET acoustic pipeline to identify birds from their calls, which means
the node has a microphone that is always listening. The National Park Service
does not want park visitors recorded, and their default request was simple:
turn the microphone off.

Turning the microphone off also turns off the bird science. This project
offers an alternative: automatically detect and erase human speech at the
edge, so the node can continue classifying birdsong while preventing detected human speech from being written to disk or uploaded.

The privacy requirement is strict, so the system is designed to fail closed. The system is designed to fail closed: if speech detection cannot complete successfully, the audio is redacted rather than written to disk.

## The Approach

The core insight came from reading how the existing BirdNET node application
actually handles audio: **it writes every recording to disk before it
classifies anything.** A speech filter added at the end of the pipeline would
be too late, because the raw audio, human speech included, is already saved
by the time classification runs.

So redaction has to happen before the audio is persisted, while it is still
an in-memory array. The pipeline is:

```
microphone capture (in-memory audio array)
   -> YAMNet speech scoring (one score per 0.48s frame)
   -> RedactionGate (turns scores into speech time-windows)
   -> zero out the speech windows in the array, in place
   -> BirdNET classification and publish
   -> publish redaction event (timestamp, duration)
   (the raw, un-redacted audio never touches disk)
```

**Detecting speech.** BirdNET's own human-sound classes do not reliably catch
real speech, so this project uses **YAMNet**, a general audio classifier, for
the speech-detection step. YAMNet produces a speech score for every short
frame of audio. These scores are the input to the decision logic.

**Deciding what to erase.** A `RedactionGate` turns those per-frame scores
into the actual time-windows to erase. It uses hysteresis: a frame has to
clear a higher threshold to *start* a speech window, and drop below a lower
threshold to *end* one, so the gate does not flicker on and off in the middle
of a sentence. Each detected window is padded on both sides (pre-roll and
post-roll) so the edges of an utterance are never clipped and leaked.

**Failing safe.** The safe default state of the system is *redacting*.
Successful speech classification is what permits recording, not what triggers
redaction. The gate raises on missing scores rather than silently redacting
nothing, and in the integrated microphone path, if YAMNet errors, returns
nothing, or the gate hits any unexpected failure, the system zeroes the
entire audio buffer before saving rather than risk letting speech through.

## My Contribution

- **Found the architectural constraint** that redaction must happen before
  persistence, by tracing both audio paths in the BirdNET application and
  finding they write to disk before classifying.
- **Built and tested the redaction components**: a YAMNet speech-scoring
  wrapper, a verified list of YAMNet's speech-related classes (checked
  against the canonical class map rather than assumed), and the
  `RedactionGate` hysteresis state machine. Caught and fixed a frame-timing
  bug in the gate that was under-redacting the tail of each speech segment.
- **Integrated the gate into the microphone recording path** of the BirdNET
  application, so the in-memory audio array is redacted in place before the
  file is ever written, with the fail-closed behavior described above.
- **Verified the pipeline on the target hardware** (an NVIDIA Jetson AGX
  Thor) and produced a before/after demo on real speech audio.

## Key finding, in detail: the pipeline persists audio before it classifies it

The existing `flint-pete/birdnet` node application writes every capture to a
temporary FLAC file before any classification runs, on both audio paths:

- **Microphone path** (`record_from_microphone`, app.py:160-178): pywaggle's
  `Microphone.record()` returns an in-memory `AudioSample`, but the code
  immediately calls `sample.save()` to a temp FLAC before doing anything
  else.
- **Camera path** (`record_from_camera`, app.py:181-237): ffmpeg writes the
  capture straight to disk; the PCM samples never exist as a Python array.

This means redaction cannot be a downstream filter: by the time BirdNET
runs, raw speech is already on disk. Redaction has to happen **before
persistence**.

The microphone path is fixable, because `sample.data` is a 1-D float32 array
available in memory before `save()`. Redaction can operate on that array in
place, and only the already-redacted array is written. The camera path is
harder and would require piping ffmpeg to stdout to decode in-process.

## What I built

Three tested Python modules (all with unit tests, verified against source
rather than assumed).

**`RedactionGate`** is a hysteresis state machine that takes per-frame speech
scores and returns the time ranges to redact. It has configurable enter/exit
thresholds (low bar to enter redaction, higher bar to exit, so it does not
flicker mid-sentence), pre-roll padding, hangover (gap tolerance), and
post-roll padding. It fails closed on empty input.

- Caught and fixed a real bug: YAMNet frames are 0.96s long with a 0.48s
  hop, so they overlap. Naive frame-hop math under-redacted the tail of
  every speech segment by 0.48s. Fixed by tracking frame duration separately
  from hop.

**`speech_classes.py`** holds the verified YAMNet AudioSet speech class
indices, checked against the canonical `yamnet_class_map.csv`, split into
core speech classes and ambiguous ones (crowd, chatter, children). It
reduces a 521-class YAMNet frame vector to a single speech score by taking
the max across the speech family. (Note: "Male speech" and "Female speech"
are not YAMNet classes, a detail worth verifying rather than assuming.)

**`yamnet_speech.py`** wraps YAMNet to turn a raw audio array into per-frame
speech scores: resample to 16kHz mono, run YAMNet, reduce each frame through
`speech_classes`. Model load is lazy and cached. The deployment uses a persistent TFLite model location on the Thor so the model survives node reboots.

## What has been verified on hardware (Jetson AGX Thor, aarch64)

The complete detection-and-redaction pipeline has been validated end-to-end on the NVIDIA Jetson AGX Thor. YAMNet runs through LiteRT/TFLite on the ARM CPU and produces reliable speech scores on real recordings. The RedactionGate correctly identifies speech regions, applies the configured padding and hangover behavior, and removes speech while preserving surrounding ambient audio. A before/after demonstration confirmed that speech is removed while the remaining soundscape is preserved.

Together these confirm the runtime half of the design: the speech detector
runs on the target hardware, and the redaction gate fires correctly on real
speech and stays quiet on real non-speech.

**Mic-path integration: done.** The in-memory, never-persists integration is
now landed on the birdnet fork (hermes-mighdz/birdnet): `redact_speech` is
wired into `record_from_microphone` so the raw array is redacted before the
save call, failing closed if scoring is unavailable. 43 tests pass on the
fork, and a before/after demo (raw vs redacted output on the same speech
clip) was produced.

**Threshold tuning: harness built and verified; real tuning pending better
data.** A sweep harness (`redaction/scripts/tune_thresholds.py` on the
fork's `redaction-mic-integration` branch) sweeps `enter_threshold` and
reports recall vs false-positive rate on labeled clips. It runs correctly on
real audio, but has only been exercised on a 3-clip labeled set so far,
which is too cleanly separable to pick an operating point from. A richer
labeled set spanning borderline cases (distant or mumbled speech, ambient
sounds that trip YAMNet's speech classes) is needed before it yields a real
tuned threshold.

**Camera path: design proposal, not built.**
`redaction/CAMERA-PATH-DESIGN.md` on the fork proposes extending redaction
to the camera path via an ffmpeg stdout-pipe (so raw audio never touches
disk), plus the audio/video time-windowing idea (detect an audio event at
time t, snip the video around it). None of it is implemented yet.

## Grounded parameter choices

Padding and threshold choices are backed by sourced research on voice
activity detection hangover timing (WebRTC VAD, NVIDIA Riva/Silero, 3GPP AMR
specs) rather than guessed. Telephony VAD uses 60-580ms hangover, but that
is tuned for the opposite cost trade-off (don't waste bandwidth on silence).
For privacy redaction, where under-redacting is far more costly, the
recommended guard is ~1s post-utterance with a longer hold when the signal
is non-stationary.

## Redaction as a data product

Rather than filling redacted spans with comfort noise (which would corrupt
the bioacoustic record BirdNET and downstream soundscape analysis depend
on), the proposal, pending sign-off from Pete, is to write silence plus
publish a separate `audio.redacted` measurement with timestamp and duration.
This makes each redaction auditable and distinguishable from a dead sensor,
and gives NPS a verifiable log of when redaction occurred, without revealing
what was said. It also yields free statistics on human presence at the site.

## Current status and next steps

- **Done:** the microphone-path integration, the tested redaction
  components, on-Thor validation of YAMNet, and the before/after demo on
  real speech.
- **Pending:** refactor the microphone-path implementation into a standalone     producer/consumer Sage plugin that consumes audio from the media-sampler       cache and publishes a redacted audio product for downstream applications       such as BirdNET.
- **Proposed, not built, camera path:** RTSP audio from the Reolink is
  confirmed (AAC 16kHz mono, verified live) and a design proposal is written
  (`redaction/CAMERA-PATH-DESIGN.md` on the fork); implementing the ffmpeg
  stdout-pipe decode is the open item. None of it is implemented yet.
- **Pending, threshold tuning:** build a richer labeled clip set
  (distant/mumbled speech, YAMNet-confusable ambient) and run
  `tune_thresholds.py` over it to pick an operating point: target recall
  99%+, report the precision cost. The system currently ships on
  conservative defaults that err toward over-redacting.
- **Pending, redaction data product:** define the `audio.redacted`
  measurement schema for Beehive.

## Repositories

- Notes, design docs, and redaction modules:
  https://github.com/Miguel-Hernandez1/project-sage-notes
- Integration analysis (birdnet fork):
  https://github.com/hermes-mighdz/birdnet

## Acknowledgment

This work was supported in part by the National Science Foundation under
Awards No. 2331263 and 2436842.
