"""YAMNet speech-scoring front-end for the redaction pipeline.

Adapted from notes-ref/code/redaction/yamnet_speech.py: the public API
``speech_scores(audio_1d, samplerate, include_ambiguous=False)`` is identical,
but the model backend is swapped from ``tensorflow_hub`` (requires network +
full TF, not available on this Thor) to the YAMNet ``.tflite`` model run via
``ai_edge_litert.interpreter.Interpreter`` (already a transitive dependency
of ``birdnet>=0.2.16``, verified working on aarch64: see
REDACTION-INTEGRATION-NOTES.md §2).

Build-time: the yamnet.tflite model is produced once from the TF SavedModel
(`google/yamnet/tensorFlow2/yamnet` via kagglehub, kaggle path: also already
a birdnet dep) and baked into the plugin image alongside the BirdNET models.
See the Dockerfile changes in the integration plan.

Runtime: no network, no tensorflow_hub. Loads the .tflite from
``YAMNET_TFLITE_PATH`` (env override ``BIRDNET_YAMNET_TFLITE``).
"""

import os

import numpy as np

from .speech_classes import speech_score

YAMNET_SAMPLE_RATE = 16000

# Resolve the .tflite path once. Env override takes precedence; then the path
# baked into the plugin image (see Dockerfile); then the persistent dev
# location (survives reboots on a bare-metal Thor); then the dev scratch
# location used during early validation. The persistent path is ahead of /tmp
# so a dev Thor that has neither env override nor a Dockerfile-baked model
# still lands on the reboot-persistent copy rather than the volatile one: see
# REDACTION-INTEGRATION-NOTES.md §3 "Reboot persistence of the .tflite".
#
# NOTE: literal absolute paths, not `~` expansion. On a Sage plugin container
# AND in dev sandboxes where HOME is not /home/mighdz, os.path.expanduser("~")
# would resolve to the wrong home and the file would never be found. Coded
# against the known real path; for any other dev machine, set
# BIRDNET_YAMNET_TFLITE.
#
# NOTE: existence-filtered, NOT an `or` chain. A bare
#   `env.get(...) or "/app/models/..." or "/tmp/..."`
# returns the first truthy string unconditionally: "/app/models/yamnet.tflite"
# is non-empty even on a dev Thor where that file does NOT exist, so the chain
# would resolve to a non-existent /app/models/ path and never fall through to
# /tmp/. Filter by os.path.exists so the chain reflects where the model
# actually is. _load_model still raises FileNotFoundError below if every
# path misses.
_YAMNET_TFLITE_PATHS = [
    os.environ.get("BIRDNET_YAMNET_TFLITE"),        # highest precedence
    "/app/models/yamnet.tflite",                     # plugin container (Dockerfile COPY)
    "/home/mighdz/AI-Projects/models/yamnet.tflite", # persistent dev (survives reboots)
    "/tmp/yamnet.tflite",                            # volatile dev/validation scratch
]
YAMNET_TFLITE_PATH = next(
    (p for p in _YAMNET_TFLITE_PATHS if p and os.path.exists(p)), None
)

# Interpreter instance is lazily constructed and reused across calls. YAMNet
# is stateless across waveforms (no carry-over), so a single Interpreter is
# safe to share; resize_tensor_input() is called per waveform.
_interp = None
_interp_model_path = None


def _load_model():
    """Return a singleton ai_edge_litert Interpreter for the YAMNet .tflite.

    Raises FileNotFoundError if the model is missing. The caller (speech_scores)
    catches that and propagates RedactionFailure so record_from_microphone can
    fail closed -- NEVER return [] (which would mean "no speech -> publish raw").
    """
    global _interp, _interp_model_path
    if not YAMNET_TFLITE_PATH or not os.path.exists(YAMNET_TFLITE_PATH):
        raise FileNotFoundError(
            f"YAMNet .tflite not found. Checked: "
            f"{', '.join(p or '<env unset>' for p in _YAMNET_TFLITE_PATHS)}. "
            f"Set BIRDNET_YAMNET_TFLITE or bake the model into the image "
            f"(see REDACTION-INTEGRATION-NOTES.md §3)."
        )
    # Re-create only if the path changed (e.g. env override flipped at runtime).
    if _interp is None or _interp_model_path != YAMNET_TFLITE_PATH:
        from ai_edge_litert.interpreter import Interpreter  # deferred: heavy import
        _interp = Interpreter(model_path=YAMNET_TFLITE_PATH)
        _interp_model_path = YAMNET_TFLITE_PATH
    return _interp


class RedactionFailure(Exception):
    """Raised when speech scoring cannot run. Caller MUST fail closed."""


def _prepare_waveform(audio_1d, samplerate: int) -> np.ndarray:
    """Normalize to 1-D float32 mono in [-1, 1] and resample to 16 kHz.

    Mirrors notes-ref yamnet_speech._prepare_waveform. Linear interpolation is
    a first-pass resampler (no anti-aliasing); a one-line swap to
    scipy.signal.resample_poly is noted in REDACTION-INTEGRATION-NOTES.md §5.
    """
    audio = np.asarray(audio_1d)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # downmix interleaved stereo / multichannel
    elif audio.ndim != 1:
        raise ValueError(f"expected 1-D (or 2-D multichannel) audio, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("empty audio buffer")
    if samplerate <= 0:
        raise ValueError(f"invalid samplerate {samplerate}")

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)  # YAMNet expects float in [-1, 1]
    audio = audio.astype(np.float32)

    if samplerate != YAMNET_SAMPLE_RATE:
        n_out = int(round(audio.size * YAMNET_SAMPLE_RATE / samplerate))
        t_out = np.arange(n_out) * (samplerate / YAMNET_SAMPLE_RATE)
        audio = np.interp(t_out, np.arange(audio.size), audio).astype(np.float32)

    return audio


def speech_scores(audio_1d, samplerate: int, include_ambiguous: bool = False) -> list:
    """Return one float speech score per YAMNet frame (0.96s window, 0.48s hop).

    Raises RedactionFailure on ANY model/load error. Never returns [] silently
    -- the caller (record_from_microphone) treats that as "fail closed, redact
    everything".

    Args:
        audio_1d: 1-D (or 2-D multichannel) numpy array or array-like PCM.
        samplerate: integer sample rate of audio_1d.
        include_ambiguous: if True, also take max over the 4 ambiguous speech
            classes (Chatter, Crowd, Child singing, Children playing). Default
            False == CORE_SPEECH only, matching REDACTION-INTEGRATION-NOTES.md.
    """
    try:
        waveform = _prepare_waveform(audio_1d, samplerate)
    except ValueError as e:
        # Bad input (empty array, multidim, ...) -- can't score, must fail closed.
        raise RedactionFailure(f"could not prepare waveform: {e}") from e

    try:
        interp = _load_model()
    except FileNotFoundError as e:
        raise RedactionFailure(str(e)) from e

    try:
        in_det = interp.get_input_details()[0]
        # The TFLite converter froze the input dim to [1]; resize per waveform
        # length so we can feed arbitrary-duration captures.
        interp.resize_tensor_input(in_det["index"], [waveform.size])
        interp.allocate_tensors()
        interp.set_tensor(in_det["index"], waveform)
        interp.invoke()
    except Exception as e:
        # Interpreter blew up mid-inference. Treat as fail-closed (we have no
        # speech scores -> cannot safely redact nothing).
        raise RedactionFailure(f"YAMNet .tflite inference failed: {e}") from e

    # Find the scores tensor by shape (..., 521). The converter-mangled output
    # names ('StatefulPartitionedCall:0' etc.) are unreliable across builds;
    # the 521-wide column is YAMNet's class count and is stable.
    scores_arr = None
    for t in interp.get_output_details():
        arr = interp.get_tensor(t["index"])
        if arr.shape[-1] == 521:
            scores_arr = arr
            break
    if scores_arr is None:
        raise RedactionFailure("YAMNet .tflite produced no (N,521) scores tensor")

    if scores_arr.ndim == 2 and scores_arr.shape[0] == 1:
        scores_arr = scores_arr[0]  # squeeze leading batch dim if the converter added one

    return [speech_score(frame, include_ambiguous=include_ambiguous)
            for frame in scores_arr]
