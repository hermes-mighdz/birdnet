"""Apply speech redaction to an in-memory audio array.

The privacy-critical entry point used by ``record_from_microphone``:
takes the raw ``AudioSample.data`` numpy array, runs YAMNet speech scoring +
RedactionGate to compute speech windows, and ZEROS those sample ranges IN
PLACE-then-return so the caller persists only the redacted array.

Invariant (enforced by this module):
  On ANY failure (YAMNet error, model missing, RedactionGate empty-scores),
  this module either (a) returns a fully-zeroed buffer, or (b) raises
  ``RedactionFailure`` -- it NEVER returns the original unredacted array
  and NEVER returns a "no redaction" result on an error.
"""

import numpy as np

from .redaction_gate import RedactionGate, RedactionGateFailure
from .yamnet_speech import RedactionFailure, speech_scores

# Default gate parameters -- the notes-ref design defaults, biased hard toward
# recall (under-redacting is catastrophic). Overridable via a future CLI flag
# (not added in this change -- per REDACTION-INTEGRATION-NOTES.md §4 step 7).
_DEFAULT_GATE = dict(
    enter_threshold=0.25,
    exit_threshold=0.15,
    pre_roll_seconds=1.5,
    hangover_seconds=0.75,
    post_roll_seconds=0.75,
    # frame_hop / frame_duration default to YAMNet's 0.48 / 0.96 inside RedactionGate
    fail_closed=True,
)


def redact_speech(audio_1d, samplerate: int, gate: RedactionGate | None = None):
    """Zero out speech windows in an audio array. IN-PLACE.

    Args:
        audio_1d: 1-D numpy float array (the AudioSample.data buffer). Modified
            in place AND returned. Caller must NOT have written this array to
            disk yet.
        samplerate: integer sampling rate of audio_1d.
        gate: optional RedactionGate instance (else the notes' defaults are used).

    Returns:
        (redacted_audio, windows, fail_closed_reason) where:
          - ``redacted_audio`` is the SAME array object, zeroed across the
            redaction windows. It is NEVER the raw unredacted buffer on an
            error path.
          - ``windows`` is a list of (start_s, end_s) float pairs, always
            (float, float). On a fail-closed path this is [(0.0, duration)].
          - ``fail_closed_reason`` is None on the normal path, or a string
            describing the failure when the whole buffer was zeroed because
            YAMNet / RedactionGate could not run. The caller should log it.

    Failure handling (fail-closed):
      If speech_scores or RedactionGate raises (model missing, inference
      error, empty scores with fail_closed=True), the ENTIRE buffer is zeroed
      and returned with windows=[(0.0, duration)] and a non-None reason. The
      raw array NEVER leaves this function on an error path. Privacy posture:
      if we cannot classify speech, assume all of it is speech.
    """
    gate = gate or RedactionGate(**_DEFAULT_GATE)
    n = audio_1d.size
    duration_s = n / float(samplerate) if samplerate else 0.0

    try:
        scores = speech_scores(audio_1d, samplerate)
        windows = gate.get_redaction_windows(scores)
    except (RedactionFailure, RedactionGateFailure) as e:
        # Fail closed: zero the ENTIRE buffer. No raw audio leaves this function
        # on an error path. Caller logs the reason and may publish a redaction
        # event measurement covering the whole capture.
        audio_1d.fill(0.0)
        return audio_1d, [(0.0, duration_s)], str(e)

    # Zero each window in place. Windows are in seconds (from YAMNet's 0.48s
    # hop / 0.96s frames); convert back to sample indices at the ORIGINAL
    # samplerate (48k or whatever the mic was set to) -- speech_scores already
    # handled the 48k->16k resample internally, windows are in wall-clock time.
    for start_s, end_s in windows:
        i0 = max(0, int(start_s * samplerate))
        i1 = min(n, int(end_s * samplerate))
        if i0 < i1:
            audio_1d[i0:i1] = 0.0

    return audio_1d, windows, None
