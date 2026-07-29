"""
Unit tests for the redaction package — run locally, no GPU / no pywaggle /
no LiteRT model file required. The LiteRT-backed path (yamnet_speech /
redact_speech calling the real .tflite) is exercised with monkeypatching only;
the model file itself is NOT loaded here.

    python3 -m pytest tests/test_redaction.py -v
    # or, dependency-free:
    python3 tests/test_redaction.py
"""

import os
import sys

import numpy as np
import pytest

# Repo-root on sys.path so `from redaction.X import ...` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaction.redaction_gate import RedactionGate, RedactionGateFailure  # noqa: E402
from redaction.speech_classes import (  # noqa: E402
    AMBIGUOUS, CORE_SPEECH, NUM_YAMNET_CLASSES, speech_score,
)
from redaction.apply import redact_speech  # noqa: E402
from redaction import yamnet_speech  # noqa: E402


# ── speech_classes (indices verified against canonical CSV this session) ──────

def test_core_speech_index_count():
    assert len(CORE_SPEECH) == 12


def test_ambiguous_index_count():
    assert len(AMBIGUOUS) == 4


def test_no_overlap_between_core_and_ambiguous():
    assert set(CORE_SPEECH).isdisjoint(AMBIGUOUS)


def test_speech_score_uses_core_only_by_default():
    scores = [0.0] * NUM_YAMNET_CLASSES
    scores[0] = 0.7    # Speech
    scores[64] = 0.9   # Crowd (ambiguous) — ignored by default
    assert speech_score(scores) == 0.7


def test_speech_score_raises_on_wrong_length_vector():
    with pytest.raises(ValueError):
        speech_score([0.0] * 10)


# ── RedactionGate (pure-Python hysteresis — tests ported from notes-ref) ──────

def test_isolated_spike():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=1.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.0, 0.0, 0.9, 0.0, 0.0]) == [(1.0, 4.0)]


def test_sustained_speech():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=1.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.0, 0.0] + [0.9] * 5 + [0.0] * 3) == [(1.0, 8.0)]


def test_flickering_scores_merge_into_one_window():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.9, 0.2, 0.9, 0.2, 0.9]) == [(0.0, 5.0)]


def test_empty_scores_fail_closed_by_default():
    with pytest.raises(RedactionGateFailure):
        RedactionGate().get_redaction_windows([])


def test_empty_scores_can_opt_into_fail_open():
    gate = RedactionGate(fail_closed=False)
    assert gate.get_redaction_windows([]) == []


# ── apply.redact_speech — the privacy-critical contract ──────────────────────
#
# These tests are the heart of the integration verification: redact_speech must
# (a) zero the windows returned by the gate, in place, on the success path;
# (b) NEVER return the raw unredacted buffer on ANY failure path — it must zero
#     the whole buffer and surface the failure.
# We monkeypatch yamnet_speech.speech_scores so no model file is needed.

def _fake_speech_scores_factory(scores):
    """Return a fake speech_scores replacement that ignores its audio args
    and returns the given per-frame scores (mimicking yamnet_speech.speech_scores)."""
    def _fake(audio_1d, samplerate, include_ambiguous=False):
        return list(scores)
    return _fake


def test_redact_speech_zeros_windows_in_place_on_success(monkeypatch):
    # 3s of audio @ 1000 Hz; scores [0, 0.9, 0] with hop=1.0/dur=1.0 means frame 1
    # covers [1.0s, 2.0s] and the gate returns window (1.0, 2.0). That maps to
    # samples [1000, 2000) at 1000 Hz.
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.9, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(3000, dtype=np.float32) * 0.5  # 3s @ 1000 Hz
    original_first = audio[0]
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate)
    assert reason is None
    assert windows == [(1.0, 2.0)]
    assert redacted is audio  # in place — same array object
    # window (1.0, 2.0) maps to samples [1000, 2000) at 1000 Hz — zeroed.
    assert redacted[1000:2000].sum() == 0.0
    # Untouched regions preserved.
    assert redacted[0] == original_first          # first sample intact
    assert redacted[800] == original_first        # pre-window intact
    assert redacted[2500] == original_first        # post-window intact


def test_redact_speech_fail_closed_on_yamnet_error(monkeypatch):
    """If YAMNet raises, redact_speech must zero the ENTIRE buffer and return
    a fail-closed reason — it must NEVER return the original array unmodified."""
    def _raising_speech_scores(audio_1d, samplerate, include_ambiguous=False):
        raise yamnet_speech.RedactionFailure("simulated YAMNet inference crash")

    # NOTE: apply.py does `from .yamnet_speech import speech_scores`, binding the
    # name into its own namespace. Patching redaction.yamnet_speech.speech_scores
    # would not affect what apply.redact_speech actually calls. Patch the name
    # inside redaction.apply instead.
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores", _raising_speech_scores)
    gate = RedactionGate()
    audio = np.ones(24000, dtype=np.float32) * 0.5  # 1.5s @ 16k simulating mic
    raw_sum = audio.sum()
    redacted, windows, reason = redact_speech(audio, 16000, gate=gate)
    # Critical: fail closed, the raw array did NOT leak.
    assert redacted.sum() == 0.0
    assert redacted.sum() != raw_sum
    assert reason is not None and "simulated YAMNet inference crash" in reason
    assert windows == [(0.0, 1.5)]  # whole buffer


def test_redact_speech_fail_closed_on_empty_scores(monkeypatch):
    """Empty scores -> RedactionGate(fail_closed=True) raises RedactionGateFailure;
    redact_speech must catch it and zero the whole buffer, not let raw audio through."""
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([]))
    gate = RedactionGate()  # fail_closed=True default
    audio = np.ones(24000, dtype=np.float32) * 0.5
    redacted, windows, reason = redact_speech(audio, 16000, gate=gate)
    assert redacted.sum() == 0.0
    assert reason is not None
    # Reason message comes from RedactionGateFailure("no classification scores...").
    assert "no classification scores" in reason.lower()


def test_redact_speech_no_speech_leaves_buffer_untouched(monkeypatch):
    """Success path with NO speech detected: windows is [], reason is None, and
    the buffer is NOT modified — that's correct (ambient audio, no redaction)."""
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.0, 0.0, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(4000, dtype=np.float32) * 0.5
    pre_sum = audio.sum()
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate)
    assert reason is None
    assert windows == []
    assert redacted.sum() == pre_sum  # untouched — ambient preserved


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
