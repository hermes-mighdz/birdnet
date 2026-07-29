"""Speech-triggered redaction package for the BirdNET Sage plugin.

Privacy-critical: zero out speech windows in the in-memory audio array BEFORE
the array is written to disk (sample.save()) or uploaded. The raw unredacted
array must never hit disk.

Modules:
  speech_classes   — verified YAMNet speech-family class indices + speech_score()
  redaction_gate   — hysteresis state machine turning per-frame scores into padded
                     [start, end) redaction windows. Pure Python, fully unit-tested.
  yamnet_speech    — YAMNet front-end producing per-frame speech scores, backed by
                     the YAMNet .tflite model run via ai_edge_litert (no
                     tensorflow_hub, no network at runtime). Public API mirrors
                     the notes-ref design doc.
"""
