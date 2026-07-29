import math
from typing import List, Tuple


class RedactionGateFailure(Exception):
    pass


class RedactionGate:
    def __init__(
        self,
        enter_threshold: float = 0.25,
        exit_threshold: float = 0.15,
        pre_roll_seconds: float = 1.5,
        hangover_seconds: float = 0.75,
        post_roll_seconds: float = 0.75,
        frame_hop: float = 0.48,
        frame_duration: float = 0.96,
        fail_closed: bool = True,
    ):
        if exit_threshold > enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        if frame_hop <= 0:
            raise ValueError("frame_hop must be positive")
        if frame_duration < frame_hop:
            raise ValueError("frame_duration must be >= frame_hop for contiguous coverage")
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.pre_roll_seconds = pre_roll_seconds
        self.hangover_seconds = hangover_seconds
        self.post_roll_seconds = post_roll_seconds
        self.frame_hop = frame_hop
        self.frame_duration = frame_duration
        self.fail_closed = fail_closed
        # frame counts, not float seconds, so the gap check can't drift at odd hop values
        self._hangover_frames = math.ceil(hangover_seconds / frame_hop)

    def get_redaction_windows(self, scores: List[float]) -> List[Tuple[float, float]]:
        n = len(scores)
        if n == 0:
            if self.fail_closed:
                raise RedactionGateFailure(
                    "no classification scores available - caller must redact the full "
                    "buffer duration explicitly; silently redacting nothing is not safe"
                )
            return []

        # last frame covers [i*hop, i*hop + frame_duration), which overlaps the next
        # frame's start when frame_duration > frame_hop (true for YAMNet's 0.96s/0.48s)
        total_duration = (n - 1) * self.frame_hop + self.frame_duration
        raw_segments: List[Tuple[int, int]] = []

        active = False
        seg_start_frame = 0
        last_active_frame = 0

        for i, s in enumerate(scores):
            if active:
                if s >= self.exit_threshold:
                    last_active_frame = i
                elif (i - last_active_frame) > self._hangover_frames:
                    raw_segments.append((seg_start_frame, last_active_frame))
                    active = False
            if not active and s >= self.enter_threshold:
                active = True
                seg_start_frame = i
                last_active_frame = i

        if active:
            raw_segments.append((seg_start_frame, last_active_frame))

        windows = []
        for start_frame, last_frame in raw_segments:
            start_time = max(0.0, start_frame * self.frame_hop - self.pre_roll_seconds)
            end_time = min(
                total_duration,
                last_frame * self.frame_hop + self.frame_duration + self.post_roll_seconds,
            )
            windows.append((start_time, end_time))

        return self._merge_windows(windows)

    @staticmethod
    def _merge_windows(windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not windows:
            return []
        windows = sorted(windows)
        merged = [windows[0]]
        for start, end in windows[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
