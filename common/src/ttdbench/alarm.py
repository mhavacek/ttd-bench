"""K-consecutive alarm logic.

Alarm definition (configurable, see configs/experiment.yaml):
  a detection of a weapon class with confidence >= threshold in K CONSECUTIVE
  processed frames raises the alarm; t_alarm is the monotonic timestamp of the
  decision immediately after the K-th frame's postprocessing.

"Consecutive" refers to consecutively PROCESSED frames (the operational
system only sees frames that survive backpressure dropping) — dropped frames
do not reset the counter, a processed frame without a qualifying detection
does. This matches how a deployed alarm module would behave and is documented
in the paper's methods.
"""

from __future__ import annotations

from typing import Any, Optional

from .logio import now_ns


class AlarmLogic:
    def __init__(
        self,
        weapon_class_names: list[str],
        confidence_threshold: float = 0.5,
        k_consecutive: int = 3,
    ):
        self.weapon_classes = {c.lower() for c in weapon_class_names}
        self.threshold = confidence_threshold
        self.k = k_consecutive
        self.streak = 0
        self.fired = False
        self.t_alarm: Optional[int] = None
        self.alarm_frame_idx: Optional[int] = None

    def _qualifies(self, detections: list[dict[str, Any]]) -> bool:
        return any(
            d["cls"].lower() in self.weapon_classes and d["conf"] >= self.threshold
            for d in detections
        )

    def update(self, frame_idx: int, detections: list[dict[str, Any]]) -> Optional[int]:
        """Feed one processed frame; returns t_alarm (ns) when the alarm fires."""
        if self.fired:
            return None
        if self._qualifies(detections):
            self.streak += 1
            if self.streak >= self.k:
                self.fired = True
                self.t_alarm = now_ns()
                self.alarm_frame_idx = frame_idx
                return self.t_alarm
        else:
            self.streak = 0
        return None
