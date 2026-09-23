"""Hand tracking helpers: pinch detection and the two knob interaction models."""

import math
from dataclasses import dataclass, field

import numpy as np

# MediaPipe hand landmark indices we care about.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, PINKY_TIP = 0, 4, 8, 9, 20

# Skeleton edges, for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]

PINCH_ON = 0.42    # normalized distance below which the pinch "closes"
PINCH_OFF = 0.55   # and above which it opens again (hysteresis kills flicker)

# Thumb-to-pinky tap: the gesture that steps this hand to the next effect.
# An open hand sits well above 1.0 here, so the thresholds have plenty of room.
CYCLE_ON = 0.50
CYCLE_OFF = 0.72
CYCLE_COOLDOWN = 0.45   # seconds, so one deliberate tap fires exactly once

# How far you must twist your hand to sweep the knob from 0 to 1.
TWIST_SWEEP = math.radians(150.0)

# Squeeze mode range: normalized pinch distance mapped onto 0..1.
SQUEEZE_MIN, SQUEEZE_MAX = 0.12, 0.90


def to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Normalized landmarks -> (21, 2) float array of pixel coordinates."""
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)


@dataclass
class HandState:
    """Everything the UI needs to know about one tracked hand this frame."""
    present: bool = False
    points: np.ndarray | None = None
    pinch_norm: float = 1.0
    pinching: bool = False
    pinch_point: tuple[float, float] = (0.0, 0.0)
    cycle_norm: float = 2.0
    cycling: bool = False


def read_hand(points: np.ndarray, was_pinching: bool,
              was_cycling: bool = False) -> HandState:
    """Measure both gestures, normalized by the hand's own size so distance to
    the camera never changes the thresholds."""
    thumb, index, pinky = points[THUMB_TIP], points[INDEX_TIP], points[PINKY_TIP]
    span = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST])) or 1.0

    norm = float(np.linalg.norm(thumb - index)) / span
    pinching = norm < PINCH_ON if not was_pinching else norm < PINCH_OFF

    cycle_norm = float(np.linalg.norm(thumb - pinky)) / span
    cycling = cycle_norm < CYCLE_ON if not was_cycling else cycle_norm < CYCLE_OFF
    # A closed fist puts thumb and pinky together too, so require the index
    # finger to be clear of the thumb — a tap, not a grip.
    cycling = cycling and norm > PINCH_OFF

    mid = tuple(((thumb + index) / 2.0).tolist())
    return HandState(True, points, norm, pinching, mid, cycle_norm, cycling)


def _angle(points: np.ndarray, pinch_point: tuple[float, float]) -> float:
    """Orientation of the wrist -> pinch vector. This is what you 'twist'."""
    dx = pinch_point[0] - points[WRIST][0]
    dy = pinch_point[1] - points[WRIST][1]
    return math.atan2(dy, dx)


def _wrap(a: float) -> float:
    """Shortest signed angular difference, so crossing +/-pi doesn't jump."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Knob:
    """A 0..1 control driven by one hand.

    TWIST  : pinch to grab, then rotate your hand like a real dial.
    SQUEEZE: how open your pinch is maps straight onto the value.
    """
    label: str
    value: float = 0.0
    display: float = 0.0
    grabbed: bool = False
    _grab_angle: float = 0.0
    _grab_value: float = 0.0
    _last_angle: float = 0.0
    _was_pinching: bool = False
    was_cycling: bool = False
    _last_cycle_t: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, state: HandState | None, mode: str) -> None:
        if state is None or not state.present:
            self.grabbed = False
            self._was_pinching = False
            self._settle()
            return

        if mode == "squeeze":
            t = (state.pinch_norm - SQUEEZE_MIN) / (SQUEEZE_MAX - SQUEEZE_MIN)
            self.value = float(np.clip(1.0 - t, 0.0, 1.0))
            self.grabbed = state.pinch_norm < SQUEEZE_MAX
        else:
            angle = _angle(state.points, state.pinch_point)
            if state.pinching and not self._was_pinching:
                # Fresh grab: anchor the dial where it currently sits.
                self._grab_angle = angle
                self._grab_value = self.value
                self._last_angle = angle
            if state.pinching:
                # Accumulate incrementally so >180 deg of total twist still works.
                self._grab_value += _wrap(angle - self._last_angle) / TWIST_SWEEP
                self._grab_value = float(np.clip(self._grab_value, 0.0, 1.0))
                self._last_angle = angle
                self.value = self._grab_value
            self.grabbed = state.pinching
            self._was_pinching = state.pinching

        self._settle()

    def _settle(self) -> None:
        """Exponential smoothing: tracking jitter should not make the look flicker."""
        self.display += (self.value - self.display) * 0.35
        self.history.append(self.display)
        if len(self.history) > 120:
            self.history.pop(0)

    def tapped(self, state: HandState | None, now: float) -> bool:
        """True once per thumb-to-pinky tap, on the closing edge."""
        if state is None or not state.present:
            self.was_cycling = False
            return False
        fired = (state.cycling and not self.was_cycling
                 and now - self._last_cycle_t > CYCLE_COOLDOWN)
        self.was_cycling = state.cycling
        if fired:
            self._last_cycle_t = now
        return fired

    def glide_to(self, value: float) -> None:
        """Retarget the knob without snapping: `display` eases there over a few
        frames, which is what makes a tap read as a smooth effect change."""
        self.value = float(np.clip(value, 0.0, 1.0))
        self._grab_value = self.value

    def reset(self) -> None:
        self.value = self.display = 0.0
        self.grabbed = False
        self._was_pinching = False
        self.was_cycling = False
        self.history.clear()
