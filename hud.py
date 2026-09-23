"""Console-style overlay.

OpenCV's Hershey fonts are proportional, so text drawn with one putText call
never lines up in columns. Everything here draws character by character at a
fixed pitch instead, which is what makes the panels read as a terminal rather
than as captions.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_PLAIN
BARS = " .:-=+*#%@"      # density ramp, reused for the history trace


def metrics(scale: float) -> tuple[int, int]:
    """(pitch, line height) in pixels for a given text scale."""
    (w, h), _ = cv2.getTextSize("M", FONT, scale, 1)
    return w + 1, int(h * 1.9)


def mono(canvas: np.ndarray, text: str, x: int, y: int, scale: float,
         color, thickness: int = 1) -> None:
    """Draw `text` on a fixed grid so columns align between lines."""
    pitch, _ = metrics(scale)
    for i, char in enumerate(text):
        if char == " ":
            continue
        cv2.putText(canvas, char, (x + i * pitch, y), FONT, scale, color,
                    thickness, cv2.LINE_AA)


def bar(value: float, width: int = 14) -> str:
    """[########............] as plain characters."""
    filled = int(round(np.clip(value, 0.0, 1.0) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def trace(history: list[float], width: int = 16) -> str:
    """The knob's recent movement as a density ramp, oldest to newest."""
    if not history:
        return " " * width
    samples = history[-width * 3:]
    step = max(1, len(samples) // width)
    picks = samples[::step][-width:]
    return "".join(BARS[int(np.clip(v, 0, 1) * (len(BARS) - 1))] for v in picks)


def panel_size(lines: list[str], scale: float = 1.0) -> tuple[int, int]:
    """Pixel footprint a panel will take, so callers can place it before drawing."""
    pitch, lh = metrics(scale)
    inner = max(len(s) for s in lines)
    return (inner + 4) * pitch + 6, (len(lines) + 2) * lh + 8


def fit_scale(line_sets: list[list[str]], budget: int, gap: int = 48,
              start: float = 0.95, floor: float = 0.6) -> float:
    """Largest text scale at which every panel still fits side by side."""
    scale = start
    while scale > floor:
        total = sum(panel_size(lines, scale)[0] for lines in line_sets) + gap
        if total <= budget:
            break
        scale -= 0.05
    return round(scale, 2)


def panel(canvas: np.ndarray, org: tuple[int, int], lines: list[str], color,
          scale: float = 1.0, alpha: float = 0.55, accent_rows: tuple = ()) -> tuple[int, int]:
    """Draw an ASCII-framed box at `org` (top-left). Returns its pixel size."""
    pitch, lh = metrics(scale)
    inner = max(len(s) for s in lines)
    framed = ["+" + "-" * (inner + 2) + "+"]
    framed += [f"| {s.ljust(inner)} |" for s in lines]
    framed.append("+" + "-" * (inner + 2) + "+")

    pw, ph = (inner + 4) * pitch + 6, len(framed) * lh + 8
    x, y = org
    x = max(0, min(x, canvas.shape[1] - pw))
    y = max(0, min(y, canvas.shape[0] - ph))

    # Dim the area behind the panel so it stays readable over a bright frame.
    roi = canvas[y:y + ph, x:x + pw]
    if roi.size:
        cv2.addWeighted(roi, 1.0 - alpha, np.zeros_like(roi), alpha, 0, roi)

    for i, line in enumerate(framed):
        is_frame = i in (0, len(framed) - 1)
        row_color = color if (is_frame or (i - 1) in accent_rows) else (205, 205, 205)
        mono(canvas, line, x + 4, y + (i + 1) * lh, scale, row_color)
    return pw, ph
