"""Terminal telemetry.

The window shows what the effects look like; this shows what the program is
*doing* — device negotiation, model load, every gesture the tracker resolves,
and a periodic status line. Colours are 24-bit escapes taken straight from each
effect's accent, so the terminal and the HUD agree, and they switch themselves
off when stdout is not a TTY.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"

BLOCKS = "░▒▓█"

TAG_COLORS = {
    "BOOT": (120, 200, 255),
    "OK": (110, 230, 140),
    "CAM": (150, 200, 255),
    "MDL": (190, 160, 255),
    "HAND": (150, 220, 230),
    "GRAB": (255, 210, 110),
    "REL": (180, 170, 140),
    "TAP": (255, 130, 220),
    "FX": (255, 190, 90),
    "MODE": (140, 190, 255),
    "VIEW": (160, 180, 200),
    "STAT": (150, 150, 150),
    "WARN": (255, 120, 110),
    "EXIT": (200, 200, 200),
}


class Console:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # NO_COLOR is the de facto opt-out; a pipe or file gets plain text.
        self.color = (enabled and sys.stdout.isatty()
                      and os.environ.get("NO_COLOR") is None)
        self.t0 = time.perf_counter()
        self.width = shutil.get_terminal_size((90, 24)).columns

    # ------------------------------------------------------------- plumbing

    def _rgb(self, rgb: tuple[int, int, int]) -> str:
        if not self.color:
            return ""
        r, g, b = rgb
        return f"\x1b[38;2;{r};{g};{b}m"

    def _c(self, text: str, rgb: tuple[int, int, int] | None = None,
           style: str = "") -> str:
        if not self.color:
            return text
        return f"{style}{self._rgb(rgb) if rgb else ''}{text}{RESET}"

    def log(self, tag: str, message: str,
            rgb: tuple[int, int, int] | None = None) -> None:
        if not self.enabled:
            return
        stamp = f"[{time.perf_counter() - self.t0:8.3f}s]"
        colored_tag = self._c(f"{tag:<4}", rgb or TAG_COLORS.get(tag, (200, 200, 200)),
                              BOLD if self.color else "")
        print(f"{self._c(stamp, (110, 110, 110), DIM)} {colored_tag}  {message}",
              flush=True)

    # --------------------------------------------------------------- pieces

    def banner(self) -> None:
        if not self.enabled:
            return
        accent = (255, 150, 90)
        bar = "─" * 54
        print()
        print(self._c(f"┌{bar}┐", accent))
        print(self._c("│", accent)
              + self._c("  P I N C H   V I S I O N".ljust(54), (255, 255, 255), BOLD)
              + self._c("│", accent))
        print(self._c("│", accent)
              + self._c("  gesture-controlled optics · mediapipe + opencv".ljust(54),
                        (150, 150, 150))
              + self._c("│", accent))
        print(self._c(f"└{bar}┘", accent))
        print()

    def rack(self, effects, fx_a: int, fx_b: int) -> None:
        """List the effect rack, marking what each knob currently holds."""
        if not self.enabled:
            return
        self.log("BOOT", self._c("effect rack", (255, 255, 255), BOLD))
        for i, (name, _, bgr) in enumerate(effects):
            rgb = (bgr[2], bgr[1], bgr[0])
            marks = []
            if i == fx_a:
                marks.append("K1")
            if i == fx_b:
                marks.append("K2")
            held = self._c(" ← " + "+".join(marks), (255, 255, 255)) if marks else ""
            swatch = self._c("███", rgb)
            print(f"{'':11} {swatch} {self._c(f'{i}', (120, 120, 120))} "
                  f"{self._c(name, rgb)}{held}", flush=True)

    def controls(self) -> None:
        if not self.enabled:
            return
        rows = [
            ("pinch index+thumb", "grab the knob, twist to turn it"),
            ("tap thumb+pinky", "step that hand to the next effect"),
            ("C", "morph mode — knobs sweep the whole rack"),
            ("H / M / R / L", "swap hands / input mode / reset / skeleton"),
            ("+ - / F", "window size / fullscreen"),
            ("Q", "quit"),
        ]
        self.log("BOOT", self._c("controls", (255, 255, 255), BOLD))
        for keys, what in rows:
            print(f"{'':11} {self._c(keys.rjust(18), (180, 210, 255))}  "
                  f"{self._c(what, (150, 150, 150))}", flush=True)
        print(flush=True)

    def meter(self, value: float, width: int = 10) -> str:
        filled = int(round(max(0.0, min(1.0, value)) * width))
        return BLOCKS[3] * filled + BLOCKS[0] * (width - filled)

    def stat(self, fps: float, hands_seen: int, slots: list[tuple]) -> None:
        """One periodic line: rate, hands in frame, and both knobs."""
        if not self.enabled:
            return
        parts = [f"{fps:5.1f} fps", f"hands {hands_seen}"]
        for slot, name, value, bgr, grabbed in slots:
            rgb = (bgr[2], bgr[1], bgr[0])
            tick = self._c("●", (255, 210, 110)) if grabbed else self._c("○", (110, 110, 110))
            parts.append(f"{tick} K{slot} {self._c(name, rgb)} "
                         f"{self._c(self.meter(value), rgb)} {value * 100:3.0f}%")
        self.log("STAT", self._c(" │ ", (90, 90, 90)).join(parts))

    def outro(self, frames: int, elapsed: float) -> None:
        if not self.enabled:
            return
        rate = frames / elapsed if elapsed > 0 else 0.0
        self.log("EXIT", f"{frames} frames in {elapsed:.1f}s "
                         f"({rate:.1f} fps average)")
        print()
