"""Pinch-to-turn camera effects rack.

Each hand is a knob: pinch thumb and index together to grab it, then twist your
hand to dial it. In fixed mode each knob drives one effect; in morph mode a knob
sweeps the whole rack, dissolving from clean through thermal, night vision, edge
glow, echo trails, glitch, ascii and halftone.

    python main.py [--camera 0] [--width 1280] [--height 720] [--window-scale 1.5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import hud
from console import Console
from effects import (EFFECTS, SHORT, EffectCache, morph, morph_position,
                     morph_station)
from hands import HAND_CONNECTIONS, HandState, Knob, read_hand, to_pixels

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "hand_landmarker.task")

# Hand tracking runs on a downscaled copy — landmarks come back normalized, so
# they map straight onto the full-res frame and we save real milliseconds.
DETECT_WIDTH = 640

WINDOW = "pinch vision"
KEYS = ("[1/2] FX  [C] MORPH  [H] SWAP  [M] INPUT  [R] RESET  [L] SKEL  "
        "[+/-] SIZE  [F] FULL  [Q] QUIT")


# ------------------------------------------------------------------ drawing

def draw_hand(canvas: np.ndarray, state: HandState, color) -> None:
    pts = state.points.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(canvas, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)

    thumb, index = tuple(pts[4]), tuple(pts[8])
    if state.pinching:
        cx, cy = int(state.pinch_point[0]), int(state.pinch_point[1])
        cv2.circle(canvas, (cx, cy), 14, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)
    else:
        cv2.line(canvas, thumb, index, color, 1, cv2.LINE_AA)


def knob_readout(knob: Knob, slot: int, hand: str, fx_index: int,
                 morphing: bool) -> tuple[list[str], tuple[int, int, int]]:
    """Build the text block and accent colour for one knob."""
    state = "[GRAB]" if knob.grabbed else "[ -- ]"
    if morphing:
        i, f = morph_position(knob.display)
        name_lo, _, color_lo = morph_station(i)
        name_hi, _, color_hi = morph_station(i + 1)
        # Blend the two stations' accents so the panel colour crossfades too.
        color = tuple(int(a * (1 - f) + b * f) for a, b in zip(color_lo, color_hi))
        lines = [
            f"KNOB {slot} :: {hand} {state}",
            f"FX  {SHORT[name_lo]} >> {SHORT[name_hi]}   [{i}/{len(EFFECTS)}]",
            f"MIX {hud.bar(f)} {int(round(f * 100)):3d}%",
        ]
    else:
        name, _, color = EFFECTS[fx_index]
        lines = [
            f"KNOB {slot} :: {hand} {state}",
            f"FX  {name}",
            f"LVL {hud.bar(knob.display)} {int(round(knob.display * 100)):3d}%",
        ]
    lines.append(f"TRC {hud.trace(knob.history)}")
    return lines, color


def draw_status(canvas: np.ndarray, fps: float, mode: str, morphing: bool,
                swapped: bool, seen: int) -> None:
    lines = [
        f"PINCH VISION            {fps:5.1f} FPS",
        f"HANDS {seen}   INPUT {mode.upper():<7} "
        f"MAP {'MORPH' if morphing else 'FIXED'}"
        + ("   SWAPPED" if swapped else ""),
    ]
    hud.panel(canvas, (16, 14), lines, (235, 235, 235), scale=0.85, alpha=0.5)


# --------------------------------------------------------------------- main

def _rgb_of(name: str) -> tuple[int, int, int]:
    """Terminal RGB for an effect name (the HUD stores BGR)."""
    for fx_name, _, bgr in EFFECTS:
        if fx_name == name:
            return (bgr[2], bgr[1], bgr[0])
    return (200, 200, 200)


def next_station(value: float) -> float:
    """Knob value of the next pure station, wrapping past halftone to clean."""
    n = len(EFFECTS)
    k = int(round(float(np.clip(value, 0.0, 1.0)) * n))
    return ((k + 1) % (n + 1)) / n


def build_landmarker(num_hands: int) -> vision.HandLandmarker:
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"Missing model: {MODEL_PATH}\n"
            "Download it with:\n"
            "  curl -L -o models/hand_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = vision.HandLandmarkerOptions(
        # Force the CPU delegate: MediaPipe 1.x on macOS aborts trying to
        # bring up its Metal helper inside a plain Python process.
        base_options=BaseOptions(model_asset_path=MODEL_PATH,
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(index: int, width: int, height: int,
                log: Console) -> cv2.VideoCapture:
    log.log("CAM", f"opening camera {index} ...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {index}.\n"
            "If you saw 'not authorized to capture video', macOS is blocking the\n"
            "program that launched this script. Open System Settings -> Privacy &\n"
            "Security -> Camera, enable the app you ran it from (Terminal, iTerm,\n"
            "VS Code, ...), fully quit that app, then run again.\n"
            "If you have several cameras, try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = cap.get(cv2.CAP_PROP_FPS) or 0.0
    log.log("CAM", f"camera {index} -> {got_w}x{got_h}"
                   + (f" @ {rate:.0f} fps" if rate else "")
                   + f" [{cap.getBackendName()}]")
    if (got_w, got_h) != (width, height):
        log.log("WARN", f"asked for {width}x{height}, device gave "
                        f"{got_w}x{got_h} — using what it gave")
    return cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--window-scale", type=float, default=1.5,
                    help="window size relative to the capture resolution")
    ap.add_argument("--mode", choices=["twist", "squeeze"], default="twist")
    ap.add_argument("--morph", action="store_true",
                    help="start with both knobs sweeping the whole rack")
    ap.add_argument("--left", type=int, default=0, help="effect index for the left hand")
    ap.add_argument("--right", type=int, default=1, help="effect index for the right hand")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the terminal telemetry")
    args = ap.parse_args()

    log = Console(enabled=not args.quiet)
    log.banner()

    cap = open_camera(args.camera, args.width, args.height, log)

    t_model = time.perf_counter()
    landmarker = build_landmarker(num_hands=2)
    log.log("MDL", f"hand landmarker ready in "
                   f"{(time.perf_counter() - t_model) * 1000:.0f} ms "
                   f"(CPU delegate, video mode, 2 hands)")

    knob_a, knob_b = Knob("A"), Knob("B")
    fx_a, fx_b = args.left % len(EFFECTS), args.right % len(EFFECTS)
    log.rack(EFFECTS, fx_a, fx_b)
    log.controls()

    mode = args.mode
    morphing = args.morph
    swapped = False
    show_skeleton = True
    win_scale = max(0.5, args.window_scale)
    fullscreen = False
    cache: EffectCache | None = None
    fps, last_t, t0 = 0.0, time.perf_counter(), time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(args.width * win_scale), int(args.height * win_scale))
    log.log("VIEW", f"window at {win_scale:.1f}x "
                    f"({int(args.width * win_scale)}x{int(args.height * win_scale)})")
    log.log("OK", "running — tap thumb+pinky to change an effect, "
                  "pinch index+thumb to turn the knob")

    frames = 0
    last_stat = time.perf_counter()
    seen_before: set[str] = set()
    grabbed_before = {1: False, 2: False}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera returned no frame; stopping", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)            # mirror: moving right moves right
        h, w = frame.shape[:2]
        if cache is None or cache.size != (w, h):
            cache = EffectCache(w, h)

        det = frame
        if w > DETECT_WIDTH:
            det = cv2.resize(frame, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)),
                             interpolation=cv2.INTER_AREA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
        # VIDEO mode wants strictly increasing timestamps to track across frames.
        result = landmarker.detect_for_video(mp_image,
                                             int((time.perf_counter() - t0) * 1000))

        # The frame is mirrored, so MediaPipe's label is the opposite of the
        # user's real hand — flip it back so "left" means their left hand.
        states: dict[str, HandState] = {}
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            side = "right" if handed[0].category_name == "Left" else "left"
            prior = knob_a if (side == "left") != swapped else knob_b
            states[side] = read_hand(to_pixels(lms, w, h), prior.grabbed,
                                     prior.was_cycling)

        for side in states.keys() - seen_before:
            log.log("HAND", f"{side} hand acquired")
        for side in seen_before - states.keys():
            log.log("HAND", f"{side} hand lost")
        seen_before = set(states.keys())

        src_a, src_b = states.get("left"), states.get("right")
        if swapped:
            src_a, src_b = src_b, src_a
        knob_a.update(src_a, mode)
        knob_b.update(src_b, mode)

        # Thumb-to-pinky tap: step this hand's effect, no keyboard involved.
        now_t = time.perf_counter()
        for slot, knob, src in ((1, knob_a, src_a), (2, knob_b, src_b)):
            if not knob.tapped(src, now_t):
                continue
            if morphing:
                knob.glide_to(next_station(knob.value))
                i, _ = morph_position(knob.value)
                name = morph_station(i if knob.value > 0 else 0)[0]
                log.log("TAP", f"K{slot} thumb+pinky -> gliding to "
                               f"{log._c(name, _rgb_of(name))}")
            elif slot == 1:
                fx_a = (fx_a + 1) % len(EFFECTS)
                log.log("TAP", f"K1 thumb+pinky -> "
                               f"{log._c(EFFECTS[fx_a][0], _rgb_of(EFFECTS[fx_a][0]))}")
            else:
                fx_b = (fx_b + 1) % len(EFFECTS)
                log.log("TAP", f"K2 thumb+pinky -> "
                               f"{log._c(EFFECTS[fx_b][0], _rgb_of(EFFECTS[fx_b][0]))}")

        for slot, knob in ((1, knob_a), (2, knob_b)):
            if knob.grabbed and not grabbed_before[slot]:
                log.log("GRAB", f"K{slot} engaged at {knob.display * 100:.0f}%")
            elif not knob.grabbed and grabbed_before[slot]:
                log.log("REL", f"K{slot} released at {knob.display * 100:.0f}%")
            grabbed_before[slot] = knob.grabbed

        if morphing:
            out = morph(frame, knob_a.display, cache)
            out = morph(out, knob_b.display, cache)
        else:
            out = EFFECTS[fx_a][1](frame, knob_a.display, cache)
            out = EFFECTS[fx_b][1](out, knob_b.display, cache)

        hand_a = "RIGHT HAND" if swapped else "LEFT HAND"
        hand_b = "LEFT HAND" if swapped else "RIGHT HAND"
        lines_a, color_a = knob_readout(knob_a, 1, hand_a, fx_a, morphing)
        lines_b, color_b = knob_readout(knob_b, 2, hand_b, fx_b, morphing)

        if show_skeleton:
            for side, state in states.items():
                is_a = (side == "left") != swapped
                draw_hand(out, state, color_a if is_a else color_b)

        # Both panels hang off the bottom edge, above the key legend. The text
        # scale is chosen so the pair always fits the frame width, whatever the
        # effect names happen to be.
        scale = hud.fit_scale([lines_a, lines_b], int(w * 0.78))
        aw, ah = hud.panel_size(lines_a, scale)
        bw, bh = hud.panel_size(lines_b, scale)
        base = h - max(ah, bh) - 26
        hud.panel(out, (16, base), lines_a, color_a, scale=scale, accent_rows=(1,))
        hud.panel(out, (w - bw - 16, base), lines_b, color_b, scale=scale,
                  accent_rows=(1,))

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
        last_t = now
        frames += 1
        if now - last_stat >= 2.0:
            last_stat = now
            slots = []
            for slot, knob, fx in ((1, knob_a, fx_a), (2, knob_b, fx_b)):
                if morphing:
                    i, f = morph_position(knob.display)
                    nm, _, col = morph_station(i + (1 if f > 0.5 else 0))
                else:
                    nm, _, col = EFFECTS[fx]
                slots.append((slot, nm, knob.display, col, knob.grabbed))
            log.stat(fps, len(states), slots)
        draw_status(out, fps, mode, morphing, swapped, len(states))
        hud.mono(out, KEYS, 18, h - 9, 0.85, (150, 150, 150))

        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("1"):
            fx_a = (fx_a + 1) % len(EFFECTS)
            log.log("FX", f"K1 -> {log._c(EFFECTS[fx_a][0], _rgb_of(EFFECTS[fx_a][0]))}")
        elif key == ord("2"):
            fx_b = (fx_b + 1) % len(EFFECTS)
            log.log("FX", f"K2 -> {log._c(EFFECTS[fx_b][0], _rgb_of(EFFECTS[fx_b][0]))}")
        elif key == ord("c"):
            morphing = not morphing
            log.log("MODE", f"map = {'MORPH (knobs sweep the rack)' if morphing else 'FIXED'}")
        elif key == ord("h"):
            swapped = not swapped
            log.log("MODE", f"hands {'swapped' if swapped else 'normal'}")
        elif key == ord("m"):
            mode = "squeeze" if mode == "twist" else "twist"
            log.log("MODE", f"input = {mode}")
        elif key == ord("r"):
            knob_a.reset()
            knob_b.reset()
            log.log("MODE", "knobs reset to 0%")
        elif key == ord("l"):
            show_skeleton = not show_skeleton
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if not fullscreen:
                cv2.resizeWindow(WINDOW, int(w * win_scale), int(h * win_scale))
            log.log("VIEW", f"fullscreen {'on' if fullscreen else 'off'}")
        elif key in (ord("+"), ord("=")) and not fullscreen:
            win_scale = min(3.0, win_scale + 0.1)
            cv2.resizeWindow(WINDOW, int(w * win_scale), int(h * win_scale))
            log.log("VIEW", f"window {win_scale:.1f}x "
                            f"({int(w * win_scale)}x{int(h * win_scale)})")
        elif key == ord("-") and not fullscreen:
            win_scale = max(0.5, win_scale - 0.1)
            cv2.resizeWindow(WINDOW, int(w * win_scale), int(h * win_scale))
            log.log("VIEW", f"window {win_scale:.1f}x "
                            f"({int(w * win_scale)}x{int(h * win_scale)})")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    log.outro(frames, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
