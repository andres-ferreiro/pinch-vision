"""The effect rack.

Every look is a function `(img_bgr, amount, cache) -> img_bgr` where `amount` is
0..1 and 0 means "return the frame untouched". They take the *running* image, not
the original, so stacking two knobs composes them (ASCII over thermal, glitch
over night vision, ...).

Anything expensive — colour ramps, vignettes, noise, the glyph atlas — is built
once in `EffectCache` and reused, so per-frame work is a handful of uint8 ops.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------- lookup tables

def _ramp(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Build a 256-entry BGR lookup table from (position, colour) stops."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, 256)
    pos = np.array([s[0] for s in stops])
    for ch in range(3):
        vals = np.array([s[1][ch] for s in stops], dtype=np.float32)
        lut[:, 0, ch] = np.interp(xs, pos, vals).astype(np.uint8)
    return lut


# Ironbow: the palette actual thermal cameras ship with.
THERMAL_LUT = _ramp([
    (0.00, (40, 8, 12)),      # cold: deep blue-black
    (0.16, (110, 20, 30)),    # blue
    (0.34, (140, 25, 110)),   # violet
    (0.52, (90, 40, 200)),    # magenta-red
    (0.70, (30, 110, 255)),   # orange
    (0.86, (70, 220, 255)),   # yellow
    (1.00, (255, 255, 255)),  # white hot
])

NIGHT_LUT = _ramp([
    (0.00, (0, 0, 0)),
    (0.45, (25, 120, 35)),
    (0.80, (60, 225, 80)),
    (1.00, (200, 255, 210)),
])

EDGE_LUT = _ramp([
    (0.00, (0, 0, 0)),
    (0.40, (90, 40, 10)),
    (0.75, (255, 180, 40)),
    (1.00, (255, 255, 235)),
])

ECHO_LUT = _ramp([
    (0.00, (0, 0, 0)),
    (0.35, (150, 20, 90)),
    (0.62, (255, 60, 180)),
    (0.88, (255, 150, 235)),
    (1.00, (255, 235, 255)),
])

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 8x8 clustered-dot matrix: ordered dithering that clumps into round dots, which
# is what makes it read as print halftone instead of noise.
_HALFTONE = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37],
    [8,  0,  2, 14, 45, 59, 61, 51],
    [22,  6,  4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [34, 46, 48, 36, 25, 11, 13, 27],
    [44, 58, 60, 50,  9,  1,  3, 15],
    [42, 56, 62, 52, 23,  7,  5, 17],
    [32, 40, 54, 38, 31, 21, 19, 29],
], dtype=np.float32) * 4.0

ASCII_CHARS = " .:-=+*#%@"


class EffectCache:
    """Per-resolution scratch space. One instance lives for the whole session."""

    def __init__(self, width: int, height: int, noise_frames: int = 12):
        self.size = (width, height)
        self.rng = np.random.default_rng(7)

        # Radial falloff, 1.0 at the centre -> ~0.25 in the corners.
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width / 2.0, height / 2.0
        r = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
        self.vignette = np.clip(1.15 - 0.62 * r ** 2, 0.25, 1.0).astype(np.float32)

        lines = np.ones((height, 1), dtype=np.float32)
        lines[::3] = 0.82
        self.scanlines = np.repeat(lines, width, axis=1)

        # Noise at third-res and upscaled: cheap, and the grain reads as sensor
        # noise rather than salt-and-pepper pixels.
        small = (max(1, height // 3), max(1, width // 3))
        self.noise = [
            cv2.resize(self.rng.normal(0.0, 1.0, small).astype(np.float32),
                       (width, height), interpolation=cv2.INTER_LINEAR)
            for _ in range(noise_frames)
        ]
        self._noise_i = 0

        # Tiled halftone threshold field.
        reps = (height // 8 + 1, width // 8 + 1)
        self.halftone = np.tile(_HALFTONE, reps)[:height, :width]

        # Solarize curves, built on demand and keyed by quantised knob position.
        self.invert_luts: dict[int, np.ndarray] = {}

        # Thermal auto-range, smoothed over time so the palette doesn't pump.
        self.heat_lo, self.heat_hi = 0.0, 1.0
        # Night-vision auto gain, and the echo effect's trail accumulators.
        self.nv_gain = 1.0
        self.echo: np.ndarray | None = None
        self.echo_prev: np.ndarray | None = None

        self.ascii_atlas, self.cell = _build_ascii_atlas()

    def next_noise(self) -> np.ndarray:
        self._noise_i = (self._noise_i + 1) % len(self.noise)
        return self.noise[self._noise_i]


def _build_ascii_atlas(cw: int = 10, ch: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    """Pre-render each glyph into a (n, ch, cw) mask, so drawing the ASCII frame
    is one fancy-index instead of thousands of putText calls."""
    atlas = np.zeros((len(ASCII_CHARS), ch, cw), dtype=np.uint8)
    for i, char in enumerate(ASCII_CHARS):
        if char == " ":
            continue
        cv2.putText(atlas[i], char, (0, ch - 3), cv2.FONT_HERSHEY_PLAIN,
                    0.9, 255, 1, cv2.LINE_AA)
    return atlas, (cw, ch)


def _blend(base: np.ndarray, styled: np.ndarray, amount: float) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - amount, styled, amount, 0)


# ------------------------------------------------------------------- the looks

def thermal(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Ironbow false colour driven by a *skin-weighted* heat estimate.

    Mapping plain luminance to heat is what makes naive thermal look wrong: a
    white wall becomes the hottest thing in frame and dark hair goes cold. Skin
    sits far apart from almost everything else in the Cr-Cb chroma plane, so we
    estimate heat mostly from chroma and only lightly from brightness.
    """
    if amount <= 0.001:
        return img
    h, w = img.shape[:2]

    # Real thermal sensors are low-res and optically soft; working small gives us
    # that blobby quality for free and keeps the cost down.
    sw, sh = max(32, w // 6), max(24, h // 6)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]

    # Skin/warm-body score: reddish chroma, peaked around Cr ~ 150.
    skin = np.clip(1.0 - np.abs(cr - 152.0) / 28.0, 0.0, 1.0)
    skin *= np.clip((cr - cb + 20.0) / 55.0, 0.0, 1.0)
    skin = cv2.GaussianBlur(skin, (0, 0), 2.0)

    heat = 0.70 * skin + 0.30 * (y / 255.0) * (0.35 + 0.65 * skin)

    # Auto-range like a real camera, but smoothed: a sudden bright object
    # shouldn't re-colour the whole scene in one frame.
    lo, hi = np.percentile(heat, (4.0, 99.0))
    cache.heat_lo += (float(lo) - cache.heat_lo) * 0.15
    cache.heat_hi += (float(hi) - cache.heat_hi) * 0.15
    span = max(cache.heat_hi - cache.heat_lo, 1e-3)
    heat = np.clip((heat - cache.heat_lo) / span, 0.0, 1.0)

    # Edge detail from the real image, so faces keep their features.
    detail = _CLAHE.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    detail = cv2.Laplacian(detail, cv2.CV_32F, ksize=3) / 255.0
    heat = np.clip(heat + 0.10 * detail, 0.0, 1.0)

    heat = cv2.GaussianBlur(heat, (0, 0), 1.1)
    heat_u8 = cv2.resize((heat * 255).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    out = cv2.LUT(cv2.cvtColor(heat_u8, cv2.COLOR_GRAY2BGR), THERMAL_LUT)

    hot = cv2.threshold(heat_u8, 215, 255, cv2.THRESH_TOZERO)[1]
    bloom = cv2.GaussianBlur(cv2.cvtColor(hot, cv2.COLOR_GRAY2BGR), (0, 0), 9)
    out = cv2.addWeighted(out, 1.0, bloom, 0.35, 0)
    return _blend(img, out, amount)


def nightvision(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Light amplification: auto gain, soft-knee highlights, noise, phosphor.

    Straight multiplication clips instantly in a lit room — a white wall pins to
    255 and the whole frame goes flat. Real intensifier tubes roll off instead,
    so this gains toward a target exposure and compresses through
    `1 - exp(-gain*x)`, which approaches white asymptotically and never clips.
    """
    if amount <= 0.001:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Auto-exposure, smoothed so the image doesn't pump frame to frame.
    mean = float(gray.mean()) + 1e-3
    target = 0.34
    cache.nv_gain += (np.clip(target / mean, 0.35, 6.0) - cache.nv_gain) * 0.12
    amp = 1.0 - np.exp(-(cache.nv_gain * (0.9 + 1.6 * amount)) * gray)

    amp = amp * 255.0 + cache.next_noise() * (22.0 * amount)
    amp *= cache.vignette * cache.scanlines
    amp = np.clip(amp, 0, 255).astype(np.uint8)

    green = cv2.LUT(cv2.cvtColor(amp, cv2.COLOR_GRAY2BGR), NIGHT_LUT)
    glow = cv2.GaussianBlur(green, (0, 0), 6)
    green = cv2.addWeighted(green, 1.0, glow, 0.35, 0)
    return _blend(img, green, amount)


def edge_glow(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Neon wireframe: gradient magnitude on black, bloomed."""
    if amount <= 0.001:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.4)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag * (0.9 + 1.6 * amount) / 255.0, 0.0, 1.0)
    mag_u8 = (mag * 255).astype(np.uint8)

    neon = cv2.LUT(cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR), EDGE_LUT)
    glow = cv2.GaussianBlur(neon, (0, 0), 7)
    neon = cv2.addWeighted(neon, 1.0, glow, 0.8, 0)
    return _blend(img, neon, amount)


def echo(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Motion trails: a decaying record of *what moved*, screened back on top.

    A max-blend of past frames only shows up against a dark room. Accumulating
    frame-to-frame difference instead means the trail follows motion, so it reads
    the same whether you are lit by a window or a monitor.
    """
    if amount <= 0.001:
        cache.echo = cache.echo_prev = None
        return img
    h, w = img.shape[:2]
    # The trail layer is soft and bloomy, so half-res costs a quarter as much
    # and looks the same once it is blurred and upscaled.
    gray = cv2.cvtColor(cv2.resize(img, (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)
    if cache.echo_prev is None or cache.echo_prev.shape != gray.shape:
        cache.echo_prev = gray.copy()
        cache.echo = np.zeros_like(gray)

    motion = cv2.absdiff(gray, cache.echo_prev) * (1.1 + 2.0 * amount)
    cache.echo_prev = gray
    decay = 0.72 + 0.26 * amount          # longer tail as the knob opens
    cache.echo = np.maximum(cache.echo * decay, np.clip(motion, 0, 255))

    trail = cv2.GaussianBlur(cache.echo, (0, 0), 1.5).astype(np.uint8)
    neon = cv2.LUT(cv2.cvtColor(trail, cv2.COLOR_GRAY2BGR), ECHO_LUT)
    neon = cv2.addWeighted(neon, 1.0, cv2.GaussianBlur(neon, (0, 0), 5), 0.7, 0)
    neon = cv2.resize(neon, (w, h), interpolation=cv2.INTER_LINEAR)

    # Alpha-composite rather than screen: adding light to a bright room just
    # blows out to white and the trail loses its colour, so the ghost is painted
    # *over* the frame with its own opacity instead.
    alpha = cv2.resize(np.clip(cache.echo / 255.0, 0.0, 1.0), (w, h),
                       interpolation=cv2.INTER_LINEAR)
    alpha = (alpha * (0.55 + 0.45 * amount))[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + neon.astype(np.float32) * alpha
    return _blend(img, np.clip(out, 0, 255).astype(np.uint8),
                  min(1.0, 0.5 + 0.5 * amount))


def glitch(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Datamosh: RGB channel separation plus displaced horizontal slices."""
    if amount <= 0.001:
        return img
    h, w = img.shape[:2]
    out = img.copy()

    shift = int(amount * 0.025 * w)
    if shift:
        out[..., 2] = np.roll(out[..., 2], shift, axis=1)    # red right
        out[..., 0] = np.roll(out[..., 0], -shift, axis=1)   # blue left

    for _ in range(int(amount * 14)):
        y0 = int(cache.rng.integers(0, h - 8))
        y1 = min(h, y0 + int(cache.rng.integers(6, 70)))
        dx = int(cache.rng.integers(-1, 2) * cache.rng.integers(0, int(amount * 0.12 * w) + 1))
        out[y0:y1] = np.roll(out[y0:y1], dx, axis=1)
        if cache.rng.random() < 0.18:      # occasional blown-out band
            out[y0:y1] = cv2.convertScaleAbs(out[y0:y1], alpha=1.6, beta=40)

    return _blend(img, out, min(1.0, amount * 1.3))


def ascii_art(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Terminal view: one glyph per cell, tinted with the cell's own colour."""
    if amount <= 0.001:
        return img
    h, w = img.shape[:2]
    cw, ch = cache.cell
    cols, rows = max(1, w // cw), max(1, h // ch)

    small = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Stretch between percentiles so a flat bright wall doesn't saturate every
    # cell to '@' — the glyph range should track the scene, not absolute levels.
    lo, hi = np.percentile(gray, (6.0, 96.0))
    gray = np.clip((gray - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    idx = (gray * (len(ASCII_CHARS) - 1)).astype(np.int32)

    # One fancy-index builds every glyph at once, then a transpose stitches the
    # (rows, cols, ch, cw) block grid into a single (rows*ch, cols*cw) image.
    tiles = cache.ascii_atlas[idx]
    mask = tiles.transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    tint = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    tint = cv2.convertScaleAbs(tint, alpha=1.5, beta=25)
    styled = (tint.astype(np.float32) * (mask[..., None].astype(np.float32) / 255.0))
    styled = styled.astype(np.uint8)
    return _blend(img, styled, amount)


def halftone(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Pop-art print: posterised colour knocked out by clustered ink dots."""
    if amount <= 0.001:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=10).astype(np.float32)

    ink = (gray > cache.halftone).astype(np.float32)   # 1 = paper, 0 = ink dot
    posterized = (img // 52) * 52 + 26                 # 5 levels per channel
    paper = cv2.addWeighted(posterized, 0.75,
                            np.full_like(posterized, 255), 0.25, 0)
    styled = (paper.astype(np.float32) * ink[..., None]).astype(np.uint8)
    return _blend(img, styled, amount)


# Abbreviations for the HUD, where a morph line has to name two effects at once.
SHORT = {
    "THERMAL": "THERM", "NIGHT VISION": "NIGHT", "EDGE GLOW": "EDGE",
    "ECHO TRAILS": "ECHO", "GLITCH": "GLTCH", "ASCII": "ASCII",
    "HALFTONE": "HALFT", "INVERT": "INVRT", "CLEAN": "CLEAN",
}

def _solarize_lut(amount: float) -> np.ndarray:
    """Curve that inverts everything above a threshold which falls as `amount`
    rises: highlights first, then midtones, finally a full negative at 1.0."""
    x = np.arange(256, dtype=np.float32)
    threshold = 255.0 * (1.0 - amount)
    t = np.clip((x - threshold) / 26.0 + 0.5, 0.0, 1.0)   # soft knee, no banding
    t = t * t * (3.0 - 2.0 * t)
    y = np.clip(x * (1.0 - t) + (255.0 - x) * t, 0, 255).astype(np.uint8)
    return np.repeat(y.reshape(256, 1, 1), 3, axis=2)


def invert(img: np.ndarray, amount: float, cache: EffectCache) -> np.ndarray:
    """Photo negative, approached through solarization.

    Crossfading an image with its own negative passes through flat grey at 50%,
    which is the dullest possible midpoint for a knob. Sweeping an inversion
    *threshold* down from white instead means the dial walks highlights ->
    midtones -> full negative, and every position on the way looks like
    something.
    """
    if amount <= 0.001:
        return img
    # 32 steps is finer than the eye tracks on a moving image, and it keeps the
    # LUT table small enough to stay cached for the whole session.
    key = int(round(np.clip(amount, 0.0, 1.0) * 32))
    lut = cache.invert_luts.get(key)
    if lut is None:
        lut = _solarize_lut(key / 32.0)
        cache.invert_luts[key] = lut
    return cv2.LUT(img, lut)


# name, function, accent colour (BGR) for the HUD
EFFECTS: list[tuple[str, object, tuple[int, int, int]]] = [
    ("THERMAL",      thermal,     (60, 150, 255)),
    ("NIGHT VISION", nightvision, (120, 255, 120)),
    ("EDGE GLOW",    edge_glow,   (255, 190, 60)),
    ("ECHO TRAILS",  echo,        (255, 110, 220)),
    ("GLITCH",       glitch,      (90, 90, 255)),
    ("ASCII",        ascii_art,   (200, 255, 255)),
    ("HALFTONE",     halftone,    (230, 230, 230)),
    ("INVERT",       invert,      (255, 80, 140)),
]


def morph_station(index: int) -> tuple[str, object, tuple[int, int, int]]:
    """Station 0 is the untouched camera; station k is EFFECTS[k-1] at full."""
    if index <= 0:
        return ("CLEAN", None, (200, 200, 200))
    return EFFECTS[min(index, len(EFFECTS)) - 1]


def morph_position(value: float) -> tuple[int, float]:
    """Knob value -> (station index, crossfade into the next station)."""
    n = len(EFFECTS)
    p = float(np.clip(value, 0.0, 1.0)) * n
    i = min(int(p), n - 1)
    f = np.clip(p - i, 0.0, 1.0)
    return i, float(f * f * (3.0 - 2.0 * f))     # smoothstep the crossfade


def morph(img: np.ndarray, value: float, cache: EffectCache) -> np.ndarray:
    """Sweep the whole rack with one knob, crossfading between neighbours.

    Turning from 0 to 1 walks clean -> thermal -> night vision -> ... -> halftone,
    dissolving between each pair. Only the stations actually visible get
    evaluated, so a knob parked on one effect costs the same as fixed mode.
    """
    i, f = morph_position(value)
    _, fn_lo, _ = morph_station(i)
    low = img if fn_lo is None else fn_lo(img, 1.0, cache)
    if f < 0.02:
        return low
    _, fn_hi, _ = morph_station(i + 1)
    high = img if fn_hi is None else fn_hi(img, 1.0, cache)
    if f > 0.98:
        return high
    return cv2.addWeighted(low, 1.0 - f, high, f, 0)
