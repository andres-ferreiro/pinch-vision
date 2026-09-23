# pinch vision

**Live in the browser → https://andres-ferreiro.github.io/pinch-vision/**
(works on a phone; camera stays on your device)

Webcam toy: MediaPipe tracks both hands, and **each hand is a knob**. Pinch thumb
and index together to grab your knob, then twist your hand to dial an effect in
or out. Left hand drives knob 1, right hand knob 2; each knob can be pointed at
any effect in the rack, and the two stack.

## The rack

| effect | what it is |
| --- | --- |
| `THERMAL` | ironbow false colour driven by a skin-weighted heat estimate |
| `NIGHT VISION` | auto-gain light amplification, grain, scanlines, phosphor green |
| `EDGE GLOW` | neon wireframe from gradient magnitude, bloomed |
| `ECHO TRAILS` | magenta ghosts of whatever moved, decaying over time |
| `GLITCH` | RGB channel separation plus displaced horizontal slices |
| `ASCII` | one glyph per cell, tinted with that cell's own colour |
| `HALFTONE` | posterised colour knocked out by clustered print dots |
| `INVERT` | photo negative, reached by sweeping a solarization threshold |

**Tap your thumb to your pinky** to step that hand to the next effect — no
keyboard. (`1` and `2` still do the same thing if you prefer keys.) `H` swaps
which hand owns which knob. Stacking order is knob 1 then knob 2, so ASCII on knob 2 renders the
thermal image as text, while ASCII on knob 1 gets heat-mapped.

The tap is measured the same scale-invariant way as the pinch — thumb-to-pinky
distance over wrist-to-knuckle span — with its own hysteresis, a 0.45 s cooldown
so one deliberate tap fires once, and a guard that the index finger is *not*
pinching, since a closed fist also puts thumb and pinky together.

### Morph mode (`C`)

Instead of each knob holding one effect, the knob position sweeps the *whole*
rack and dissolves between neighbours:

```
clean -> thermal -> night vision -> edge glow -> echo trails -> glitch -> ascii
      -> halftone -> invert
```

Both hands morph at once, so one hand can sit mid-dissolve between edge glow and
echo trails while the other pushes through glitch. Tapping thumb-to-pinky here
*glides* to the next pure effect instead of jumping — the knob retargets and the
existing smoothing eases it across over a few frames. Only the two stations either
side of the knob are evaluated, and the crossfade is smoothstepped so the dwell
at each pure effect feels longer than the transition. Worst case (both hands
mid-dissolve between the two heaviest effects) is ~33 ms/frame; parked on a
single effect it costs exactly what fixed mode does.

## Run

```bash
./run.sh
```

First run creates `.venv`, installs the deps and downloads the hand-landmark
model (~7 MB) into `models/`. After that it just starts.

Manual equivalent:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
./.venv/bin/python main.py
```

### macOS camera permission

The first run will likely fail with `not authorized to capture video`. macOS
grants camera access to the **app that launched the script**, so open
System Settings → Privacy & Security → Camera, enable your terminal
(Terminal / iTerm / VS Code), fully quit and reopen it, then run again.

## Controls

| key | action |
| --- | --- |
| thumb + pinky tap | step that hand to the next effect |
| `1` / `2` | same, from the keyboard |
| `C` | toggle morph mode (knobs sweep the whole rack) |
| `H` | swap which hand owns which knob |
| `+` / `-` | grow / shrink the window |
| `F` | fullscreen |
| `M` | switch knob mode (twist ↔ squeeze) |
| `R` | reset both knobs to 0 |
| `L` | toggle the hand skeleton overlay |
| `Q` / `Esc` | quit |

Flags: `--camera 1`, `--width/--height`, `--mode squeeze`, `--morph`,
`--window-scale 2.0`, `--quiet`, and `--left/--right N` to pick each knob's
starting effect.

The window opens at 1.5x the capture resolution by default (1920x1080 from a
720p camera) — capture stays at 720p because effect cost scales with pixels, and
the upscale is free. Raise `--window-scale` or press `+` for more.

## The two knob modes

**twist** (default) — pinching *grabs* the dial; the rotation of your hand
(measured as the angle from wrist to the pinch point) turns it. About 150° of
twist sweeps 0→100%. Release the pinch and the value stays where you left it,
like a real knob.

**squeeze** — no grabbing; how open your pinch is maps straight onto the value.
Fingers together = 100%, spread = 0%. Snappier, but it never holds a value.

Pinch detection normalizes the thumb↔index distance by the wrist→middle-knuckle
length, so the threshold doesn't change as you move toward or away from the
camera, and it uses separate close/open thresholds (0.42 / 0.55) so a value
hovering on the edge doesn't flicker.

## Terminal output

The run logs what it is actually doing — device negotiation, model load, every
gesture the tracker resolves, and a status line every two seconds:

```
[   0.412s] CAM   camera 0 -> 1280x720 @ 30 fps [avfoundation]
[   0.626s] MDL   hand landmarker ready in 214 ms (CPU delegate, video mode, 2 hands)
[   3.150s] HAND  left hand acquired
[   4.002s] TAP   K1 thumb+pinky -> EDGE GLOW
[   5.771s] GRAB  K2 engaged at 12%
[   8.004s] STAT   28.4 fps | hands 2 | K1 EDGE GLOW ████░░░░░░ 41% | K2 NIGHT ███████░░░ 72%
[  22.110s] EXIT  631 frames in 22.1s (28.6 fps average)
```

Colours are 24-bit escapes taken from each effect's HUD accent, so the terminal
and the window agree on what THERMAL looks like. They turn themselves off when
stdout is not a TTY or `NO_COLOR` is set; `--quiet` silences the whole thing.

## Web build (`docs/`)

The browser version is the same instrument with a simpler control scheme: it is
**pinch-only** — how closed your pinch is *is* the effect strength, no grabbing
or twisting — plus the thumb-to-pinky tap to change effect, and clip recording.

| file | what's in it |
| --- | --- |
| `docs/index.html` | the page shell |
| `docs/app.js` | camera, MediaPipe, gesture maths, UI, recording |
| `docs/effects.js` | WebGL2 renderer: all eight effects as fragment shaders |
| `docs/style.css` | mobile-first dark UI |
| `docs/selftest.html` | renders every shader against a synthetic frame and reports timings — open it after changing a shader |

The gesture logic is a direct port (same normalisation, same hysteresis, same
cooldown). The effects are not: numpy per-pixel work does not exist in the
browser at speed, so each one is a fragment shader. Notable differences:

* **Echo trails keeps its state in a ping-pong framebuffer pair**, because a
  shader cannot read and write one texture in a single pass. The state texture
  packs the trail in `r` and the previous frame's luma in `g`, which removes a
  whole copy pass, and it is seeded on first use — diffing against an empty
  texture marks every pixel as motion and washes the frame.
* **Scene averages come from mipmaps.** Thermal's auto-range and night vision's
  auto-exposure need a mean brightness; sampling the 1x1 mip with `textureLod`
  gives it for free, with no readback stall.
* **The ASCII glyph atlas is drawn once to a 2D canvas** and uploaded as a
  texture, so a frame is one texture sample per pixel rather than thousands of
  draw calls.

Recording uses `canvas.captureStream()` into `MediaRecorder`, capped at 20s
(`MAX_RECORD_MS` in `app.js`). It prefers `video/mp4` where supported — Safari
and iOS — and falls back to WebM, then offers Save plus the native share sheet.

### Deploying

GitHub Pages serves this repo from `main` → `/docs`, which is why the folder is
named that. For Vercel, import the repo and set **Root Directory** to `docs`
with no build command. Any static host works; the only requirement is **HTTPS**,
since browsers refuse camera access otherwise (localhost is exempt, so
`python3 -m http.server --directory docs` is fine for local work).

## Layout

Desktop app:

| file | what's in it |
| --- | --- |
| `main.py` | capture loop, MediaPipe wiring, key handling, panel layout |
| `hud.py` | the on-screen overlay: fixed-pitch text, bars, ASCII frames |
| `console.py` | the terminal telemetry: banner, event log, status lines |
| `hands.py` | pinch measurement, hysteresis, the `Knob` interaction model |
| `effects.py` | thermal and night-vision looks, with cached LUTs/masks |

Every effect has the signature `(img, amount, cache) -> img` and blends back over
what it was given, so `amount = 0` is a pass-through and the two knobs compose by
simple function chaining. Each takes the *running* image, not the original, which
is why stacking reads as one effect applied to the other rather than a crossfade.

Two details worth knowing:

* **Thermal does not map brightness to heat.** That is the naive version, and it
  is why a lit wall came out white-hot while dark hair went cold. Skin separates
  cleanly from almost everything else in the Cr-Cb chroma plane, so heat is
  estimated mostly from chroma, auto-ranged per frame (smoothed, so the palette
  does not pump), and computed at 1/6 resolution — which also buys the blobby
  low-res quality real thermal sensors have.
* **Invert never passes through flat grey.** Crossfading an image with its own
  negative is 50% grey at the midpoint — the dullest thing a knob can do. The
  dial sweeps an inversion *threshold* down from white instead, so it walks
  highlights, then midtones, then a full negative, solarizing on the way. It is
  a plain 256-entry LUT per knob position (cached, 32 steps), which makes it the
  cheapest effect in the rack at 0.2 ms.
* **Night vision rolls off instead of clipping.** Multiplying by a gain pins a
  bright room to 255 instantly; `1 - exp(-gain * x)` approaches white
  asymptotically, so highlights compress the way an intensifier tube does.

## Notes

* Pinned to `mediapipe==0.10.35`. Version 1.x installs fine on Python 3.14 but
  its macOS hand-landmarker graph aborts in a plain Python process
  (`DrishtiMetalHelper ... Service is unavailable`).
* MediaPipe runs on the CPU delegate, on a 640px-wide copy of the frame —
  landmarks come back normalized, so they map straight onto the full-res image.
  Effects cost roughly 2-15 ms each at 720p (glitch cheapest, thermal dearest).
  Drop to `--width 960 --height 540` if the frame rate bothers you.
* The frame is mirrored for a natural selfie view, which inverts MediaPipe's
  handedness label — `main.py` flips it back so "left" means your left hand.
