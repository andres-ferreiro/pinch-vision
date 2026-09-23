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

Three toggles sit in the top bar, and all three persist in `localStorage`:

* **SKEL** — draw the tracked hand skeleton over the picture, with a pinch ring
  that closes as your fingers meet. It is a separate 2D canvas stacked on the
  WebGL one, so the effects never have to know about it.
* **BLEND** — replaces per-hand effects with a two-handed instrument: one hand
  **sweeps** the rack, the other **fades** the whole chain back toward the clean
  camera. The sweep is rate-limited to `SWEEP_RATE` (0.22/s, so a full pass
  takes ~4.5s) because mapping eight effects straight onto one pinch turned
  every twitch into a jump cut.
* **WINDOW** (a gesture, not a button) — touch your two index fingertips and
  your two thumb tips together and a window **opens**. Spread your hands and the
  masked region opens out with them: it is the convex hull of those same four
  fingertips, so the quad tracks your hands wherever they go.

  **It only opens.** Closing used to be the same gesture, and that was the flaw:
  while shaping a window your hands pass near each other constantly, and any of
  those passes read as a second edge and shut it off by itself. Dismissing is
  the twist, which cannot happen by accident.

  The two gaps are **averaged** rather than both having to pass — lining up four
  fingertips at once is fussy, and one pair landing squarely should carry the
  other being a little wide. `handOpen()` is deliberately generous too, since a
  hand tilted toward the camera foreshortens; it only has to rule out a curled
  hand, not certify a flat one.

  While the window is up, thumb and index are busy holding the frame, so
  **strength moves to the middle finger**: extended is full, curled into the
  palm is off. The cards read `WIN`.

* **Thumb to the back fingers** does two jobs, told apart by *how long you
  hold it* rather than by which fingertip you touch:

  * **touch and release** — that hand steps to the next effect
  * **touch and hold one second** — the window pins into the scene, with the
    effects it had, and your hands are free for the next one (up to
    `MAX_PINNED`, 4)

  While you hold, a ring fills at the contact point on your hand and the top bar
  counts up `PINNING 60%`.

  **Duration, not finger identity, is what separates them.** Ring and pinky
  travel together — curling the pinky to meet the thumb drags the ring along, and
  in a 2D projection either can read as nearer — so asking the tracker which one
  the thumb is on is a coin toss, and it stopped the effect tap from working at
  all. `min(thumb→ring, thumb→pinky)` gets used instead, and the clock decides.

  There is **no upper limit on a tap**: any release that did not reach a pin
  counts as one. An earlier 600 ms cutoff left a dead zone where holding a shade
  too long did nothing, which feels identical to the gesture being broken.

  Verified: quick tap with and without a window → effect steps, no pin; an
  awkward 800 ms hold → effect steps, no pin; a full hold → pins and leaves the
  effect alone, including on release; three taps in a row → three effects.

* **Turn both palms away from the camera and hold a second** to dismiss. It is
  context sensitive: if you are holding a live window it drops that one and
  leaves your pinned ones alone; if you are not, it wipes every pinned window. The badge counts up
  `CLEARING 60%` while you hold.

  Orientation is the one property of a hand that no other gesture here uses, so
  it cannot collide with anything. It is read as the sign of the
  wrist→index-knuckle × wrist→pinky-knuckle cross product, which flips when a
  hand turns over.

  **Nothing assumes which sign means "palm forward"** — that depends on which
  hand it is and whether the image is mirrored. Each hand's usual orientation is
  learned at runtime and a *departure* from it is the gesture, so it
  self-calibrates. Edge-on hands report no opinion (`PALM_EPS`) rather than a
  coin-flip sign, and the gesture re-arms only once the hands come back, so
  turning over and back is one clear rather than two.

  Verified: three seconds of palms-toward-camera clears nothing; 600 ms turned
  away charges to 59% without firing; 1100 ms clears everything; a 400 ms flip
  clears nothing.

  A `CLEAR n` pill in the top bar does the same by hand — tap removes the most
  recent window, press and hold removes them all.

  `?debug=1` shows both thumb distances and the live hold time per hand.
* **⇄** — swap which hand drives which card, carrying each hand's chosen effect
  across with it. MediaPipe reports handedness as if the frame were already
  mirrored; this covers devices that disagree.

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

### Installing it as an app

`docs/manifest.webmanifest` plus `docs/sw.js` make it installable — Add to Home
Screen on iOS, Install on Android/Chrome — where it runs full screen with no
browser chrome, which is worth a lot when the whole interface is your hands.

The service worker splits its caching by what the file *is*:

* the app's own files are **network-first**, so a deploy is never masked by a
  stale cache (the module imports also carry a `?v=` you bump on release)
* the CDN payload — MediaPipe's WASM and the ~8 MB landmark model — is
  **cache-first**, because it is immutable and version-pinned. After one visit
  the tracker starts warm, which is the difference between a six second wait and
  an instant one on a phone.

Icons are generated at `docs/icon-*.png` (192, 512, and a 512 maskable with the
safe-zone padding Android needs), plus a 180px `apple-touch-icon.png`.

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
