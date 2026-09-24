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
| `PIXELATE` | square cells on a flattened palette, both coarsening together |
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
  takes ~4.5s) because mapping the whole rack straight onto one pinch turned
  every twitch into a jump cut.
* **WINDOW** (a gesture, not a button) — touch your two index fingertips and
  your two thumb tips together and a window **opens**. Spread your hands and the
  masked region opens out with them: it is the convex hull of those same four
  fingertips, so the quad tracks your hands wherever they go.

  **It only opens.** Closing used to be the same gesture, and that was the flaw:
  while shaping a window your hands pass near each other constantly, and any of
  those passes read as a second edge and shut it off by itself. Dismissing is
  the fist reset, which cannot happen by accident.

  The two gaps are **averaged** rather than both having to pass — lining up four
  fingertips at once is fussy, and one pair landing squarely should carry the
  other being a little wide. `handOpen()` is deliberately generous too, since a
  hand tilted toward the camera foreshortens; it only has to rule out a curled
  hand, not certify a flat one.

  While the window is up, thumb and index are busy holding the frame, so
  **strength moves to the ring finger**: extended is full, curled into the palm
  is off. The cards read `WIN`.

  The middle finger used to have the job, and it sits right beside the index —
  which is the shutter pose — so dialling a window fired photos by accident. The
  dial moved to the ring finger and **pinning became pinky-only** to make room
  for it: `backTouch()` drops the ring from its reckoning while a window is up,
  or every adjustment would read as a thumb-to-back-finger tap. Only while a
  window is up, though — outside window mode the ring is idle and the
  `min(pinky, ring)` still earns its keep, because telling the two apart in a 2D
  projection is a coin toss and the minimum is what keeps the effect tap working
  at all. Pinning only ever happens with a window up, so pinning is pinky-only,
  full stop. The ring finger is shorter and further from the middle knuckle
  everything is measured against, so it gets its own band (`DIAL_OPEN` /
  `DIAL_SHUT`) rather than the middle finger's.

  Verified: a full sweep of the ring dial, closed to open, fires no tap and pins
  nothing while a window is up; the pinky still latches and pins on a 1s hold;
  a quick pinky tap outside window mode still steps the effect; and outside
  window mode the ring is still counted.

* **Thumb to the back fingers** does two jobs, told apart by *how long you
  hold it* rather than by which fingertip you touch:

  * **touch and release** — that hand steps to the next effect
  * **touch and hold one second** — the window pins into the scene, with the
    effects it had, and your hands are free for the next one (up to
    `MAX_PINNED`, 4). Pinning is **pinky-only**, since the ring finger is the
    window's strength dial

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

* **Press your index and middle fingertips together and hold 0.7s** to take a
  photo. The screen blinks white and the frame shrinks down into a film strip
  above the dock, Photo Booth style; tap any frame to open it. One
  hand is enough, so the other stays free to keep dialling the look you are
  photographing. The badge counts up `PHOTO 60%` while you hold.

  The shot is the **rendered** frame — effects, live window and pinned windows
  all included. The skeleton is drawn on a separate overlay canvas, so it is
  never in the picture. The WebGL context is created with
  `preserveDrawingBuffer`, which is what makes reading the canvas back safe at
  any point, and the capture is taken right after the draw so it lands on the
  frame that was actually on screen.

  **The fist test is relative, not absolute.** A fist puts those same two
  fingertips side by side, so "close together" alone would fire on every reset.
  The first version answered that by also demanding the two fingers be
  *extended* — and that was wrong twice over. People make this pose as a relaxed
  claw with the fingers bent, and a hand tilted toward the camera foreshortens,
  so the extension threshold was rarely met; it also had no hysteresis, so near
  its boundary it chattered frame to frame. The gesture almost never fired.

  What it compares now is two gaps on the same hand: the index-to-middle tip gap
  must be small *and* the middle-to-ring gap clearly wider (`SHUTTER_PART_ON`).
  Curl or tilt the hand and every distance shrinks together, leaving the
  comparison intact — so the test survives any amount of foreshortening. It also
  rejects a flat palm with all four fingers adjacent, which the extension test
  would have accepted. A thumb-to-index pinch is excluded by
  `pinch > PINCH_OPEN`, a hand mid thumb-to-back-finger contact is reaching for
  a different gesture and is skipped, and a hand framing a window holds index
  and middle together while the two hands touch across — which no single-hand
  measurement can tell apart, so the shutter stands down whenever the cross-hand
  contact is live.

  **The pose destroys the value it is photographing**, so the strength is taken
  from before the fingers moved. `strengthFrom()` maps the thumb-index gap to
  effect strength, and the shutter needs `pinch > PINCH_OPEN` to keep a pinch
  from being mistaken for the pose — substitute that threshold in and the effect
  is already down to 45% at the moment the pose becomes legal, and an open hand
  with index and middle together reads 0%. Dial a look to 100%, shoot it, and
  the photo came back blank.

  Each hand therefore keeps `TRAIL_MS` of recent strength, and the rising edge of
  the pose latches the value from `SHUTTER_LOOKBACK_MS` earlier — the same trick
  `PIN_LOOKBACK_MS` uses for a window ruined by the clench that pins it. The
  value is *restored*, not eased, because it decayed over the frames it took to
  recognise the pose and easing back would photograph the climb. Releasing the
  pose hands the dial back to the live pinch. Nothing is invented: a hand that
  was at 50% holds 50%, and one that was never pinching holds 0%.

  This also fixed an ordering bug. `updateShutter` ran *after* `renderer.draw`,
  so the restored strength would have landed a frame late and the capture would
  have read a canvas drawn at the decayed value. Detection and the strength
  restore now run before the draw, and the capture itself is deferred through
  `state.shotPending` until the frame it asked for is on screen.

  **The shutter stands down for the whole time a window is live.** While one is
  up, thumb and index hold the frame corners and strength moves to the middle
  finger — so dialling parks the middle fingertip right beside the index one,
  which is this pose exactly, and windows fired photos by accident.
  `state.contact` does not cover this: it only marks the moment the fingertips
  touch and goes false again as soon as the window latches open and the hands
  spread. Pin the window and the shutter comes back, with the pinned frame in
  the shot.

  The window dial has since moved to the ring finger, with pinning narrowed to
  the pinky to make room — see the WINDOW entry above.

  **A hold that one dropped frame can reset is a hold that never completes.**
  The tracker loses the pose for a frame here and there, and the first version
  cleared the charge the instant it did, restarting the 700 ms clock from zero.
  `SHUTTER_GRACE_MS` carries the hold across those gaps; a real release, longer
  than the grace, still resets and re-arms as before. Measured: with one frame
  in five losing the pose, the charge still reaches 91% in 640 ms — it reached
  nothing at all before.

  The strip keeps the last `MAX_SHOTS` (8) frames. Object URLs pin their blob in
  memory until revoked, so evicted frames are revoked as they fall off the end —
  an unbounded strip would be an unbounded leak. **A reset does not touch the
  photos**: windows are scratch state, but a photo is output.

  Thumbnails carry the frame's own aspect ratio, which is what lets the flying
  copy scale into place uniformly instead of squashing. The flyer and the flash
  are removed by their animation's `onfinish` *and* by an independent timeout —
  `onfinish` never arriving (a backgrounded tab defers animations, and a browser
  without `element.animate` never starts one) would otherwise leave a full-frame
  photo pasted over the UI. Both the flash and the flight are skipped under
  `prefers-reduced-motion`.

  Verified: 350 ms charges to 50% without firing; 700 ms captures; holding the
  pose does not machine-gun; separating the fingers re-arms; either hand works;
  a curled claw with the tips meeting is detected while a fist, a flat palm with
  the fingers together, a splayed hand, a peace sign, a hard pinch and a
  thumb-to-pinky tap are all rejected — each held 1.3 s without firing — and a
  fist still reads as a fist; the strip caps at 8; and the flight lands within a
  pixel of its slot.

  `?debug=1` prints `spr` (index-to-middle), `spl` (middle-to-ring) and `gap`
  (the margin between them) per hand, which are the three numbers this test
  decides on.

* **Tapping a frame opens the carousel** — a coverflow over the whole strip,
  with the selected photo square on and its neighbours turned away and pushed
  back. Drag it, swipe it, use the arrows or the arrow keys, click a card at the
  side to bring it forward; Escape or the backdrop closes it. Save writes a
  JPEG, and Share appears only where `navigator.canShare` accepts a file.

  Cards are laid out from a **fractional** position rather than a selected
  index. That is what lets a drag move the row with the finger instead of
  snapping only on release: the pointer offset is divided by a card's travel and
  fed straight into the same layout function, with the ends rubber-banded so the
  row cannot run off into space. The perspective lives on the container, not on
  the cards, so all of them share one vanishing point and the row reads as a
  single receding object rather than a fan of independently skewed rectangles.

  Pointer capture is wrapped in `try`/`catch` at both ends. `releasePointerCapture`
  throws when the pointer is no longer captured — which happens for real after a
  `pointercancel`, and optional chaining does not catch a throw — and letting it
  escape abandoned the drag before it was committed, leaving the row parked
  mid-swipe. Nothing about committing a drag depends on capture succeeding.

  Evicting a frame off the end of the strip shifts every index, so `photoIndex`
  is walked back with it; otherwise an open viewer would silently re-point at
  the wrong photo.

  Verified: opens at the tapped frame; arrows, arrow keys, side-card clicks and
  drags in both directions all move the selection; a nudge under the threshold
  does not; the ends clamp and disable their arrow; `pointercancel` followed by
  `pointerup` commits exactly once; Escape and the backdrop close while a click
  on a card does not; and closing drops the images so the decoded bitmaps go.

* **Close both hands into fists and hold a second** to reset. It drops the live
  window, drops every pinned one, and returns blend mode to fixed — one gesture
  that puts the scene back to plain. The badge counts up `RESETTING 60%` while
  you hold, and both effects fall to zero on the way in rather than on the way
  out.

  **This replaced turning both palms away**, which could not be made reliable:

  * Palm orientation was read as the sign of a 2D cross product, and rotating a
    palm *has* to sweep through edge-on, where that sign is genuinely noise.
    `updateTwist` treated that "no opinion" frame as both not-flipped and
    not-home, so it cleared the charge — you had to flip faster than the blind
    spot.
  * Each hand's baseline orientation was learned once and never revised, so a
    baseline captured from a tilted hand stayed inverted for the whole session
    and the gesture worked backwards.
  * A palm turned away is the pose MediaPipe tracks *worst*, so the gesture
    needed maximum confidence exactly where confidence was lowest.
  * It also read the *present* hands rather than both, so with one hand in frame
    a single flipped palm fired a reset the docs said needed two.

  A fist has none of these: it is a high-confidence pose, it needs no learned
  baseline, and there is no ambiguous state to pass through.

  **The thumb is what makes it safe.** Four curled fingers also describe a hard
  pinch — this app's primary gesture — so the test additionally requires the
  thumb home across the knuckles (`FIST_THUMB_ON`). Pinching holds thumb and
  index out in front of the palm; clenching brings both in. Every threshold has
  hysteresis (`FIST_ON`/`FIST_OFF`), held per hand, so a hand resting at the
  boundary settles instead of chattering the charge.

  **Both hands must be tracked**, not merely both fisted: a reset discards up to
  four pinned windows, so one hand leaving the frame must not be able to finish
  it. It re-arms only once both hands open again, so holding the fists longer is
  one reset rather than a repeating one.

  A `CLEAR n` pill in the top bar still removes windows by hand — tap removes the
  most recent, press and hold removes them all.

  `?debug=1` shows the two numbers the test decides on per hand — `fng` (the
  furthest of the four fingertips) and `thb` — plus a `FIST` flag and the live
  reset charge, so the thresholds can be tuned against a real hand.
* **⇄** — swap which hand drives which card, carrying each hand's chosen effect
  across with it. MediaPipe reports handedness as if the frame were already
  mirrored; this covers devices that disagree.

## Air controls, lock and drawing

A rail on the left edge carries three controls that work **two ways**: tap them,
or hold a fingertip over them in the camera. The air path is the same hit test a
mouse would do — each control's `getBoundingClientRect()` against the index
fingertip mapped into viewport pixels — so there is one set of controls, not a
touch set and a gesture set.

* **LOCK** fixes the current look. Strengths hold where they stand, the effect
  tap is refused, the window latch and the back-finger gestures stand down, and
  the blend sweep stops. Your hands are then free to move without the picture
  following them. The two-fist reset still works and unlocks as it goes — a
  lock with no escape hatch is a trap.
* **DRAW** turns a pinch into a pen: thumb to index puts it down at the index
  tip, opening the hand lifts it. That reuses the one measurement this app
  already trusts most rather than inventing another pose. A painting hand also
  stops driving the dial, or every stroke would drag the effect strength with
  it.
* **BRUSH** cycles `LINE → DASH → DOTS → ASCII`, borrowing the rack's own
  vocabulary rather than inventing a second one.
* **The hue slider** picks the pen colour. A dwell *grabs* it and then the
  fingertip tracks it continuously through a wider box than the one that
  grabbed it, so running the colour to either end does not drop the grab at the
  edge.

**The dwell is what makes air control safe.** A hand crossing the rail on the
way somewhere else passes through in a few frames and never commits, and only a
*pointing* hand is read at all — a pinching hand is drawing or dialling, not
aiming, so it is skipped. One dwell buys one action: the finger has to leave
before a control will take another, or resting on a toggle would flip it every
`AIR_DWELL_MS`. A reticle is drawn at the fingertip whether or not the skeleton
is on, because an air control with no visible cursor is a guessing game.

BRUSH and the colour slider only appear once the pen is out — they mean nothing
otherwise, and leaving them off keeps the rail short enough to clear the dock on
a landscape phone. That conditional display needs care on the air side: a
`display: none` element reports a **zero rect at the origin**, so `airRect()`
returns null for it rather than a padded box around (0, 0) that would make every
hidden control activatable by pointing at the top-left corner of the screen.

### Strokes

Two things make a stroke look drawn rather than sampled.

The pen **chases** the fingertip (`PEN_EASE`) instead of snapping to it. Raw
landmarks shake a pixel or two every frame and a polyline straight through them
reads as a seismograph.

Continuous brushes then **curve rather than corner**: each frame's segment is a
quadratic through the *previous* point, running from the midpoint before it to
the midpoint after. Successive segments share both endpoints and tangents, which
is what turns a chain of short lines into one smooth path — the standard
midpoint trick, and the reason the line no longer shows its frame rate.

Discrete brushes walk the segment at a fixed spacing instead, carrying the
leftover distance across frames so marks stay evenly spread however far the hand
moved between two of them. `DASH` sets its `lineDashOffset` from distance
travelled for the same reason: without it every frame restarts the pattern and
the line looks chewed rather than dotted.

### Layers

The drawing lives on its own 2D canvas between the picture and the skeleton, in
**screen** space — `object-fit: fill`, because the letterboxing that the view
canvas uses would shift every stroke off the picture.

That means the two layers do not share a coordinate space, which matters as soon
as a frame leaves the screen. Photos and recordings are composited: the paint
canvas is drawn into the rectangle of the view canvas that screen space maps to
(`-ox * rw/dw`, `-oy * rh/dh`, `cw * rw/dw`, `ch * rh/dh`), which is what keeps
a stroke on the same part of the picture in the file as it was under the finger.
Recording always streams the composite even when nothing is drawn yet, because
`captureStream` is bound to the canvas it is handed and the first stroke may
come later. Photos skip the copy entirely when nothing has been painted.

## Sound## Sound

Six samples in `docs/sounds/`, one per *kind* of event, mapped in one table at
the top of `app.js` so they can be swapped without hunting through the code:

| cue | sample | fires when |
|---|---|---|
| `step` | `tap_03.wav` | a hand steps to the next effect |
| `on` | `toggle_on.wav` | a window opens, a toggle comes on, recording starts |
| `off` | `toggle_off.wav` | a window is dismissed or reset, a toggle goes off, recording stops |
| `commit` | `select.wav` | a window is pinned, a photo is saved |
| `shutter` | `button.wav` | a photo is taken |
| `nope` | `disabled.wav` | the app declined — nothing to reset, capture failed, recording unsupported |

The rule that keeps this from becoming noise: **anything that repeats while a
hand simply moves stays silent.** The strength dial, the charge meters, the
carousel and the hint line make no sound at all, because a cue that can fire
every frame stops carrying information. What is left is the set of moments where
something happened that moving your hand back will not undo. `step` is the one
frequent cue, and it is also the shortest sample at 10 ms.

Playback is Web Audio with the buffers decoded once, not `<audio>` elements:
lower latency, and overlapping plays do not fight over one element. Audio may
only start inside a user gesture, so the context is created on the first
`pointerdown` — in practice the Start camera tap — and a sample that fails to
load simply never plays. Master gain is 0.3, and the same cue cannot retrigger
within 60 ms, or a gesture sitting on a threshold would rattle.

`♪` in the top bar mutes, and the choice is remembered. Making room for a fourth
toggle is why the wordmark now drops out below 460px rather than 360px.

Verified: silent before any gesture; the context opens on the first tap and all
six samples decode; the same cue is throttled while a different one is not;
muting silences playback and persists across a reload.

## Landscape

**iPhone Safari cannot do this at all.** It has no element Fullscreen API —
only `<video>` can go fullscreen there, via `webkitEnterFullscreen` — so
`requestFullscreen` does not exist to call and no amount of code will remove the
URL bar from a normal Safari tab. Installing to the home screen runs the app
without browser chrome, which is the same result by another route, so the app
says so once in landscape and then never again. Everywhere that *does* support
it:

A phone on its side goes fullscreen. A rotation is not a user gesture, so
`requestFullscreen` from the orientation change is usually refused — when it is,
the request is *armed* and the next touch spends it, which costs nothing because
using the app in landscape means touching it. Only what we entered is exited
again on rotating back; a fullscreen the user asked for is theirs to keep.
iPhone Safari has no element fullscreen at all, so there the installed PWA is
the route, which is what `display_override: ["fullscreen"]` in the manifest is
for.

The layout follows: below 540px of height in landscape the start screen uses the
same two-column split as the desktop one with everything wound in, and the top
bar, dock, record button, strip and photo viewer all give back the height that
landscape does not have to spare.

Filling the screen in landscape happens in two places, because the first one is
allowed to fail. `reshapeCamera()` asks the camera to turn with the phone, and
`fitFor()` fills the screen whatever shape the stream turns out to be — on a
phone held sideways it crops rather than letterboxing, since a thumbnail of
picture stranded in a field of black is worse than losing the top and bottom.
With the reshape working, 82% of the frame height stays visible; with the camera
refusing outright, 26% does, and the screen is still full either way. Portrait
is untouched: a landscape stream on a portrait phone still letterboxes rather
than cutting hands out of view.

The camera is **re-shaped on rotation**. The stream is asked for in the shape of
the screen, but that shape used to be chosen once, when it opened: start in
portrait, rotate to landscape, and a 720x1280 stream was left on an 852x393
screen, where `fitFor()` computes `0.26` against its `0.74` threshold and can
only pillarbox it into a sliver with most of the display black. `applyConstraints`
is tried first and a fresh stream opened if the camera ignores it, which plenty
do for a swap of their own sensor orientation — iOS in particular is entitled to
hand back the same sensor orientation no matter what `ideal` it is given, which
is why `fitFor()` has to cope on its own rather than trusting this to work.
Rotation is read as three signals (the media query, `resize` and
`orientationchange`) on a delay, since iOS reports the track's new dimensions a
beat after the rotation and a PWA does not always deliver the first of them.

A notch sits on one of the short edges in landscape, so `safe-area-inset-left`
and `-right` stop being zero there. Everything pinned to an edge now clears
them — the rail was being clipped by exactly this.

Verified at 844x390: the start screen fits with no scrolling at all, the dock
takes 126px of 390 with a photo strip in it, and nothing overflows
horizontally at 320, 375, 430 or 480px with the top bar fully loaded.

| file | what's in it |
| --- | --- |
| `docs/index.html` | the page shell |
| `docs/app.js` | camera, MediaPipe, gesture maths, UI, recording |
| `docs/effects.js` | WebGL2 renderer: every effect as a fragment shader |
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
