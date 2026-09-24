/**
 * Pinch Vision — web build.
 *
 * Hand tracking is the same MediaPipe model the desktop app uses, and the
 * gesture maths is a direct port: distances normalised by the hand's own size,
 * hysteresis on every threshold, and a cooldown so one tap fires once.
 *
 * Interaction is pinch-only. How closed your index-to-thumb pinch is sets the
 * effect strength for that hand; tapping thumb to pinky steps that hand to the
 * next effect.
 */

import { FilesetResolver, HandLandmarker }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';
import { Renderer, EFFECTS, CLEAN } from './effects.js?v=11';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker'
            + '/hand_landmarker/float16/1/hand_landmarker.task';

const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9, PINKY_TIP = 20;
const THUMB_MCP = 2, INDEX_MCP = 5;

// Pinch openness -> strength. Same window as the desktop "squeeze" mode.
const SQUEEZE_MIN = 0.12, SQUEEZE_MAX = 0.90;
// Thumb-to-pinky tap, with hysteresis and a cooldown so a held tap fires once.
const TAP_ON = 0.50, TAP_OFF = 0.72, TAP_COOLDOWN = 450;
const PINCH_OPEN = 0.55;     // index must be clear of the thumb: a tap, not a fist
// Touching the two hands' fingertips together toggles the window on and off.
// Distances are normalised by hand span, so it works at any distance from the
// camera, and the two thresholds keep one touch from flickering the latch.
const TOUCH_ON = 0.52;
const TOUCH_OFF = 0.88;
const MASK_EASE = 0.16;      // how fast the window opens and closes

// While the window is up, thumb and index are busy holding the frame, so
// strength moves to the middle finger: extended is full, curled is off.
const MIDDLE_TIP = 12, RING_TIP = 16;

// Pinning: touch your thumb to your RING finger — the one beside the pinky —
// and hold it. Same shape as the pinky tap that changes effect, one finger
// over, so the two are told apart by which fingertip is actually closer.
const EXTENDED = 0.75;       // index still out: the frame is still being held
// Thumb to the back fingers does both jobs, told apart by duration rather than
// by which fingertip is nearer: ring and pinky travel together, so asking the
// tracker which one the thumb is on is a coin toss. A quick touch changes the
// effect; holding it a full second pins the window.
const PIN_HOLD_MS = 1000;

// Resetting: close both hands into fists and hold.
//
// This replaced turning both palms away, which could not be made reliable. That
// gesture read palm orientation as the sign of a 2D cross product, and rotating
// a palm has to sweep through edge-on, where the sign is genuinely noise — so
// the charge kept resetting mid-turn. It also learned each hand's "normal"
// orientation once and never revised it, so a baseline captured from a tilted
// hand stayed inverted for the whole session.
//
// A fist has neither problem: it is a high-confidence pose for the tracker, it
// needs no learned baseline, and there is no ambiguous state to pass through.
const RESET_HOLD_MS = 1000;
// After a pin the hands are still in frame position, so the fingertip latch
// would immediately open a new window (or toggle one shut). Ignore it until the
// hands have plainly moved on.
const LATCH_LOCK_MS = 900;
// Fist thresholds, all as fingertip -> middle knuckle in hand spans, with
// hysteresis so a hand hovering at the boundary cannot chatter.
//
// The thumb term is the one that matters: four curled fingers alone also
// describe a hard pinch, which is this app's primary gesture and must never be
// mistaken for a reset. Pinching holds thumb and index out in front of the
// palm; clenching brings the thumb home across the knuckles.
const FIST_ON = 0.60;
const FIST_OFF = 0.82;
const FIST_THUMB_ON = 0.85;
const FIST_THUMB_OFF = 1.05;
// Clenching drags the mask corners inward, so the shape at the instant the fist
// lands is already ruined. Pin what the hands were holding a moment earlier.
const PIN_LOOKBACK_MS = 220;
const HISTORY_MS = 600;
const MAX_PINNED = 4;
// The window-mode dial is the RING finger. The middle finger used to have the
// job, but it sits next to the index, which is the shutter pose — dialling a
// window fired photos by accident. The ring finger is shorter and sits further
// from the middle knuckle we measure against, so it needs its own band rather
// than the middle finger's.
const DIAL_OPEN = 0.92;      // |ring tip - middle knuckle| / span, extended
const DIAL_SHUT = 0.42;      // the same, folded into the palm
const SMOOTHING = 0.30;      // per-frame easing on every strength value

// Shutter: press your index and middle fingertips together and hold.
//
// Two collisions decide the shape of this test. A fist also puts those two
// tips side by side, so "close together" alone would fire on every reset —
// hence both fingers must also be *extended*. And a hand framing a window
// naturally holds them together while the two hands touch across, which no
// single-hand measurement can tell apart, so the shutter stands down whenever
// the cross-hand contact is live.
const SHUTTER_HOLD_MS = 700;
const SHUTTER_ON = 0.45;     // index tip -> middle tip, in hand spans
const SHUTTER_OFF = 0.68;
// The fist test is relative, not absolute: the middle-to-ring gap has to be
// clearly wider than the index-to-middle one. An earlier version demanded the
// two fingers be *extended*, which a hand tilted toward the camera or curled
// into a claw never satisfies — and people make this pose as a claw.
// Hysteresis here too: a bare threshold on the margin would chatter at its own
// boundary exactly the way the extension gate did.
const SHUTTER_PART_ON = 0.20;
const SHUTTER_PART_OFF = 0.10;
// The tracker drops the pose for a frame here and there. Without this, every
// gap restarted the hold and the gesture almost never completed.
const SHUTTER_GRACE_MS = 180;
// Posing for the shutter means opening the pinch, and the pinch *is* the dial:
// by the time the pose is legal the effect has already fallen to 45% or less.
// So the strength is taken from just before the fingers started moving, the
// same trick `PIN_LOOKBACK_MS` uses for a window ruined by the clench that
// pins it.
const SHUTTER_LOOKBACK_MS = 320;
const TRAIL_MS = 700;        // how much strength history each hand keeps
const MAX_SHOTS = 8;         // strip length; older frames are revoked

// Air controls: hold a fingertip over an on-screen control to work it. The
// dwell is what makes it safe — a hand crossing the rail on its way somewhere
// else passes through in a few frames and never commits.
const AIR_DWELL_MS = 650;
const AIR_PAD = 14;          // px of slack around a control's box
const AIR_KEEP = 40;         // once the slider is grabbed, a wider box holds it
// The reticle only fades up as a hand approaches the rail. Drawn all the time
// it was a dot following your finger across the whole picture for no reason.
const RAIL_NEAR = 150;
const PEN_WIDTH = 5;
// Raw landmarks shake a pixel or two every frame, and a polyline straight
// through them reads as a seismograph. The pen chases the fingertip instead.
const PEN_EASE = 0.4;
// Opening the hand to stop drawing leaves index and middle together — which is
// the shutter pose. Without a beat after the pen lifts, putting the pen down
// costs you a photo you did not ask for.
const PEN_LIFT_LOCK_MS = 700;
// Brushes borrow the rack's own vocabulary rather than inventing a second one.
const BRUSHES = ['LINE', 'DASH', 'DOTS', 'ASCII'];
const DOT_GAP = 13;          // px between stamps, measured along the path
const ASCII_GAP = 17;
const ASCII_GLYPHS = '@%#*+=-:.';
const ASCII_FONT = '700 15px "JetBrains Mono", ui-monospace, monospace';

const MAX_RECORD_MS = 20000;
const STATIONS = ['clean', ...EFFECTS.map((e) => e.id)];

// Blend mode sweeps the rack at a hard speed limit rather than tracking the
// pinch directly: a rackful of effects across one pinch made every twitch a
// jump cut.
const SWEEP_RATE = 0.22;      // full 0 -> 1 sweep takes about 4.5 seconds

// Skeleton overlay.
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(`pv.${key}`); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`pv.${key}`, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

/* --------------------------------------------------------------------- sound */

/**
 * Six samples, one per *kind* of event, and nothing continuous.
 *
 * The rule that keeps this from becoming noise: anything that repeats while a
 * hand simply moves — the strength dial, the charge meters, the carousel — stays
 * silent. A sound that can fire every frame stops carrying information. What is
 * left are the moments something actually happened and could not be undone by
 * moving your hand back.
 */
const SOUNDS = {
  shutter: 'button.wav',      // a photo was taken
  step:    'tap_03.wav',      // this hand stepped to the next effect
  on:      'toggle_on.wav',   // a mode came on
  off:     'toggle_off.wav',  // a mode went off
  commit:  'select.wav',      // something was kept: a pin, a saved file
  nope:    'disabled.wav',    // the app declined
};

const sfx = {
  ctx: null,
  gain: null,
  buffers: new Map(),
  last: new Map(),
  on: store.get('sound', true),

  /**
   * Audio may only start inside a user gesture, so this runs from the first
   * tap rather than at load. Decoding is fire-and-forget: a sample that fails
   * to load simply never plays.
   */
  unlock() {
    if (this.ctx) { this.ctx.resume?.().catch(() => {}); return; }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    try { this.ctx = new Ctx(); } catch { return; }
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 0.3;
    this.gain.connect(this.ctx.destination);
    for (const [name, file] of Object.entries(SOUNDS)) {
      fetch(`sounds/${file}`)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
        .then((raw) => this.ctx.decodeAudioData(raw))
        .then((buf) => this.buffers.set(name, buf))
        .catch(() => { /* never worth breaking the app over */ });
    }
  },

  play(name) {
    if (!this.on || !this.ctx || this.ctx.state !== 'running') return;
    const buf = this.buffers.get(name);
    if (!buf) return;
    // Two of the same sample a frame apart is a rattle, not a cue.
    const now = this.ctx.currentTime;
    if (now - (this.last.get(name) ?? -1) < 0.06) return;
    this.last.set(name, now);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gain);
    src.start();
  },

  setEnabled(on) {
    this.on = on;
    store.set('sound', on);
  },
};

const $ = (sel) => document.querySelector(sel);
const clamp01 = (v) => Math.min(1, Math.max(0, v));

const ui = {
  stage: $('#stage'),
  canvas: $('#view'),
  video: $('#feed'),
  start: $('#start'),
  startBtn: $('#start-btn'),
  startNote: $('#start-note'),
  fps: $('#fps'),
  handL: $('#hand-l'),
  handR: $('#hand-r'),
  hint: $('#hint'),
  maskBadge: $('#mask-badge'),
  clearBtn: $('#clear'),
  probe: $('#probe'),
  blendBtn: $('#blend'),
  skelBtn: $('#skel'),
  swapBtn: $('#swap'),
  soundBtn: $('#sound'),
  overlay: $('#overlay'),
  paint: $('#paint'),
  rail: $('#rail'),
  lockBtn: $('#lock'),
  drawBtn: $('#draw'),
  brushBtn: $('#brush'),
  hue: $('#hue'),
  airEls: [],
  recordBtn: $('#record'),
  ring: $('#ring-progress'),
  recTime: $('#rec-time'),
  result: $('#result'),
  resultVideo: $('#result-video'),
  download: $('#download'),
  rackStrip: $('#rack-strip'),
  clipMeta: $('#clip-meta'),
  clipFx: $('#clip-fx'),
  share: $('#share'),
  again: $('#again'),
  toast: $('#toast'),
  strip: $('#strip'),
  photos: $('#photos'),
  carousel: $('#carousel'),
  photoCount: $('#photo-count'),
  photoMeta: $('#photo-meta'),
  photoPrev: $('#photo-prev'),
  photoNext: $('#photo-next'),
  photoSave: $('#photo-save'),
  photoShare: $('#photo-share'),
  photoClose: $('#photo-close'),
  slots: [
    { root: $('#slot-1'), name: $('#slot-1 .fx-name'), pct: $('#slot-1 .fx-pct'),
      meter: $('#slot-1 .meter'), hand: $('#slot-1 .slot-hand'), segs: [] },
    { root: $('#slot-2'), name: $('#slot-2 .fx-name'), pct: $('#slot-2 .fx-pct'),
      meter: $('#slot-2 .meter'), hand: $('#slot-2 .slot-hand'), segs: [] },
  ],
};

const state = {
  renderer: null,
  landmarker: null,
  running: false,
  mode: store.get('mode', 'fixed'),          // 'fixed' | 'blend'
  swap: store.get('swap', false),
  skeleton: store.get('skeleton', true),
  blendPos: 0,
  blendFade: 0,
  fit: 'cover',
  contact: false,
  contactPrev: false,
  masked: false,
  pinCharge: 0,
  resetCharge: 0,
  resetArmed: true,
  resetStart: -1,        // -1, not 0: a timestamp of 0 is a valid start
  shotCharge: 0,
  shotArmed: true,
  shotStart: -1,
  shotSeen: 0,
  shotPending: false,
  shots: [],
  photoIndex: 0,
  locked: false,
  drawing: false,
  hue: store.get('hue', 42),
  brush: store.get('brush', 0),
  painted: false,
  penLiftUntil: 0,
  airPoint: null,
  airEl: null,
  airStart: -1,
  airGrab: null,
  airDone: null,
  airNear: 0,
  composite: null,
  latchLockedUntil: 0,
  pinned: [],
  history: [],
  maskStrength: 0,
  maskPoints: null,
  fps: 0,
  lastFrame: 0,
  handsSeen: 0,
  slots: [
    { fx: 0, amount: 0, target: 0, pinch: 1, wasTapping: false, lastTap: 0,
      present: false, points: null, pinching: false, backHeld: false,
      backStart: 0, backFired: false, pinSnapshot: null, measure: null,
      reach: null, fist: false, shutter: false,
      trail: [], heldAmount: -1, penDown: false, penX: 0, penY: 0,
      penMX: 0, penMY: 0, penDist: 0, penGlyph: 0 },
    { fx: 1, amount: 0, target: 0, pinch: 1, wasTapping: false, lastTap: 0,
      present: false, points: null, pinching: false, backHeld: false,
      backStart: 0, backFired: false, pinSnapshot: null, measure: null,
      reach: null, fist: false, shutter: false,
      trail: [], heldAmount: -1, penDown: false, penX: 0, penY: 0,
      penMX: 0, penMY: 0, penDist: 0, penGlyph: 0 },
  ],
  recorder: null,
  recording: false,
  recDuration: 0,
  recStart: 0,
  chunks: [],
  blobUrl: null,
};

// ------------------------------------------------------------------ gestures

/** Landmarks -> the two measurements we care about, both scale-invariant. */
function readHand(points) {
  const wrist = points[WRIST], mcp = points[MIDDLE_MCP];
  const span = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y) || 1e-4;
  const t = points[THUMB_TIP], i = points[INDEX_TIP], p = points[PINKY_TIP];
  const r = points[RING_TIP], m = points[MIDDLE_TIP];
  return {
    pinch: Math.hypot(t.x - i.x, t.y - i.y) / span,
    tap: Math.hypot(t.x - p.x, t.y - p.y) / span,
    ring: Math.hypot(t.x - r.x, t.y - r.y) / span,
    spread: Math.hypot(i.x - m.x, i.y - m.y) / span,
    split: Math.hypot(m.x - r.x, m.y - r.y) / span,
  };
}

function strengthFrom(pinch) {
  return clamp01(1 - (pinch - SQUEEZE_MIN) / (SQUEEZE_MAX - SQUEEZE_MIN));
}

/** Ring-finger extension, 0 (curled) to 1 (straight). */
function curlStrength(points) {
  const span = spanOf(points);
  const reach = Math.hypot(points[RING_TIP].x - points[MIDDLE_MCP].x,
                           points[RING_TIP].y - points[MIDDLE_MCP].y) / span;
  return clamp01((reach - DIAL_SHUT) / (DIAL_OPEN - DIAL_SHUT));
}

/** A short rolling record of each hand's strength, for the shutter lookback. */
function pushTrail(slot, now) {
  slot.trail.push({ t: now, amount: slot.amount });
  while (slot.trail.length && now - slot.trail[0].t > TRAIL_MS) slot.trail.shift();
}

/** The newest recorded strength at or before `t`, or the oldest one we have. */
function amountBefore(slot, t) {
  for (let i = slot.trail.length - 1; i >= 0; i--) {
    if (slot.trail[i].t <= t) return slot.trail[i].amount;
  }
  return slot.trail.length ? slot.trail[0].amount : slot.amount;
}

/**
 * Every fingertip folded to the knuckles, with the thumb home across them.
 *
 * Hysteresis is per hand and lives on the slot, so a hand resting near the
 * threshold settles instead of flickering the charge on and off.
 */
function isFist(slot) {
  if (!slot.present || !slot.reach) { slot.fist = false; return false; }
  const r = slot.reach;
  const fingers = Math.max(r.index, r.middle, r.ring, r.pinky);
  slot.fist = fingers < (slot.fist ? FIST_OFF : FIST_ON)
           && r.thumb < (slot.fist ? FIST_THUMB_OFF : FIST_THUMB_ON);
  return slot.fist;
}

/**
 * Both hands clenched and held for a second: drop the live window, drop every
 * pinned one, leave blend mode, and let both effects fall back to zero.
 *
 * It re-arms only once both hands have opened again, so keeping the fists up
 * is one reset rather than a repeating one. Both hands are required and both
 * must be tracked — a reset throws away up to four pinned windows, so a single
 * hand leaving the frame must not be able to complete it.
 */
function updateReset(now) {
  const [a, b] = state.slots;
  // Evaluated every frame for both hands, not just when the gesture is live,
  // so neither hand's hysteresis goes stale while the other is away.
  const fistA = isFist(a), fistB = isFist(b);

  if (!fistA && !fistB) state.resetArmed = true;

  if (!fistA || !fistB || !state.resetArmed) {
    state.resetStart = -1;
    state.resetCharge = 0;
    return;
  }

  if (state.resetStart < 0) state.resetStart = now;
  state.resetCharge = clamp01((now - state.resetStart) / RESET_HOLD_MS);
  if (state.resetCharge < 1) return;

  state.resetArmed = false;
  state.resetStart = -1;
  state.resetCharge = 0;
  fireReset(now);
}

function fireReset(now) {
  const had = state.masked || state.pinned.length > 0 || state.mode === 'blend'
           || state.locked || state.painted;

  if (state.locked) { state.locked = false; setPill(ui.lockBtn, false); }
  clearPaint();

  state.masked = false;
  state.maskPoints = null;
  state.history.length = 0;
  state.pinned.length = 0;
  paintPinUI();

  if (state.mode === 'blend') {
    state.mode = 'fixed';
    store.set('mode', 'fixed');
    setPill(ui.blendBtn, false);
    state.blendPos = 0;
    state.blendFade = 0;
  }

  // Clenching drops the thumb-index gap below PINCH_OPEN, which reads as a
  // release of any thumb-to-back-finger contact — and an unconsumed release is
  // a tap. Mark it fired, the same way pinning does, so a reset cannot also
  // step the effect on its way out.
  for (const slot of state.slots) slot.backFired = true;

  // The strength loop already holds a clenched hand at zero, but `amount` only
  // eases there — a reset should land on zero rather than decay toward it.
  for (const slot of state.slots) { slot.amount = 0; slot.target = 0; }

  // Unclenching passes back through a closing hand, which the fingertip latch
  // would read as a fresh contact. Hold it shut while the hands open.
  state.contact = true;
  state.contactPrev = true;
  state.latchLockedUntil = now + LATCH_LOCK_MS;
  navigator.vibrate?.([14, 50, 14]);

  sfx.play(had ? 'off' : 'nope');
  toast(had ? 'RESET' : 'NOTHING TO RESET');
}

/**
 * One hand with index and middle pressed together, both fingers out straight.
 *
 * The extension gate is what separates this from a fist, which puts the same
 * two tips side by side with everything folded. `pinch > PINCH_OPEN` keeps a
 * thumb-to-index pinch out of it, and a hand mid thumb-to-back-finger contact
 * is reaching for a different gesture entirely.
 */
function shutterHand(slot, now) {
  // Stood down for the whole time a window is live. While one is up, thumb and
  // index hold the frame corners and strength moves to the middle finger — so
  // dialling parks the middle fingertip right beside the index one, which is
  // this pose exactly. `state.contact` does not cover it: that only marks the
  // moment the fingertips touch, and it goes false again the instant the window
  // latches open and the hands spread. Pin the window and the shutter comes
  // back, with the pinned frame in the shot.
  if (!slot.present || !slot.measure || slot.backHeld || state.masked) {
    slot.shutter = false;
    slot.heldAmount = -1;
    return false;
  }
  const was = slot.shutter;
  const m = slot.measure;
  // Two fingertips meeting, with the ring finger plainly left behind. Both
  // terms are gaps between fingertips on the same hand, so the test survives
  // any amount of foreshortening: curl the hand or tilt it toward the camera
  // and every distance shrinks together, leaving the comparison intact.
  slot.shutter = m.pinch > PINCH_OPEN
              && m.spread < (slot.shutter ? SHUTTER_OFF : SHUTTER_ON)
              && m.split - m.spread
                 > (slot.shutter ? SHUTTER_PART_OFF : SHUTTER_PART_ON);

  // On the way in, recover the strength this hand was holding before the
  // fingers moved; on the way out, hand the dial back to the live pinch.
  if (slot.shutter && !was) {
    slot.heldAmount = amountBefore(slot, now - SHUTTER_LOOKBACK_MS);
  } else if (!slot.shutter) {
    slot.heldAmount = -1;
  }
  return slot.shutter;
}

/**
 * Hold the shutter pose and a still is captured. Either hand will do — one is
 * enough, so the other stays free to keep dialling the look being photographed.
 *
 * Stood down while the hands are touching across each other, because a framing
 * hand holds index and middle together too and no single-hand measurement can
 * tell the two apart.
 */
function updateShutter(now) {
  const a = shutterHand(state.slots[0], now);
  const b = shutterHand(state.slots[1], now);
  const pose = (a || b) && !state.contact && state.resetCharge === 0
            && now >= state.penLiftUntil;

  if (pose) state.shotSeen = now;

  // A hold that a single dropped frame can reset is a hold that never
  // completes. Carry it across short gaps instead.
  const holding = pose
    || (state.shotStart >= 0 && now - state.shotSeen <= SHUTTER_GRACE_MS);

  if (!holding) state.shotArmed = true;

  if (!holding || !state.shotArmed) {
    state.shotStart = -1;
    state.shotCharge = 0;
    return;
  }

  if (state.shotStart < 0) state.shotStart = now;
  state.shotCharge = clamp01((now - state.shotStart) / SHUTTER_HOLD_MS);
  if (state.shotCharge < 1) return;

  state.shotArmed = false;
  state.shotStart = -1;
  state.shotCharge = 0;
  // Not captured here: this runs before the draw, so the canvas still holds the
  // previous frame. The flag is consumed once this frame has been rendered.
  state.shotPending = true;
}

/**
 * How close the thumb is to the back of the hand.
 *
 * Normally the nearer of ring and pinky. Telling the two apart in a 2D
 * projection is a coin toss — curling the pinky to meet the thumb drags the
 * ring along and either can read as nearer — so taking the minimum is what
 * keeps the tap working at all.
 *
 * While a window is up the ring finger is the strength dial, so it has to come
 * out of the reckoning or every adjustment would read as a tap. Only there:
 * outside window mode the ring is idle and the minimum still earns its keep.
 * Pinning only ever happens with a window up, so pinning is pinky-only, full
 * stop.
 */
function backTouch(slot) {
  if (!slot.measure) return 1;
  return state.masked ? slot.measure.tap
                      : Math.min(slot.measure.tap, slot.measure.ring);
}

/** Rack position -> the two stations either side of it, and the crossfade. */
function morphAt(amount) {
  const n = EFFECTS.length;
  const p = clamp01(amount) * n;
  const i = Math.min(Math.floor(p), n - 1);
  const f = clamp01(p - i);
  return { from: STATIONS[i], to: STATIONS[i + 1], mix: f * f * (3 - 2 * f), index: i };
}

function stationMeta(id) {
  return id === 'clean' ? CLEAN : EFFECTS.find((e) => e.id === id) || CLEAN;
}

/** Convex hull (monotone chain), returned counter-clockwise. */
function hull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const out = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...half(pts), ...half([...pts].reverse())];
}

const spanOf = (pts) => Math.hypot(pts[MIDDLE_MCP].x - pts[WRIST].x,
                                   pts[MIDDLE_MCP].y - pts[WRIST].y) || 1e-4;

/** Index and thumb both extended — the hand is framing, not clenched. */
function handOpen(points) {
  const span = spanOf(points);
  const reach = (tip, base) => Math.hypot(points[tip].x - points[base].x,
                                          points[tip].y - points[base].y) / span;
  // Generous, because a hand tilted toward the camera foreshortens: these only
  // need to rule out a curled hand, not certify a perfectly flat one.
  return reach(INDEX_TIP, INDEX_MCP) > 0.50 && reach(THUMB_TIP, THUMB_MCP) > 0.25;
}

/** How far a fingertip sits from the knuckles, in hand spans. */
function fingerReach(points, tip) {
  return Math.hypot(points[tip].x - points[MIDDLE_MCP].x,
                    points[tip].y - points[MIDDLE_MCP].y) / spanOf(points);
}

/**
 * Both index tips touching each other *and* both thumb tips touching, with both
 * hands open. The open-hand gate keeps "framing" apart from any curled shape,
 * so a hand that is reaching for another gesture can never latch the window.
 */
function fingertipsTouching() {
  const [a, b] = state.slots;
  if (!a.points || !b.points || !a.present || !b.present) return false;
  if (!handOpen(a.points) || !handOpen(b.points)) return false;
  const span = (spanOf(a.points) + spanOf(b.points)) / 2;
  const gap = (i) => Math.hypot(a.points[i].x - b.points[i].x,
                                a.points[i].y - b.points[i].y) / span;
  // Averaged rather than both-must-pass: lining up four fingertips at once is
  // fussy, and one pair landing squarely should carry the other being a little
  // wide. This is what made opening a window feel slow.
  const limit = state.contact ? TOUCH_OFF : TOUCH_ON;
  state.contact = (gap(INDEX_TIP) + gap(THUMB_TIP)) / 2 < limit;
  return state.contact;
}

/**
 * Fingertips together *opens* a window. It does not close one.
 *
 * Toggling with the same gesture was the real flaw: while shaping a window your
 * hands come near each other all the time, and any of those passes read as a
 * second edge and shut it off. Dismissing is now the twist gesture, which you
 * cannot perform by accident.
 */
function updateWindowLatch(now) {
  // Locked means the look is fixed — that includes which overlays are up.
  if (state.locked) { state.contactPrev = fingertipsTouching(); return; }
  const contact = fingertipsTouching();
  // Still tracked, just not acted on: when the lock lifts, an unbroken contact
  // is not a fresh edge and so cannot fire.
  if (now < state.latchLockedUntil) {
    state.contactPrev = contact;
    return;
  }
  if (contact && !state.contactPrev && !state.masked) {
    state.masked = true;
    toast('WINDOW ON — hold thumb to pinky to pin');
    navigator.vibrate?.(14);
    sfx.play('on');
  }
  state.contactPrev = contact;
}

/**
 * Thumb to the back fingers, resolved by how long it is held.
 *
 * Released quickly -> change this hand's effect. Held a full second with a
 * window up -> pin it. The duration split is what makes both reliable: telling
 * ring from pinky in a 2D projection failed often enough that the tap stopped
 * working, because curling the pinky to meet the thumb drags the ring along
 * with it and either can read as nearer.
 *
 * The pin snapshot is taken when contact *begins*, since reaching back there
 * moves the thumb, and the thumb is one of the window's own corners.
 */
function updateBackGestures(now) {
  if (state.locked) { state.pinCharge = 0; return; }
  let charge = 0;
  for (const [index, slot] of state.slots.entries()) {
    if (!slot.present || !slot.measure) {
      slot.backHeld = false;
      slot.backStart = 0;
      slot.backFired = false;
      continue;
    }

    const limit = slot.backHeld ? TAP_OFF : TAP_ON;
    // The index must be clear of the thumb: a reach to the back fingers, not a
    // closing hand.
    const touching = backTouch(slot) < limit && slot.measure.pinch > PINCH_OPEN;

    if (touching && !slot.backHeld) {
      slot.backHeld = true;
      slot.backStart = now;
      slot.backFired = false;
      slot.pinSnapshot = snapshotBefore(now);
    } else if (!touching && slot.backHeld) {
      // Any release that did not reach a pin is a tap — no upper limit, or
      // holding a shade too long (600-1000ms) would do nothing at all, which
      // feels exactly like the gesture being broken.
      if (!slot.backFired && now - slot.lastTap > TAP_COOLDOWN) {
        slot.lastTap = now;
        cycleSlot(index, true);
      }
      slot.backHeld = false;
      slot.backStart = 0;
      slot.backFired = false;
    }

    if (slot.backHeld && state.masked && !slot.backFired) {
      const held = now - slot.backStart;
      charge = Math.max(charge, clamp01(held / PIN_HOLD_MS));
      if (held >= PIN_HOLD_MS) {
        slot.backFired = true;          // no effect change on the way out
        pinWindow(slot.pinSnapshot, now);
        state.pinCharge = 0;
        return;
      }
    }
  }
  state.pinCharge = charge;
}

/** The newest recorded frame that predates `now` by the lookback. */
function snapshotBefore(now) {
  return [...state.history].reverse().find((h) => now - h.t >= PIN_LOOKBACK_MS)
    || state.history[state.history.length - 1]
    || null;
}

function pinWindow(snap, now) {
  if (!snap || !snap.poly) return;

  if (state.pinned.length >= MAX_PINNED) state.pinned.shift();
  state.pinned.push({ poly: snap.poly, slots: snap.slots });
  toast(`PINNED ${state.pinned.length}/${MAX_PINNED}`);
  navigator.vibrate?.([10, 40, 18]);
  sfx.play('commit');

  // Hands are free now; touch the fingertips again to start the next one.
  state.masked = false;
  state.maskPoints = null;
  state.history.length = 0;
  state.pinCharge = 0;
  // Treat the current contact as already consumed, and hold the latch shut for
  // a moment, so pinning never reads as "window off" or spawns a new window.
  state.contact = true;
  state.contactPrev = true;
  state.latchLockedUntil = now + LATCH_LOCK_MS;
  paintPinUI();
}

/** What the hands can usefully do right now, in one line. */
// With nine gestures there is more to say than one line holds, so the idle
// line rotates. Only the idle one: every other state has exactly one useful
// next move and saying anything else there would be noise.
const IDLE_TIPS = [
  'TOUCH FINGERTIPS TO OPEN A WINDOW',
  'INDEX AND MIDDLE TOGETHER TAKES A PHOTO',
  'POINT AT THE RAIL TO LOCK OR DRAW',
];
const HINT_ROTATE_MS = 4200;

function hintText() {
  if (state.recording) return '';
  if (!state.handsSeen) return 'SHOW YOUR HANDS';
  // Locked and drawing come first: in both, most of the vocabulary is either
  // refused or means something else, so the old lines were actively wrong.
  if (state.locked) return 'LOOK LOCKED · POINT AT LOCK TO RELEASE · BOTH FISTS TO RESET';
  if (state.drawing) return 'PINCH TO PAINT · POINT AT BRUSH TO CHANGE IT';
  if (state.masked) return 'HOLD THUMB TO PINKY TO PIN · BOTH FISTS TO RESET';
  if (state.handsSeen < 2) return 'PINCH TO DIAL · TAP THUMB TO PINKY FOR THE NEXT EFFECT';
  if (state.pinned.length) return 'TOUCH FINGERTIPS FOR ANOTHER · BOTH FISTS TO RESET';
  const i = Math.floor(performance.now() / HINT_ROTATE_MS) % IDLE_TIPS.length;
  return IDLE_TIPS[i];
}

function paintPinUI() {
  const count = state.pinned.length;
  ui.clearBtn.hidden = count === 0;
  ui.clearBtn.textContent = `CLEAR ${count}`;
}

function clearPinned(all) {
  if (all) state.pinned.length = 0;
  else state.pinned.pop();
  toast(all ? 'ALL WINDOWS CLEARED' : 'WINDOW REMOVED');
  sfx.play('off');
  paintPinUI();
}

/**
 * The masked region: the quad spanned by the four fingertips that made the
 * frame — each hand's thumb tip and index tip. Spread your hands and it opens
 * out with them.
 *
 * The convex hull keeps the polygon non-self-intersecting however the hands are
 * turned or crossed; a hard-coded corner order would bow-tie into an hourglass
 * and invert the inside test.
 *
 * Landmark space is the raw image with y pointing down; the render target is
 * mirrored with y up, hence (1 - x, 1 - y).
 */
function windowPolygon() {
  const [a, b] = state.slots;
  if (!a.points || !b.points) return null;
  const corners = [
    a.points[THUMB_TIP], a.points[INDEX_TIP],
    b.points[THUMB_TIP], b.points[INDEX_TIP],
  ].map((lm) => [1 - lm.x, 1 - lm.y]);

  const poly = hull(corners);
  if (poly.length < 3) return null;
  let area = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    area += x1 * y2 - x2 * y1;
  }
  // While the hands are still touching the quad is a sliver; keep the previous
  // shape until they open out into something worth looking through.
  return Math.abs(area) / 2 > 0.004 ? poly : null;
}

// ------------------------------------------------------------------ rendering

/**
 * A landscape camera frame inside a portrait phone loses most of its width to
 * `object-fit: cover` — and for a hand app that means your hands are tracked
 * but not visible. When the aspects disagree badly, letterbox instead.
 */
function fitFor(rw, rh, cw, ch) {
  const ratio = (rw / rh) / (cw / ch);
  return (ratio > 1.35 || ratio < 0.74) ? 'contain' : 'cover';
}

function sizeFor(video) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const isPhone = Math.min(screen.width, screen.height) < 600;
  const cap = isPhone ? 900 : 1280;
  const scale = Math.min(1, cap / Math.max(vw, vh));
  return [Math.round(vw * scale), Math.round(vh * scale)];
}

/* ----------------------------------------------------------- view mapping */

/**
 * Where the rendered frame actually sits on screen.
 *
 * The view is letterboxed or cropped by `object-fit`, so landmark space and
 * screen space differ by an offset and a scale. The skeleton, the air controls
 * and the paint layer all have to agree on this or they drift off the hand.
 * The stage is a fixed, full-bleed element, so these CSS pixels are viewport
 * pixels too.
 */
function viewBox() {
  const cw = ui.overlay.clientWidth, ch = ui.overlay.clientHeight;
  const rw = state.renderer?.width || 0, rh = state.renderer?.height || 0;
  if (!rw || !rh || !cw || !ch) return null;
  const scale = state.fit === 'contain'
    ? Math.min(cw / rw, ch / rh)
    : Math.max(cw / rw, ch / rh);
  const dw = rw * scale, dh = rh * scale;
  return { ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh, cw, ch, rw, rh };
}

/** A landmark in viewport CSS pixels. x is mirrored, as the view is. */
function screenPoint(lm, box) {
  return [box.ox + (1 - lm.x) * box.dw, box.oy + lm.y * box.dh];
}

/* ------------------------------------------------------------------ paint */

function paintCtx() {
  const canvas = ui.paint;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
  // Resizing a canvas clears it, so only ever do it when the size really moved.
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    state.painted = false;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const penColor = () => `hsl(${state.hue} 95% 62%)`;

function clearPaint() {
  const canvas = ui.paint;
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  state.painted = false;
  for (const slot of state.slots) slot.penDown = false;
}

/** One stamp of a discrete brush. */
function stamp(ctx, slot, x, y, brush) {
  if (brush === 'DOTS') {
    ctx.beginPath();
    ctx.arc(x, y, PEN_WIDTH * 0.62, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.fillText(ASCII_GLYPHS[slot.penGlyph % ASCII_GLYPHS.length], x, y);
  slot.penGlyph++;
}

/**
 * One frame's worth of stroke.
 *
 * Continuous brushes curve rather than corner: the segment is drawn as a
 * quadratic through the *previous* point, from the midpoint before it to the
 * midpoint after. Successive segments then share endpoints and tangents, which
 * is what turns a chain of short lines into one smooth path.
 *
 * Discrete brushes walk the segment at a fixed spacing instead, so marks stay
 * evenly spread however far the hand moved between two frames.
 */
function paintStroke(ctx, slot, x, y, brush) {
  const px = slot.penX, py = slot.penY;
  const mx = (px + x) / 2, my = (py + y) / 2;
  const seg = Math.hypot(x - px, y - py);

  if (brush === 'DOTS' || brush === 'ASCII') {
    const gap = brush === 'DOTS' ? DOT_GAP : ASCII_GAP;
    // Carry the leftover distance across frames, or every frame boundary would
    // reset the spacing and the marks would bunch.
    for (let t = gap - (slot.penDist % gap); t <= seg; t += gap) {
      const f = t / seg;
      stamp(ctx, slot, px + (x - px) * f, py + (y - py) * f, brush);
    }
  } else {
    ctx.save();
    if (brush === 'DASH') {
      ctx.setLineDash([1.5, 9]);
      // Phase follows distance travelled. Without it each frame restarts the
      // pattern and the line looks chewed rather than dotted.
      ctx.lineDashOffset = -slot.penDist;
    }
    ctx.beginPath();
    ctx.moveTo(slot.penMX, slot.penMY);
    ctx.quadraticCurveTo(px, py, mx, my);
    ctx.stroke();
    ctx.restore();
  }

  slot.penDist += seg;
  slot.penMX = mx;
  slot.penMY = my;
}

/**
 * Draw mode: a pinch is the pen. Thumb to index puts it down at the index tip
 * and opening the hand lifts it, which reuses the one measurement this app
 * already trusts most rather than inventing another pose.
 */
function updatePaint(box) {
  if (!state.drawing || !box) {
    for (const slot of state.slots) slot.penDown = false;
    return;
  }
  const ctx = paintCtx();
  const brush = BRUSHES[state.brush] || 'LINE';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = PEN_WIDTH;
  ctx.strokeStyle = penColor();
  ctx.fillStyle = penColor();
  if (brush === 'ASCII') {
    ctx.font = ASCII_FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
  }

  for (const slot of state.slots) {
    if (!slot.present || !slot.points || !slot.measure
        || slot.measure.pinch >= PINCH_OPEN) {
      if (slot.penDown) state.penLiftUntil = performance.now() + PEN_LIFT_LOCK_MS;
      slot.penDown = false;
      continue;
    }
    const [rx, ry] = screenPoint(slot.points[INDEX_TIP], box);

    if (!slot.penDown) {
      // A pen that has only touched down once has no direction yet.
      slot.penDown = true;
      slot.penX = rx; slot.penY = ry;
      slot.penMX = rx; slot.penMY = ry;
      slot.penDist = 0;
      slot.penGlyph = 0;
      continue;
    }

    const x = slot.penX + (rx - slot.penX) * PEN_EASE;
    const y = slot.penY + (ry - slot.penY) * PEN_EASE;
    paintStroke(ctx, slot, x, y, brush);
    slot.penX = x;
    slot.penY = y;
    state.painted = true;
  }
}

/* ------------------------------------------------------------ air controls */

/**
 * A control's box, padded — or null if it is not on screen.
 *
 * The null case matters: a `display: none` element reports a zero rect at the
 * origin, and padding that would quietly make every hidden control activatable
 * by pointing at the top-left corner.
 */
function airRect(el, pad) {
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return { l: r.left - pad, t: r.top - pad, r: r.right + pad, b: r.bottom + pad };
}

const inRect = (box, x, y) =>
  !!box && x >= box.l && x <= box.r && y >= box.t && y <= box.b;

/** The pointing fingertip, if any hand is pointing rather than pinching. */
function airTip(box) {
  for (const slot of state.slots) {
    if (!slot.present || !slot.points || !slot.measure) continue;
    // A pinching hand is drawing or dialling, not aiming at a control.
    if (slot.measure.pinch <= PINCH_OPEN) continue;
    return screenPoint(slot.points[INDEX_TIP], box);
  }
  return null;
}

function setCharge(el, v) {
  el.style.setProperty('--charge', v.toFixed(3));
}

function releaseAir() {
  if (state.airEl) setCharge(state.airEl, 0);
  state.airEl = null;
  state.airStart = -1;
}

function setHueFrom(el, y) {
  const r = el.getBoundingClientRect();
  const pos = clamp01((y - r.top) / (r.height || 1));
  state.hue = Math.round(pos * 360);
  store.set('hue', state.hue);
  paintHueUI();
}

function paintHueUI() {
  ui.hue.style.setProperty('--pos', (state.hue / 360).toFixed(4));
  ui.hue.style.setProperty('--picked', penColor());
  ui.stage.style.setProperty('--ink-accent', penColor());
  ui.drawBtn.style.setProperty('--accent', penColor());
  ui.hue.setAttribute('aria-valuenow', String(state.hue));
}

/**
 * One fingertip, one control, one dwell. Overlays take the screen when they are
 * up, so the rail stands down rather than reacting to hands behind them.
 */
function updateAir(now, box) {
  const live = !!box && ui.start.hidden && ui.photos.hidden && ui.result.hidden;
  const tip = live ? airTip(box) : null;
  state.airPoint = tip;

  if (!tip) {
    state.airGrab = null;
    state.airNear = 0;
    ui.rail.classList.remove('near');
    releaseAir();
    return;
  }
  const [x, y] = tip;

  // Distance to the rail's box, zero inside it. The rail brightens as a hand
  // comes for it, which is the only thing that says it can be pointed at.
  const rr = ui.rail.getBoundingClientRect();
  const gap = Math.hypot(Math.max(rr.left - x, 0, x - rr.right),
                         Math.max(rr.top - y, 0, y - rr.bottom));
  state.airNear = clamp01(1 - gap / RAIL_NEAR);
  ui.rail.classList.toggle('near', state.airNear > 0.35);

  // A grabbed slider keeps the finger through a wider box, so tracking it to
  // the ends does not drop the grab at the edge.
  if (state.airGrab) {
    if (inRect(airRect(state.airGrab, AIR_KEEP), x, y)) { setHueFrom(state.airGrab, y); return; }
    state.airGrab = null;
  }

  const el = ui.airEls.find((n) => inRect(airRect(n, AIR_PAD), x, y));
  if (!el) { state.airDone = null; releaseAir(); return; }

  // One dwell, one action: the finger has to leave before this control will
  // take another. Without it, resting on a toggle flips it every 650ms.
  if (el === state.airDone) { setCharge(el, 0); return; }

  if (state.airEl !== el) { releaseAir(); state.airEl = el; state.airStart = now; }
  const charge = clamp01((now - state.airStart) / AIR_DWELL_MS);
  setCharge(el, charge);
  if (charge < 1) return;

  releaseAir();
  state.airDone = el;
  if (el === ui.hue) { state.airGrab = el; sfx.play('on'); setHueFrom(el, y); }
  else if (el === ui.lockBtn) toggleLock();
  else if (el === ui.drawBtn) toggleDraw();
  else if (el === ui.brushBtn) cycleBrush();
}

function toggleLock() {
  state.locked = !state.locked;
  setPill(ui.lockBtn, state.locked);
  sfx.play(state.locked ? 'on' : 'off');
  toast(state.locked ? 'LOOK LOCKED' : 'LOOK LIVE');
}

function paintBrushUI() {
  ui.brushBtn.querySelector('.air-label').textContent = BRUSHES[state.brush] || 'LINE';
}

function cycleBrush() {
  state.brush = (state.brush + 1) % BRUSHES.length;
  store.set('brush', state.brush);
  paintBrushUI();
  sfx.play('step');
  toast(`BRUSH — ${BRUSHES[state.brush]}`);
}

function toggleDraw() {
  state.drawing = !state.drawing;
  setPill(ui.drawBtn, state.drawing);
  ui.stage.classList.toggle('drawing', state.drawing);
  sfx.play(state.drawing ? 'on' : 'off');
  toast(state.drawing ? 'DRAW — pinch to paint' : 'DRAW OFF');
}

function renderPlan() {
  if (state.mode === 'blend') {
    const m = morphAt(state.blendPos);
    return {
      slots: [{ morph: { from: m.from, to: m.to, mix: m.mix }, amount: 1 }],
      fade: state.blendFade,
    };
  }
  return {
    slots: state.slots.map((slot) => ({ effect: EFFECTS[slot.fx].id, amount: slot.amount })),
    fade: 1,
  };
}

/** In blend mode one hand fades, the other sweeps. Which is which follows the
 *  same swap toggle as everything else, so the cards never lie. */
function blendRoles() {
  return state.swap ? { fade: 1, sweep: 0 } : { fade: 0, sweep: 1 };
}

const SEGMENTS = 14;

/** The start panel's colour key, built from the rack itself. */
function buildRackStrip() {
  ui.rackStrip.innerHTML = '';
  for (const fx of EFFECTS) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.setProperty('--chip', fx.color);
    chip.innerHTML = '<i></i>';
    chip.append(fx.name);
    ui.rackStrip.appendChild(chip);
  }
}

function buildMeters() {
  for (const el of ui.slots) {
    el.meter.innerHTML = '';
    el.segs = [];
    for (let i = 0; i < SEGMENTS; i++) {
      const seg = document.createElement('i');
      el.meter.appendChild(seg);
      el.segs.push(seg);
    }
  }
}

function paintSlotUI(index) {
  const slot = state.slots[index];
  const el = ui.slots[index];
  let meta, label, value, role;

  if (state.mode === 'blend') {
    const roles = blendRoles();
    if (index === roles.fade) {
      meta = CLEAN;
      label = 'FADE';
      value = state.blendFade;
      role = 'depth of the whole chain';
    } else {
      const m = morphAt(state.blendPos);
      const from = stationMeta(m.from), to = stationMeta(m.to);
      meta = m.mix > 0.5 ? to : from;
      label = `${from.short} › ${to.short}`;
      value = state.blendPos;
      role = 'sweep';
    }
  } else {
    meta = EFFECTS[slot.fx];
    label = meta.name;
    value = slot.amount;
  }

  el.hand.textContent = state.masked
    ? `${index === 0 ? 'LEFT' : 'RIGHT'} · WIN`
    : (index === 0 ? 'LEFT' : 'RIGHT');
  el.name.textContent = label;
  el.pct.textContent = `${Math.round(value * 100)}`;
  el.root.style.setProperty('--accent', meta.color);
  el.root.classList.toggle('active', slot.present);
  el.root.classList.toggle('idle', value < 0.02 && !slot.present);

  const lit = Math.round(value * SEGMENTS);
  for (let i = 0; i < SEGMENTS; i++) el.segs[i].classList.toggle('on', i < lit);
}

function loop() {
  if (!state.running) return;
  requestAnimationFrame(loop);

  const video = ui.video;
  if (video.readyState < 2) return;

  const [w, h] = sizeFor(video);
  state.renderer.setSize(w, h);

  const fit = fitFor(w, h, ui.canvas.clientWidth, ui.canvas.clientHeight);
  if (fit !== state.fit) {
    state.fit = fit;
    ui.canvas.style.objectFit = fit;
  }

  const now = performance.now();
  const result = state.landmarker.detectForVideo(video, now);

  const seen = [false, false];
  for (let i = 0; i < result.landmarks.length; i++) {
    // The JS API renamed this field; accept either spelling.
    const handed = result.handedness || result.handednesses || [];
    const label = handed[i]?.[0]?.categoryName;
    // MediaPipe labels handedness as if the frame were already mirrored, which
    // is exactly what the viewer sees — so "Left" is the card on the left. The
    // swap toggle covers devices that report it the other way round.
    let slotIndex = label === 'Left' ? 0 : 1;
    if (state.swap) slotIndex = 1 - slotIndex;
    if (seen[slotIndex]) continue;
    seen[slotIndex] = true;

    const slot = state.slots[slotIndex];
    slot.points = result.landmarks[i];
    const measure = readHand(result.landmarks[i]);
    slot.pinch = measure.pinch;
    slot.pinching = measure.pinch < PINCH_OPEN;
    slot.measure = measure;
    slot.reach = {
      thumb: fingerReach(slot.points, THUMB_TIP),
      index: fingerReach(slot.points, INDEX_TIP),
      middle: fingerReach(slot.points, MIDDLE_TIP),
      ring: fingerReach(slot.points, RING_TIP),
      pinky: fingerReach(slot.points, PINKY_TIP),
    };
  }

  const dtSec = Math.min(0.1, (now - state.lastFrame) / 1000) || 0.016;

  for (let i = 0; i < 2; i++) {
    const slot = state.slots[i];
    slot.present = seen[i];
    if (!seen[i]) {
      slot.wasTapping = false;
      slot.points = null;
      slot.pinching = false;
      slot.backHeld = false;
      slot.backStart = 0;
      slot.measure = null;
      // Stale reach would otherwise keep a departed hand looking clenched.
      slot.reach = null;
      slot.fist = false;
      slot.shutter = false;
      slot.heldAmount = -1;
      slot.trail.length = 0;
      state.contact = false;
    }
  }

  updateWindowLatch(now);
  // Before the strength loop, so a clench zeroes the dial on the same frame it
  // is recognised rather than one frame later.
  updateReset(now);
  // Before the strength loop: the shutter pose has to restore this hand's dial
  // on the same frame it is recognised, or the photo catches the decay.
  updateShutter(now);

  // Exactly one rule decides each hand's strength, and it eases exactly once.
  // (Setting a pinch target here and then a second target below made the two
  // pulls meet in the middle and park at a fixed 59%.)
  for (const slot of state.slots) {
    if (state.locked) { pushTrail(slot, now); continue; }
    // A hand that is painting is not dialling: the pinch has been borrowed.
    if (state.drawing && slot.penDown) { pushTrail(slot, now); continue; }
    if (!slot.present) slot.target = 0;
    // A fist closes thumb on index, which reads as a maximum pinch. Without
    // this the reset gesture would ramp both effects to full on its way in.
    else if (slot.fist) slot.target = 0;
    else if (state.masked) slot.target = curlStrength(slot.points);
    else slot.target = strengthFrom(slot.pinch);

    if (slot.heldAmount >= 0) {
      // Restored, not eased: the value decayed over the few frames it took to
      // recognise the pose, and easing back would photograph the climb.
      slot.target = slot.heldAmount;
      slot.amount = slot.heldAmount;
    } else {
      slot.amount += (slot.target - slot.amount) * SMOOTHING;
    }
    pushTrail(slot, now);
  }

  if (state.mode === 'blend' && !state.locked) {
    const roles = blendRoles();
    state.blendFade += (state.slots[roles.fade].amount - state.blendFade) * SMOOTHING;
    // Rate-limit the sweep so the rack drifts between effects instead of
    // snapping the moment a finger moves a few millimetres.
    const wanted = state.slots[roles.sweep].amount;
    const step = Math.max(-SWEEP_RATE * dtSec,
      Math.min(SWEEP_RATE * dtSec, wanted - state.blendPos));
    state.blendPos = clamp01(state.blendPos + step);
  }

  if (state.masked) {
    state.maskPoints = windowPolygon() || state.maskPoints;
    // Keep a short rolling record so a pin can reach back past the clench.
    state.history.push({
      t: now,
      poly: state.maskPoints,
      slots: state.slots.map((slot) => ({ effect: EFFECTS[slot.fx].id, amount: slot.amount })),
    });
    while (state.history.length && now - state.history[0].t > HISTORY_MS) {
      state.history.shift();
    }
  }
  updateBackGestures(now);
  const showMask = state.masked && state.maskPoints;
  state.maskStrength += ((showMask ? 1 : 0) - state.maskStrength) * MASK_EASE;
  if (state.maskStrength < 0.004) {
    state.maskStrength = 0;
    if (!state.masked) state.maskPoints = null;
  }
  ui.maskBadge.hidden = !state.masked;
  if (state.resetCharge > 0.02) {
    ui.maskBadge.hidden = false;
    ui.maskBadge.textContent = `RESETTING ${Math.round(state.resetCharge * 100)}%`;
    ui.maskBadge.classList.add('charging');
  } else if (state.shotCharge > 0.02) {
    ui.maskBadge.hidden = false;
    ui.maskBadge.textContent = `PHOTO ${Math.round(state.shotCharge * 100)}%`;
    ui.maskBadge.classList.add('charging');
  } else if (state.pinCharge > 0.02) {
    ui.maskBadge.textContent = `PINNING ${Math.round(state.pinCharge * 100)}%`;
    ui.maskBadge.classList.add('charging');
  } else {
    ui.maskBadge.textContent = 'WINDOW';
    ui.maskBadge.classList.remove('charging');
  }
  if (ui.probe) {
    // The two numbers the fist test actually decides on, so the thresholds can
    // be tuned against a real hand instead of guessed at.
    const fmt = (slot) => {
      if (!slot.present || !slot.reach) return '—';
      const r = slot.reach;
      const fingers = Math.max(r.index, r.middle, r.ring, r.pinky);
      const m = slot.measure;
      return `fng ${fingers.toFixed(2)} thb ${r.thumb.toFixed(2)} `
        + `spr ${m ? m.spread.toFixed(2) : '—'} `
        + `spl ${m ? m.split.toFixed(2) : '—'} `
        + `gap ${m ? (m.split - m.spread).toFixed(2) : '—'} `
        + `dial ${r.ring.toFixed(2)}`
        + (slot.fist ? ' FIST' : '')
        + (slot.shutter ? ' SHUTTER' : '')
        + (slot.backHeld ? ` HELD ${Math.round(performance.now() - slot.backStart)}ms` : '');
    };
    const charge = state.resetCharge > 0.02
      ? `  ·  RESET ${Math.round(state.resetCharge * 100)}%`
      : state.shotCharge > 0.02
        ? `  ·  PHOTO ${Math.round(state.shotCharge * 100)}%` : '';
    ui.probe.textContent =
      `L ${fmt(state.slots[0])}  |  R ${fmt(state.slots[1])}${charge}`;
  }

  paintSlotUI(0);
  paintSlotUI(1);

  state.handsSeen = seen.filter(Boolean).length;
  const plan = renderPlan();
  state.renderer.draw(video, plan.slots, {
    mirror: true,
    time: now / 1000,
    fade: plan.fade,
    mask: state.maskPoints ? { points: state.maskPoints, strength: state.maskStrength } : null,
    pinned: state.pinned,
  });
  const box = viewBox();
  updatePaint(box);
  updateAir(now, box);
  drawSkeleton();
  if (state.recording) compositeFrame(true);
  // The frame on screen is now the one the shutter asked for, strength and all.
  if (state.shotPending) {
    state.shotPending = false;
    capture();
  }

  const dt = now - state.lastFrame;
  state.lastFrame = now;
  if (dt > 0) state.fps = state.fps * 0.9 + (1000 / dt) * 0.1;
  ui.fps.innerHTML = `${state.fps.toFixed(0)}<b>fps</b>`;
  ui.handL.style.background = state.slots[0].present
    ? EFFECTS[state.slots[0].fx].color : '';
  ui.handR.style.background = state.slots[1].present
    ? EFFECTS[state.slots[1].fx].color : '';
  ui.handL.classList.toggle('lit', state.slots[0].present);
  ui.handR.classList.toggle('lit', state.slots[1].present);

  const hint = hintText();
  ui.hint.textContent = hint;
  ui.hint.classList.toggle('show', hint !== '');
  if (state.recording) tickRecording(now);
}

/**
 * Hand skeleton, drawn on a 2D canvas stacked over the WebGL view.
 *
 * The view is `object-fit: cover`, so landmark coordinates (0..1 of the video)
 * have to be mapped through the same cover transform to land on the right
 * pixels, and mirrored on x because the picture is.
 */
function drawSkeleton() {
  const canvas = ui.overlay;
  const ctx = canvas.getContext('2d');
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);

  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  // The aiming reticle is not part of the skeleton: without something showing
  // where the fingertip is being read, an air control is a guessing game.
  if (state.airPoint && state.airNear > 0.02) {
    const [ax, ay] = state.airPoint;
    ctx.save();
    ctx.globalAlpha = state.airNear;
    ctx.strokeStyle = state.airEl ? penColor() : 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(ax, ay, state.airEl ? 13 : 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.beginPath();
    ctx.arc(ax, ay, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  if (!state.skeleton) return;

  const rw = state.renderer.width, rh = state.renderer.height;
  if (!rw || !rh) return;
  // Mirror whichever fit the view is using, or the skeleton drifts off the hand.
  const scale = state.fit === 'contain'
    ? Math.min(cw / rw, ch / rh)
    : Math.max(cw / rw, ch / rh);
  const dw = rw * scale, dh = rh * scale;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
  const at = (lm) => [ox + (1 - lm.x) * dw, oy + lm.y * dh];

  for (let i = 0; i < 2; i++) {
    const slot = state.slots[i];
    if (!slot.points) continue;
    const accent = getComputedStyle(ui.slots[i].root).getPropertyValue('--accent').trim()
      || '#ffffff';

    ctx.lineWidth = 2;
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      const [ax, ay] = at(slot.points[a]);
      const [bx, by] = at(slot.points[b]);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    ctx.globalAlpha = 1;
    ctx.fillStyle = '#ffffff';
    for (const lm of slot.points) {
      const [x, y] = at(lm);
      ctx.beginPath();
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pin charge, drawn where the thumb meets the pinky so the wait has an
    // obvious anchor on the hand doing it.
    if (slot.backHeld && state.masked) {
      const [rx, ry] = at(slot.points[PINKY_TIP]);
      const [thx, thy] = at(slot.points[THUMB_TIP]);
      const cx = (rx + thx) / 2, cy = (ry + thy) / 2;
      const progress = clamp01((performance.now() - slot.backStart) / PIN_HOLD_MS);
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, 17, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 17, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      ctx.stroke();
    }

    // The pinch itself: a ring that closes as the fingers meet.
    const [tx, ty] = at(slot.points[THUMB_TIP]);
    const [ix, iy] = at(slot.points[INDEX_TIP]);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc((tx + ix) / 2, (ty + iy) / 2, 6 + 16 * (1 - slot.amount), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// -------------------------------------------------------------------- actions

function cycleSlot(index, fromGesture) {
  if (state.locked) { sfx.play('nope'); toast('LOCKED'); return; }
  const slot = state.slots[index];
  const hand = index === 0 ? 'LEFT' : 'RIGHT';
  if (state.mode === 'blend') {
    // Nothing to advance here — nudge the sweep to the next station and let the
    // rate limiter glide it across rather than cutting.
    const n = EFFECTS.length;
    const k = Math.round(clamp01(state.blendPos) * n);
    const next = (k + 1) % (n + 1);
    state.slots[blendRoles().sweep].target = next / n;
    toast(`SWEEPING → ${stationMeta(STATIONS[next]).name}`);
  } else {
    slot.fx = (slot.fx + 1) % EFFECTS.length;
    toast(`${hand} → ${EFFECTS[slot.fx].name}`);
  }
  sfx.play('step');
  if (fromGesture) navigator.vibrate?.(12);
  paintSlotUI(index);
}

let toastTimer = null;
function toast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove('show'), 1400);
}

function pickMime() {
  const candidates = [
    'video/mp4;codecs=avc1.42E01E',    // Safari, including iOS
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return candidates.find((m) => MediaRecorder.isTypeSupported?.(m)) || '';
}

function startRecording() {
  const mime = pickMime();
  // Composited once up front, then every frame in the loop: captureStream is
  // bound to the canvas it is given, so it has to be the one that gets the
  // drawing even if the first stroke comes later.
  const stream = compositeFrame(true).captureStream(30);
  try {
    state.recorder = new MediaRecorder(stream, mime
      ? { mimeType: mime, videoBitsPerSecond: 6_000_000 }
      : undefined);
  } catch (err) {
    sfx.play('nope');
    toast('recording unsupported here');
    return;
  }
  state.chunks = [];
  state.recorder.ondataavailable = (e) => { if (e.data.size) state.chunks.push(e.data); };
  state.recorder.onstop = finishRecording;
  state.recorder.start(200);
  state.recording = true;
  state.recStart = performance.now();
  sfx.play('on');
  ui.recordBtn.classList.add('recording');
  ui.stage.classList.add('recording');
}

function tickRecording(now) {
  const elapsed = now - state.recStart;
  const frac = Math.min(1, elapsed / MAX_RECORD_MS);
  // The ring is a circle of circumference 2*pi*r with r = 26.
  ui.ring.style.strokeDashoffset = `${163.4 * (1 - frac)}`;
  ui.recTime.textContent = `${Math.ceil((MAX_RECORD_MS - elapsed) / 1000)}s`;
  if (elapsed >= MAX_RECORD_MS) stopRecording();
}

function stopRecording() {
  if (!state.recording) return;
  // MediaRecorder blobs often report an unseekable duration, so trust our own
  // clock rather than the video element's metadata.
  state.recDuration = (performance.now() - state.recStart) / 1000;
  state.recording = false;
  sfx.play('off');
  ui.recordBtn.classList.remove('recording');
  ui.stage.classList.remove('recording');
  ui.ring.style.strokeDashoffset = '163.4';
  ui.recTime.textContent = '';
  state.recorder?.stop();
}

/* ------------------------------------------------------------------ photos */

/**
 * Capture the rendered frame — effects, window and pinned windows included.
 * The skeleton lives on a separate overlay canvas, so it is never in the shot.
 *
 * The aspect is read from the element rather than the drawing buffer: the
 * canvas is `object-fit` fitted to the stage, and the thumbnail and the flying
 * copy both have to match what the viewer actually saw.
 */
function capture() {
  navigator.vibrate?.(20);
  sfx.play('shutter');
  flash();
  const aspect = (ui.canvas.clientWidth / ui.canvas.clientHeight) || 1.5;
  const fit = ui.canvas.style.objectFit || 'cover';
  compositeFrame().toBlob((blob) => {
    if (!blob) { sfx.play('nope'); toast('CAPTURE FAILED'); return; }
    addShot(blob, aspect, fit);
  }, 'image/jpeg', 0.92);
}

/**
 * Remove an element when its animation ends — or regardless, shortly after.
 *
 * `onfinish` is not a guarantee: a backgrounded tab defers animations, and a
 * browser without `element.animate` never starts one at all. Both leave a
 * full-frame photo pasted over the UI, so the timeout is the floor.
 */
function cleanUpAfter(el, anim, ms) {
  let done = false;
  const drop = () => { if (done) return; done = true; el.remove(); };
  if (anim) { anim.onfinish = drop; anim.oncancel = drop; }
  setTimeout(drop, ms);
  return drop;
}

/**
 * The rendered frame with the drawing on top, in the view canvas's own pixels.
 *
 * The two layers do not share a coordinate space: the paint canvas covers the
 * whole screen, while the view canvas is letterboxed or cropped inside it by
 * `object-fit`. So the paint layer is drawn into the rectangle that screen
 * space maps to, which is what keeps a stroke on the same part of the picture
 * in the file as it was under the finger.
 */
function compositeFrame(force) {
  // A recording is locked to whichever canvas it was handed, so once it starts
  // it always gets the composite — the drawing may not exist yet.
  if (!state.painted && !force) return ui.canvas; // nothing to add; skip the copy
  const box = viewBox();
  if (!box) return ui.canvas;

  const w = ui.canvas.width, h = ui.canvas.height;
  let c = state.composite;
  if (!c) { c = state.composite = document.createElement('canvas'); }
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }

  const ctx = c.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(ui.canvas, 0, 0);

  if (state.painted) {
    const sx = box.rw / box.dw, sy = box.rh / box.dh;
    ctx.drawImage(ui.paint, -box.ox * sx, -box.oy * sy, box.cw * sx, box.ch * sy);
  }
  return c;
}

/** The white blink. Separate element so it covers the UI as well as the feed. */
function flash() {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const el = document.createElement('div');
  el.className = 'flash';
  document.body.appendChild(el);
  const anim = el.animate?.(
    [{ opacity: 0 }, { opacity: 0.9, offset: 0.1 }, { opacity: 0 }],
    { duration: 420, easing: 'ease-out' },
  );
  cleanUpAfter(el, anim, 900);
}

function addShot(blob, aspect, fit) {
  const shot = { blob, url: URL.createObjectURL(blob), aspect, fit, at: new Date() };
  state.shots.push(shot);
  // Object URLs pin their blob in memory until revoked, so an unbounded strip
  // is an unbounded leak.
  while (state.shots.length > MAX_SHOTS) {
    URL.revokeObjectURL(state.shots.shift().url);
    if (state.photoIndex > 0) state.photoIndex--;
  }
  renderStrip();
  flyToStrip(shot, ui.strip.lastElementChild);
}

function renderStrip() {
  ui.strip.innerHTML = '';
  for (const shot of state.shots) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'shot';
    b.style.backgroundImage = `url("${shot.url}")`;
    b.style.aspectRatio = String(shot.aspect);
    b.style.backgroundSize = shot.fit === 'contain' ? 'contain' : 'cover';
    b.setAttribute('aria-label', 'open this photo');
    const index = state.shots.indexOf(shot);
    b.addEventListener('click', () => openPhotos(index));
    ui.strip.appendChild(b);
  }
  const any = state.shots.length > 0;
  ui.strip.hidden = !any;
  ui.stage.classList.toggle('has-shots', any);
  ui.strip.scrollLeft = ui.strip.scrollWidth;
}

/**
 * Photo Booth's move: the frame you just shot shrinks into its slot in the
 * strip. The flying copy is a real element over the page rather than a
 * transition on the thumbnail, so it can start at full-frame size and travel.
 */
function flyToStrip(shot, thumb) {
  if (!thumb) return;
  const to = thumb.getBoundingClientRect();
  const from = ui.canvas.getBoundingClientRect();
  if (!to.width || !from.width
      || matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const flyer = document.createElement('img');
  flyer.className = 'flyer';
  flyer.src = shot.url;
  flyer.style.cssText = `left:${from.left}px; top:${from.top}px;`
    + `width:${from.width}px; height:${from.height}px;`
    + `object-fit:${shot.fit};`;
  document.body.appendChild(flyer);

  // Uniform scale: the thumbnail carries the frame's own aspect ratio, so the
  // photo never squashes on its way down.
  const scale = to.width / from.width;
  const dx = (to.left + to.width / 2) - (from.left + from.width / 2);
  const dy = (to.top + to.height / 2) - (from.top + from.height / 2);

  thumb.style.opacity = '0';
  const anim = flyer.animate?.([
    { transform: 'translate(0px, 0px) scale(1)' },
    { transform: `translate(${dx}px, ${dy}px) scale(${scale})` },
  ], { duration: 620, easing: 'cubic-bezier(.5,.02,.2,1)', fill: 'forwards' });

  // The thumbnail must come back whether or not the flight ever completed.
  const drop = cleanUpAfter(flyer, anim, 1200);
  const land = () => { thumb.style.opacity = ''; drop(); };
  if (anim) { anim.onfinish = land; anim.oncancel = land; }
  setTimeout(land, 1200);
}

/* ---------------------------------------------------------- photo carousel */

/**
 * A coverflow over the strip. Cards are laid out from a *fractional* position
 * rather than a selected index, which is what lets a drag move them with the
 * finger instead of snapping only on release.
 */
function layoutCarousel(pos) {
  const cards = ui.carousel.children;
  for (let i = 0; i < cards.length; i++) {
    const off = i - pos;
    const dist = Math.min(Math.abs(off), 4);
    const dir = Math.sign(off);
    const card = cards[i];
    // Depth and turn ramp over the first step, then hold: past one card away
    // the angle stops changing and only the offset keeps growing, which is what
    // reads as a receding row rather than a fan.
    const turn = -dir * Math.min(dist, 1) * 42;
    card.style.transform =
      `translate(-50%, -50%) translateX(${dir * (32 + dist * 24)}%) `
      + `translateZ(${-dist * 190}px) rotateY(${turn}deg) scale(${1 - dist * 0.07})`;
    card.style.opacity = dist > 3.2 ? '0' : String(1 - dist * 0.17);
    card.style.zIndex = String(100 - Math.round(dist * 10));
    card.style.pointerEvents = dist > 3.2 ? 'none' : 'auto';
    card.classList.toggle('front', Math.abs(off) < 0.5);
  }
}

function buildCarousel() {
  ui.carousel.innerHTML = '';
  for (const [i, shot] of state.shots.entries()) {
    const card = document.createElement('figure');
    card.className = 'card';
    card.style.aspectRatio = String(shot.aspect);
    const img = document.createElement('img');
    img.src = shot.url;
    img.alt = '';
    img.draggable = false;
    card.appendChild(img);
    card.addEventListener('click', () => {
      if (i === state.photoIndex) return;
      state.photoIndex = i;
      paintPhotos();
    });
    ui.carousel.appendChild(card);
  }
}

function paintPhotos() {
  const n = state.shots.length;
  state.photoIndex = Math.max(0, Math.min(n - 1, state.photoIndex));
  layoutCarousel(state.photoIndex);
  ui.photoCount.textContent = n ? `${state.photoIndex + 1} / ${n}` : '—';
  const shot = state.shots[state.photoIndex];
  ui.photoMeta.textContent = shot
    ? shot.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '';
  ui.photoPrev.disabled = state.photoIndex <= 0;
  ui.photoNext.disabled = state.photoIndex >= n - 1;
  ui.photoShare.hidden = !canShareShot(shot);
}

function canShareShot(shot) {
  if (!shot || !navigator.canShare) return false;
  try {
    return navigator.canShare({ files: [shotFile(shot)] });
  } catch { return false; }
}

function shotFile(shot) {
  const stamp = shot.at.toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return new File([shot.blob], `pinch-vision-${stamp}.jpg`, { type: 'image/jpeg' });
}

function openPhotos(index) {
  if (!state.shots.length) return;
  state.photoIndex = index;
  buildCarousel();
  ui.photos.hidden = false;
  paintPhotos();
}

function closePhotos() {
  ui.photos.hidden = true;
  // The images hold the blobs open; dropping them frees the decoded bitmaps.
  ui.carousel.innerHTML = '';
}

function stepPhoto(delta) {
  const next = Math.max(0, Math.min(state.shots.length - 1, state.photoIndex + delta));
  if (next === state.photoIndex) return;
  state.photoIndex = next;
  paintPhotos();
}

/** Drag the row with a finger; release decides how many cards it travelled. */
function wirePhotoDrag() {
  let id = null, x0 = 0, moved = 0, width = 1;

  ui.carousel.addEventListener('pointerdown', (e) => {
    if (id !== null || !state.shots.length) return;
    id = e.pointerId;
    x0 = e.clientX;
    moved = 0;
    width = Math.max(120, ui.carousel.clientWidth * 0.38);
    // Capture is a nicety — it keeps the drag alive past the element's edge —
    // so it must never be able to take the drag down with it.
    try { ui.carousel.setPointerCapture(id); } catch { /* not capturable */ }
    ui.carousel.classList.add('dragging');
  });

  ui.carousel.addEventListener('pointermove', (e) => {
    if (e.pointerId !== id) return;
    moved = e.clientX - x0;
    // Rubber-band at the ends rather than letting the row run off into space.
    let pos = state.photoIndex - moved / width;
    if (pos < 0) pos *= 0.35;
    else if (pos > state.shots.length - 1) {
      pos = (state.shots.length - 1) + (pos - (state.shots.length - 1)) * 0.35;
    }
    layoutCarousel(pos);
  });

  const end = (e) => {
    if (e.pointerId !== id) return;
    // Releasing a pointer that is no longer captured throws, and `?.` does not
    // catch a throw — letting it escape here would abandon the drag before it
    // was committed and leave the row parked mid-swipe.
    try {
      if (ui.carousel.hasPointerCapture?.(id)) ui.carousel.releasePointerCapture(id);
    } catch { /* already released */ }
    ui.carousel.classList.remove('dragging');
    id = null;
    const steps = Math.round(-moved / width);
    state.photoIndex = Math.max(0, Math.min(state.shots.length - 1,
      state.photoIndex + steps));
    paintPhotos();
    // A drag that moved the row is not also a click on a card.
    if (Math.abs(moved) > 6) {
      const swallow = (ev) => ev.stopPropagation();
      ui.carousel.addEventListener('click', swallow, { capture: true, once: true });
      setTimeout(() => ui.carousel.removeEventListener('click', swallow, true), 0);
    }
  };
  ui.carousel.addEventListener('pointerup', end);
  ui.carousel.addEventListener('pointercancel', end);
}

async function sharePhoto() {
  const shot = state.shots[state.photoIndex];
  if (!shot) return;
  try { await navigator.share({ files: [shotFile(shot)], title: 'Pinch Vision' }); }
  catch { /* dismissed */ }
}

function saveShot(shot) {
  const stamp = shot.at.toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const a = document.createElement('a');
  a.href = shot.url;
  a.download = `pinch-vision-${stamp}.jpg`;
  a.click();
  sfx.play('commit');
  toast('PHOTO SAVED');
}

/** What was on screen while recording, in the rack's own vocabulary. */
function activeLabel() {
  if (state.mode === 'blend') {
    const m = morphAt(state.blendPos);
    return `BLEND · ${stationMeta(m.from).short} › ${stationMeta(m.to).short}`;
  }
  return state.slots.map((slot) => EFFECTS[slot.fx].short).join('  ×  ');
}

function finishRecording() {
  const type = state.recorder?.mimeType || 'video/webm';
  const blob = new Blob(state.chunks, { type });
  if (state.blobUrl) URL.revokeObjectURL(state.blobUrl);
  state.blobUrl = URL.createObjectURL(blob);

  const ext = type.includes('mp4') ? 'mp4' : 'webm';
  const name = `pinch-vision-${Date.now()}.${ext}`;
  ui.resultVideo.src = state.blobUrl;
  ui.download.href = state.blobUrl;
  ui.download.download = name;

  const secs = Math.round(state.recDuration);
  const mb = (blob.size / 1048576).toFixed(1);
  ui.clipMeta.textContent =
    `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')} · ${mb} MB · ${ext.toUpperCase()}`;
  ui.clipFx.textContent = activeLabel();
  ui.result.hidden = false;

  const file = new File([blob], name, { type });
  if (navigator.canShare?.({ files: [file] })) {
    ui.share.hidden = false;
    ui.share.onclick = () => navigator.share({ files: [file], title: 'Pinch Vision' })
      .catch(() => {});
  } else {
    ui.share.hidden = true;
  }
}

function setPill(btn, on) {
  btn.classList.toggle('on', on);
  btn.setAttribute('aria-pressed', String(on));
}

function toggleBlend() {
  state.mode = state.mode === 'blend' ? 'fixed' : 'blend';
  store.set('mode', state.mode);
  setPill(ui.blendBtn, state.mode === 'blend');
  sfx.play(state.mode === 'blend' ? 'on' : 'off');
  if (state.mode === 'blend') {
    const roles = blendRoles();
    state.blendPos = state.slots[roles.sweep].amount;
    state.blendFade = state.slots[roles.fade].amount;
    toast(`BLEND — ${roles.fade === 0 ? 'left' : 'right'} hand fades, other sweeps`);
  } else {
    toast('FIXED — one effect per hand');
  }
  paintSlotUI(0);
  paintSlotUI(1);
}

function toggleSound() {
  sfx.setEnabled(!sfx.on);
  setPill(ui.soundBtn, sfx.on);
  // Announce it with the sound itself when turning on; silence says the rest.
  if (sfx.on) sfx.play('on');
  toast(sfx.on ? 'SOUND ON' : 'SOUND OFF');
}

function toggleSkeleton() {
  state.skeleton = !state.skeleton;
  store.set('skeleton', state.skeleton);
  setPill(ui.skelBtn, state.skeleton);
  sfx.play(state.skeleton ? 'on' : 'off');
  toast(state.skeleton ? 'SKELETON ON' : 'SKELETON OFF');
}

function toggleSwap() {
  state.swap = !state.swap;
  store.set('swap', state.swap);
  setPill(ui.swapBtn, state.swap);
  sfx.play(state.swap ? 'on' : 'off');
  // Carry each hand's settings across so the swap feels like moving the cards,
  // not like resetting them.
  const [a, b] = state.slots;
  [a.fx, b.fx] = [b.fx, a.fx];
  [a.points, b.points] = [b.points, a.points];
  toast('HANDS SWAPPED');
  paintSlotUI(0);
  paintSlotUI(1);
}

// ----------------------------------------------------------------- lifecycle

/**
 * Load the tracker immediately, not on tap. It is ~8 MB of model plus WASM and
 * takes several seconds cold; starting now means it downloads while the intro
 * is being read, so "Start camera" feels instant on a warm connection.
 */
async function loadTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

const trackerPromise = loadTracker().catch((err) => err);

async function begin() {
  ui.startBtn.disabled = true;
  ui.startNote.textContent = 'requesting camera…';
  try {
    const portrait = window.matchMedia('(orientation: portrait)').matches
      && Math.min(screen.width, screen.height) < 600;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        // Ask for a stream shaped like the screen. Phones usually honour this;
        // where they do not, fitFor() letterboxes rather than cropping hands away.
        width: { ideal: portrait ? 720 : 1280 },
        height: { ideal: portrait ? 1280 : 720 },
      },
      audio: false,
    });
    ui.video.srcObject = stream;
    await ui.video.play();
  } catch (err) {
    ui.startNote.textContent = `camera blocked: ${err.name}. Check the site permission and reload.`;
    ui.startBtn.disabled = false;
    return;
  }

  ui.startNote.textContent = 'starting hand tracker…';
  const tracker = await trackerPromise;
  if (tracker instanceof Error) {
    ui.startNote.textContent = `tracker failed to load: ${tracker.message}`;
    ui.startBtn.disabled = false;
    return;
  }
  state.landmarker = tracker;

  try {
    state.renderer = new Renderer(ui.canvas);
  } catch (err) {
    ui.startNote.textContent = err.message;
    ui.startBtn.disabled = false;
    return;
  }

  ui.start.hidden = true;
  state.running = true;
  state.lastFrame = performance.now();
  setPill(ui.skelBtn, state.skeleton);
  setPill(ui.blendBtn, state.mode === 'blend');
  setPill(ui.swapBtn, state.swap);
  paintSlotUI(0);
  paintSlotUI(1);
  requestAnimationFrame(loop);
}

ui.startBtn.addEventListener('click', begin);
ui.blendBtn.addEventListener('click', toggleBlend);
ui.skelBtn.addEventListener('click', toggleSkeleton);
ui.swapBtn.addEventListener('click', toggleSwap);

// Tap clears the most recent window, press and hold clears them all. Removal
// stays on a button: one false positive should not wipe your work.
let holdTimer = null;
let heldAll = false;
const startHold = () => {
  heldAll = false;
  holdTimer = setTimeout(() => { heldAll = true; clearPinned(true); }, 550);
};
const endHold = () => {
  clearTimeout(holdTimer);
  if (!heldAll && state.pinned.length) clearPinned(false);
};
ui.clearBtn.addEventListener('pointerdown', startHold);
ui.clearBtn.addEventListener('pointerup', endHold);
ui.clearBtn.addEventListener('pointerleave', () => clearTimeout(holdTimer));
ui.slots.forEach((el, i) => el.root.addEventListener('click', () => cycleSlot(i, false)));
ui.recordBtn.addEventListener('click', () =>
  state.recording ? stopRecording() : startRecording());
ui.again.addEventListener('click', () => { ui.result.hidden = true; });
ui.soundBtn.addEventListener('click', toggleSound);
ui.lockBtn.addEventListener('click', toggleLock);
// Tap toggles, press and hold wipes the sheet — the same idiom the CLEAR pill
// already uses. The air path can only toggle, so this is the one control with
// a second action reachable by touch alone.
{
  let timer = null;
  let held = false;
  ui.drawBtn.addEventListener('pointerdown', () => {
    held = false;
    timer = setTimeout(() => {
      held = true;
      if (state.painted) { clearPaint(); sfx.play('off'); toast('DRAWING CLEARED'); }
      else { sfx.play('nope'); toast('NOTHING DRAWN'); }
    }, 550);
  });
  ui.drawBtn.addEventListener('pointerup', () => {
    clearTimeout(timer);
    if (!held) toggleDraw();
  });
  ui.drawBtn.addEventListener('pointerleave', () => clearTimeout(timer));
}
ui.brushBtn.addEventListener('click', cycleBrush);

// The rail works by touch as well as by fingertip; same handlers either way.
ui.airEls = [ui.lockBtn, ui.drawBtn, ui.brushBtn, ui.hue];
{
  let dragging = false;
  const track = (e) => { setHueFrom(ui.hue, e.clientY); };
  ui.hue.addEventListener('pointerdown', (e) => {
    dragging = true;
    try { ui.hue.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
    track(e);
  });
  ui.hue.addEventListener('pointermove', (e) => { if (dragging) track(e); });
  const stop = (e) => {
    if (!dragging) return;
    dragging = false;
    try {
      if (ui.hue.hasPointerCapture?.(e.pointerId)) ui.hue.releasePointerCapture(e.pointerId);
    } catch { /* already released */ }
  };
  ui.hue.addEventListener('pointerup', stop);
  ui.hue.addEventListener('pointercancel', stop);
}

/* ---------------------------------------------------- audio + fullscreen */

// Audio can only start inside a gesture. Any tap will do, and the first one is
// almost always the start button.
addEventListener('pointerdown', () => sfx.unlock(), { once: true, capture: true });

/**
 * Turning a phone sideways goes fullscreen.
 *
 * A rotation is not a user gesture, so `requestFullscreen` from the orientation
 * change is usually refused. When it is, the request is *armed* and the next
 * touch spends it — which costs the user nothing, since using the app in
 * landscape means touching it. iPhone Safari has no element fullscreen at all;
 * there the installed PWA is the route, which is what `display_override:
 * ["fullscreen"]` in the manifest is for.
 */
const landscapeQuery = matchMedia('(orientation: landscape)');
const coarseQuery = matchMedia('(pointer: coarse)');
let armedFullscreen = false;
let weWentFullscreen = false;

function wantsFullscreen() {
  return landscapeQuery.matches && coarseQuery.matches;
}

function enterFullscreen() {
  const el = document.documentElement;
  if (!el.requestFullscreen || document.fullscreenElement) return;
  el.requestFullscreen({ navigationUI: 'hide' })
    .then(() => { weWentFullscreen = true; })
    .catch(() => { armedFullscreen = true; });   // needs a gesture — wait for one
}

function syncFullscreen() {
  if (wantsFullscreen()) {
    enterFullscreen();
    return;
  }
  armedFullscreen = false;
  // Only undo what we did: a fullscreen the user asked for is theirs to keep.
  if (weWentFullscreen && document.fullscreenElement) {
    weWentFullscreen = false;
    document.exitFullscreen?.().catch(() => {});
  }
}

addEventListener('pointerdown', () => {
  if (!armedFullscreen || !wantsFullscreen()) return;
  armedFullscreen = false;
  enterFullscreen();
}, { capture: true });

landscapeQuery.addEventListener?.('change', syncFullscreen);
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) weWentFullscreen = false;
});

wirePhotoDrag();
ui.photoClose.addEventListener('click', closePhotos);
ui.photoPrev.addEventListener('click', () => stepPhoto(-1));
ui.photoNext.addEventListener('click', () => stepPhoto(1));
ui.photoSave.addEventListener('click', () => saveShot(state.shots[state.photoIndex]));
ui.photoShare.addEventListener('click', sharePhoto);
// Clicking the backdrop closes; clicking the row itself must not.
ui.photos.addEventListener('pointerdown', (e) => {
  if (e.target === ui.photos) closePhotos();
});
addEventListener('keydown', (e) => {
  if (ui.photos.hidden) return;
  if (e.key === 'Escape') closePhotos();
  else if (e.key === 'ArrowLeft') stepPhoto(-1);
  else if (e.key === 'ArrowRight') stepPhoto(1);
  else return;
  e.preventDefault();
});

// With ?debug in the URL, expose the internals so the pieces that need a real
// camera to exercise (skeleton mapping, card states, blend roles) can still be
// driven from the console. See selftest.html for the shader side.
if (location.search.includes('debug')) {
  // A live readout of every finger's extension, so a gesture that misfires on
  // camera can be diagnosed from the numbers rather than from guesswork.
  ui.probe = document.createElement('div');
  ui.probe.id = 'probe';
  document.body.appendChild(ui.probe);
  window.__pv = { state, ui, drawSkeleton, paintSlotUI, toggleBlend, toggleSkeleton,
                  toggleSwap, morphAt, stationMeta, fitFor, sizeFor,
                  fingertipsTouching, updateWindowLatch, windowPolygon, hull,
                  curlStrength, backTouch, fingerReach, handOpen, updateBackGestures,
                  isFist, updateReset, fireReset, shutterHand, updateShutter,
                  capture, addShot, renderStrip, saveShot, openPhotos,
                  closePhotos, stepPhoto, paintPhotos, layoutCarousel,
                  buildCarousel, pushTrail, amountBefore, strengthFrom,
                  sfx, toggleSound, syncFullscreen, wantsFullscreen,
                  viewBox, screenPoint, updateAir, updatePaint, clearPaint,
                  toggleLock, toggleDraw, cycleBrush, setHueFrom, airRect,
                  compositeFrame, airTip, paintStroke, BRUSHES,
                  pinWindow, hintText,
                  paintPinUI, clearPinned, Renderer };
}

buildRackStrip();
paintPinUI();
buildMeters();
setPill(ui.skelBtn, state.skeleton);
setPill(ui.blendBtn, state.mode === 'blend');
setPill(ui.swapBtn, state.swap);
setPill(ui.soundBtn, sfx.on);
setPill(ui.lockBtn, state.locked);
setPill(ui.drawBtn, state.drawing);
paintHueUI();
paintBrushUI();

// Installable, and the CDN payload is cached so the tracker starts warm.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => { /* file:// or denied */ });
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === '1') cycleSlot(0, false);
  if (e.key === '2') cycleSlot(1, false);
  if (e.key.toLowerCase() === 'c') toggleBlend();
  if (e.key.toLowerCase() === 'l') toggleSkeleton();
  if (e.key.toLowerCase() === 'h') toggleSwap();
  if (e.key.toLowerCase() === 'r') state.recording ? stopRecording() : startRecording();
});

// iOS reclaims the camera when the tab is hidden; restart cleanly on return.
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.recording) stopRecording();
});
