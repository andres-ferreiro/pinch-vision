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
import { Renderer, EFFECTS, CLEAN } from './effects.js?v=10';

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

// Clearing: turn both palms away from the camera and hold. Orientation is the
// one thing about a hand that no other gesture here touches.
const CLEAR_HOLD_MS = 1000;
// After a pin the hands are still in frame position, so the fingertip latch
// would immediately open a new window (or toggle one shut). Ignore it until the
// hands have plainly moved on.
const LATCH_LOCK_MS = 900;
const PALM_EPS = 0.06;       // below this the hand is edge-on; sign is noise
const PINKY_MCP = 17;
// Clenching drags the mask corners inward, so the shape at the instant the fist
// lands is already ruined. Pin what the hands were holding a moment earlier.
const PIN_LOOKBACK_MS = 220;
const HISTORY_MS = 600;
const MAX_PINNED = 4;
const CURL_OPEN = 1.02;      // |middle tip - knuckle| / span, finger extended
const CURL_SHUT = 0.42;      // the same, finger folded into the palm
const SMOOTHING = 0.30;      // per-frame easing on every strength value

const MAX_RECORD_MS = 20000;
const STATIONS = ['clean', ...EFFECTS.map((e) => e.id)];

// Blend mode sweeps the rack at a hard speed limit rather than tracking the
// pinch directly: eight effects across one pinch made every twitch a jump cut.
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
  overlay: $('#overlay'),
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
  clearCharge: 0,
  clearArmed: true,
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
      palmHome: 0, flipStart: 0 },
    { fx: 1, amount: 0, target: 0, pinch: 1, wasTapping: false, lastTap: 0,
      present: false, points: null, pinching: false, backHeld: false,
      backStart: 0, backFired: false, pinSnapshot: null, measure: null,
      palmHome: 0, flipStart: 0 },
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
  const r = points[RING_TIP];
  return {
    pinch: Math.hypot(t.x - i.x, t.y - i.y) / span,
    tap: Math.hypot(t.x - p.x, t.y - p.y) / span,
    ring: Math.hypot(t.x - r.x, t.y - r.y) / span,
  };
}

function strengthFrom(pinch) {
  return clamp01(1 - (pinch - SQUEEZE_MIN) / (SQUEEZE_MAX - SQUEEZE_MIN));
}

/** Middle-finger extension, 0 (curled) to 1 (straight). */
function curlStrength(points) {
  const span = spanOf(points);
  const reach = Math.hypot(points[MIDDLE_TIP].x - points[MIDDLE_MCP].x,
                           points[MIDDLE_TIP].y - points[MIDDLE_MCP].y) / span;
  return clamp01((reach - CURL_SHUT) / (CURL_OPEN - CURL_SHUT));
}

/**
 * Which way the palm faces, as the sign of the wrist→index-knuckle ×
 * wrist→pinky-knuckle cross product. Turning a hand over flips it.
 *
 * The sign itself is meaningless without knowing which hand it is and whether
 * the image is mirrored — so nothing here assumes a convention. Each hand's
 * usual orientation is learned at runtime, and a *change* from it is the
 * gesture. Near zero the hand is edge-on and the sign is noise, so it reports
 * 0 and the caller holds its last opinion.
 */
function palmSign(points) {
  const w = points[WRIST], i = points[INDEX_MCP], p = points[PINKY_MCP];
  const span = spanOf(points);
  const cross = ((i.x - w.x) * (p.y - w.y) - (i.y - w.y) * (p.x - w.x)) / (span * span);
  return Math.abs(cross) < PALM_EPS ? 0 : Math.sign(cross);
}

/**
 * Both palms turned away from their usual orientation, held for a second,
 * clears every window and returns to normal.
 *
 * It re-arms only once the hands come back, so turning them over and back is
 * one clear, not two — the baseline orientation is never rewritten by the
 * gesture itself.
 */
function updateTwist(now) {
  const live = state.slots.filter((slot) => slot.present && slot.points);
  if (!live.length) { state.clearCharge = 0; return; }

  let flippedSince = 0;
  let allFlipped = true;
  let allHome = true;

  for (const slot of live) {
    const sign = palmSign(slot.points);
    if (sign === 0) { allFlipped = false; allHome = false; continue; }
    if (slot.palmHome === 0) slot.palmHome = sign;        // learn "normal" once

    if (sign === slot.palmHome) {
      slot.flipStart = 0;
      allFlipped = false;
    } else {
      allHome = false;
      if (!slot.flipStart) slot.flipStart = now;
      flippedSince = Math.max(flippedSince, slot.flipStart);
    }
  }

  if (allHome) state.clearArmed = true;
  if (!allFlipped || !state.clearArmed || !flippedSince) {
    state.clearCharge = 0;
    return;
  }

  state.clearCharge = clamp01((now - flippedSince) / CLEAR_HOLD_MS);
  if (state.clearCharge < 1) return;

  state.clearArmed = false;
  state.clearCharge = 0;
  state.contactPrev = true;
  state.latchLockedUntil = now + LATCH_LOCK_MS;
  navigator.vibrate?.([14, 50, 14]);

  // Dismiss what you are holding; if you are holding nothing, wipe the board.
  if (state.masked) {
    state.masked = false;
    state.maskPoints = null;
    state.history.length = 0;
    toast('WINDOW DISMISSED');
  } else if (state.pinned.length) {
    clearPinned(true);
  } else {
    toast('NOTHING TO CLEAR');
  }
}

/** How close the thumb is to the back of the hand — ring or pinky, whichever. */
function backTouch(slot) {
  return slot.measure ? Math.min(slot.measure.tap, slot.measure.ring) : 1;
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
  const contact = fingertipsTouching();
  // Still tracked, just not acted on: when the lock lifts, an unbroken contact
  // is not a fresh edge and so cannot fire.
  if (now < state.latchLockedUntil) {
    state.contactPrev = contact;
    return;
  }
  if (contact && !state.contactPrev && !state.masked) {
    state.masked = true;
    toast('WINDOW ON — hold thumb to ring to pin');
    navigator.vibrate?.(14);
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
function hintText() {
  if (state.recording) return '';
  if (!state.handsSeen) return 'SHOW YOUR HANDS';
  if (state.masked) return 'HOLD THUMB TO RING TO PIN · TURN PALMS AWAY TO DISMISS';
  if (state.handsSeen < 2) return 'PINCH TO DIAL · TAP THUMB TO PINKY FOR THE NEXT EFFECT';
  if (state.pinned.length) return 'TOUCH FINGERTIPS FOR ANOTHER WINDOW';
  return 'TOUCH FINGERTIPS TO OPEN A WINDOW';
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
      slot.flipStart = 0;
      slot.measure = null;
      state.contact = false;
    }
  }

  updateWindowLatch(now);

  // Exactly one rule decides each hand's strength, and it eases exactly once.
  // (Setting a pinch target here and then a second target below made the two
  // pulls meet in the middle and park at a fixed 59%.)
  for (const slot of state.slots) {
    if (!slot.present) slot.target = 0;
    else if (state.masked) slot.target = curlStrength(slot.points);
    else slot.target = strengthFrom(slot.pinch);
    slot.amount += (slot.target - slot.amount) * SMOOTHING;
  }

  if (state.mode === 'blend') {
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
  updateTwist(now);
  const showMask = state.masked && state.maskPoints;
  state.maskStrength += ((showMask ? 1 : 0) - state.maskStrength) * MASK_EASE;
  if (state.maskStrength < 0.004) {
    state.maskStrength = 0;
    if (!state.masked) state.maskPoints = null;
  }
  ui.maskBadge.hidden = !state.masked;
  if (state.clearCharge > 0.02) {
    ui.maskBadge.hidden = false;
    ui.maskBadge.textContent = `CLEARING ${Math.round(state.clearCharge * 100)}%`;
    ui.maskBadge.classList.add('charging');
  } else if (state.pinCharge > 0.02) {
    ui.maskBadge.textContent = `PINNING ${Math.round(state.pinCharge * 100)}%`;
    ui.maskBadge.classList.add('charging');
  } else {
    ui.maskBadge.textContent = 'WINDOW';
    ui.maskBadge.classList.remove('charging');
  }
  if (ui.probe) {
    const fmt = (slot) => slot.reach && slot.present
      ? `thumb→ring ${slot.measure ? slot.measure.ring.toFixed(2) : '—'} `
        + `thumb→pinky ${slot.measure ? slot.measure.tap.toFixed(2) : '—'}`
        + (slot.backHeld ? ` HELD ${Math.round(performance.now() - slot.backStart)}ms` : '')
      : '—';
    ui.probe.textContent = `L ${fmt(state.slots[0])}  |  R ${fmt(state.slots[1])}`;
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
  drawSkeleton();

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

    // Pin charge, drawn where the thumb meets the ring finger so the wait has
    // an obvious anchor on the hand doing it.
    if (slot.backHeld && state.masked) {
      const [rx, ry] = at(slot.points[RING_TIP]);
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
  const stream = ui.canvas.captureStream(30);
  try {
    state.recorder = new MediaRecorder(stream, mime
      ? { mimeType: mime, videoBitsPerSecond: 6_000_000 }
      : undefined);
  } catch (err) {
    toast('recording unsupported here');
    return;
  }
  state.chunks = [];
  state.recorder.ondataavailable = (e) => { if (e.data.size) state.chunks.push(e.data); };
  state.recorder.onstop = finishRecording;
  state.recorder.start(200);
  state.recording = true;
  state.recStart = performance.now();
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
  ui.recordBtn.classList.remove('recording');
  ui.stage.classList.remove('recording');
  ui.ring.style.strokeDashoffset = '163.4';
  ui.recTime.textContent = '';
  state.recorder?.stop();
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

function toggleSkeleton() {
  state.skeleton = !state.skeleton;
  store.set('skeleton', state.skeleton);
  setPill(ui.skelBtn, state.skeleton);
  toast(state.skeleton ? 'SKELETON ON' : 'SKELETON OFF');
}

function toggleSwap() {
  state.swap = !state.swap;
  store.set('swap', state.swap);
  setPill(ui.swapBtn, state.swap);
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
                  palmSign, updateTwist, pinWindow, hintText, paintPinUI,
                  clearPinned, Renderer };
}

buildRackStrip();
paintPinUI();
buildMeters();
setPill(ui.skelBtn, state.skeleton);
setPill(ui.blendBtn, state.mode === 'blend');
setPill(ui.swapBtn, state.swap);

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
