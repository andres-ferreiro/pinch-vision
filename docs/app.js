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
import { Renderer, EFFECTS, CLEAN } from './effects.js?v=8';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker'
            + '/hand_landmarker/float16/1/hand_landmarker.task';

const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9, PINKY_TIP = 20;

// Pinch openness -> strength. Same window as the desktop "squeeze" mode.
const SQUEEZE_MIN = 0.12, SQUEEZE_MAX = 0.90;
// Thumb-to-pinky tap, with hysteresis and a cooldown so a held tap fires once.
const TAP_ON = 0.50, TAP_OFF = 0.72, TAP_COOLDOWN = 450;
const PINCH_OPEN = 0.55;     // index must be clear of the thumb: a tap, not a fist
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
  handsBadge: $('#hands'),
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
  fps: 0,
  lastFrame: 0,
  handsSeen: 0,
  slots: [
    { fx: 0, amount: 0, target: 0, wasTapping: false, lastTap: 0, present: false, points: null },
    { fx: 1, amount: 0, target: 0, wasTapping: false, lastTap: 0, present: false, points: null },
  ],
  recorder: null,
  recording: false,
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
  return {
    pinch: Math.hypot(t.x - i.x, t.y - i.y) / span,
    tap: Math.hypot(t.x - p.x, t.y - p.y) / span,
  };
}

function strengthFrom(pinch) {
  return clamp01(1 - (pinch - SQUEEZE_MIN) / (SQUEEZE_MAX - SQUEEZE_MIN));
}

/** Rising edge of the thumb-to-pinky tap, with cooldown. */
function tapped(slot, measure, now) {
  const closed = slot.wasTapping ? measure.tap < TAP_OFF : measure.tap < TAP_ON;
  const valid = closed && measure.pinch > PINCH_OPEN;
  const fired = valid && !slot.wasTapping && now - slot.lastTap > TAP_COOLDOWN;
  slot.wasTapping = valid;
  if (fired) slot.lastTap = now;
  return fired;
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
    slot.target = strengthFrom(measure.pinch);
    if (tapped(slot, measure, now)) cycleSlot(slotIndex, true);
  }

  const dtSec = Math.min(0.1, (now - state.lastFrame) / 1000) || 0.016;

  for (let i = 0; i < 2; i++) {
    const slot = state.slots[i];
    slot.present = seen[i];
    if (!seen[i]) { slot.target = 0; slot.wasTapping = false; slot.points = null; }
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

  paintSlotUI(0);
  paintSlotUI(1);

  state.handsSeen = seen.filter(Boolean).length;
  const plan = renderPlan();
  state.renderer.draw(video, plan.slots,
    { mirror: true, time: now / 1000, fade: plan.fade });
  drawSkeleton();

  const dt = now - state.lastFrame;
  state.lastFrame = now;
  if (dt > 0) state.fps = state.fps * 0.9 + (1000 / dt) * 0.1;
  ui.fps.textContent = `${state.fps.toFixed(0)} FPS`;
  ui.handsBadge.textContent = `${state.handsSeen} ✋`;
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
  state.recording = false;
  ui.recordBtn.classList.remove('recording');
  ui.stage.classList.remove('recording');
  ui.ring.style.strokeDashoffset = '163.4';
  ui.recTime.textContent = '';
  state.recorder?.stop();
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
ui.slots.forEach((el, i) => el.root.addEventListener('click', () => cycleSlot(i, false)));
ui.recordBtn.addEventListener('click', () =>
  state.recording ? stopRecording() : startRecording());
ui.again.addEventListener('click', () => { ui.result.hidden = true; });

// With ?debug in the URL, expose the internals so the pieces that need a real
// camera to exercise (skeleton mapping, card states, blend roles) can still be
// driven from the console. See selftest.html for the shader side.
if (location.search.includes('debug')) {
  window.__pv = { state, ui, drawSkeleton, paintSlotUI, toggleBlend, toggleSkeleton,
                  toggleSwap, morphAt, stationMeta, Renderer };
}

buildMeters();
setPill(ui.skelBtn, state.skeleton);
setPill(ui.blendBtn, state.mode === 'blend');
setPill(ui.swapBtn, state.swap);

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
