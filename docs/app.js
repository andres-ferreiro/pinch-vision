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
import { Renderer, EFFECTS, CLEAN } from './effects.js';

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
  morphBtn: $('#morph'),
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
      fill: $('#slot-1 .bar-fill'), hand: $('#slot-1 .slot-hand') },
    { root: $('#slot-2'), name: $('#slot-2 .fx-name'), pct: $('#slot-2 .fx-pct'),
      fill: $('#slot-2 .bar-fill'), hand: $('#slot-2 .slot-hand') },
  ],
};

const state = {
  renderer: null,
  landmarker: null,
  running: false,
  morph: false,
  fps: 0,
  lastFrame: 0,
  handsSeen: 0,
  slots: [
    { fx: 0, amount: 0, target: 0, wasTapping: false, lastTap: 0, present: false },
    { fx: 1, amount: 0, target: 0, wasTapping: false, lastTap: 0, present: false },
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

/** Morph: strength walks the chain clean -> ... -> invert, crossfading. */
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

function sizeFor(video) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const isPhone = Math.min(screen.width, screen.height) < 600;
  const cap = isPhone ? 900 : 1280;
  const scale = Math.min(1, cap / Math.max(vw, vh));
  return [Math.round(vw * scale), Math.round(vh * scale)];
}

function slotDescriptor(slot) {
  if (state.morph) {
    const m = morphAt(slot.amount);
    return { morph: { from: m.from, to: m.to, mix: m.mix }, amount: 1 };
  }
  return { effect: EFFECTS[slot.fx].id, amount: slot.amount };
}

function paintSlotUI(index) {
  const slot = state.slots[index];
  const el = ui.slots[index];
  let meta, label, value;
  if (state.morph) {
    const m = morphAt(slot.amount);
    const from = stationMeta(m.from), to = stationMeta(m.to);
    meta = m.mix > 0.5 ? to : from;
    label = `${from.short} › ${to.short}`;
    value = slot.amount;
  } else {
    meta = EFFECTS[slot.fx];
    label = meta.name;
    value = slot.amount;
  }
  el.name.textContent = label;
  el.pct.textContent = `${Math.round(value * 100)}%`;
  el.fill.style.width = `${value * 100}%`;
  el.root.style.setProperty('--accent', meta.color);
  el.root.classList.toggle('active', slot.present);
}

function loop() {
  if (!state.running) return;
  requestAnimationFrame(loop);

  const video = ui.video;
  if (video.readyState < 2) return;

  const [w, h] = sizeFor(video);
  state.renderer.setSize(w, h);

  const now = performance.now();
  const result = state.landmarker.detectForVideo(video, now);

  const seen = [false, false];
  for (let i = 0; i < result.landmarks.length; i++) {
    // The JS API renamed this field; accept either spelling.
    const handed = result.handedness || result.handednesses || [];
    const label = handed[i]?.[0]?.categoryName;
    // The preview is mirrored, so MediaPipe's label is the opposite of the
    // user's real hand — flip it back.
    const slotIndex = label === 'Left' ? 1 : 0;
    if (seen[slotIndex]) continue;
    seen[slotIndex] = true;

    const slot = state.slots[slotIndex];
    const measure = readHand(result.landmarks[i]);
    slot.target = strengthFrom(measure.pinch);
    if (tapped(slot, measure, now)) cycleSlot(slotIndex, true);
  }

  for (let i = 0; i < 2; i++) {
    const slot = state.slots[i];
    slot.present = seen[i];
    if (!seen[i]) { slot.target = 0; slot.wasTapping = false; }
    slot.amount += (slot.target - slot.amount) * SMOOTHING;
    paintSlotUI(i);
  }

  state.handsSeen = seen.filter(Boolean).length;
  state.renderer.draw(video, state.slots.map(slotDescriptor),
    { mirror: true, time: now / 1000 });

  const dt = now - state.lastFrame;
  state.lastFrame = now;
  if (dt > 0) state.fps = state.fps * 0.9 + (1000 / dt) * 0.1;
  ui.fps.textContent = `${state.fps.toFixed(0)} FPS`;
  ui.handsBadge.textContent = `${state.handsSeen} ✋`;
  if (state.recording) tickRecording(now);
}

// -------------------------------------------------------------------- actions

function cycleSlot(index, fromGesture) {
  const slot = state.slots[index];
  if (state.morph) {
    // In morph mode there is no fixed effect to advance — nudge the strength to
    // the next station instead, and let the smoothing glide it there.
    const n = EFFECTS.length;
    const k = Math.round(clamp01(slot.amount) * n);
    slot.target = ((k + 1) % (n + 1)) / n;
    slot.amount = slot.target;      // gesture nudges are instant targets
    toast(`${index === 0 ? 'LEFT' : 'RIGHT'} → ${stationMeta(STATIONS[(k + 1) % (n + 1)]).name}`);
  } else {
    slot.fx = (slot.fx + 1) % EFFECTS.length;
    toast(`${index === 0 ? 'LEFT' : 'RIGHT'} → ${EFFECTS[slot.fx].name}`);
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

function toggleMorph() {
  state.morph = !state.morph;
  ui.morphBtn.classList.toggle('on', state.morph);
  ui.morphBtn.setAttribute('aria-pressed', String(state.morph));
  toast(state.morph ? 'MORPH — pinch sweeps the whole rack' : 'FIXED — one effect per hand');
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
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
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
  paintSlotUI(0);
  paintSlotUI(1);
  requestAnimationFrame(loop);
}

ui.startBtn.addEventListener('click', begin);
ui.morphBtn.addEventListener('click', toggleMorph);
ui.slots.forEach((el, i) => el.root.addEventListener('click', () => cycleSlot(i, false)));
ui.recordBtn.addEventListener('click', () =>
  state.recording ? stopRecording() : startRecording());
ui.again.addEventListener('click', () => { ui.result.hidden = true; });

document.addEventListener('keydown', (e) => {
  if (e.key === '1') cycleSlot(0, false);
  if (e.key === '2') cycleSlot(1, false);
  if (e.key.toLowerCase() === 'c') toggleMorph();
  if (e.key.toLowerCase() === 'r') state.recording ? stopRecording() : startRecording();
});

// iOS reclaims the camera when the tab is hidden; restart cleanly on return.
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.recording) stopRecording();
});
