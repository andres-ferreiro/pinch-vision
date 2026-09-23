/**
 * WebGL2 effect rack.
 *
 * Every look from the desktop build re-expressed as a fragment shader. The
 * chain works on a pool of framebuffers: the video lands in one, each active
 * effect reads the previous target and writes the next, and the last pass draws
 * to the canvas. Echo trails is the one effect with state, so it keeps its own
 * ping-pong pair — you cannot read and write the same texture in one pass.
 */

export const EFFECTS = [
  { id: 'thermal', name: 'THERMAL', short: 'THERM', color: '#ff9c3c' },
  { id: 'night', name: 'NIGHT VISION', short: 'NIGHT', color: '#78ff78' },
  { id: 'edge', name: 'EDGE GLOW', short: 'EDGE', color: '#3cbeff' },
  { id: 'echo', name: 'ECHO TRAILS', short: 'ECHO', color: '#ff6edc' },
  { id: 'glitch', name: 'GLITCH', short: 'GLTCH', color: '#ff5a5a' },
  { id: 'ascii', name: 'ASCII', short: 'ASCII', color: '#d8ffc8' },
  { id: 'halftone', name: 'HALFTONE', short: 'HALFT', color: '#e6e6e6' },
  { id: 'invert', name: 'INVERT', short: 'INVRT', color: '#8c50ff' },
];

export const CLEAN = { id: 'clean', name: 'CLEAN', short: 'CLEAN', color: '#9aa0a6' };

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uAmount;
uniform float uTime;
uniform vec2 uRes;
in vec2 vUv;
out vec4 fragColor;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
vec3 blend(vec3 a, vec3 b, float t) { return mix(a, b, clamp(t, 0.0, 1.0)); }
`;

// --------------------------------------------------------------- the shaders

const SRC = HEAD + `
uniform float uMirror;
void main() {
  vec2 uv = vec2(uMirror > 0.5 ? 1.0 - vUv.x : vUv.x, vUv.y);
  fragColor = vec4(texture(uTex, uv).rgb, 1.0);
}`;

const PRESENT = HEAD + `
void main() { fragColor = vec4(texture(uTex, vUv).rgb, 1.0); }`;

// Final composite. Without a window it is a plain crossfade; with one, the
// effect chain is clipped to the quad the two pinches make, and the boundary
// gets a hairline so the window reads as a real object held in the hands.
const MASK_MIX = HEAD + `
uniform sampler2D uTexB;
uniform vec2 uPoly[8];
uniform int uCount;
uniform float uMask;

// Signed distance to a directed edge: positive on its left, so a
// counter-clockwise polygon is positive everywhere inside it.
float edgeDist(vec2 p, vec2 a, vec2 b) {
  vec2 e = b - a;
  return (e.x * (p.y - a.y) - e.y * (p.x - a.x)) / max(length(e), 1e-5);
}

void main() {
  vec3 clean = texture(uTex, vUv).rgb;
  vec3 styled = texture(uTexB, vUv).rgb;

  // Work in pixels, or the feather would stretch with the aspect ratio.
  vec2 p = vUv * uRes;
  float d = 1e9;
  for (int i = 0; i < 8; i++) {
    if (i >= uCount) break;
    int j = i + 1;
    if (j >= uCount) j = 0;
    d = min(d, edgeDist(p, uPoly[i] * uRes, uPoly[j] * uRes));
  }

  float inside = smoothstep(-1.5, 1.5, d);
  float keep = mix(1.0, inside, uMask);        // uMask 0 -> the whole frame
  vec3 col = mix(clean, styled, clamp(uAmount, 0.0, 1.0) * keep);

  float line = (1.0 - smoothstep(0.0, 2.0, abs(d))) * uMask;
  col = mix(col, vec3(1.0), line * 0.8);
  fragColor = vec4(col, 1.0);
}`;

const MIXER = HEAD + `
uniform sampler2D uTexB;
void main() {
  vec3 a = texture(uTex, vUv).rgb;
  vec3 b = texture(uTexB, vUv).rgb;
  fragColor = vec4(mix(a, b, clamp(uAmount, 0.0, 1.0)), 1.0);
}`;

// Ironbow ramp, same stops as the Python LLUT.
const IRONBOW = `
vec3 ironbow(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = mix(vec3(0.05, 0.03, 0.16), vec3(0.12, 0.08, 0.43), smoothstep(0.00, 0.16, t));
  c = mix(c, vec3(0.43, 0.10, 0.55), smoothstep(0.16, 0.34, t));
  c = mix(c, vec3(0.78, 0.16, 0.35), smoothstep(0.34, 0.52, t));
  c = mix(c, vec3(1.00, 0.43, 0.12), smoothstep(0.52, 0.70, t));
  c = mix(c, vec3(1.00, 0.86, 0.27), smoothstep(0.70, 0.86, t));
  c = mix(c, vec3(1.00, 1.00, 1.00), smoothstep(0.86, 1.00, t));
  return c;
}`;

const THERMAL = HEAD + IRONBOW + `
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  // Real sensors are low-res and soft: a blurred mip gives that blobbiness free.
  vec3 soft = textureLod(uTex, vUv, 2.5).rgb;
  float y = luma(soft);
  float cr = (soft.r - y) * 0.713 + 0.5;
  float cb = (soft.b - y) * 0.564 + 0.5;

  // Skin sits in a tight pocket of the Cr-Cb plane, far from walls and screens,
  // which is why heat is estimated from chroma and only lightly from brightness.
  float skin = clamp(1.0 - abs(cr * 255.0 - 152.0) / 28.0, 0.0, 1.0);
  skin *= clamp(((cr - cb) * 255.0 + 20.0) / 55.0, 0.0, 1.0);
  float heat = 0.70 * skin + 0.30 * y * (0.35 + 0.65 * skin);

  // Auto-range against the scene average, read from the smallest mip.
  float mean = luma(textureLod(uTex, vec2(0.5), 12.0).rgb);
  float lo = mean * 0.22;
  // Range to 0.78 rather than 0.55: skin should land high on the ramp, not
  // clip flat white and lose every feature in the face.
  heat = clamp((heat - lo) / max(0.78 - lo, 0.30), 0.0, 1.0);
  heat += 0.22 * (luma(base) - y);            // put the sharp detail back
  heat = clamp(heat, 0.0, 1.0);

  vec3 hot = ironbow(heat);
  hot += vec3(0.30, 0.18, 0.08) * smoothstep(0.88, 1.0, heat);   // bloom
  fragColor = vec4(blend(base, hot, uAmount), 1.0);
}`;

const NIGHT = HEAD + `
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  float g = luma(base);

  // Auto-exposure from the 1x1 mip, then a soft knee. Multiplying by a gain
  // would pin a lit room to white; 1 - exp(-gain*x) approaches it instead.
  float mean = max(luma(textureLod(uTex, vec2(0.5), 12.0).rgb), 0.02);
  float gain = clamp(0.34 / mean, 0.35, 6.0);
  float amp = 1.0 - exp(-gain * (0.9 + 1.6 * uAmount) * g);

  amp += (hash(vUv * uRes + uTime) - 0.5) * 0.18 * uAmount;      // tube noise
  vec2 d = vUv - 0.5;
  amp *= clamp(1.15 - 0.62 * dot(d, d) * 4.0, 0.25, 1.0);        // vignette
  amp *= mod(floor(vUv.y * uRes.y), 3.0) < 1.0 ? 0.82 : 1.0;     // scanlines
  amp = clamp(amp, 0.0, 1.0);

  float s = pow(amp, 0.8);
  vec3 green = vec3(s * 0.25, s, s * 0.32) + vec3(0.05, 0.12, 0.06) * s * s;
  fragColor = vec4(blend(base, green, uAmount), 1.0);
}`;

const EDGE = HEAD + `
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  vec2 px = 1.0 / uRes;
  float tl = luma(texture(uTex, vUv + px * vec2(-1, -1)).rgb);
  float tc = luma(texture(uTex, vUv + px * vec2( 0, -1)).rgb);
  float tr = luma(texture(uTex, vUv + px * vec2( 1, -1)).rgb);
  float ml = luma(texture(uTex, vUv + px * vec2(-1,  0)).rgb);
  float mr = luma(texture(uTex, vUv + px * vec2( 1,  0)).rgb);
  float bl = luma(texture(uTex, vUv + px * vec2(-1,  1)).rgb);
  float bc = luma(texture(uTex, vUv + px * vec2( 0,  1)).rgb);
  float br = luma(texture(uTex, vUv + px * vec2( 1,  1)).rgb);
  float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  float mag = clamp(length(vec2(gx, gy)) * (1.4 + 2.2 * uAmount), 0.0, 1.0);

  vec3 neon = mix(vec3(0.0), vec3(0.04, 0.16, 0.35), smoothstep(0.0, 0.40, mag));
  neon = mix(neon, vec3(0.16, 0.71, 1.0), smoothstep(0.40, 0.75, mag));
  neon = mix(neon, vec3(0.92, 1.0, 1.0), smoothstep(0.75, 1.0, mag));
  fragColor = vec4(blend(base, neon, uAmount), 1.0);
}`;

// Two passes: one advances the trail state, one composites it over the frame.
const ECHO_STATE = HEAD + `
uniform sampler2D uState;
uniform float uPrime;
void main() {
  float cur = luma(textureLod(uTex, vUv, 1.0).rgb);
  // First frame has no history. Diffing against an empty texture would mark
  // the entire image as motion and wash the frame in trail colour, so seed it.
  if (uPrime > 0.5) { fragColor = vec4(0.0, cur, 0.0, 1.0); return; }
  vec2 st = texture(uState, vUv).rg;           // r = trail, g = previous luma
  float motion = clamp(abs(cur - st.g) * (3.0 + 6.0 * uAmount), 0.0, 1.0);
  float decay = 0.72 + 0.26 * uAmount;
  fragColor = vec4(max(st.r * decay, motion), cur, 0.0, 1.0);
}`;

const ECHO_COMP = HEAD + `
uniform sampler2D uState;
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  float t = 0.0;
  vec2 px = 1.0 / uRes;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      t += texture(uState, vUv + px * vec2(float(i), float(j)) * 2.0).r;
    }
  }
  t = clamp(t / 9.0, 0.0, 1.0);

  vec3 neon = mix(vec3(0.0), vec3(0.35, 0.08, 0.59), smoothstep(0.0, 0.35, t));
  neon = mix(neon, vec3(1.0, 0.24, 0.71), smoothstep(0.35, 0.62, t));
  neon = mix(neon, vec3(1.0, 0.92, 1.0), smoothstep(0.62, 1.0, t));

  // Painted over rather than added: adding light to a bright room just blows
  // out to white and the trail loses its colour.
  float alpha = t * (0.55 + 0.45 * uAmount);
  fragColor = vec4(mix(base, neon, alpha * uAmount), 1.0);
}`;

const GLITCH = HEAD + `
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  float step_t = floor(uTime * 12.0);
  float row = floor(vUv.y * 42.0);
  float roll = hash(vec2(row, step_t));
  vec2 uv = vUv;
  if (roll < 0.25 + 0.2 * uAmount) {
    uv.x += (hash(vec2(row, step_t + 7.0)) - 0.5) * 0.18 * uAmount;
  }
  float sh = uAmount * 0.022;
  float r = texture(uTex, uv + vec2(sh, 0.0)).r;
  float g = texture(uTex, uv).g;
  float b = texture(uTex, uv - vec2(sh, 0.0)).b;
  vec3 styled = vec3(r, g, b);
  if (hash(vec2(row * 3.0, step_t)) < 0.05 * uAmount) styled = styled * 1.6 + 0.15;
  fragColor = vec4(blend(base, styled, clamp(uAmount * 1.3, 0.0, 1.0)), 1.0);
}`;

const ASCII = HEAD + `
uniform sampler2D uAtlas;
const float GLYPHS = 10.0;
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  vec2 cellPx = vec2(10.0, 16.0);
  vec2 grid = uRes / cellPx;
  vec2 cell = floor(vUv * grid);
  vec2 inCell = fract(vUv * grid);

  vec3 cellColor = textureLod(uTex, (cell + 0.5) / grid, 3.0).rgb;
  float mean = luma(textureLod(uTex, vec2(0.5), 12.0).rgb);
  // Stretch around the scene mean so a bright wall does not saturate every
  // cell to the densest glyph.
  float g = clamp((luma(cellColor) - mean * 0.55) / max(mean * 0.9, 0.18), 0.0, 1.0);

  float idx = floor(g * (GLYPHS - 1.0) + 0.5);
  vec2 auv = vec2((idx + inCell.x) / GLYPHS, 1.0 - inCell.y);
  float mask = texture(uAtlas, auv).r;
  vec3 styled = cellColor * 1.5 * mask;
  fragColor = vec4(blend(base, styled, uAmount), 1.0);
}`;

const HALFTONE = HEAD + `
float clustered(vec2 p) {
  int x = int(mod(p.x, 8.0));
  int y = int(mod(p.y, 8.0));
  int i = y * 8 + x;
  float m[64] = float[64](
    24., 10., 12., 26., 35., 47., 49., 37.,
     8.,  0.,  2., 14., 45., 59., 61., 51.,
    22.,  6.,  4., 16., 43., 57., 63., 53.,
    30., 20., 18., 28., 33., 41., 55., 39.,
    34., 46., 48., 36., 25., 11., 13., 27.,
    44., 58., 60., 50.,  9.,  1.,  3., 15.,
    42., 56., 62., 52., 23.,  7.,  5., 17.,
    32., 40., 54., 38., 31., 21., 19., 29.);
  return m[i] / 64.0;
}
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  float g = clamp(luma(base) * 1.15 + 0.04, 0.0, 1.0);
  float ink = step(clustered(gl_FragCoord.xy), g);
  vec3 poster = floor(base * 5.0) / 5.0 + 0.1;
  vec3 paper = mix(poster, vec3(1.0), 0.25);
  fragColor = vec4(blend(base, paper * ink, uAmount), 1.0);
}`;

const INVERT = HEAD + `
void main() {
  vec3 base = texture(uTex, vUv).rgb;
  // Crossfading an image with its own negative is flat grey at 50%. Sweeping
  // the inversion threshold down from white solarizes on the way instead.
  float th = 1.0 - uAmount;
  vec3 t = smoothstep(vec3(th - 0.05), vec3(th + 0.05), base);
  fragColor = vec4(mix(base, 1.0 - base, t), 1.0);
}`;

const SHADERS = {
  src: SRC, present: PRESENT, mixer: MIXER, maskMix: MASK_MIX,
  thermal: THERMAL, night: NIGHT, edge: EDGE, glitch: GLITCH,
  ascii: ASCII, halftone: HALFTONE, invert: INVERT,
  echoState: ECHO_STATE, echoComp: ECHO_COMP,
};

// ------------------------------------------------------------------ renderer

function compile(gl, type, source) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
  }
  return sh;
}

function program(gl, fragSource, name) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  try {
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragSource));
  } catch (err) {
    throw new Error(`${name}: ${err.message}`);
  }
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`${name}: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

function buildAtlas() {
  // One canvas of ten glyphs, uploaded once. Sampling a texture is far cheaper
  // than drawing thousands of characters per frame.
  const chars = ' .:-=+*#%@';
  const cw = 20, ch = 32;
  const c = document.createElement('canvas');
  c.width = cw * chars.length;
  c.height = ch;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = '#fff';
  ctx.font = `600 ${Math.round(ch * 0.82)}px ui-monospace, "SF Mono", Menlo, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < chars.length; i++) {
    ctx.fillText(chars[i], i * cw + cw / 2, ch / 2 + 1);
  }
  return c;
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    this.canvas = canvas;
    this.width = 0;
    this.height = 0;

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.programs = {};
    for (const [name, source] of Object.entries(SHADERS)) {
      this.programs[name] = program(gl, source, name);
    }

    this.videoTex = this.makeTexture(false);
    this.atlasTex = this.makeTexture(false);
    gl.bindTexture(gl.TEXTURE_2D, this.atlasTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, buildAtlas());

    this.pool = [];
    this.echo = [];
    this.echoIndex = 0;
    this.poolIndex = 0;
    this.echoPrimed = false;
  }

  makeTexture(mip) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
      mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    return tex;
  }

  makeTarget(w, h) {
    const gl = this.gl;
    const tex = this.makeTexture(true);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo };
  }

  setSize(w, h) {
    if (w === this.width && h === this.height) return;
    const gl = this.gl;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    for (const t of [...this.pool, ...this.echo]) {
      gl.deleteTexture(t.tex);
      gl.deleteFramebuffer(t.fbo);
    }
    this.pool = [0, 1, 2, 3, 4, 5, 6, 7].map(() => this.makeTarget(w, h));
    this.echo = [0, 1].map(() => this.makeTarget(w, h));
    this.echoPrimed = false;              // new textures, no history yet
    this.poolIndex = 0;
  }

  nextTarget(avoid = []) {
    for (let i = 0; i < this.pool.length; i++) {
      const t = this.pool[(this.poolIndex + i) % this.pool.length];
      if (!avoid.includes(t)) {
        this.poolIndex = (this.poolIndex + i + 1) % this.pool.length;
        return t;
      }
    }
    return this.pool[0];
  }

  /** Run one program into `target` (or the canvas when target is null). */
  pass(name, target, uniforms = {}, textures = {}) {
    const gl = this.gl;
    const p = this.programs[name];
    gl.useProgram(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, this.width, this.height);

    let unit = 0;
    for (const [uname, tex] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(p, uname), unit);
      unit++;
    }
    gl.uniform2f(gl.getUniformLocation(p, 'uRes'), this.width, this.height);
    for (const [uname, value] of Object.entries(uniforms)) {
      const loc = gl.getUniformLocation(p, uname);
      // A Float32Array is a vec2[] (the mask polygon); a name ending in Count is
      // an int; everything else is a plain float.
      if (value instanceof Float32Array) gl.uniform2fv(loc, value);
      else if (Array.isArray(value)) gl.uniform2f(loc, value[0], value[1]);
      else if (uname.endsWith('Count')) gl.uniform1i(loc, value);
      else gl.uniform1f(loc, value);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (target) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.generateMipmap(gl.TEXTURE_2D);       // later passes sample coarse mips
    }
  }

  /** Apply one effect id to a texture, returning the new target. */
  applyEffect(id, sourceTex, amount, time, keep) {
    if (id === 'clean' || amount <= 0.002) return null;
    if (id === 'echo') {
      if (!this.echoPrimed) {
        this.pass('echoState', this.echo[this.echoIndex], { uPrime: 1 },
          { uTex: sourceTex, uState: this.echo[1 - this.echoIndex].tex });
        this.echoPrimed = true;
      }
      const prev = this.echo[this.echoIndex];
      const next = this.echo[1 - this.echoIndex];
      this.pass('echoState', next, { uAmount: amount, uTime: time, uPrime: 0 },
        { uTex: sourceTex, uState: prev.tex });
      this.echoIndex = 1 - this.echoIndex;
      const out = this.nextTarget(keep);
      this.pass('echoComp', out, { uAmount: amount, uTime: time },
        { uTex: sourceTex, uState: next.tex });
      return out;
    }
    const out = this.nextTarget(keep);
    const textures = { uTex: sourceTex };
    if (id === 'ascii') textures.uAtlas = this.atlasTex;
    this.pass(id, out, { uAmount: amount, uTime: time }, textures);
    return out;
  }

  /**
   * Run a list of slots over `from`, returning the last target written (or
   * `from` itself when nothing was applied). `hold` lists targets the pool must
   * not recycle while this runs.
   */
  chain(from, slots, time, hold = []) {
    let current = from;
    for (const slot of slots) {
      const keep = [...hold, current];
      if (slot.morph) {
        const { from: lo, to: hi, mix } = slot.morph;
        const a = this.applyEffect(lo, current.tex, 1.0, time, keep);
        const lowTex = a ? a.tex : current.tex;
        if (mix <= 0.02) { if (a) current = a; continue; }
        const b = this.applyEffect(hi, current.tex, 1.0, time, [...keep, a].filter(Boolean));
        if (!b) { if (a) current = a; continue; }
        const out = this.nextTarget([...keep, a, b].filter(Boolean));
        this.pass('mixer', out, { uAmount: mix }, { uTex: lowTex, uTexB: b.tex });
        current = out;
      } else {
        const out = this.applyEffect(slot.effect, current.tex, slot.amount, time, keep);
        if (out) current = out;
      }
    }
    return current;
  }

  /** Flatten a polygon into the vec2[8] the mask shader expects. */
  static packPoly(poly) {
    const flat = new Float32Array(16);
    for (let i = 0; i < Math.min(poly.length, 8); i++) {
      flat[i * 2] = poly[i][0];
      flat[i * 2 + 1] = poly[i][1];
    }
    return flat;
  }

  /**
   * slots: [{ effect: id|'clean', amount, morph: {from, to, mix} | null }]
   * Rendered in order, each reading the previous result.
   *
   * `pinned` is a list of { poly, slots } windows stamped into the scene, each
   * carrying the effects it was pinned with; `mask` is the live one. Every
   * region runs its own chain from the clean source and is then composited over
   * the running result inside its own polygon.
   */
  draw(source, slots, { mirror = true, time = 0, fade = 1, mask = null,
                        pinned = [] } = {}) {
    const gl = this.gl;
    // Callers normally pick the render size (it is capped per device), but fall
    // back to the source's own dimensions so a bare draw() still works.
    if (!this.pool.length) {
      this.setSize(source.videoWidth || source.width || 1280,
                   source.videoHeight || source.height || 720);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    const srcTarget = this.nextTarget();
    this.pass('src', srcTarget, { uMirror: mirror ? 1 : 0 }, { uTex: this.videoTex });

    const regions = pinned
      .filter((w) => w.poly && w.poly.length >= 3)
      .map((w) => ({ poly: w.poly, slots: w.slots, strength: 1 }));
    if (mask && mask.strength > 0.002 && mask.points && mask.points.length >= 3) {
      regions.push({ poly: mask.points, slots, strength: mask.strength });
    }

    // Nothing is windowed: the chain simply covers the whole frame.
    if (!regions.length) {
      const current = this.chain(srcTarget, slots, time, [srcTarget]);
      if (fade < 0.999 && current !== srcTarget) {
        this.pass('maskMix', null, {
          uAmount: fade, uMask: 0,
          uPoly: Renderer.packPoly([[0, 0], [1, 0], [1, 1], [0, 1]]), uCount: 4,
        }, { uTex: srcTarget.tex, uTexB: current.tex });
      } else {
        this.pass('present', null, {}, { uTex: current.tex });
      }
      return;
    }

    let base = srcTarget;
    let drawn = false;
    for (let i = 0; i < regions.length; i++) {
      const region = regions[i];
      const styled = this.chain(srcTarget, region.slots, time, [srcTarget, base]);
      if (styled === srcTarget) continue;          // that window has no effect up
      const last = i === regions.length - 1;
      const target = last ? null : this.nextTarget([srcTarget, base, styled]);
      this.pass('maskMix', target, {
        uAmount: fade,
        uMask: region.strength,
        uPoly: Renderer.packPoly(region.poly),
        uCount: Math.min(region.poly.length, 8),
      }, { uTex: base.tex, uTexB: styled.tex });
      if (!last) base = target;
      drawn = true;
    }
    if (!drawn) this.pass('present', null, {}, { uTex: base.tex });
  }
}