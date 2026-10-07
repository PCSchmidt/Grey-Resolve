/**
 * degrade.js -- Grey-Resolve live degradation bench panel.
 *
 * Fills #degrade-panel with a procedural synthetic tile on a canvas plus mode
 * buttons: CLEAN / BLUR / LOW-RES / BRIGHTNESS / OFF-ANGLE. Applying a mode
 * visibly degrades the tile LIVE with real canvas ops:
 *   BLUR       -> ctx.filter blur (downscale/upscale fallback)
 *   LOW-RES    -> imageSmoothing off + nearest-neighbour upscale (x0.1)
 *   BRIGHTNESS -> brightness filter / lighter composite
 *   OFF-ANGLE  -> rotate transform (45 deg)
 * Each mode shows the measured effect labels from the real runs (EER,
 * FMR@FNMR=1%, detection failures, gate pass rate) sourced from ctx.data.runs.
 * Emits 'mode' on every user mode change. Redraws only on interaction and
 * during the short (<=380 ms) switch transition, so it stays well below
 * 60 fps cost at idle.
 *
 * All imagery is procedural and labeled SYNTHETIC -- no faces, no external
 * images.
 */

import { mulberry32, hashSeed } from './data.js';

const TRANSITION_MS = 380;

/**
 * Degradation modes exposed on the panel. `key` maps to rows in
 * createDemoData().runs.degradedSweep / runs.gateAblation.
 */
export const MODES = [
  { id: 'CLEAN', label: 'CLEAN', key: 'clean' },
  { id: 'BLUR', label: 'BLUR s8', key: 'blur8' },
  { id: 'LOW-RES', label: 'LOW-RES x0.1', key: 'down01' },
  { id: 'BRIGHTNESS', label: 'BRIGHTNESS d0.75', key: 'bright075' },
  { id: 'OFF-ANGLE', label: 'OFF-ANGLE 45deg', key: 'angle45' },
];

/** Create a DOM element with optional class list string. */
function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

/** Fixed 3-decimal formatter (n/a for nullish). */
function f3(x) {
  return x == null ? 'n/a' : Number(x).toFixed(3);
}

/**
 * Draw the base procedural tile: gradient sky, silhouette blocks/figures,
 * seeded grain, grid overlay, and SYNTHETIC labeling. Deterministic (seed 1337).
 */
function drawBaseTile(canvas) {
  const g = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const rnd = mulberry32(hashSeed('grey-resolve-demo-tile-1337'));

  const sky = g.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#0a1424');
  sky.addColorStop(0.62, '#16263c');
  sky.addColorStop(1, '#22344a');
  g.fillStyle = sky;
  g.fillRect(0, 0, w, h);

  // Glow blob (abstract light source).
  const glow = g.createRadialGradient(w * 0.72, h * 0.3, 4, w * 0.72, h * 0.3, w * 0.4);
  glow.addColorStop(0, 'rgba(120, 190, 220, 0.35)');
  glow.addColorStop(1, 'rgba(120, 190, 220, 0)');
  g.fillStyle = glow;
  g.fillRect(0, 0, w, h);

  // Silhouette blocks (abstract skyline).
  for (let i = 0; i < 14; i++) {
    const bw = w * (0.05 + rnd() * 0.08);
    const bh = h * (0.18 + rnd() * 0.5);
    const x = (w / 14) * i + rnd() * 6;
    g.fillStyle = `hsl(215, ${18 + rnd() * 14}%, ${5 + rnd() * 8}%)`;
    g.fillRect(x, h - bh, bw, bh);
  }

  // Figure blobs (abstract, near-bottom).
  for (let i = 0; i < 3; i++) {
    const cx = w * (0.18 + i * 0.3 + rnd() * 0.08);
    const cy = h * (0.82 + rnd() * 0.06);
    g.fillStyle = `hsla(210, 25%, ${4 + rnd() * 6}%, 0.92)`;
    g.beginPath();
    g.ellipse(cx, cy, w * 0.055, h * 0.14, 0, 0, Math.PI * 2);
    g.fill();
    g.beginPath();
    g.ellipse(cx, cy - h * 0.13, w * 0.03, h * 0.055, 0, 0, Math.PI * 2);
    g.fill();
  }

  // Seeded grain.
  for (let i = 0; i < 380; i++) {
    g.fillStyle = `rgba(255,255,255,${(0.015 + rnd() * 0.05).toFixed(3)})`;
    g.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2);
  }

  // Grid overlay.
  g.strokeStyle = 'rgba(120, 220, 180, 0.08)';
  g.lineWidth = 1;
  for (let x = 0; x <= w; x += 30) {
    g.beginPath();
    g.moveTo(x + 0.5, 0);
    g.lineTo(x + 0.5, h);
    g.stroke();
  }
  for (let y = 0; y <= h; y += 30) {
    g.beginPath();
    g.moveTo(0, y + 0.5);
    g.lineTo(w, y + 0.5);
    g.stroke();
  }

  // Labels.
  g.fillStyle = 'rgba(120, 220, 180, 0.85)';
  g.font = '11px monospace';
  g.fillText('SYNTHETIC TILE -- PROCEDURAL, NO REAL IMAGERY', 10, h - 10);
  g.fillText('GREY-RESOLVE // DEGRADATION BENCH', 10, 18);
}

/** Blur: ctx.filter when available, else repeated downscale/upscale. */
function renderBlur(base, dest, small, t) {
  const g = dest.getContext('2d');
  const w = dest.width;
  const h = dest.height;
  g.clearRect(0, 0, w, h);
  if ('filter' in g) {
    g.filter = `blur(${(4 * t).toFixed(2)}px)`;
    g.drawImage(base, 0, 0);
    g.filter = 'none';
    return;
  }
  // Fallback: progressive box blur via downscale/upscale passes.
  const passes = 3;
  let src = base;
  for (let p = 0; p < passes; p++) {
    const f = 1 + 3 * t;
    small.width = Math.max(2, Math.round(w / f));
    small.height = Math.max(2, Math.round(h / f));
    const sg = small.getContext('2d');
    sg.imageSmoothingEnabled = true;
    sg.drawImage(src, 0, 0, small.width, small.height);
    src = small;
  }
  g.imageSmoothingEnabled = true;
  g.drawImage(small, 0, 0, w, h);
}

/** Low-res: downsample then nearest-neighbour upscale (x0.1 at t=1). */
function renderLowRes(base, dest, small, t) {
  const g = dest.getContext('2d');
  const w = dest.width;
  const h = dest.height;
  const factor = 2 + 8 * t; // 2x .. 10x chunking (x0.1 at full severity)
  small.width = Math.max(2, Math.round(w / factor));
  small.height = Math.max(2, Math.round(h / factor));
  const sg = small.getContext('2d');
  sg.imageSmoothingEnabled = true;
  sg.drawImage(base, 0, 0, small.width, small.height);
  g.imageSmoothingEnabled = false;
  g.drawImage(small, 0, 0, w, h);
  g.imageSmoothingEnabled = true;
}

/** Brightness: filter when available, else 'lighter' white composite. */
function renderBrightness(base, dest, t) {
  const g = dest.getContext('2d');
  const w = dest.width;
  const h = dest.height;
  g.clearRect(0, 0, w, h);
  if ('filter' in g) {
    g.filter = `brightness(${(1 + 1.6 * t).toFixed(2)}) contrast(${(1 - 0.35 * t).toFixed(2)})`;
    g.drawImage(base, 0, 0);
    g.filter = 'none';
    return;
  }
  g.drawImage(base, 0, 0);
  g.globalCompositeOperation = 'lighter';
  g.fillStyle = `rgba(255,255,255,${(0.45 * t).toFixed(3)})`;
  g.fillRect(0, 0, w, h);
  g.globalCompositeOperation='source-over';
}

/** Off-angle: rotate transform up to 45 deg, scaled to cover the frame. */
function renderOffAngle(base, dest, t) {
  const g = dest.getContext('2d');
  const w = dest.width;
  const h = dest.height;
  g.fillStyle = '#070b12';
  g.fillRect(0, 0, w, h);
  g.save();
  g.translate(w / 2, h / 2);
  g.rotate(((-45 * Math.PI) / 180) * t);
  const s = 1 + 0.6 * t;
  g.scale(s, s);
  g.drawImage(base, -w / 2, -h / 2);
  g.restore();
}

/**
 * Build the measured effect label lines for one mode key.
 * Exported for reuse/testing.
 * @param {object} runs createDemoData().runs
 * @param {string} key one of MODES[].key
 * @returns {{primary: string, secondary: string, metrics: object|null}}
 */
export function effectLabels(runs, key) {
  const row = runs.degradedSweep.find((r) => r.key === key) || null;
  const gate = runs.gateAblation.find((r) => r.key === key) || null;
  if (!row) {
    return { primary: 'no measured data for this mode', secondary: '', metrics: null };
  }
  const primary = `${row.label}: FMR@FNMR=1% ${f3(row.fmr1)}  EER ${f3(row.eer)}`;
  const gateTxt = gate
    ? gate.fnmrGated == null
      ? `gate pass ${(gate.passRate * 100).toFixed(1)}% (too few passing -- gated metrics N/A)`
      : `gate FNMR ${f3(gate.fnmrFull)}->${f3(gate.fnmrGated)} (pass ${(gate.passRate * 100).toFixed(0)}%)`
    : 'gate n/a';
  const secondary =
    `scored ${row.scored}/${row.total} · detection failures ${row.dropped} · ${gateTxt}` +
    ` · SOURCE run ${runs.runIds.degraded} · ${runs.disclaimer}`;
  return { primary, secondary, metrics: row };
}

/**
 * Initialize the degradation bench panel in #degrade-panel.
 * @param {{bus: {on: Function, emit: Function}, data: object}} ctx
 * @returns {{destroy: () => void}|null} cleanup handle (null when DOM missing)
 */
export function init(ctx) {
  const bus = ctx && ctx.bus;
  const data = ctx && ctx.data;
  const host = document.querySelector('#degrade-panel');
  if (!bus || !data || !data.runs || !host) {
    console.warn('[demo-degrade] missing ctx.data.runs or #degrade-panel container');
    return null;
  }

  const panel = el('div', 'panel');
  const title = el('div', 'panel-title');
  title.textContent = 'DEGRADATION BENCH // LIVE CANVAS OPS';
  panel.append(title);

  const viewWrap = el('div', 'scanlines');
  const canvas = el('canvas');
  canvas.width = 360;
  canvas.height = 240;
  viewWrap.append(canvas);
  panel.append(viewWrap);

  const base = document.createElement('canvas');
  base.width = canvas.width;
  base.height = canvas.height;
  const small = document.createElement('canvas');
  drawBaseTile(base);

  const controls = el('div');
  const labelEl = el('div', 'mono');
  labelEl.style.marginTop = '6px';
  panel.append(controls, labelEl);
  host.append(panel);

  let rafId = 0;
  let current = MODES[0];

  /** Short smooth transition between modes (never a persistent loop). */
  function animateRender(render) {
    if (rafId) cancelAnimationFrame(rafId);
    const start = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - start) / TRANSITION_MS);
      const smooth = t * t * (3 - 2 * t);
      render(smooth);
      rafId = t < 1 ? requestAnimationFrame(frame) : 0;
    };
    rafId = requestAnimationFrame(frame);
  }

  function renderMode(mode, animate) {
    const render = (t) => {
      switch (mode.id) {
        case 'BLUR':
          renderBlur(base, canvas, small, t);
          break;
        case 'LOW-RES':
          renderLowRes(base, canvas, small, t);
          break;
        case 'BRIGHTNESS':
          renderBrightness(base, canvas, t);
          break;
        case 'OFF-ANGLE':
          renderOffAngle(base, canvas, t);
          break;
        default:
          canvas.getContext('2d').drawImage(base, 0, 0);
      }
    };
    if (animate) animateRender(render);
    else render(1);
  }

  function applyMode(mode, animate) {
    current = mode;
    renderMode(mode, animate);
    const { primary, secondary, metrics } = effectLabels(data.runs, mode.key);
    labelEl.textContent = '';
    const p = el('div', 'glow');
    p.textContent = primary;
    const s = el('div');
    s.textContent = secondary;
    labelEl.append(p, s);

    for (const btn of controls.children) {
      btn.classList.toggle('mode-btn-active', btn.dataset.mode === mode.id);
    }
    bus.emit('mode', {
      mode: mode.id,
      key: mode.key,
      label: mode.label,
      metrics,
      runId: data.runs.runIds.degraded,
    });
  }

  for (const mode of MODES) {
    const btn = el('button', 'mode-btn');
    btn.type = 'button';
    btn.dataset.mode = mode.id;
    btn.textContent = mode.label;
    btn.addEventListener('click', () => {
      if (current.id === mode.id) return;
      applyMode(mode, true);
    });
    controls.append(btn);
  }

  // Initial state: CLEAN tile drawn, no 'mode' emit (fires on user change only).
  renderMode(MODES[0], false);
  for (const btn of controls.children) {
    btn.classList.toggle('mode-btn-active', btn.dataset.mode === MODES[0].id);
  }
  const initial = effectLabels(data.runs, MODES[0].key);
  labelEl.textContent = '';
  const p0 = el('div', 'glow');
  p0.textContent = initial.primary;
  const s0 = el('div');
  s0.textContent = initial.secondary;
  labelEl.append(p0, s0);

  // Refresh labels if the real run JSONs land after init.
  if (data.runsReady && typeof data.runsReady.then === 'function') {
    data.runsReady.then(() => {
      const { primary, secondary } = effectLabels(data.runs, current.key);
      p0.textContent = primary;
      s0.textContent = secondary;
    }).catch(() => {});
  }

  return {
    destroy() {
      if (rafId) cancelAnimationFrame(rafId);
    },
  };
}
