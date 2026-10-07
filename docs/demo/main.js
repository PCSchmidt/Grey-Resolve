/**
 * main.js -- console bootstrap for the Grey-Resolve demo.
 *
 * Wires the shared context {bus, data} and starts the modules in contract order:
 * data.js -> panels.js -> degrade.js -> manifold.js. Each module exposes
 * `export function init(ctx)` with ctx = {bus, data}; the bus comes from data.js
 * ({on(evt, fn), emit(evt, payload)}) and this file adapts to the common export
 * shapes (named `bus`/`data`, `createContext()`, or `init(ctx)` in-place/return).
 *
 * Also runs the typed boot overlay (~2.5s) and the honest synthetic telemetry
 * cluster (ephemeral run-id, UTC clock, live FPS). Every step degrades gracefully:
 * a missing or broken sibling module only logs a warning.
 */

/* ------------------------------------------------------------------ */
/* Fallback bus (used only if data.js provides none)                  */
/* ------------------------------------------------------------------ */

/**
 * Minimal synchronous pub/sub matching the {on, emit} contract.
 * @returns {{on: (evt: string, fn: Function) => void, emit: (evt: string, payload: any) => void}}
 */
function makeFallbackBus() {
  const listeners = new Map();
  return {
    on(evt, fn) {
      if (typeof fn !== 'function') return;
      if (!listeners.has(evt)) listeners.set(evt, []);
      listeners.get(evt).push(fn);
    },
    emit(evt, payload) {
      for (const fn of listeners.get(evt) ?? []) {
        try {
          fn(payload);
        } catch (err) {
          console.warn('[main.js] bus listener failed for', evt, err);
        }
      }
    },
  };
}

/* ------------------------------------------------------------------ */
/* Context assembly                                                   */
/* ------------------------------------------------------------------ */

/**
 * Load data.js and build the shared ctx = {bus, data}.
 * Accepts: named `bus`/`data` exports, `createContext()`, or `init(ctx)`
 * that fills ctx in place and/or returns {bus, data}.
 * @returns {Promise<{bus: object, data: object}>}
 */
async function loadContext() {
  const ctx = { bus: null, data: null };
  let mod = null;
  try {
    mod = await import('./data.js');
  } catch (err) {
    console.warn('[main.js] data.js unavailable, using fallback context:', err);
    ctx.bus = makeFallbackBus();
    ctx.data = { personas: [] };
    return ctx;
  }

  // Seed from named exports first so init() sees the canonical objects.
  if (mod.bus && typeof mod.bus.on === 'function') ctx.bus = mod.bus;
  if (mod.data && typeof mod.data === 'object') ctx.data = mod.data;

  try {
    // Factory-style data.js: build canonical bus/dataset before init().
    if (typeof mod.createBus === 'function' && !ctx.bus) ctx.bus = mod.createBus();
    if (typeof mod.createDemoData === 'function' && !ctx.data) ctx.data = mod.createDemoData();
    if (typeof mod.createContext === 'function') {
      const created = await mod.createContext();
      if (created?.bus) ctx.bus = created.bus;
      if (created?.data) ctx.data = created.data;
    }
    if (typeof mod.init === 'function') {
      const res = await mod.init(ctx);
      if (res && typeof res === 'object') {
        // init() may return {bus, data}, a bare dataset, or fill ctx in place.
        if (res.bus && typeof res.bus.on === 'function') ctx.bus = res.bus;
        if (res.data && typeof res.data === 'object') ctx.data = res.data;
        else if (Array.isArray(res.personas)) ctx.data = res;
      }
    }
  } catch (err) {
    console.warn('[main.js] data.js init failed, keeping partial context:', err);
  }

  if (!ctx.bus || typeof ctx.bus.on !== 'function') ctx.bus = makeFallbackBus();
  if (!ctx.data || typeof ctx.data !== 'object') ctx.data = { personas: [] };
  return ctx;
}

/**
 * Import a sibling module and run its init(ctx); never throws.
 * @param {string} path - module specifier relative to this file
 * @param {string} label - log label
 * @param {{bus: object, data: object}} ctx
 */
async function startModule(path, label, ctx) {
  try {
    const mod = await import(path);
    if (typeof mod.init !== 'function') {
      console.warn(`[main.js] ${label} has no init(ctx); skipped.`);
      return;
    }
    await mod.init(ctx);
    console.info(`[main.js] ${label} started.`);
  } catch (err) {
    console.warn(`[main.js] ${label} failed to start (console stays up):`, err);
  }
}

/* ------------------------------------------------------------------ */
/* Boot overlay: typed init lines, then fade                          */
/* ------------------------------------------------------------------ */

/**
 * Type the #boot lines char-by-char (~2.5s total), then fade the overlay out.
 * Click/keypress skips ahead; reduced-motion users see lines instantly.
 * @returns {Promise<void>}
 */
function runBootSequence() {
  const boot = document.getElementById('boot');
  if (!boot) return Promise.resolve();
  const lines = Array.from(boot.querySelectorAll('.boot-line[data-text]'));
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

  return new Promise((resolve) => {
    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      for (const line of lines) line.textContent = line.dataset.text ?? '';
      boot.classList.add('boot-done');
      window.setTimeout(() => {
        boot.remove();
        resolve();
      }, 700);
    };

    if (reduced || lines.length === 0) {
      window.setTimeout(finish, reduced ? 500 : 0);
      return;
    }

    const totalChars = lines.reduce((n, l) => n + (l.dataset.text ?? '').length, 0);
    const charDelay = Math.max(2, Math.min(6, 2300 / Math.max(totalChars, 1)));
    const caret = document.createElement('span');
    caret.className = 'boot-caret';

    let li = 0;
    let ci = 0;
    const step = () => {
      if (finished) return;
      if (li >= lines.length) {
        caret.remove();
        window.setTimeout(finish, 380);
        return;
      }
      const line = lines[li];
      const text = line.dataset.text ?? '';
      if (ci === 0) line.appendChild(caret);
      ci += 1;
      line.textContent = text.slice(0, ci);
      line.appendChild(caret);
      if (ci >= text.length) {
        li += 1;
        ci = 0;
        window.setTimeout(step, 90);
      } else {
        window.setTimeout(step, charDelay);
      }
    };
    step();

    window.addEventListener('pointerdown', finish, { once: true });
    window.addEventListener('keydown', finish, { once: true });
    window.setTimeout(finish, 4200); // hard cap: overlay never traps the user
  });
}

/* ------------------------------------------------------------------ */
/* HUD telemetry (honest: synthetic, local, ephemeral)                */
/* ------------------------------------------------------------------ */

/** Generate a human-readable ephemeral run id. */
function makeRunId() {
  const t = Date.now().toString(36).toUpperCase().slice(-5);
  const r = Math.random().toString(36).toUpperCase().slice(2, 6);
  return `GR-${t}-${r}`;
}

/** Wire the corner telemetry cluster: run-id, UTC clock, live FPS. */
function startTelemetry() {
  const runEl = document.querySelector('[data-tel="run"]');
  const clockEl = document.querySelector('[data-tel="clock"]');
  const fpsEl = document.querySelector('[data-tel="fps"]');
  if (runEl) runEl.textContent = makeRunId();

  const tickClock = () => {
    if (!clockEl) return;
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    clockEl.textContent = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}Z`;
  };
  tickClock();
  window.setInterval(tickClock, 1000);

  // Rolling FPS from rAF timestamps (skipped under reduced motion).
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  if (fpsEl && !reduced && typeof window.requestAnimationFrame === 'function') {
    let frames = 0;
    let last = performance.now();
    const loop = (now) => {
      frames += 1;
      if (now - last >= 500) {
        const fps = (frames * 1000) / (now - last);
        fpsEl.textContent = fps.toFixed(0);
        frames = 0;
        last = now;
      }
      window.requestAnimationFrame(loop);
    };
    window.requestAnimationFrame(loop);
  } else if (fpsEl) {
    fpsEl.textContent = 'n/a';
  }
}

/* ------------------------------------------------------------------ */
/* Boot                                                               */
/* ------------------------------------------------------------------ */

/** Wire all console modules and start the overlays. */
async function boot() {
  const ctx = await loadContext();

  await startModule('./panels.js', 'panels.js', ctx);
  await startModule('./degrade.js', 'degrade.js', ctx);
  await startModule('./manifold.js', 'manifold.js', ctx);

  startTelemetry();

  // Debug/integration handle; harmless in production.
  window.__GREY_RESOLVE__ = { ctx, ready: true };

  await runBootSequence();
}

boot();
