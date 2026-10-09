/**
 * manifold.js -- 3D entity-resolution manifold for the Grey-Resolve console.
 *
 * Renders the "gods-eye" cluster space over #viewport with three.js:
 *   (a) ~12 persona clusters as glowing point constellations (from ctx.data.personas),
 *   (b) query contacts as streaking particles that settle near their candidate clusters,
 *   (c) near-tie dashed lines between clusters with a resolution flash,
 *   (d) pointer raycast selection that emits 'select-persona' on the bus,
 *   (e) subtle fog + additive "bloom-like" materials, no postprocessing deps.
 *
 * Contract: `export function init(ctx)` with ctx = {bus, data}. Events consumed:
 *   'contact' (feed item -> streak), 'tie' (tie-break -> dashed line + flash),
 *   'select-persona' (highlight a cluster), 'mode' (fog/tint response).
 * Robustness: WebGL failure, missing data, malformed payloads and context loss
 * all degrade to a CSS fallback message or a no-op instead of throwing.
 */

import * as THREE from 'three';

/* ------------------------------------------------------------------ */
/* Deterministic helpers                                              */
/* ------------------------------------------------------------------ */

/** FNV-1a style string hash -> uint32, used for deterministic colors/positions. */
function hash32(str) {
  let h = 2166136261 >>> 0;
  const s = String(str);
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/** Small seeded PRNG (mulberry32) so a persona always looks the same per load. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Clamp helper. */
function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

/** Normalize an id-ish payload value (string | number | {id}|{personaId}|{name}) to string. */
function normId(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'string' || typeof v === 'number') return String(v);
  const id = v.id ?? v.personaId ?? v.persona_id ?? v.name ?? v.label ?? null;
  return id === null || id === undefined ? null : String(id);
}

/* ------------------------------------------------------------------ */
/* Procedural textures (no external images)                           */
/* ------------------------------------------------------------------ */

/** Radial glow sprite texture used for points, cores and streak heads. */
function makeGlowTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.22, 'rgba(255,255,255,0.62)');
  grad.addColorStop(0.55, 'rgba(255,255,255,0.16)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

/** HUD-style label texture: mono text with detection-box corner brackets. */
function makeLabelTexture(text, colorCss) {
  const w = 256;
  const h = 64;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, w, h);
  g.fillStyle = 'rgba(4, 10, 12, 0.55)';
  g.fillRect(6, 14, w - 12, h - 28);
  g.strokeStyle = colorCss;
  g.lineWidth = 2;
  const b = 10;
  const arm = 16;
  // corner brackets (detection-box motif)
  const corners = [
    [b, b, 1, 1], [w - b, b, -1, 1],
    [b, h - b, 1, -1], [w - b, h - b, -1, -1],
  ];
  for (const [cx, cy, sx, sy] of corners) {
    g.beginPath();
    g.moveTo(cx + sx * arm, cy);
    g.lineTo(cx, cy);
    g.lineTo(cx, cy + sy * arm);
    g.stroke();
  }
  g.font = '700 22px "IBM Plex Mono", Consolas, monospace';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = colorCss;
  g.fillText(String(text).slice(0, 18), w / 2, h / 2 + 1);
  return new THREE.CanvasTexture(canvas);
}

/** Full detection-box bracket sprite (selection / flash marker). */
function makeBracketTexture(colorCss) {
  const s = 128;
  const canvas = document.createElement('canvas');
  canvas.width = s;
  canvas.height = s;
  const g = canvas.getContext('2d');
  g.strokeStyle = colorCss;
  g.lineWidth = 6;
  const b = 12;
  const arm = 34;
  const corners = [
    [b, b, 1, 1], [s - b, b, -1, 1],
    [b, s - b, 1, -1], [s - b, s - b, -1, -1],
  ];
  for (const [cx, cy, sx, sy] of corners) {
    g.beginPath();
    g.moveTo(cx + sx * arm, cy);
    g.lineTo(cx, cy);
    g.lineTo(cx, cy + sy * arm);
    g.stroke();
  }
  return new THREE.CanvasTexture(canvas);
}

/* ------------------------------------------------------------------ */
/* Small CSS fallback                                                 */
/* ------------------------------------------------------------------ */

/**
 * Show a CSS fallback message inside #viewport (keeps existing HUD tags).
 * @param {HTMLElement} container - the #viewport element
 * @param {string} title - bold headline
 * @param {string} detail - explanatory line
 */
function showFallback(container, title, detail) {
  const box = document.createElement('div');
  box.className = 'webgl-fallback';
  const t = document.createElement('div');
  t.className = 'fb-title';
  t.textContent = title;
  const d = document.createElement('div');
  d.textContent = detail;
  box.appendChild(t);
  box.appendChild(d);
  container.appendChild(box);
}

/* ------------------------------------------------------------------ */
/* Persona plumbing                                                   */
/* ------------------------------------------------------------------ */

/**
 * Extract a persona array from loosely-shaped data (array | {personas} | {people} | map).
 * @param {object} data
 * @returns {object[]}
 */
function personasFromData(data) {
  const raw = data?.personas ?? data?.people ?? data?.entities ?? data;
  if (Array.isArray(raw)) return raw.filter((p) => p !== null && p !== undefined);
  if (raw && typeof raw === 'object') {
    return Object.entries(raw).map(([k, v]) =>
      v && typeof v === 'object' ? (v.id !== undefined ? v : { id: k, ...v }) : { id: k, value: v });
  }
  return [];
}

/**
 * Resolve a stable 3D anchor for a cluster: explicit coords if present,
 * otherwise a deterministic golden-angle spiral derived from the id.
 */
function clusterPosition(persona, index, total, seed) {
  const p = persona?.centroid3d ?? persona?.centroid ?? persona?.position ??
    persona?.pos ?? persona?.coords ?? persona?.xyz ?? persona?.cluster;
  if (Array.isArray(p) && p.length >= 3 && p.every((n) => Number.isFinite(n))) {
    const v = new THREE.Vector3(p[0], p[1], p[2]);
    // normalized centroids (|v| <= 1.5) get scaled up to a readable spread
    const maxAbs = Math.max(Math.abs(v.x), Math.abs(v.y), Math.abs(v.z));
    if (maxAbs > 0 && maxAbs <= 1.5) v.multiplyScalar(11);
    return v;
  }
  if (p && typeof p === 'object' && Number.isFinite(p.x)) {
    return new THREE.Vector3(p.x, p.y ?? 0, p.z ?? 0);
  }
  if (Number.isFinite(persona?.x)) {
    return new THREE.Vector3(persona.x, persona.y ?? 0, persona.z ?? 0);
  }
  const i = index + 1;
  const rand = mulberry32(seed);
  const radius = 6 + 8 * Math.sqrt(i / Math.max(total, 1));
  const angle = i * 2.39996323 + (rand() - 0.5) * 0.35;
  return new THREE.Vector3(
    Math.cos(angle) * radius,
    (rand() - 0.5) * 7.5,
    Math.sin(angle) * radius,
  );
}

/* ------------------------------------------------------------------ */
/* init                                                               */
/* ------------------------------------------------------------------ */

/**
 * Build and run the 3D manifold over #viewport.
 * Safe to call once per page; all internals are closure-scoped.
 * @param {{bus: {on: Function, emit: Function}, data: object}} ctx
 * @returns {{renderer: object}|null} live handles (for debugging), or null on fallback
 */
export function init(ctx) {
  const container = document.getElementById('viewport');
  if (!container) {
    console.warn('[manifold.js] #viewport not found; 3D manifold disabled.');
    return null;
  }

  // --- WebGL guard -------------------------------------------------
  let glOk = true;
  try {
    const probe = document.createElement('canvas');
    glOk = !!(window.WebGLRenderingContext &&
      (probe.getContext('webgl2') || probe.getContext('webgl')));
  } catch (err) {
    glOk = false;
  }

  let renderer;
  if (glOk) {
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (err) {
      glOk = false;
    }
  }
  if (!glOk || !renderer) {
    showFallback(container, 'WEBGL OFFLINE',
      '3D manifold unavailable in this browser/session. Feed, resolver and degradation panels stay live.');
    console.warn('[manifold.js] WebGL unavailable; using CSS fallback.');
    return null;
  }

  try {
    return buildScene(ctx, container, renderer);
  } catch (err) {
    console.warn('[manifold.js] scene init failed:', err);
    showFallback(container, 'MANIFOLD FAULT',
      '3D manifold could not initialize. Remaining console panels stay live.');
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Scene construction                                                 */
/* ------------------------------------------------------------------ */

function buildScene(ctx, container, renderer) {
  const bus = ctx?.bus ?? { on() {}, emit() {} };
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

  // --- renderer / camera ------------------------------------------
  renderer.setClearColor(0x05080b, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05080b, 0.028);

  const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 220);
  camera.position.set(0, 10, 27);

  // --- shared assets ----------------------------------------------
  const glowTex = makeGlowTexture();
  const bracketTex = makeBracketTexture('#67d9ff');

  // --- ambient dust field (depth cue) -----------------------------
  {
    const dustCount = 240;
    const pos = new Float32Array(dustCount * 3);
    const rand = mulberry32(0xd057);
    for (let i = 0; i < dustCount; i += 1) {
      pos[i * 3] = (rand() - 0.5) * 70;
      pos[i * 3 + 1] = (rand() - 0.5) * 34;
      pos[i * 3 + 2] = (rand() - 0.5) * 70;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.18, map: glowTex, color: 0x2c6f63,
      transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    scene.add(new THREE.Points(geo, mat));
  }

  // --- clusters ----------------------------------------------------
  const personas = personasFromData(ctx?.data);
  const clusters = [];
  const pickTargets = [];
  const palette = [0x6ef7b0, 0x67d9ff, 0x9be8ff, 0x55e0a2, 0xffc257, 0x7fffd4];

  personas.slice(0, 24).forEach((persona, index) => {
    const id = normId(persona) ?? `P-${String(index).padStart(2, '0')}`;
    const seed = hash32(id);
    const rand = mulberry32(seed);
    const color = new THREE.Color(palette[seed % palette.length]);
    if (Number.isFinite(persona?.hue) && typeof color.setHSL === 'function') {
      color.setHSL(((persona.hue % 360) + 360) % 360 / 360, 0.52, 0.62);
    }
    const colorHex = color.getHex();
    const center = clusterPosition(persona, index, personas.length, seed);
    const group = new THREE.Group();
    group.position.copy(center);

    // constellation points (gaussian blob)
    const count = 64;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const r = 0.35 + rand() * 1.75;
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.cos(phi) * 0.72;
      pos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const ptsGeo = new THREE.BufferGeometry();
    ptsGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const ptsMat = new THREE.PointsMaterial({
      size: 0.3, map: glowTex, color: colorHex,
      transparent: true, opacity: 0.8,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    group.add(new THREE.Points(ptsGeo, ptsMat));

    // additive core glow ("bloom-like")
    const core = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: colorHex,
      transparent: true, opacity: 0.42,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    core.scale.setScalar(4.6);
    group.add(core);

    // faint orbit ring
    const ringPts = [];
    for (let i = 0; i <= 72; i += 1) {
      const a = (i / 72) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a) * 2.25, 0, Math.sin(a) * 2.25));
    }
    const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPts);
    const ring = new THREE.Line(ringGeo, new THREE.LineBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.22,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    group.add(ring);

    // HUD label sprite
    const labelTex = makeLabelTexture(
      String(persona?.name ?? persona?.label ?? id).toUpperCase(),
      `#${color.getHexString()}`,
    );
    const label = new THREE.Sprite(new THREE.SpriteMaterial({
      map: labelTex, transparent: true, opacity: 0.85, depthWrite: false,
    }));
    label.scale.set(5.2, 1.3, 1);
    label.position.y = 3.1;
    group.add(label);

    // invisible pick sphere for reliable raycast
    const pick = new THREE.Mesh(
      new THREE.SphereGeometry(2.4, 10, 10),
      new THREE.MeshBasicMaterial({ visible: false }),
    );
    pick.position.copy(center);
    pick.userData.clusterIndex = clusters.length;
    scene.add(pick);
    pickTargets.push(pick);

    scene.add(group);
    clusters.push({
      id, persona, center, group, core, ring, label, pick,
      color, baseCoreScale: 4.6, pulse: 0, phase: rand() * Math.PI * 2,
    });
  });

  // --- selection marker -------------------------------------------
  const selectMarker = new THREE.Sprite(new THREE.SpriteMaterial({
    map: bracketTex, color: 0x67d9ff,
    transparent: true, opacity: 0, depthWrite: false,
    blending: THREE.AdditiveBlending,
  }));
  selectMarker.scale.setScalar(6.2);
  scene.add(selectMarker);
  let selectedCluster = null;

  // --- effect containers ------------------------------------------
  const streaks = [];   // {head, trail, history, t, dur, from, ctrl, to, fade}
  const ties = [];      // {line, pulses, a, b, winnerId, born, resolveAt, resolved, fade}
  const flashes = [];   // {mesh, born, life, kind}

  /** Expand a detection-box/ring flash at a world position. */
  function spawnFlash(position, colorHex, kind = 'ring') {
    let mesh;
    if (kind === 'ring') {
      const pts = [];
      for (let i = 0; i <= 48; i += 1) {
        const a = (i / 48) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
      }
      mesh = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: colorHex, transparent: true, opacity: 0.95,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }),
      );
    } else {
      mesh = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: colorHex,
        transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
    }
    mesh.position.copy(position);
    scene.add(mesh);
    flashes.push({ mesh, born: performance.now(), life: kind === 'ring' ? 950 : 700, kind });
  }

  /** Find a cluster by loose id. */
  function clusterById(id) {
    if (id === null || id === undefined) return null;
    const key = String(id);
    return clusters.find((c) => c.id === key || String(c.persona?.id) === key ||
      String(c.persona?.name ?? '') === key) ?? null;
  }

  /* ---------------------------------------------------------------- */
  /* Query contacts -> streaking particles                            */
  /* ---------------------------------------------------------------- */

  /** Extract the primary candidate id from a loose 'contact' payload. */
  function contactTargetId(payload) {
    const query = payload?.query ?? payload?.q ?? null;
    return normId(query?.candidates?.[0]) ??
      normId(payload?.contact?.personaId ?? payload?.contact?.candidate ?? payload?.contact?.top) ??
      normId(payload?.personaId ?? payload?.persona_id ?? payload?.selected ??
        payload?.candidate ?? payload?.winner ?? payload?.top) ??
      normId(payload?.candidates?.[0] ?? payload?.matches?.[0] ?? payload?.topCandidates?.[0]) ??
      normId(payload?.persona) ?? normId(payload);
  }

  /** Launch one streaking contact particle toward a cluster center. */
  function spawnStreak(target) {
    const rand = Math.random;
    const from = new THREE.Vector3(
      (rand() - 0.5) * 2, 0.25 + rand() * 0.6, (rand() - 0.5) * 2,
    ).normalize().multiplyScalar(34 + rand() * 10);
    const to = target.clone().add(new THREE.Vector3(
      (rand() - 0.5) * 1.2, (rand() - 0.5) * 1.2, (rand() - 0.5) * 1.2));
    const mid = from.clone().add(to).multiplyScalar(0.5);
    const ctrl = mid.add(new THREE.Vector3(
      (rand() - 0.5) * 12, 6 + rand() * 6, (rand() - 0.5) * 12));

    const trailLen = 16;
    const history = [];
    for (let i = 0; i < trailLen; i += 1) history.push(from.clone());
    const trailGeo = new THREE.BufferGeometry().setFromPoints(history);
    const colors = new Float32Array(trailLen * 3);
    for (let i = 0; i < trailLen; i += 1) {
      const f = Math.pow(i / (trailLen - 1), 1.6);
      colors[i * 3] = 0.25 + 0.75 * f;
      colors[i * 3 + 1] = 0.85;
      colors[i * 3 + 2] = 0.95;
    }
    trailGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.8,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(trail);

    const head = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0xa8f4ff,
      transparent: true, opacity: 0.95,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    head.scale.setScalar(1.7);
    head.position.copy(from);
    scene.add(head);

    streaks.push({
      head, trail, history, t: 0,
      dur: reduced ? 0.7 : 1.55,
      from, ctrl, to, settled: false, fade: 1,
    });
  }

  function updateStreaks(dt) {
    for (let i = streaks.length - 1; i >= 0; i -= 1) {
      const s = streaks[i];
      if (!s.settled) {
        s.t = Math.min(1, s.t + dt / s.dur);
        const t = s.t;
        const omt = 1 - t;
        const p = new THREE.Vector3(
          omt * omt * s.from.x + 2 * omt * t * s.ctrl.x + t * t * s.to.x,
          omt * omt * s.from.y + 2 * omt * t * s.ctrl.y + t * t * s.to.y,
          omt * omt * s.from.z + 2 * omt * t * s.ctrl.z + t * t * s.to.z,
        );
        s.head.position.copy(p);
        s.history.pop();
        s.history.unshift(p.clone());
        const attr = s.trail.geometry.getAttribute('position');
        for (let k = 0; k < s.history.length; k += 1) {
          attr.setXYZ(k, s.history[k].x, s.history[k].y, s.history[k].z);
        }
        attr.needsUpdate = true;
        if (s.t >= 1) {
          s.settled = true;
          spawnFlash(s.to.clone(), 0x67d9ff, 'ring');
          const hit = clusters.find((c) => c.center.distanceTo(s.to) < 3);
          if (hit) hit.pulse = 1;
        }
      } else {
        s.fade = Math.max(0, s.fade - dt * 1.6);
        s.trail.material.opacity = 0.8 * s.fade;
        s.head.material.opacity = 0.95 * s.fade;
        if (s.fade <= 0) {
          scene.remove(s.head);
          scene.remove(s.trail);
          s.head.material.dispose();
          s.trail.geometry.dispose();
          s.trail.material.dispose();
          streaks.splice(i, 1);
        }
      }
    }
  }

  /* ---------------------------------------------------------------- */
  /* Near-tie dashed lines + resolution flash                          */
  /* ---------------------------------------------------------------- */

  /** Extract participant ids from a loose 'tie' payload. */
  function tieParticipants(payload) {
    const out = [];
    const push = (v) => {
      const id = normId(v);
      if (id && !out.includes(id)) out.push(id);
    };
    if (Array.isArray(payload)) {
      payload.forEach(push);
    } else {
      const group = payload?.candidates ?? payload?.contestants ?? payload?.ids ??
        payload?.between ?? payload?.participants ?? payload?.tied;
      if (Array.isArray(group)) group.forEach(push);
      push(payload?.a); push(payload?.b); push(payload?.left); push(payload?.right);
    }
    return out;
  }

  /** Extract a winner id from a payload when the tie already resolved. */
  function tieWinner(payload) {
    return normId(payload?.winner ?? payload?.winnerId ??
      payload?.chosen ?? payload?.selected ?? payload?.decision);
  }

  /**
   * Delayed resolution winner: the tiebreak's pick when the payload carries
   * one (it may be wrong), else the fabricated ground truth.
   */
  function tieGroundTruth(payload) {
    return normId(payload?.pick ?? payload?.groundTruth ?? payload?.truth);
  }

  /** Open an animated dashed tie line between (up to 3) clusters. */
  function spawnTie(ids, winnerId, immediate) {
    let matched = ids.map(clusterById).filter(Boolean);
    if (matched.length < 2) {
      // payload was too thin: pick a deterministic near pair so the event stays visible
      const seed = hash32(JSON.stringify(ids ?? []) + performance.now());
      const rand = mulberry32(seed);
      if (clusters.length >= 2) {
        const a = clusters[Math.floor(rand() * clusters.length)];
        let b = clusters[Math.floor(rand() * clusters.length)];
        if (b === a) b = clusters[(clusters.indexOf(a) + 1) % clusters.length];
        matched = [a, b];
      } else {
        return;
      }
    }
    const a = matched[0].center.clone().add(new THREE.Vector3(0, 0.4, 0));
    const b = matched[1].center.clone().add(new THREE.Vector3(0, 0.4, 0));

    const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
    const mat = new THREE.LineDashedMaterial({
      color: 0x67d9ff, dashSize: 0.6, gapSize: 0.42,
      transparent: true, opacity: 0.7,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    scene.add(line);

    const pulses = [];
    for (let i = 0; i < 2; i += 1) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xd9f6ff,
        transparent: true, opacity: 0.9,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }));
      s.scale.setScalar(1.15);
      s.userData.t = i * 0.5;
      scene.add(s);
      pulses.push(s);
    }

    ties.push({
      line, pulses, a, b, winnerId: winnerId ?? null,
      born: performance.now(), resolveAt: performance.now() + 7000,
      resolved: false, fade: 1,
    });

    if (winnerId !== null && winnerId !== undefined) {
      const tie = ties[ties.length - 1];
      tie.winnerId = winnerId;
      if (immediate) {
        resolveTie(tie);
      } else {
        tie.resolveAt = performance.now() + 3200; // let the tie-break beat play first
      }
    }
  }

  /** Flash + tear down a tie line. */
  function resolveTie(tie) {
    if (tie.resolved) return;
    tie.resolved = true;
    const winner = clusterById(tie.winnerId);
    if (winner) {
      spawnFlash(winner.center.clone(), 0xeafff6, 'ring');
      spawnFlash(winner.center.clone(), 0x6ef7b0, 'glow');
      winner.pulse = 1.4;
    }
    tie.line.material.color.setHex(0xeafff6);
    tie.line.material.opacity = 0.95;
    tie.resolveAt = performance.now(); // start fade now
  }

  function updateTies(dt, now) {
    for (let i = ties.length - 1; i >= 0; i -= 1) {
      const tie = ties[i];
      if (!tie.resolved && now > tie.resolveAt) {
        resolveTie(tie); // timeout: fade (no flash unless a winner was scheduled)
      }
      for (const s of tie.pulses) {
        s.userData.t = (s.userData.t + dt * (reduced ? 0.12 : 0.32)) % 1;
        s.position.lerpVectors(tie.a, tie.b, s.userData.t);
      }
      if (tie.resolved) {
        tie.fade = Math.max(0, tie.fade - dt * 0.9);
        tie.line.material.opacity = 0.95 * tie.fade;
        for (const s of tie.pulses) s.material.opacity = 0.9 * tie.fade;
        if (tie.fade <= 0) {
          scene.remove(tie.line);
          tie.line.geometry.dispose();
          tie.line.material.dispose();
          for (const s of tie.pulses) {
            scene.remove(s);
            s.material.dispose();
          }
          ties.splice(i, 1);
        }
      }
    }
  }

  function updateFlashes(now) {
    for (let i = flashes.length - 1; i >= 0; i -= 1) {
      const f = flashes[i];
      const k = clamp((now - f.born) / f.life, 0, 1);
      if (f.kind === 'ring') {
        f.mesh.scale.setScalar(1 + k * 5.5);
        f.mesh.material.opacity = 0.95 * (1 - k);
      } else {
        f.mesh.scale.setScalar(2 + k * 6);
        f.mesh.material.opacity = 0.95 * (1 - k);
      }
      if (k >= 1) {
        scene.remove(f.mesh);
        if (f.mesh.geometry) f.mesh.geometry.dispose();
        f.mesh.material.dispose();
        flashes.splice(i, 1);
      }
    }
  }

  /* ---------------------------------------------------------------- */
  /* Bus wiring                                                       */
  /* ---------------------------------------------------------------- */

  bus.on('contact', (payload) => {
    try {
      const id = contactTargetId(payload);
      const target = clusterById(id) ?? clusters[hash32(String(id ?? 'x')) % Math.max(clusters.length, 1)];
      if (target) spawnStreak(target.center);
    } catch (err) {
      console.warn('[manifold.js] contact handler error:', err);
    }
  });

  bus.on('tie', (payload) => {
    try {
      const explicit = tieWinner(payload);
      const winner = explicit ?? tieGroundTruth(payload);
      spawnTie(tieParticipants(payload), winner, explicit !== null && explicit !== undefined);
    } catch (err) {
      console.warn('[manifold.js] tie handler error:', err);
    }
  });

  bus.on('select-persona', (payload) => {
    try {
      const id = normId(payload);
      const c = clusterById(id);
      if (c) {
        selectedCluster = c;
        selectMarker.position.copy(c.center);
        selectMarker.material.color.copy(c.color);
      }
    } catch (err) {
      console.warn('[manifold.js] select handler error:', err);
    }
  });

  let targetFog = 0.028;
  bus.on('mode', (payload) => {
    try {
      const name = typeof payload === 'string' ? payload : (payload?.mode ?? payload?.name ?? '');
      const MODE_FOG = {
        CLEAN: 0.12, BLUR: 0.55, 'LOW-RES': 0.75, BRIGHTNESS: 0.45, 'OFF-ANGLE': 0.6,
      };
      let level = Number(payload?.level ?? payload?.severity);
      if (!Number.isFinite(level)) {
        if (MODE_FOG[String(name).toUpperCase()] !== undefined) {
          level = MODE_FOG[String(name).toUpperCase()];
        } else if (/severe|heavy|hard|deep|low/i.test(String(name))) {
          level = 0.9;
        } else if (/mild|light|clean|none/i.test(String(name))) {
          level = 0.15;
        } else {
          level = 0.5;
        }
      }
      level = clamp(level, 0, 1);
      targetFog = 0.02 + level * 0.022;
      const tag = document.querySelector('[data-tel="viewport-mode"]');
      if (tag) tag.textContent = `MODE: ${String(name || 'CUSTOM').toUpperCase()}`;
    } catch (err) {
      console.warn('[manifold.js] mode handler error:', err);
    }
  });

  /* ---------------------------------------------------------------- */
  /* Pointer: orbit drag + click raycast selection                    */
  /* ---------------------------------------------------------------- */

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let downX = 0;
  let downY = 0;
  let lastX = 0;
  let dragging = false;
  let pointerIn = false;
  let parallaxX = 0;
  let parallaxY = 0;

  function onSelectClick(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(pickTargets, false);
    if (hits.length > 0) {
      const c = clusters[hits[0].object.userData.clusterIndex];
      if (c) {
        selectedCluster = c;
        selectMarker.position.copy(c.center);
        selectMarker.material.color.copy(c.color);
        c.pulse = 1;
        bus.emit('select-persona', c.persona);
      }
    }
  }

  const dom = renderer.domElement;
  dom.addEventListener('pointerdown', (e) => {
    downX = e.clientX;
    downY = e.clientY;
    lastX = e.clientX;
    dragging = false;
  });
  dom.addEventListener('pointermove', (e) => {
    const rect = dom.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0 && !dragging) {
      parallaxX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      parallaxY = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    }
    if (e.buttons & 1) {
      const dx = e.clientX - lastX;
      if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 5) dragging = true;
      if (dragging) {
        orbitAngle -= dx * 0.005;
        cameraDriftY = clamp(cameraDriftY + (e.movementY ?? 0) * -0.02, -6, 8);
      }
      lastX = e.clientX;
    }
    pointerIn = true;
  });
  dom.addEventListener('pointerleave', () => {
    pointerIn = false;
  });
  dom.addEventListener('pointerup', (e) => {
    if (!dragging) onSelectClick(e.clientX, e.clientY);
    dragging = false;
  });

  // Keyboard fallback: cycle selection with arrow keys when canvas is focused.
  dom.tabIndex = 0;
  dom.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    if (clusters.length === 0) return;
    const step = e.key === 'ArrowRight' ? 1 : -1;
    const idx = selectedCluster ? clusters.indexOf(selectedCluster) : -1;
    const next = clusters[(idx + step + clusters.length) % clusters.length];
    selectedCluster = next;
    selectMarker.position.copy(next.center);
    selectMarker.material.color.copy(next.color);
    next.pulse = 1;
    bus.emit('select-persona', next.persona);
  });

  /* ---------------------------------------------------------------- */
  /* Resize + context loss                                            */
  /* ---------------------------------------------------------------- */

  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  window.addEventListener('resize', resize);

  dom.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    showFallback(container, 'WEBGL CONTEXT LOST',
      'GPU context dropped. Remaining console panels stay live.');
  });

  /* ---------------------------------------------------------------- */
  /* Render loop                                                      */
  /* ---------------------------------------------------------------- */

  let orbitAngle = 0.6;
  let cameraDriftY = 0;
  let last = performance.now();
  let elapsed = 0;

  function tick(now) {
    const dt = clamp((now - last) / 1000, 0.001, 0.05);
    last = now;
    elapsed += dt;

    // slow auto-orbit + gentle bob; manual drag takes over immediately
    if (!dragging) orbitAngle += dt * (reduced ? 0.012 : 0.05);
    const radius = 27;
    const bob = Math.sin(elapsed * 0.12) * 1.4;
    const px = pointerIn && !dragging ? parallaxX : 0;
    const py = pointerIn && !dragging ? parallaxY : 0;
    camera.position.set(
      Math.cos(orbitAngle) * radius + px * 1.6,
      9.5 + bob + cameraDriftY - py * 2.2,
      Math.sin(orbitAngle) * radius,
    );
    camera.lookAt(0, 0.4, 0);

    // fog responds to degradation mode
    scene.fog.density += (targetFog - scene.fog.density) * Math.min(1, dt * 1.5);

    // cluster idle life
    for (const c of clusters) {
      const t = elapsed * (reduced ? 0.3 : 1) + c.phase;
      c.group.rotation.y = Math.sin(t * 0.12) * 0.12;
      const breathe = 1 + Math.sin(t * 0.9) * 0.05 + c.pulse * 0.45;
      c.core.scale.setScalar(c.baseCoreScale * breathe);
      c.ring.material.opacity = 0.18 + Math.sin(t * 0.7) * 0.06 + c.pulse * 0.5;
      c.pulse = Math.max(0, c.pulse - dt * 1.4);
    }

    // selection marker pulse
    if (selectedCluster) {
      const s = 6.2 + Math.sin(elapsed * 3.2) * 0.5;
      selectMarker.scale.setScalar(s);
      selectMarker.material.opacity = 0.75 + Math.sin(elapsed * 3.2) * 0.18;
    } else {
      selectMarker.material.opacity = Math.max(0, selectMarker.material.opacity - dt * 2);
    }

    updateStreaks(dt);
    updateTies(dt, now);
    updateFlashes(now);

    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(tick);

  // Public handles for debugging / integration checks.
  return { renderer, scene, camera, clusters, streaks, ties };
}
