/**
 * panels.js -- Grey-Resolve demo HUD panels (feed, resolver, ticker).
 *
 * Fills the sibling-provided containers from JS:
 *   #feed     : streaming synthetic contact cards (~1.8 s cadence) with
 *               procedural canvas thumbnails, detection-box overlay, trust
 *               badge and quality bar. Emits 'contact' per card and 'tie' when
 *               a query has gap <= 0.1. A tie contact is held back until the
 *               resolver has finished the previous tie, so none is cut off.
 *   #resolver : on 'tie' events runs the tie-break sequence -- face bars race
 *               neck-and-neck, context rows (GEO/TIME/ENTITY) light up, the
 *               fused bars show the true fused scores and the tiebreak's pick
 *               flashes. Outcomes: context confirms face #1, flips it correctly,
 *               flips it WRONG (misleading context -- the real tiebreak cannot
 *               tell), or is absent and face order is kept.
 *               On 'select-persona' it shows a small dossier block.
 *   #ticker   : scrolling telemetry strip with real measured numbers and the
 *               source run ids.
 *
 * Everything rendered is SYNTHETIC: procedural tiles only (seeded noise,
 *   gradients, silhouette blobs), no faces, no external images.
 */

import { mulberry32, hashSeed, ALPHA, BETA, fusedScore } from './data.js';

export { fusedScore };

const FEED_INTERVAL_MS = 1800;
const MAX_FEED_CARDS = 14;
const VERDICT_AT_MS = 2800; // tie sequence: verdict appears
const VERDICT_HOLD_MS = 2600; // ... and stays readable before the next tie may start

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

/** Resolved persona record for an id/label/payload of unknown shape. */
function resolvePersona(personas, payload) {
  if (!payload) return null;
  if (typeof payload === 'string') {
    return personas.find((p) => p.id === payload || p.label === payload) || null;
  }
  const id = payload.personaId || payload.id;
  if (id) {
    const hit = personas.find((p) => p.id === id || p.label === id);
    if (hit) return hit;
  }
  if (payload.label) {
    return personas.find((p) => p.label === payload.label) || null;
  }
  return null;
}

/**
 * Draw a procedural synthetic tile (gradients, seeded noise, silhouette
 * blobs) plus a detection-box overlay into `canvas`; returns a data URL.
 * Deterministic for a given (seed, hue).
 */
function makeThumbUrl(canvas, seed, hue) {
  const g = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const rnd = mulberry32(hashSeed(seed));

  const grad = g.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, `hsl(${hue}, 28%, 12%)`);
  grad.addColorStop(1, `hsl(${(hue + 55) % 360}, 22%, 30%)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);

  // Silhouette blobs (abstract -- not faces).
  for (let i = 0; i < 3; i++) {
    const cx = rnd() * w;
    const cy = h * (0.45 + rnd() * 0.5);
    const rw = w * (0.18 + rnd() * 0.3);
    const rh = h * (0.25 + rnd() * 0.45);
    g.fillStyle = `hsla(${hue}, 30%, ${6 + rnd() * 12}%, 0.85)`;
    g.beginPath();
    g.ellipse(cx, cy, rw, rh, rnd() * 0.6 - 0.3, 0, Math.PI * 2);
    g.fill();
  }

  // Seeded grain.
  for (let i = 0; i < 90; i++) {
    g.fillStyle = `rgba(255,255,255,${(0.02 + rnd() * 0.06).toFixed(3)})`;
    g.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2);
  }

  // Detection-box overlay + corner ticks.
  const bx = w * 0.22;
  const by = h * 0.18;
  const bw = w * 0.52;
  const bh = h * 0.62;
  g.strokeStyle = 'hsl(140, 80%, 55%)';
  g.lineWidth = 1;
  g.strokeRect(bx + 0.5, by + 0.5, bw, bh);
  const tick = 6;
  g.lineWidth = 2;
  const corners = [
    [bx, by, 1, 1],
    [bx + bw, by, -1, 1],
    [bx, by + bh, 1, -1],
    [bx + bw, by + bh, -1, -1],
  ];
  for (const [x, y, sx, sy] of corners) {
    g.beginPath();
    g.moveTo(x + sx * tick, y);
    g.lineTo(x, y);
    g.lineTo(x, y + sy * tick);
    g.stroke();
  }
  g.fillStyle = 'hsl(140, 80%, 55%)';
  g.font = '7px monospace';
  g.fillText('SYNTHETIC', bx + 3, by + 10);

  return {
    url: canvas.toDataURL('image/png'),
    box: `${Math.round(bx)},${Math.round(by)},${Math.round(bw)}x${Math.round(bh)}`,
  };
}

/** Build a .bar element with .bar-fill width and .bar-label text. */
function barEl(fraction, labelText) {
  const wrap = el('div', 'bar');
  const fill = el('div', 'bar-fill');
  fill.style.width = `${Math.max(0, Math.min(100, Math.round(fraction * 100)))}%`;
  const label = el('div', 'bar-label');
  label.textContent = labelText;
  wrap.append(fill, label);
  return { wrap, fill, label };
}

/**
 * Build the scrolling telemetry text lines from the runs object.
 * Exported for reuse/testing. Shows source run ids and real numbers.
 * @param {object} runs as produced by createDemoData().runs
 * @returns {string[]} ticker lines
 */
export function buildTickerText(runs) {
  const sweep = runs.degradedSweep
    .map((r) => `${r.label} EER ${f3(r.eer)} FMR@FNMR=1% ${f3(r.fmr1)}`)
    .join(' | ');
  const gate = runs.gateAblation
    .map((r) => `${r.label} FNMR ${f3(r.fnmrFull)}->${f3(r.fnmrGated)}`)
    .join(' | ');
  const slices = runs.fusion.slices
    .map(
      (s) =>
        `${s.label} (n=${s.n}) face ${f3(s.face)} fused ${f3(s.fused)} tiebreak ${f3(s.tiebreak)}`
    )
    .join(' | ');
  return [
    `RUN ${runs.runIds.degraded} DEGRADED SWEEP ${sweep}`,
    `RUN ${runs.runIds.degraded} GATE ABLATION (FNMR full->gated @0.5) ${gate}`,
    `RUN ${runs.runIds.fusion} + CLOSE-OUT ${runs.runIds.tiebreak} TIE SLICES ${slices}`,
    `LATENCY FaissHNSW p95 ${f3(runs.latency.hnswP95Ms50k)}ms @${runs.latency.indexSize / 1000}k (k<=${runs.latency.k}) vs brute-force ~${f3(runs.latency.bruteForceMs)}ms | TESTS ${runs.tests} | ${runs.disclaimer}`,
  ];
}

/**
 * Initialize the HUD panels.
 * @param {{bus: {on: Function, emit: Function}, data: object}} ctx
 * @returns {{destroy: () => void}|null} cleanup handle (null when DOM missing)
 */
export function init(ctx) {
  const bus = ctx && ctx.bus;
  const data = ctx && ctx.data;
  const feedEl = document.querySelector('#feed');
  const resolverEl = document.querySelector('#resolver');
  const tickerEl = document.querySelector('#ticker');
  const complete =
    data &&
    Array.isArray(data.personas) &&
    Array.isArray(data.queries) &&
    Array.isArray(data.contacts) &&
    data.runs;
  if (!bus || !complete || !feedEl || !resolverEl || !tickerEl) {
    console.warn('[demo-panels] missing ctx.data dataset or #feed/#resolver/#ticker containers');
    return null;
  }

  const timers = new Set();
  const later = (fn, ms) => {
    const id = setTimeout(() => {
      timers.delete(id);
      fn();
    }, ms);
    timers.add(id);
    return id;
  };

  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.width = 128;
  thumbCanvas.height = 96;

  const personaById = new Map(data.personas.map((p) => [p.id, p]));
  const queryById = new Map(data.queries.map((q) => [q.id, q]));

  // ---------------------------------------------------------------- feed --
  const feedPanel = el('div', 'panel');
  const feedTitle = el('div', 'panel-title');
  feedTitle.textContent = 'CONTACT FEED // SYNTHETIC STREAM';
  const feedList = el('div');
  feedPanel.append(feedTitle, feedList);
  feedEl.append(feedPanel);

  function makeCard(contact, query) {
    const persona = query.groundTruth
      ? personaById.get(query.groundTruth)
      : personaById.get(query.candidates[0] && query.candidates[0].personaId);
    const hue = persona ? persona.hue : 200;
    const card = el('div', 'feed-card');

    const thumb = makeThumbUrl(thumbCanvas, contact.tileSeed, hue);
    const img = el('img');
    img.src = thumb.url;
    img.width = 128;
    img.height = 96;
    img.alt = 'synthetic procedural tile';
    card.append(img);

    const meta = el('div', 'mono');
    meta.textContent = `${contact.id} · ${query.id} · ${query.condition}`;
    const boxLine = el('div', 'mono');
    const detConf = 0.82 + ((contact.tileSeed % 17) / 100);
    boxLine.textContent = query.trust === 'REJECT'
      ? 'BOX -- none -- SCRFD no detection'
      : `BOX ${thumb.box} · det ${detConf.toFixed(2)}`;
    card.append(meta, boxLine);

    const badge = el('div', `badge-trust-${query.trust.toLowerCase()} mono`);
    badge.textContent = `TRUST ${query.trust}`;
    card.append(badge);

    const q = barEl(query.quality, `QUALITY ${query.quality.toFixed(2)}`);
    card.append(q.wrap);

    const gapLine = el('div', 'mono');
    gapLine.textContent = query.gap == null
      ? 'gap -- (no ranking)'
      : `gap ${query.gap.toFixed(3)} · ${query.verdict}`;
    card.append(gapLine);
    return card;
  }

  const isTie = (query) => query.gap != null && query.gap <= 0.1;

  let feedIndex = 0;
  let feedTimer = 0;
  let resolverFreeAt = 0; // performance.now() time the current tie sequence ends

  function pushContact() {
    const contact = data.contacts[feedIndex % data.contacts.length];
    feedIndex += 1;
    const query = queryById.get(contact.queryId);
    if (!query) return;

    feedList.prepend(makeCard(contact, query));
    while (feedList.childElementCount > MAX_FEED_CARDS) {
      feedList.removeChild(feedList.lastChild);
    }

    bus.emit('contact', { contact, query });
    if (isTie(query)) {
      bus.emit('tie', query);
    }
  }

  /** Self-scheduling feed: a tie contact waits until the resolver is free. */
  function scheduleFeed() {
    const next = queryById.get(data.contacts[feedIndex % data.contacts.length].queryId);
    let delay = FEED_INTERVAL_MS;
    if (next && isTie(next)) delay = Math.max(delay, resolverFreeAt - performance.now());
    feedTimer = setTimeout(() => {
      pushContact();
      scheduleFeed();
    }, delay);
  }

  // ------------------------------------------------------------ resolver --
  const resPanel = el('div', 'panel');
  const resTitle = el('div', 'panel-title');
  resTitle.textContent = 'RESOLVER // EVIDENCE-ONLY TOP-2 TIEBREAK';
  const dossier = el('div', 'mono');
  dossier.style.display = 'none';
  const stage = el('div');
  const idle = el('div', 'mono');
  idle.textContent =
    'AWAITING AMBIGUOUS CONTACT ... gap<=0.1 triggers tie-break sequence ... outcomes are scripted: confirmed / flipped correct / flipped wrong / absent ... ALL DATA SYNTHETIC';
  stage.append(idle);
  resPanel.append(resTitle, dossier, stage);
  resolverEl.append(resPanel);

  let raceRaf = 0;

  function clearTie() {
    for (const id of Array.from(timers)) {
      clearTimeout(id);
      timers.delete(id);
    }
    if (raceRaf) {
      cancelAnimationFrame(raceRaf);
      raceRaf = 0;
    }
  }

  /** Animate two face bars racing neck-and-neck toward their targets. */
  function raceFaceBars(items) {
    const start = performance.now();
    const dur = 1100;
    const frame = (now) => {
      const t = Math.min(1, (now - start) / dur);
      const ease = t * (2 - t);
      for (const it of items) {
        const jitter = (1 - t) * 0.05 * Math.sin(t * 9 + it.phase);
        const w = Math.max(0, Math.min(100, (it.target * ease + jitter) * 100));
        it.fill.style.width = `${w.toFixed(1)}%`;
      }
      if (t < 1) raceRaf = requestAnimationFrame(frame);
      else raceRaf = 0;
    };
    raceRaf = requestAnimationFrame(frame);
  }

  /** Run the money-shot tie-break sequence for one query. */
  function playTie(query) {
    clearTie();
    stage.innerHTML = '';
    resolverFreeAt = performance.now() + VERDICT_AT_MS + VERDICT_HOLD_MS;

    const head = el('div', 'mono glow');
    head.textContent = `TIE DETECTED // ${query.id} // gap ${query.gap.toFixed(3)} // ${query.condition} // SYNTHETIC`;
    stage.append(head);

    const top2 = query.candidates.slice(0, 2);
    const rows = top2.map((c, i) => {
      const row = el('div');
      const label = el('div', 'key');
      label.textContent = `${c.label} (${c.personaId})`;
      const b = barEl(0, `FACE ${c.faceScore.toFixed(3)}`);
      row.append(label, b.wrap);
      stage.append(row);
      return { row, fill: b.fill, label: b.label, cand: c, phase: i * 2.1 };
    });
    raceFaceBars(rows.map((r) => ({ fill: r.fill, target: r.cand.faceScore, phase: r.phase })));

    // Context signal rows light up one by one.
    const signals = ['geo', 'time', 'entity'];
    signals.forEach((sig, i) => {
      later(() => {
        const row = el('div', 'mono glow');
        const [a, b] = top2;
        const av = a.context ? a.context[`${sig}Score`] : null;
        const bv = b.context ? b.context[`${sig}Score`] : null;
        const at = a.context ? a.context[sig] : '--';
        const bt = b.context ? b.context[sig] : '--';
        const tag = query.contextState === 'clean'
          ? ''
          : query.contextState === 'absent'
            ? ' [ABSENT]'
            : ' [MISLEADING]';
        row.textContent = `${sig.toUpperCase()} ${at} ${f3(av)} vs ${bt} ${f3(bv)}${tag}`;
        stage.append(row);
      }, 1300 + i * 450);
    });

    // Fused resolution. Bars always show the true fused scores; the pick is
    // the persona the real top-2 tiebreak rule ranks first (data.js).
    later(() => {
      const absent = query.contextState === 'absent';
      rows.forEach((r) => {
        if (absent) {
          const skip = el('div', 'mono');
          skip.textContent = 'FUSED -- skipped: no context, no evidence';
          r.row.append(skip);
        } else {
          const fused = fusedScore(r.cand);
          r.row.append(barEl(fused, `FUSED ${fused.toFixed(3)}`).wrap);
        }
      });

      const winnerIdx = top2[1].personaId === query.pick ? 1 : 0;
      const wrong = query.outcome === 'flipped-wrong';
      const good = query.outcome === 'confirmed' || query.outcome === 'flipped-correct';
      const verdict = el('div', `blink mono${good ? ' glow' : ''}${wrong ? ' verdict-bad' : ''}`);
      verdict.textContent = `${query.verdict} -- ${top2[winnerIdx].label} RANKED #1`;
      stage.append(verdict);

      if (winnerIdx === 1) {
        const truth = el('div', `mono${wrong ? ' verdict-bad' : ''}`);
        const gt = personaById.get(query.groundTruth);
        truth.textContent = `synthetic ground truth: ${gt ? gt.label : query.groundTruth}${
          wrong ? ' -- misleading context pulled the decision away from the correct face #1' : ''
        }`;
        stage.append(truth);
      }

      const note = el('div', 'mono');
      note.textContent = {
        confirmed: `context agrees with face #1; order unchanged (alpha ${ALPHA} face / beta ${BETA} context) -- SYNTHETIC`,
        'flipped-correct': 'run 557189Z at scale: correct and wrong flips roughly cancel -- tiebreak hit@1 0.444 vs face-only 0.500 at gap<=0.05 (n=18)',
        'flipped-wrong': 'the tiebreak cannot tell misleading context from clean -- run 557189Z: hit@1 0.444 vs face-only 0.500 at gap<=0.05 (n=18)',
        'kept-order': 'absent context is no evidence: the tiebreak never disturbs face order without it',
      }[query.outcome] || '';
      stage.append(note);

      rows[winnerIdx].row.classList.add('glow');
      later(() => rows[winnerIdx].row.classList.remove('glow'), 1600);
    }, VERDICT_AT_MS);
  }

  bus.on('tie', (query) => {
    if (query && query.candidates && query.candidates.length >= 2) playTie(query);
  });

  // Start the feed only once the resolver is listening for ties.
  pushContact();
  scheduleFeed();

  // ------------------------------------------------------ persona select --
  bus.on('select-persona', (payload) => {
    const persona = resolvePersona(data.personas, payload);
    dossier.innerHTML = '';
    if (!persona) {
      dossier.style.display = 'none';
      return;
    }
    dossier.style.display = '';
    const head = el('div', 'glow');
    head.textContent = `DOSSIER // ${persona.label} // SYNTHETIC PERSONA`;
    const rows = [
      ['ID', persona.id],
      ['HUE', String(persona.hue)],
      ['CENTROID 3D', `[${persona.centroid3d.join(', ')}]`],
    ];
    dossier.append(head);
    for (const [k, v] of rows) {
      const line = el('div');
      const key = el('span', 'key');
      key.textContent = k;
      const val = el('span', 'value');
      val.textContent = v;
      line.append(key, val);
      dossier.append(line);
    }
    const nContacts = data.contacts.filter((c) => {
      const q = queryById.get(c.queryId);
      return q && (q.groundTruth === persona.id || q.candidates.some((cd) => cd.personaId === persona.id));
    }).length;
    dossier.append(barEl(nContacts / data.contacts.length, `FEED CONTACTS ${nContacts}/${data.contacts.length}`).wrap);
  });

  // -------------------------------------------------------------- ticker --
  const tickPanel = el('div', 'panel mono');
  const scroller = el('div');
  scroller.style.overflow = 'hidden';
  scroller.style.whiteSpace = 'nowrap';
  const inner = el('div');
  inner.style.display = 'inline-block';
  inner.style.whiteSpace = 'nowrap';
  scroller.append(inner);
  tickPanel.append(scroller);
  tickerEl.append(tickPanel);

  let tickX = 0;
  let tickHalf = 1;
  let lastFrame = 0;
  let tickRaf = 0;

  function renderTicker() {
    const lines = buildTickerText(data.runs);
    const text = `  ${lines.join('  //  ')}  //  `;
    inner.textContent = text + text; // two copies for seamless wrap
    tickHalf = Math.max(1, inner.scrollWidth / 2);
    tickX = 0;
    inner.style.transform = 'translateX(0px)';
  }

  function tickFrame(now) {
    if (now - lastFrame >= 33) {
      lastFrame = now;
      tickX -= 1.6;
      if (tickX <= -tickHalf) tickX += tickHalf;
      inner.style.transform = `translateX(${tickX.toFixed(1)}px)`;
    }
    tickRaf = requestAnimationFrame(tickFrame);
  }
  tickRaf = requestAnimationFrame(tickFrame);

  renderTicker();
  if (data.runsReady && typeof data.runsReady.then === 'function') {
    data.runsReady.then(renderTicker).catch(() => {});
  }

  return {
    destroy() {
      clearTimeout(feedTimer);
      clearTie();
      if (tickRaf) cancelAnimationFrame(tickRaf);
    },
  };
}
