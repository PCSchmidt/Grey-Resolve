/**
 * data.js -- Grey-Resolve demo data layer.
 *
 * Deterministic synthetic demo data for the GitHub Pages "degraded-media entity
 * resolution console". Everything here is SYNTHETIC / research-grade. No real
 * imagery, no real identities, no operational claims.
 *
 * Exports:
 *   - createBus()      : tiny event bus ({on, emit}) shared by all demo modules.
 *   - createDemoData() : seeded (seed 1337) demo dataset: personas, queries,
 *                        contacts, and measured run metrics.
 *   - fusedScore() / ALPHA / BETA : demo fusion rule shared with the resolver.
 *   - init(ctx)        : contract entry point; ensures run hydration is running.
 *   - mulberry32() / hashSeed() : deterministic PRNG helpers (used by sibling
 *                        demo modules for procedural tiles).
 *
 * Network: the ONLY calls to the network are the two documented fetches of the
 * sanitized run JSONs (results/20261006T191330Z.json, results/20261007T111729Z.json).
 * They are optional: on file:// or any failure the embedded honest constants from
 * docs/RESULTS.md stay in place (data.runs.source === 'embedded').
 */

const SEED = 1337;

/** Cap for how many embedded run rows may be shown verbatim in one place. */
const RUN_IDS = {
  degraded: '20261006T191330Z',
  fusion: '20261007T111729Z',
  tiebreak: '557189Z (close-out)',
};

/**
 * Deterministic string/number -> 32-bit seed.
 * @param {string|number} value
 * @returns {number} unsigned 32-bit seed
 */
export function hashSeed(value) {
  const s = String(value);
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * mulberry32 PRNG -- deterministic float sequence in [0, 1).
 * @param {number} seed 32-bit seed
 * @returns {() => number} generator
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Create the shared demo event bus.
 * Events emitted by demo modules: 'contact', 'tie', 'mode'.
 * Events listened to by demo modules: 'select-persona'.
 * @returns {{on: (evt: string, fn: (payload: any) => void) => () => void,
 *            emit: (evt: string, payload?: any) => void}}
 */
export function createBus() {
  const listeners = new Map();
  return {
    on(evt, fn) {
      if (typeof fn !== 'function') return () => {};
      if (!listeners.has(evt)) listeners.set(evt, new Set());
      listeners.get(evt).add(fn);
      return () => {
        const set = listeners.get(evt);
        if (set) set.delete(fn);
      };
    },
    emit(evt, payload) {
      const set = listeners.get(evt);
      if (!set) return;
      for (const fn of Array.from(set)) {
        try {
          fn(payload);
        } catch (err) {
          console.error(`[demo-bus] listener for '${evt}' threw`, err);
        }
      }
    },
  };
}

/** Fusion weights, profile strict_surveillance (face / context). */
export const ALPHA = 0.7;
export const BETA = 0.3;

/**
 * Fused score for one tie candidate (alpha face + beta context mean).
 * Context mean is 0 when context is absent.
 */
export function fusedScore(candidate) {
  const c = candidate.context;
  const ctxMean = c ? (c.geoScore + c.timeScore + c.entityScore) / 3 : 0;
  return ALPHA * candidate.faceScore + BETA * ctxMean;
}

/** Context triple used by tie-break rows (values are synthetic labels). */
function ctx3(geo, geoScore, time, timeScore, entity, entityScore) {
  return { geo, geoScore, time, timeScore, entity, entityScore };
}

/**
 * Embedded honest constants -- rounded values from docs/RESULTS.md.
 * Used verbatim until/unless the sanitized run JSONs are fetchable.
 */
function embeddedRuns() {
  return {
    source: 'embedded',
    runIds: { ...RUN_IDS },
    // Degraded-input sweep (EER / FMR@FNMR=1% per condition).
    degradedSweep: [
      { key: 'clean', condition: 'clean', severity: 0, label: 'clean', eer: 0.0, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'bright025', condition: 'brightness', severity: 0.25, label: 'brightness d0.25', eer: 0.0, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'bright050', condition: 'brightness', severity: 0.5, label: 'brightness d0.50', eer: 0.005, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'bright075', condition: 'brightness', severity: 0.75, label: 'brightness d0.75', eer: 0.022, fmr1: 0.15, scored: 96, total: 138, dropped: 42 },
      { key: 'down05', condition: 'downsample', severity: 0.5, label: 'downsample x0.5', eer: 0.0, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'down025', condition: 'downsample', severity: 0.25, label: 'downsample x0.25', eer: 0.005, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'down01', condition: 'downsample', severity: 0.1, label: 'downsample x0.1', eer: 0.046, fmr1: 0.873, scored: 119, total: 138, dropped: 19 },
      { key: 'blur1', condition: 'gaussian_blur', severity: 1, label: 'gaussian_blur s1', eer: 0.005, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'blur3', condition: 'gaussian_blur', severity: 3, label: 'gaussian_blur s3', eer: 0.005, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'blur8', condition: 'gaussian_blur', severity: 8, label: 'gaussian_blur s8', eer: 0.078, fmr1: 0.547, scored: 126, total: 138, dropped: 12 },
      { key: 'angle15', condition: 'off_angle', severity: 15, label: 'off_angle 15deg', eer: 0.0, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'angle30', condition: 'off_angle', severity: 30, label: 'off_angle 30deg', eer: 0.0, fmr1: 0.0, scored: 137, total: 138, dropped: 1 },
      { key: 'angle45', condition: 'off_angle', severity: 45, label: 'off_angle 45deg', eer: 0.014, fmr1: 0.256, scored: 137, total: 138, dropped: 1 },
    ],
    // Quality-gate ablation: FNMR full-set -> gated-set at threshold 0.5.
    gateAblation: [
      { key: 'clean', label: 'clean', fnmrFull: 0.028, fnmrGated: 0.02, passRate: 0.94 },
      { key: 'bright025', label: 'brightness d0.25', fnmrFull: 0.028, fnmrGated: 0.018, passRate: 0.56 },
      { key: 'bright050', label: 'brightness d0.50', fnmrFull: 0.108, fnmrGated: 0.0, passRate: 0.04 },
      { key: 'bright075', label: 'brightness d0.75', fnmrFull: 0.543, fnmrGated: null, passRate: 0.0 },
      { key: 'down05', label: 'downsample x0.5', fnmrFull: 0.028, fnmrGated: 0.012, passRate: 0.83 },
      { key: 'down025', label: 'downsample x0.25', fnmrFull: 0.108, fnmrGated: 0.047, passRate: 0.47 },
      { key: 'down01', label: 'downsample x0.1', fnmrFull: 0.989, fnmrGated: 1.0, passRate: 0.05 },
      { key: 'blur1', label: 'gaussian_blur s1', fnmrFull: 0.033, fnmrGated: 0.011, passRate: 0.5 },
      { key: 'blur3', label: 'gaussian_blur s3', fnmrFull: 0.127, fnmrGated: 0.0, passRate: 0.09 },
      { key: 'blur8', label: 'gaussian_blur s8', fnmrFull: 0.995, fnmrGated: null, passRate: 0.008 },
      { key: 'angle15', label: 'off_angle 15deg', fnmrFull: 0.028, fnmrGated: 0.02, passRate: 0.93 },
      { key: 'angle30', label: 'off_angle 30deg', fnmrFull: 0.033, fnmrGated: 0.025, passRate: 0.93 },
      { key: 'angle45', label: 'off_angle 45deg', fnmrFull: 0.047, fnmrGated: 0.049, passRate: 0.96 },
    ],
    // Phase 2 at-scale fusion ablation (downsample x0.1 queries).
    // tiebreak column comes from the close-out run 557189Z (embedded only).
    fusion: {
      slices: [
        { key: 'gap005', label: 'gap<=0.05', n: 18, face: 0.5, fused: 0.167, selective: null, tiebreak: 0.444 },
        { key: 'gap01', label: 'gap<=0.1', n: 27, face: 0.63, fused: 0.407, selective: null, tiebreak: 0.63 },
        { key: 'gap02', label: 'gap<=0.2', n: 66, face: 0.848, fused: 0.667, selective: null, tiebreak: 0.848 },
        { key: 'overall', label: 'overall', n: 106, face: 0.906, fused: 0.774, selective: 0.849, tiebreak: 0.906 },
      ],
    },
    latency: { hnswP95Ms50k: 1.6, bruteForceMs: 29, k: 100, dim: 512, indexSize: 50000 },
    tests: 241,
    disclaimer: 'SYNTHETIC / RESEARCH DATA -- NOT OPERATIONAL METRICS',
  };
}

const SWEEP_KEYS = {
  'clean|0': 'clean',
  'brightness|0.25': 'bright025',
  'brightness|0.5': 'bright050',
  'brightness|0.75': 'bright075',
  'downsample|0.5': 'down05',
  'downsample|0.25': 'down025',
  'downsample|0.1': 'down01',
  'gaussian_blur|1': 'blur1',
  'gaussian_blur|3': 'blur3',
  'gaussian_blur|8': 'blur8',
  'off_angle|15': 'angle15',
  'off_angle|30': 'angle30',
  'off_angle|45': 'angle45',
};

function sweepKey(condition, severity) {
  return SWEEP_KEYS[`${condition}|${Number(severity)}`] || null;
}

/** Merge fetched degraded-sweep JSON (run 191330Z) into the runs object. */
function mergeDegradedRun(runs, json) {
  const rows = Array.isArray(json && json.sweep) ? json.sweep : [];
  for (const row of rows) {
    const key = sweepKey(row.condition, row.severity);
    const target = runs.degradedSweep.find((r) => r.key === key);
    if (target) {
      target.eer = row.eer;
      target.fmr1 = row['fmr_at_fnmr_0.01'];
      target.scored = row.n_scored;
      target.total = row.n_images;
      target.dropped = row.n_query_dropped;
    }
  }
  const gate = Array.isArray(json && json.gated_vs_ungated) ? json.gated_vs_ungated : [];
  for (const row of gate) {
    const key = sweepKey(row.condition, row.severity);
    const target = runs.gateAblation.find((r) => r.key === key);
    if (target) {
      target.fnmrFull = row.full ? row.full.fnmr : null;
      target.fnmrGated = row.gated ? row.gated.fnmr : null;
      target.passRate = row.gated ? row.gated.pass_rate : 0;
    }
  }
}

/** Merge fetched fusion-ablation JSON (run 111729Z) into the runs object. */
function mergeFusionRun(runs, json) {
  const ab = json && json.ablation;
  if (!ab) return;
  const slices = runs.fusion.slices;
  const overall = slices.find((s) => s.key === 'overall');
  if (overall && ab.face_only && ab.face_only.overall) {
    overall.n = ab.face_only.overall.n_queries;
    overall.face = ab.face_only.overall.hit_at_1;
    overall.fused = ab.fused ? ab.fused.overall.hit_at_1 : overall.fused;
    overall.selective = ab.selective ? ab.selective.overall.hit_at_1 : overall.selective;
  }
  const gapLe = ab.gap_le || {};
  for (const [cutoff, sliceKey] of [['0.05', 'gap005'], ['0.1', 'gap01'], ['0.2', 'gap02']]) {
    const src = gapLe[cutoff];
    const target = slices.find((s) => s.key === sliceKey);
    if (src && target) {
      target.n = src.n_queries;
      target.face = src.face_only ? src.face_only.hit_at_1 : target.face;
      target.fused = src.fused ? src.fused.hit_at_1 : target.fused;
    }
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> HTTP ${res.status}`);
  return res.json();
}

/**
 * Try the two documented fetches; merge real values into `runs` in place.
 * Resolves even when both fetches fail (file:// etc.) -- never rejects.
 * @param {object} data as returned by createDemoData()
 * @returns {Promise<object>} the runs object (mutated in place)
 */
async function hydrateRuns(data) {
  const runs = data.runs;
  const results = await Promise.allSettled([
    fetchJson('results/20261006T191330Z.json'),
    fetchJson('results/20261007T111729Z.json'),
  ]);
  let merged = 0;
  if (results[0].status === 'fulfilled') {
    try {
      mergeDegradedRun(runs, results[0].value);
      merged += 1;
    } catch (err) {
      console.warn('[demo-data] degraded run merge failed', err);
    }
  }
  if (results[1].status === 'fulfilled') {
    try {
      mergeFusionRun(runs, results[1].value);
      merged += 1;
    } catch (err) {
      console.warn('[demo-data] fusion run merge failed', err);
    }
  }
  if (merged > 0) {
    runs.source = merged === 2 ? 'fetched (both runs)' : 'fetched (partial)';
  } else {
    runs.source = 'embedded (fetch unavailable -- docs/RESULTS.md constants)';
  }
  return runs;
}

/**
 * Create the deterministic demo dataset (seed 1337).
 *
 * Shapes:
 *   personas : 12 x {id, label, hue, centroid3d:[x,y,z]}
 *   queries  : feed/resolver queries encoding the real finding states:
 *              clean-context ties that confirm face #1 or correctly flip it,
 *              misleading-context ties that flip a correct face #1 to the
 *              wrong candidate (the at-scale failure mode -- the real tiebreak
 *              has no misleading-context detector), absent-context ties where
 *              face order is kept, detection failures (REJECT), and clear-gap
 *              contacts (no tiebreak). Each tie carries `pick`, the persona the
 *              top-2 tiebreak actually ranks first.
 *   contacts : deterministic feed stream sequence referencing queries.
 *   runs     : measured metrics -- real sanitized run values when fetchable,
 *              else embedded honest constants from docs/RESULTS.md.
 *              `runs.runsReady` is a promise that resolves after hydration.
 *
 * @returns {{personas: object[], queries: object[], contacts: object[], runs: object}}
 */
export function createDemoData() {
  const rng = mulberry32(SEED);

  const personas = [];
  for (let i = 0; i < 12; i++) {
    const n = i + 1;
    personas.push({
      id: `syn-id-${String(n).padStart(4, '0')}`,
      label: `PERSONA-${String(n).padStart(2, '0')}`,
      hue: Math.floor(rng() * 360),
      centroid3d: [
        Math.round((rng() * 2 - 1) * 1000) / 1000,
        Math.round((rng() * 2 - 1) * 1000) / 1000,
        Math.round((rng() * 2 - 1) * 1000) / 1000,
      ],
    });
  }
  const byId = new Map(personas.map((p) => [p.id, p]));
  const cand = (id, faceScore, context) => ({
    personaId: id,
    label: byId.get(id).label,
    faceScore,
    context,
  });

  const queries = [
    {
      id: 'Q-01', condition: 'downsample x0.1', trust: 'LOW', quality: 0.42,
      gap: 0.031, contextState: 'clean', groundTruth: 'syn-id-0004',
      candidates: [
        cand('syn-id-0004', 0.412, ctx3('GRID-14', 0.82, 'T-08:14Z', 0.77, 'EVT-221', 0.74)),
        cand('syn-id-0009', 0.381, ctx3('GRID-27', 0.31, 'T-15:02Z', 0.44, 'EVT-088', 0.22)),
      ],
    },
    {
      id: 'Q-02', condition: 'gaussian_blur s8', trust: 'LOW', quality: 0.38,
      gap: 0.047, contextState: 'absent', groundTruth: 'syn-id-0002',
      candidates: [
        cand('syn-id-0002', 0.357, null),
        cand('syn-id-0007', 0.31, null),
      ],
    },
    {
      id: 'Q-03', condition: 'brightness d0.75', trust: 'REJECT', quality: 0.11,
      gap: null, contextState: 'absent', groundTruth: null,
      candidates: [],
    },
    {
      id: 'Q-04', condition: 'clean', trust: 'FULL', quality: 0.71,
      gap: 0.085, contextState: 'clean', groundTruth: 'syn-id-0011',
      candidates: [
        cand('syn-id-0011', 0.521, ctx3('GRID-03', 0.79, 'T-11:40Z', 0.81, 'EVT-402', 0.7)),
        cand('syn-id-0003', 0.436, ctx3('GRID-19', 0.35, 'T-22:07Z', 0.28, 'EVT-117', 0.41)),
      ],
    },
    {
      id: 'Q-05', condition: 'downsample x0.1', trust: 'LOW', quality: 0.35,
      gap: 0.029, contextState: 'misleading', groundTruth: 'syn-id-0006',
      candidates: [
        cand('syn-id-0006', 0.398, ctx3('GRID-08', 0.42, 'T-03:33Z', 0.38, 'EVT-221', 0.31)),
        cand('syn-id-0001', 0.369, ctx3('GRID-08', 0.71, 'T-03:31Z', 0.66, 'EVT-221', 0.62)),
      ],
    },
    {
      id: 'Q-06', condition: 'clean', trust: 'FULL', quality: 0.83,
      gap: 0.34, contextState: 'clean', groundTruth: 'syn-id-0008',
      candidates: [
        cand('syn-id-0008', 0.611, ctx3('GRID-11', 0.75, 'T-09:55Z', 0.72, 'EVT-310', 0.68)),
        cand('syn-id-0005', 0.271, ctx3('GRID-31', 0.24, 'T-18:12Z', 0.19, 'EVT-055', 0.27)),
      ],
    },
    {
      id: 'Q-07', condition: 'gaussian_blur s8', trust: 'REJECT', quality: 0.16,
      gap: null, contextState: 'absent', groundTruth: null,
      candidates: [],
    },
    {
      id: 'Q-08', condition: 'downsample x0.1', trust: 'LOW', quality: 0.49,
      gap: 0.012, contextState: 'clean', groundTruth: 'syn-id-0010',
      candidates: [
        cand('syn-id-0010', 0.404, ctx3('GRID-06', 0.8, 'T-07:02Z', 0.75, 'EVT-150', 0.77)),
        cand('syn-id-0012', 0.392, ctx3('GRID-22', 0.33, 'T-19:44Z', 0.36, 'EVT-208', 0.29)),
      ],
    },
    {
      id: 'Q-09', condition: 'off_angle 45deg', trust: 'LOW', quality: 0.33,
      gap: 0.058, contextState: 'misleading', groundTruth: 'syn-id-0003',
      candidates: [
        cand('syn-id-0003', 0.377, ctx3('GRID-17', 0.36, 'T-13:19Z', 0.31, 'EVT-402', 0.25)),
        cand('syn-id-0007', 0.319, ctx3('GRID-17', 0.69, 'T-13:21Z', 0.73, 'EVT-402', 0.66)),
      ],
    },
    {
      id: 'Q-10', condition: 'brightness d0.25', trust: 'FULL', quality: 0.79,
      gap: 0.27, contextState: 'clean', groundTruth: 'syn-id-0005',
      candidates: [
        cand('syn-id-0005', 0.583, ctx3('GRID-25', 0.71, 'T-16:28Z', 0.68, 'EVT-090', 0.73)),
        cand('syn-id-0011', 0.313, ctx3('GRID-02', 0.29, 'T-05:47Z', 0.35, 'EVT-333', 0.22)),
      ],
    },
    {
      id: 'Q-11', condition: 'downsample x0.1', trust: 'LOW', quality: 0.4,
      gap: 0.093, contextState: 'absent', groundTruth: 'syn-id-0001',
      candidates: [
        cand('syn-id-0001', 0.429, null),
        cand('syn-id-0004', 0.336, null),
      ],
    },
    {
      id: 'Q-12', condition: 'downsample x0.1', trust: 'REJECT', quality: 0.09,
      gap: null, contextState: 'absent', groundTruth: null,
      candidates: [],
    },
    {
      // Clean context overturns a wrong face #1 -- the case the tiebreak exists for.
      id: 'Q-13', condition: 'gaussian_blur s8', trust: 'LOW', quality: 0.36,
      gap: 0.022, contextState: 'clean', groundTruth: 'syn-id-0002',
      candidates: [
        cand('syn-id-0012', 0.366, ctx3('GRID-30', 0.3, 'T-21:16Z', 0.27, 'EVT-064', 0.33)),
        cand('syn-id-0002', 0.344, ctx3('GRID-09', 0.81, 'T-10:05Z', 0.76, 'EVT-187', 0.72)),
      ],
    },
  ];

  // Annotate each query with its real finding-state verdict. Tie outcomes are
  // computed with the same rule as the real top-2 tiebreak: skip when context
  // is absent, otherwise the higher fused score ranks first.
  for (const q of queries) {
    if (q.trust === 'REJECT') {
      q.outcome = 'detect-fail';
      q.verdict = 'DETECTION FAILURE -- NO FACE FOUND (SCRFD)';
    } else if (q.gap != null && q.gap <= 0.1) {
      const [a, b] = q.candidates;
      if (q.contextState === 'absent') {
        q.pick = a.personaId;
        q.outcome = 'kept-order';
        q.verdict = 'TIEBREAK: CONTEXT ABSENT -- FACE ORDER KEPT';
      } else if (fusedScore(b) <= fusedScore(a)) {
        q.pick = a.personaId;
        q.outcome = 'confirmed';
        q.verdict = 'TIEBREAK: CONTEXT CONFIRMED FACE #1';
      } else {
        q.pick = b.personaId;
        const correct = q.pick === q.groundTruth;
        q.outcome = correct ? 'flipped-correct' : 'flipped-wrong';
        q.verdict = correct
          ? 'TIEBREAK: CONTEXT FLIPPED FACE #1 -- CORRECT'
          : 'TIEBREAK: CONTEXT FLIPPED FACE #1 -- WRONG';
      }
    } else {
      q.outcome = 'clear';
      q.verdict = 'FACE RANK CLEAR -- NO TIEBREAK';
    }
  }

  // Deterministic feed sequence (cycles). Ties are interleaved with non-ties;
  // the feed also holds a tie back until the resolver has finished the last one.
  const order = ['Q-06', 'Q-01', 'Q-03', 'Q-13', 'Q-10', 'Q-09', 'Q-02', 'Q-12', 'Q-05', 'Q-08', 'Q-07', 'Q-04', 'Q-11'];
  const contacts = order.map((queryId, i) => ({
    id: `C-${String(i + 1).padStart(4, '0')}`,
    seq: i,
    queryId,
    tileSeed: 1000 + i * 7919,
  }));

  const runs = embeddedRuns();
  const data = { seed: SEED, personas, queries, contacts, runs };
  // Documented optional hydration: exactly two fetches, graceful everywhere.
  data.runsReady = hydrateRuns(data);
  return data;
}

/**
 * Contract entry point: ensure ctx has a working bus and a complete demo
 * dataset, then start run hydration (the two documented fetches).
 *
 * Compatible with both caller styles:
 *   - caller passes a dataset built by createDemoData() -> reused as-is
 *     (hydration is idempotent via data.runsReady);
 *   - caller passes a bare/incomplete ctx -> the dataset is created here
 *     (seed 1337) and assigned to ctx.data.
 * Returns {bus, data} so orchestrators can merge the canonical objects.
 *
 * @param {{bus?: object, data?: object}} ctx
 * @returns {{bus: object, data: object}}
 */
export function init(ctx) {
  const c = ctx && typeof ctx === 'object' ? ctx : {};
  if (!c.bus || typeof c.bus.on !== 'function' || typeof c.bus.emit !== 'function') {
    c.bus = createBus();
  }
  const hasDataset =
    c.data &&
    Array.isArray(c.data.personas) &&
    Array.isArray(c.data.queries) &&
    Array.isArray(c.data.contacts) &&
    c.data.runs;
  if (!hasDataset) {
    c.data = createDemoData();
  } else if (!c.data.runsReady) {
    c.data.runsReady = hydrateRuns(c.data);
  }
  return { bus: c.bus, data: c.data };
}
