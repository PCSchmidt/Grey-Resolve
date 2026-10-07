
"""Diagnostic (scratch): is the fabricated context signal correct, and how does
context_score behave on absent/noisy/drifted query contexts? Pure metadata --
no images, no embeddings. Reads a saved scenario dir."""

import sys
from collections import defaultdict

from grey_resolve.config import load_threshold_profiles
from grey_resolve.fusion.linear import LinearFusionScorer
from grey_resolve.scenario.generator import load_scenario

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/out/20261007T111729Z/scenario"

scen = load_scenario(SCENARIO)
profiles = load_threshold_profiles()
scorer = LinearFusionScorer(profiles["strict_surveillance"])

idents = {i.identity_id: i for i in scen.identities}
by_persona = defaultdict(list)
for m in scen.media:
    by_persona[m.identity_id].append(m)

def state_of(m):
    ctx = m.to_context()
    ident = idents[m.identity_id]
    if ctx.timestamp is None and ctx.geo_cluster is None and not ctx.text_entities:
        return "absent"
    tags = []
    if ctx.timestamp is not None:
        from datetime import datetime
        start = datetime.fromisoformat(ident.active_period[0]).replace(tzinfo=ctx.timestamp.tzinfo)
        end = datetime.fromisoformat(ident.active_period[1]).replace(tzinfo=ctx.timestamp.tzinfo)
        if not (start <= ctx.timestamp <= end):
            tags.append("stale_ts")
    if ctx.geo_cluster is not None and ctx.geo_cluster != ident.home_geo_cluster:
        tags.append("wrong_geo")
    truth = set(ident.text_aliases) | {ident.home_geo_cluster}
    if any(e not in truth for e in ctx.text_entities):
        tags.append("misleading_ent")
    return "+".join(sorted(tags)) if tags else "clean"

rows = []
for m in scen.media:
    qctx = m.to_context()
    st = state_of(m)
    peers = [x for x in by_persona[m.identity_id] if x.media_id != m.media_id]
    if not peers:
        continue
    own = max(scorer.context_score(qctx, x.to_context()) for x in peers)
    other_scores = []
    for pid, items in by_persona.items():
        if pid == m.identity_id:
            continue
        other_scores.append(max(scorer.context_score(qctx, x.to_context()) for x in items))
    best_other = max(other_scores) if other_scores else 0.0
    rows.append((st, own, best_other, len(qctx.text_entities) == 0 and qctx.geo_cluster is None))

print(f"queries: {len(rows)}")
from collections import Counter
print("state mix:", dict(Counter(r[0] for r in rows)))
print()
print(f"{'state':<22} {'n':>4} {'own>other':>10} {'tie':>6} {'own<other':>10} {'mean own':>9} {'mean other':>11}")
for st in sorted(set(r[0] for r in rows)):
    sub = [r for r in rows if r[0] == st]
    wins = sum(1 for r in sub if r[1] > r[2])
    ties = sum(1 for r in sub if abs(r[1] - r[2]) < 1e-12)
    losses = sum(1 for r in sub if r[1] < r[2])
    mo = sum(r[1] for r in sub) / len(sub)
    mb = sum(r[2] for r in sub) / len(sub)
    print(f"{st:<22} {len(sub):>4} {wins:>10} {ties:>6} {losses:>10} {mo:>9.3f} {mb:>11.3f}")
overall_w = sum(1 for r in rows if r[1] > r[2]) / len(rows)
print(f"\noverall context-signal accuracy (own strictly higher): {overall_w:.3f}")
