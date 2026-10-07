
"""Diagnostic 2 (scratch): separate per-comparison signal from max-over-many bias."""
import sys
from collections import defaultdict, Counter

from grey_resolve.config import load_threshold_profiles
from grey_resolve.fusion.linear import LinearFusionScorer
from grey_resolve.scenario.generator import load_scenario

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/out/20261007T111729Z/scenario"
scen = load_scenario(SCENARIO)
scorer = LinearFusionScorer(load_threshold_profiles()["strict_surveillance"])
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
        s = datetime.fromisoformat(ident.active_period[0]).replace(tzinfo=ctx.timestamp.tzinfo)
        e = datetime.fromisoformat(ident.active_period[1]).replace(tzinfo=ctx.timestamp.tzinfo)
        if not (s <= ctx.timestamp <= e):
            tags.append("stale")
    if ctx.geo_cluster is not None and ctx.geo_cluster != ident.home_geo_cluster:
        tags.append("geo")
    truth = set(ident.text_aliases) | {ident.home_geo_cluster}
    if any(x not in truth for x in ctx.text_entities):
        tags.append("ent")
    return "+".join(sorted(tags)) if tags else "clean"

own_all, other_all = [], []
rank_of_own = []
per_state = defaultdict(lambda: ([], []))
for m in scen.media:
    q = m.to_context()
    st = state_of(m)
    peers = [x for x in by_persona[m.identity_id] if x.media_id != m.media_id]
    if not peers:
        continue
    own = max(scorer.context_score(q, x.to_context()) for x in peers)
    others = []
    for pid, items in by_persona.items():
        if pid == m.identity_id:
            continue
        for x in items:
            others.append(scorer.context_score(q, x.to_context()))
    own_all.append(own)
    other_all.extend(others)
    per_state[st][0].append(own)
    per_state[st][1].extend(others)
    # rank of own among persona maxima (own persona vs 59 others)
    pmax = [max(scorer.context_score(q, x.to_context()) for x in items)
            for pid, items in by_persona.items() if pid != m.identity_id]
    better = sum(1 for v in pmax if v > own)
    rank_of_own.append(1 + better)

n = len(own_all)
import statistics as stats
print(f"queries: {n}")
print(f"PER-COMPARISON own mean: {sum(own_all)/n:.3f}   other mean: {sum(other_all)/len(other_all):.3f}")
print(f"own < per-comparison other mean fraction: {sum(1 for o in own_all if o < sum(other_all)/len(other_all))/n:.3f}")
print(f"rank of own persona among persona-max scores: mean {sum(rank_of_own)/n:.1f} / 60, "
      f"top-1 fraction {sum(1 for r in rank_of_own if r == 1)/n:.3f}")
print()
print(f"{'state':<16} {'n':>4} {'own_mean':>9} {'other_mean':>11} {'own_pctl':>9}")
for st in sorted(per_state):
    o, th = per_state[st]
    mo = sum(o)/len(o)
    mth = sum(th)/len(th)
    pctl = sum(1 for v in th if v < mo)/len(th)
    print(f"{st:<16} {len(o):>4} {mo:>9.3f} {mth:>11.3f} {pctl:>9.3f}")
