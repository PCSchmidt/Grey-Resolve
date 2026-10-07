
"""Diagnostic 3 (scratch): 2-way tiebreak accuracy of context (the fix hypothesis)."""
import sys
from collections import defaultdict

from grey_resolve.config import load_threshold_profiles
from grey_resolve.fusion.linear import LinearFusionScorer
from grey_resolve.scenario.generator import load_scenario

scen = load_scenario(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/out/20261007T111729Z/scenario")
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

# P(own_max > one rival's max) and P(own_max > rival's single first-item score)
import random
rng = random.Random(42)
per_state = defaultdict(lambda: [0, 0, 0, 0])  # win, tie, loss, n
for m in scen.media:
    q = m.to_context()
    st = state_of(m)
    peers = [x for x in by_persona[m.identity_id] if x.media_id != m.media_id]
    if not peers:
        continue
    own = max(scorer.context_score(q, x.to_context()) for x in peers)
    rivals = [pid for pid in by_persona if pid != m.identity_id]
    wins = ties = losses = 0
    for pid in rivals:
        rmax = max(scorer.context_score(q, x.to_context()) for x in by_persona[pid])
        if own > rmax: wins += 1
        elif own == rmax: ties += 1
        else: losses += 1
    n = len(rivals)
    per_state[st][0] += wins
    per_state[st][1] += ties
    per_state[st][2] += losses
    per_state[st][3] += n

print(f"{'state':<16} {'P(own>rival_max)':>17} {'tie':>6} {'P(loss)':>8}")
for st in sorted(per_state):
    w, t, l, n = per_state[st]
    print(f"{st:<16} {w/n:>17.3f} {t/n:>6.3f} {l/n:>8.3f}")
w = sum(v[0] for v in per_state.values()); t = sum(v[1] for v in per_state.values())
l = sum(v[2] for v in per_state.values()); n = sum(v[3] for v in per_state.values())
print(f"{'ALL':<16} {w/n:>17.3f} {t/n:>6.3f} {l/n:>8.3f}")
