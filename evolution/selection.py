"""Tournament selection: sample candidates, prefer fitness, randomize equal ties."""
import math


def tournament(members, scores, rng, size=3):
    if not members or size < 1 or any(not math.isfinite(scores[m.id]) for m in members):
        raise ValueError('selection requires a population and finite scores')
    candidates = rng.sample(list(members), min(size, len(members)))
    best = max(scores[m.id] for m in candidates)
    return rng.choice([m for m in candidates if scores[m.id] == best])
