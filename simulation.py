import random
import statistics
from typing import Dict, Any

def run_simulation(params: Dict = None) -> Dict[str, Any]:
    params = params or {}
    steps = int(params.get('steps', 100))
    trials = int(params.get('trials', 1000))
    init = float(params.get('init', 100.0))
    growth = float(params.get('growth', 0.01))
    vol = float(params.get('vol', 0.05))

    final_values = []
    for t in range(trials):
        v = init
        for s in range(steps):
            shock = random.gauss(growth, vol)
            v = max(0.0, v * (1 + shock))
        final_values.append(v)

    mean = statistics.mean(final_values)
    median = statistics.median(final_values)
    stdev = statistics.stdev(final_values)

    summary = (
        f"Ran {trials} trials of a {steps}-step stochastic growth simulation. "
        f"Initial value {init}, mean final value {mean:.2f}, median {median:.2f}, stdev {stdev:.2f}."
    )

    return {
        'params': params,
        'mean': mean,
        'median': median,
        'stdev': stdev,
        'summary': summary,
        'samples': final_values[:20]
    }
