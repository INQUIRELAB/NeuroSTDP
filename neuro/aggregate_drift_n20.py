import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
N_SEEDS = 20
METHODS = ['snn', 'km_frozen', 'km_refit', 'streamkm', 'mbkm']
LABEL_BY_KEY = {'snn': 'snn', 'km_frozen': 'km_frozen', 'km_refit': 'km_refit', 'streamkm': 'streamkm', 'mbkm': 'mbkm'}

def seed_json_path(seed: int) -> Path:
    if seed == 0:
        return DOCS / 'streaming_drift_v2_results.json'
    return DOCS / f'streaming_drift_v2_results_seed{seed}.json'

def main():
    seeds = list(range(N_SEEDS))
    missing = [s for s in seeds if not seed_json_path(s).exists()]
    if missing:
        print(f'MISSING per-seed JSONs for seeds: {missing}')
        return
    per_seed_mean = {m: {} for m in METHODS}
    per_seed_steps = {}
    for s in seeds:
        d = json.load(open(seed_json_path(s)))
        steps = d['step_rows']
        per_seed_steps[s] = steps
        for m in METHODS:
            key = f'{m}_map30'
            vals = [r[key] * 100.0 for r in steps]
            per_seed_mean[m][s] = float(np.mean(vals))
    agg = {}
    for m in METHODS:
        vals = np.array([per_seed_mean[m][s] for s in seeds])
        agg[m] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)), 'min': float(vals.min()), 'max': float(vals.max()), 'per_seed': {str(s): per_seed_mean[m][s] for s in seeds}}
    snn_vs_streamkm_per_seed = {str(s): per_seed_mean['snn'][s] - per_seed_mean['streamkm'][s] for s in seeds}
    deltas = np.array(list(snn_vs_streamkm_per_seed.values()))
    snn_advantage = {'mean_pp': float(deltas.mean()), 'std_pp': float(deltas.std(ddof=1)), 'min_pp': float(deltas.min()), 'max_pp': float(deltas.max()), 'n_positive': int((deltas > 0).sum()), 'n_seeds': len(deltas), 'per_seed_pp': snn_vs_streamkm_per_seed}
    out = {'n_seeds': N_SEEDS, 'seeds': seeds, 'per_seed_mean_stream_map30_pp': per_seed_mean, 'agg_mean_stream_map30_pp': agg, 'snn_advantage_over_streamkm': snn_advantage}
    out_path = DOCS / 'streaming_drift_v2_multiseed_n20.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'saved: {out_path}')
    print()
    print(f'=== N={N_SEEDS} aggregate (mean stream mAP@30, %) ===')
    for m in METHODS:
        a = agg[m]
        print(f'  {LABEL_BY_KEY[m]:14s} {a['mean']:.2f} ± {a['std']:.2f}   range [{a['min']:.2f}, {a['max']:.2f}]')
    print()
    print(f'=== SNN vs streaming-k-means delta (pp) ===')
    print(f'  mean   = {snn_advantage['mean_pp']:+.2f} ± {snn_advantage['std_pp']:.2f}')
    print(f'  range  = [{snn_advantage['min_pp']:+.2f}, {snn_advantage['max_pp']:+.2f}]')
    print(f'  positive on {snn_advantage['n_positive']}/{snn_advantage['n_seeds']} seeds')
if __name__ == '__main__':
    main()
