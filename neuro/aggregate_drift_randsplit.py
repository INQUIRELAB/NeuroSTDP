import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
N_RAND = 20
METHODS = ['snn', 'km_frozen', 'km_refit', 'streamkm', 'mbkm']

def main():
    seeds = list(range(N_RAND))
    missing = [s for s in seeds if not (DOCS / f'streaming_drift_v2_randsplit_seed{s}.json').exists()]
    if missing:
        print(f'MISSING per-seed JSONs for randsplit seeds: {missing}')
        return
    per_seed_mean = {m: {} for m in METHODS}
    for s in seeds:
        d = json.load(open(DOCS / f'streaming_drift_v2_randsplit_seed{s}.json'))
        steps = d['step_rows']
        for m in METHODS:
            key = f'{m}_map30'
            vals = [r[key] * 100.0 for r in steps]
            per_seed_mean[m][s] = float(np.mean(vals))
    agg = {}
    for m in METHODS:
        vals = np.array([per_seed_mean[m][s] for s in seeds])
        agg[m] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)), 'min': float(vals.min()), 'max': float(vals.max()), 'per_seed': {str(s): per_seed_mean[m][s] for s in seeds}}
    deltas = np.array([per_seed_mean['snn'][s] - per_seed_mean['streamkm'][s] for s in seeds])
    snn_advantage = {'mean_pp': float(deltas.mean()), 'std_pp': float(deltas.std(ddof=1)), 'min_pp': float(deltas.min()), 'max_pp': float(deltas.max()), 'n_positive': int((deltas > 0).sum()), 'n_seeds': len(deltas), 'per_seed_pp': {str(s): float(d) for s, d in zip(seeds, deltas)}}
    out = {'split_mode': 'random', 'split_seed': 42, 'n_seeds': N_RAND, 'agg_mean_stream_map30_pp': agg, 'snn_advantage_over_streamkm': snn_advantage}
    out_path = DOCS / 'streaming_drift_v2_randsplit_summary.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'saved: {out_path}')
    print()
    print(f'=== Side-by-side: by-ID (N=20) vs random-50/50 (N={N_RAND}) ===')
    byid = json.load(open(DOCS / 'streaming_drift_v2_multiseed_n20.json'))
    byid_adv = byid['snn_advantage_over_streamkm']
    print(f'{'Split':<18} {'SNN-vs-streamkm Δ':<22} {'positive seeds':<18}')
    print(f'{'by-id (IDs<=113)':<18} {'+' + format(byid_adv['mean_pp'], '.2f') + ' ± ' + format(byid_adv['std_pp'], '.2f') + ' pp':<22} {byid_adv['n_positive']}/{byid_adv['n_seeds']}')
    print(f'{'random 50/50':<18} {'+' + format(snn_advantage['mean_pp'], '.2f') + ' ± ' + format(snn_advantage['std_pp'], '.2f') + ' pp':<22} {snn_advantage['n_positive']}/{snn_advantage['n_seeds']}')
    print()
    print(f'=== Random-split per-method means (mean stream mAP@30, %) ===')
    for m in METHODS:
        a = agg[m]
        print(f'  {m:14s} {a['mean']:.2f} ± {a['std']:.2f}   range [{a['min']:.2f}, {a['max']:.2f}]')
if __name__ == '__main__':
    main()
