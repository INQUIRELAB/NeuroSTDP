import json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
CONFIGS = [('v127d_plus_6ch_test', 'Six-channel union'), ('v127d_test', 'Classical cascade alone'), ('v128_M1_label_efficient_test', 'SNN label-efficient (six-recipe pool)'), ('v128_neuro_label_efficient_test', 'SNN label-efficient (four-recipe pool)')]
N_BOOT = 1000
RNG_SEED = 42

def main():
    rng = np.random.default_rng(RNG_SEED)
    d = json.load(open(DOCS / 'eval_v2_M1_test.json'))
    out = {}
    print(f'Sequence-level bootstrap CIs (n_boot={N_BOOT}, seed={RNG_SEED})')
    print(f'{'config':<60} {'mAP@30':<22} {'mAP@50':<22}')
    print('-' * 110)
    for key, label in CONFIGS:
        if key not in d:
            print(f'  MISSING: {key}')
            continue
        e = d[key]
        sids = list(e.keys())
        map30 = np.array([e[s]['mAP_30'] for s in sids]) * 100.0
        map50 = np.array([e[s]['mAP_50'] for s in sids]) * 100.0
        n = len(map30)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot30 = map30[idx].mean(axis=1)
        boot50 = map50[idx].mean(axis=1)
        ci30 = np.percentile(boot30, [2.5, 97.5])
        ci50 = np.percentile(boot50, [2.5, 97.5])
        out[key] = {'label': label, 'n_seqs': n, 'mAP30_mean': float(map30.mean()), 'mAP30_ci95': [float(ci30[0]), float(ci30[1])], 'mAP30_se': float(boot30.std(ddof=1)), 'mAP50_mean': float(map50.mean()), 'mAP50_ci95': [float(ci50[0]), float(ci50[1])], 'mAP50_se': float(boot50.std(ddof=1))}
        print(f'  {label:<58} {map30.mean():.2f} [{ci30[0]:.2f}, {ci30[1]:.2f}]   {map50.mean():.2f} [{ci50[0]:.2f}, {ci50[1]:.2f}]')
    print()
    print('=== Drift advantage (SNN vs streaming-k-means), N=20 by-id ===')
    drift = json.load(open(DOCS / 'streaming_drift_v2_multiseed_n20.json'))
    deltas_per_seed = drift['snn_advantage_over_streamkm']['per_seed_pp']
    deltas = np.array([float(v) for v in deltas_per_seed.values()])
    n_seeds = len(deltas)
    boot_idx = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))
    boot_delta = deltas[boot_idx].mean(axis=1)
    ci_delta = np.percentile(boot_delta, [2.5, 97.5])
    print(f'  drift Δ = {deltas.mean():+.2f} pp  bootstrap-over-seeds 95% CI [{ci_delta[0]:+.2f}, {ci_delta[1]:+.2f}]')
    print(f'  (this is a CI over the N={n_seeds} seeds, NOT over sequences;')
    print(f'   sequence-level uncertainty for the DRIFT advantage is not directly available')
    print(f'   without per-seed per-sequence breakdown which the streaming protocol does not produce.)')
    out['drift_advantage_pp_seed_bootstrap'] = {'n_seeds': n_seeds, 'mean_pp': float(deltas.mean()), 'ci95_pp': [float(ci_delta[0]), float(ci_delta[1])], 'note': 'bootstrap over seeds, not sequences'}
    out_path = DOCS / 'bootstrap_seq_cis.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print()
    print(f'saved: {out_path}')
if __name__ == '__main__':
    main()
