#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
P3 = Path(os.environ.get('NEUROSTDP_ROOT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(P3 / 'code'))
sys.path.insert(0, str(P3 / 'code/neuro'))
from snn_cohort_gate import SpikingCohortGate
FP_KEYS = ['ev_med', 'ev_std', 'density_std', 'pol_imbalance', 'temporal_var', 'cx_med', 'cx_std', 'cy_med', 'cy_std', 'area_med']

def load_ev_uav_fingerprints(json_path: Path) -> Dict[str, Dict]:
    if not json_path.exists():
        raise FileNotFoundError(f'EV-UAV fingerprint source {json_path} not found. Run code/neuro/ev_uav_full_eval.py first.')
    d = json.loads(json_path.read_text())
    out: Dict[str, Dict] = {}
    for s in d.get('sequence_summaries', []):
        sid = s['seq_id']
        out[sid] = {'seq_id': sid, 'fingerprint': s['fingerprint']}
    return out

def compute_zscore_params(train_fps: Dict[str, Dict]) -> Dict:
    params = {}
    for k in FP_KEYS:
        vals = np.array([fp['fingerprint'][k] for fp in train_fps.values()])
        params[k] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=0) + 1e-08)}
    return params

def vectorise(fp: Dict, zp: Dict) -> List[float]:
    vec = []
    for k in FP_KEYS:
        v = fp['fingerprint'][k]
        z = (v - zp[k]['mean']) / zp[k]['std']
        vec.append(float(np.clip(z, -5.0, 5.0)))
    return vec

def train_and_eval(epochs: int=30, n_output: int=12, seed: int=0, T: int=350, theta0: float=0.5, w_norm_target: float=5.0, update_mode: str='online'):
    train_path = P3 / 'docs' / 'ev_uav_full_train_hotfix_union_results.json'
    val_path = P3 / 'docs' / 'ev_uav_full_val_hotfix_union_results.json'
    test_path = P3 / 'docs' / 'ev_uav_full_test_hotfix_union_results.json'
    train_fps = load_ev_uav_fingerprints(train_path)
    val_fps = load_ev_uav_fingerprints(val_path)
    test_fps = load_ev_uav_fingerprints(test_path)
    print(f'EV-UAV: {len(train_fps)} train, {len(val_fps)} val, {len(test_fps)} test fingerprints.')
    zp = compute_zscore_params(train_fps)
    train_vecs = {sid: vectorise(fp, zp) for sid, fp in train_fps.items()}
    val_vecs = {sid: vectorise(fp, zp) for sid, fp in val_fps.items()}
    test_vecs = {sid: vectorise(fp, zp) for sid, fp in test_fps.items()}
    gate = SpikingCohortGate(n_in=10, n_hidden=32, n_output=n_output, theta0=theta0, w_norm_target=w_norm_target, seed=seed, update_mode=update_mode)
    print(f'Training 10-dim EV-UAV SNN ({epochs} epochs, T={T}, seed={seed})...')
    sids = sorted(train_vecs.keys())
    rng_master = torch.Generator().manual_seed(seed)
    history = []
    best_score = -1.0
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        order = torch.randperm(len(sids), generator=rng_master).tolist()
        out_spike_total = torch.zeros(n_output)
        assignments: Dict[str, int] = {}
        for idx in order:
            sid = sids[idx]
            out, _ = gate.simulate(train_vecs[sid], T=T, train=True, rng=torch.Generator().manual_seed(seed + ep * 1000 + idx))
            assignments[sid] = int(out.argmax())
            out_spike_total += out
        hist = np.bincount(list(assignments.values()), minlength=n_output)
        if len(assignments) == 0:
            entropy = 0.0
        else:
            entropy = float(-sum((p / len(assignments) * np.log(p / len(assignments) + 1e-12) for p in hist if p > 0)))
        active = int((hist > 0).sum())
        for i in range(n_output):
            if out_spike_total[i] < 1.0:
                gate.theta_o[i] *= 0.9
                gate.theta_o[i] = max(float(gate.theta_o[i]), 0.3)
        score = active * entropy
        history.append({'epoch': ep, 'active': active, 'entropy': entropy, 'score': score})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  ep {ep}: active={active} entropy={entropy:.3f} score={score:.3f} cohorts_hist={hist.tolist()}')
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in gate.state_dict().items()}
    if best_state:
        gate.load_state_dict(best_state)
    def predict_plurality(vecs: Dict[str, List[float]], n_draws: int=5):
        out = {}
        for sid, vec in vecs.items():
            votes = []
            for s in range(n_draws):
                rng = torch.Generator().manual_seed(seed * 17 + s * 991)
                osp, _ = gate.simulate(vec, T=T, train=False, rng=rng)
                votes.append(int(osp.argmax()))
            u, c = np.unique(votes, return_counts=True)
            out[sid] = int(u[np.argmax(c)])
        return out
    train_cohorts = predict_plurality(train_vecs)
    val_cohorts = predict_plurality(val_vecs)
    test_cohorts = predict_plurality(test_vecs)
    train_hist = np.bincount(list(train_cohorts.values()), minlength=n_output)
    val_hist = np.bincount(list(val_cohorts.values()), minlength=n_output)
    test_hist = np.bincount(list(test_cohorts.values()), minlength=n_output)
    result = {'n_train': len(train_vecs), 'n_val': len(val_vecs), 'n_test': len(test_vecs), 'n_input': 10, 'n_hidden': 32, 'n_output': n_output, 'epochs': epochs, 'T': T, 'seed': seed, 'theta0': theta0, 'w_norm_target': w_norm_target, 'update_mode': update_mode, 'active_cohorts_train': int((train_hist > 0).sum()), 'active_cohorts_val': int((val_hist > 0).sum()), 'active_cohorts_test': int((test_hist > 0).sum()), 'train_cohorts_hist': train_hist.tolist(), 'val_cohorts_hist': val_hist.tolist(), 'test_cohorts_hist': test_hist.tolist(), 'train_cohorts': train_cohorts, 'val_cohorts': val_cohorts, 'test_cohorts': test_cohorts, 'training_history': history, 'runtime_sec': time.time() - t0, 'zscore_params': zp, 'note': 'EV-UAV-specific SNN cohort gate trained on 99 train fingerprints (post-hot-pixel-filter, 10-dim portable schema). Replaces FRED-trained zero-shot SNN as the principled cross-dataset gate.'}
    print(f'\nActive cohorts seed={seed}: train={result['active_cohorts_train']}/{n_output}, val={result['active_cohorts_val']}/{n_output}, test={result['active_cohorts_test']}/{n_output}')
    print(f'Train hist: {result['train_cohorts_hist']}')
    print(f'Val   hist: {result['val_cohorts_hist']}')
    print(f'Test  hist: {result['test_cohorts_hist']}')
    ckpt_path = P3 / f'runs/snn_experiments/snn_ev_uav_full99_seed{seed}.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gate.state_dict(), ckpt_path)
    print(f'Saved checkpoint -> {ckpt_path}')
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-output', type=int, default=12)
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4', help='Comma-separated seed list (default 0,1,2,3,4 for N=5)')
    ap.add_argument('--theta0', type=float, default=0.5)
    ap.add_argument('--w-norm', type=float, default=5.0)
    ap.add_argument('--out-prefix', default=str(P3 / 'docs' / 'ev_uav_b15_snn_results'))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []
    for seed in seeds:
        print(f'\n========== seed {seed} ==========')
        res = train_and_eval(epochs=args.epochs, n_output=args.n_output, seed=seed, theta0=args.theta0, w_norm_target=args.w_norm)
        out_path = Path(f'{args.out_prefix}_seed{seed}.json')
        out_path.write_text(json.dumps(res, indent=2))
        print(f'Wrote {out_path}')
        all_results.append(res)
    if len(all_results) > 1:
        agg = {'n_seeds': len(all_results), 'active_cohorts_train': [r['active_cohorts_train'] for r in all_results], 'active_cohorts_val': [r['active_cohorts_val'] for r in all_results], 'active_cohorts_test': [r['active_cohorts_test'] for r in all_results], 'active_cohorts_train_mean': float(np.mean([r['active_cohorts_train'] for r in all_results])), 'active_cohorts_train_std': float(np.std([r['active_cohorts_train'] for r in all_results])), 'active_cohorts_test_mean': float(np.mean([r['active_cohorts_test'] for r in all_results])), 'active_cohorts_test_std': float(np.std([r['active_cohorts_test'] for r in all_results])), 'runtime_sec_total': float(sum((r['runtime_sec'] for r in all_results)))}
        agg_path = Path(f'{args.out_prefix}_aggregate.json')
        agg_path.write_text(json.dumps(agg, indent=2))
        print(f'\n=== N={len(all_results)} aggregate ===')
        print(f'  active_cohorts_train: {agg['active_cohorts_train_mean']:.1f} ± {agg['active_cohorts_train_std']:.2f}')
        print(f'  active_cohorts_test:  {agg['active_cohorts_test_mean']:.1f} ± {agg['active_cohorts_test_std']:.2f}')
        print(f'  total runtime: {agg['runtime_sec_total']:.1f}s')
        print(f'Wrote {agg_path}')
if __name__ == '__main__':
    main()
