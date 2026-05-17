from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon
PROJECT = Path(os.environ.get('NEUROSTDP_ROOT', str(Path(__file__).resolve().parents[2])))
P3 = PROJECT
sys.path.insert(0, str(P3 / 'code/scripts'))
sys.path.insert(0, str(P3 / 'code/neuro'))
import eval_v2_multipred_multidrone as eval_mod
METHODS = {'snn_label_efficient_L6': 'discovery_v128_M1_label_efficient_test', 'kmeans': 'discovery_v128_neuro_tta_kmeans_test', 'gmm': 'discovery_v128_neuro_tta_gmm_test', 'mlp': 'discovery_v128_neuro_tta_mlp_test', 'lame': 'discovery_v128_neuro_tta_lame_test', 'shot_im': 'discovery_v128_neuro_tta_shot_im_test', 'union_6ch': 'discovery_v127d_plus_6ch_test'}
DELTA_EQ = 0.5
ALPHA = 0.05
BASELINES = ['kmeans', 'gmm', 'mlp', 'lame', 'shot_im', 'union_6ch']
PRETTY = {'kmeans': '$k$-means', 'gmm': 'GMM', 'mlp': 'MLP', 'lame': 'LAME', 'shot_im': 'SHOT-IM', 'union_6ch': 'Union 6-ch'}

def configure_canonical_test():
    eval_mod.DATA = eval_mod._DATA_ROOT / 'canonical_test'
    eval_mod.RAW = eval_mod._RAW_ROOT / 'test'
    eval_mod.GT_PROC = eval_mod._GT_PROC_ROOT / 'canonical_test'
    eval_mod._ACTIVE_MULTI_SEQS = eval_mod.MULTI_SEQS_TEST

def run_eval_for_dir(pred_dir_name, sids):
    tasks = [(sid, pred_dir_name) for sid in sids]
    with Pool(16) as pool:
        res = pool.map(eval_mod.eval_seq, tasks)
    out = {}
    for sid, r in res:
        if r is None:
            continue
        out[sid] = float(r['mAP_30'])
    return out

def bootstrap_median_ci(deltas, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medians[i] = np.median(deltas[idx])
    return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))

def wilcoxon_paired(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    deltas = x - y
    nz = deltas[np.abs(deltas) > 1e-12]
    n_nonzero = int(len(nz))
    if n_nonzero < 2:
        return {'n_nonzero': n_nonzero, 'W': None, 'p_two_sided': None, 'Z': None, 'r': None}
    res = wilcoxon(x, y, zero_method='wilcox', method='approx', alternative='two-sided')
    Z = res.zstatistic if hasattr(res, 'zstatistic') else None
    if Z is None:
        try:
            Z = float(res.statistic)
        except Exception:
            Z = None
    r = abs(Z) / np.sqrt(n_nonzero) if Z is not None else None
    return {'n_nonzero': n_nonzero, 'W': float(res.statistic), 'p_two_sided': float(res.pvalue), 'Z': float(Z) if Z is not None else None, 'r': float(r) if r is not None else None}

def tost_one_sided(deltas, margin, alternative):
    shifted = np.asarray(deltas, dtype=float) - margin
    nz = shifted[np.abs(shifted) > 1e-12]
    if len(nz) < 2:
        return None
    res = wilcoxon(shifted, zero_method='wilcox', method='approx', alternative=alternative)
    return float(res.pvalue)

def tost_equivalence(deltas, delta_eq=DELTA_EQ):
    p_lower = tost_one_sided(deltas, -delta_eq, 'greater')
    p_upper = tost_one_sided(deltas, +delta_eq, 'less')
    if p_lower is None or p_upper is None:
        return {'p_lower': p_lower, 'p_upper': p_upper, 'p_tost': None, 'rejects': False}
    p_tost = max(p_lower, p_upper)
    return {'p_lower': p_lower, 'p_upper': p_upper, 'p_tost': p_tost, 'rejects': p_tost < ALPHA}

def main():
    configure_canonical_test()
    sids = sorted([d.name for d in eval_mod.DATA.iterdir() if d.is_dir()], key=lambda x: int(x))
    print(f'canonical_test: {len(sids)} seqs')
    per_method = {}
    for name, dir_name in METHODS.items():
        print(f'  evaluating {name} ({dir_name})...')
        per_method[name] = run_eval_for_dir(dir_name, sids)
        print(f'    got {len(per_method[name])} per-seq mAP@30 values')
    snn = 'snn_label_efficient_L6'
    common_sids = [s for s in sids if s in per_method[snn] and all((s in per_method[b] for b in BASELINES))]
    print(f'\ncommon sids: {len(common_sids)}')
    results = {}
    deltas_all = {}
    for b in BASELINES:
        x = np.array([per_method[snn][s] * 100 for s in common_sids])
        y = np.array([per_method[b][s] * 100 for s in common_sids])
        deltas = x - y
        deltas_all[b] = deltas.tolist()
        ci_lo, ci_hi = bootstrap_median_ci(deltas)
        wlx = wilcoxon_paired(x, y)
        tost = tost_equivalence(deltas)
        results[b] = {'n_compared': len(common_sids), 'n_diff': int(np.sum(np.abs(deltas) > 1e-12)), 'median_delta': float(np.median(deltas)), 'mean_delta': float(np.mean(deltas)), 'ci95_median': [ci_lo, ci_hi], 'wilcoxon': wlx, 'tost': tost}
    panel_A = [b for b in BASELINES if b != 'union_6ch']
    sorted_pa = sorted(panel_A, key=lambda b: results[b]['wilcoxon']['p_two_sided'])
    m_A = len(panel_A)
    for i, b in enumerate(sorted_pa):
        p = results[b]['wilcoxon']['p_two_sided']
        results[b]['wilcoxon']['p_holm_PanelA'] = min(1.0, p * (m_A - i))
    p_union = results['union_6ch']['wilcoxon']['p_two_sided']
    results['union_6ch']['wilcoxon']['p_holm_PanelB'] = p_union
    summary = {'lever_pool': '|L|=6 (M1 expanded)', 'snn_dir': METHODS[snn], 'n_canonical_test': len(sids), 'n_compared': len(common_sids), 'delta_eq_pp': DELTA_EQ, 'alpha': ALPHA, 'results': results}
    out_json = P3 / 'docs/wilcoxon_tost_L6_results.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {out_json}')
    print('\n=== Wilcoxon + TOST on |L|=6 ===')
    print(f'{'baseline':<12} {'n_diff':>7} {'med_Δ':>7} {'p_holm':>10} {'p_TOST':>10} {'TOST eq':>9}')
    for b in BASELINES:
        r = results[b]
        med = r['median_delta']
        nd = r['n_diff']
        if b == 'union_6ch':
            ph = r['wilcoxon']['p_holm_PanelB']
        else:
            ph = r['wilcoxon']['p_holm_PanelA']
        pt = r['tost']['p_tost']
        eq = '✓' if r['tost']['rejects'] else '✗'
        print(f'{b:<12} {nd:>7d} {med:>+7.2f} {ph:>10.4f} {pt:>10.4f} {eq:>9}')
if __name__ == '__main__':
    main()
