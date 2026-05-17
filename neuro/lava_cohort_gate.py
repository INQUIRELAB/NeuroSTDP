#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
import sys
from pathlib import Path
import numpy as np
import torch
from _stable_hash import stable_hash
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSource
from lava.proc.io.sink import RingBuffer as SpikeSink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P3 = Path(os.environ.get('NEUROSTDP_ROOT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(P3 / 'code/neuro'))
from snn_cohort_gate import SpikingCohortGate, load_fingerprints, to_rate_tensor, assign_cohorts
W_INH_HIDDEN = 0.2
W_INH_OUTPUT = 0.0
DU_MATCHING = 1.0
DV_MATCHING = 1.0 - 0.9

def build_poisson_input(vec: np.ndarray, T: int, torch_seed: int) -> np.ndarray:
    rates = to_rate_tensor(torch.as_tensor(vec, dtype=torch.float32))
    g = torch.Generator().manual_seed(int(torch_seed))
    out = np.empty((len(rates), T), dtype=np.float32)
    for t in range(T):
        out[:, t] = (torch.rand(len(rates), generator=g) < rates).float().numpy()
    return out

def run_lava_sample(W_ih: np.ndarray, W_ho: np.ndarray, vth_out: np.ndarray, input_spikes: np.ndarray, T: int, capture_hidden: bool):
    n_in, n_h = W_ih.shape
    n_o = W_ho.shape[1]
    src = SpikeSource(data=input_spikes)
    dense_ih = Dense(weights=W_ih.T.astype(np.float32))
    hidden = LIF(shape=(n_h,), du=DU_MATCHING, dv=DV_MATCHING, vth=1.0, bias_mant=0.0)
    wta = None
    if W_INH_HIDDEN > 0:
        wta = Dense(weights=(-W_INH_HIDDEN * (np.ones((n_h, n_h)) - np.eye(n_h))).astype(np.float32))
    dense_ho = Dense(weights=W_ho.T.astype(np.float32))
    output = LIF(shape=(n_o,), du=DU_MATCHING, dv=DV_MATCHING, vth=vth_out.astype(np.float32), bias_mant=0.0)
    sink_o = SpikeSink(shape=(n_o,), buffer=T)
    sink_h = SpikeSink(shape=(n_h,), buffer=T) if capture_hidden else None
    src.s_out.connect(dense_ih.s_in)
    dense_ih.a_out.connect(hidden.a_in)
    if wta is not None:
        hidden.s_out.connect(wta.s_in)
        wta.a_out.connect(hidden.a_in)
    hidden.s_out.connect(dense_ho.s_in)
    dense_ho.a_out.connect(output.a_in)
    output.s_out.connect(sink_o.a_in)
    if capture_hidden:
        hidden.s_out.connect(sink_h.a_in)
    run_cfg = Loihi2SimCfg(select_tag='floating_pt')
    src.run(condition=RunSteps(num_steps=T), run_cfg=run_cfg)
    out_spikes = sink_o.data.get()
    hid_spikes = sink_h.data.get() if capture_hidden else None
    src.stop()
    return (out_spikes, hid_spikes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-test', type=int, default=5, help='Number of test-split sequences to evaluate (Lava CPU sim is ~1-2s per T=350 step run; 5 seqs * 5 draws = ~30s).')
    ap.add_argument('--T', type=int, default=350)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-draws', type=int, default=5, help="Plurality-vote draws per seq. PyTorch reference uses 5; higher values reduce Poisson noise but don't change the fundamental agreement ceiling (~65% PT self-consistency).")
    ap.add_argument('--ckpt', default=str(P3 / 'runs/snn_experiments/snn_label_free.pt'))
    ap.add_argument('--variant', choices=['label_free', 'with_circular'], default='label_free')
    ap.add_argument('--out-json', default=str(P3 / 'docs/lava_port_results.json'))
    ap.add_argument('--fig-out', default=str(P3 / 'paper/figures/spike_raster_lava'))
    args = ap.parse_args()
    print(f'[lava] loading weights from {args.ckpt}')
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    W_ih = sd['W_ih'].numpy().astype(np.float32)
    W_ho = sd['W_ho'].numpy().astype(np.float32)
    theta_o = sd['theta_o'].numpy().astype(np.float32)
    cfg = sd['config']
    print(f'  n_in={cfg['n_in']} n_h={cfg['n_h']} n_o={cfg['n_o']} alpha={cfg['alpha']} theta_o=[{theta_o.min():.3f},{theta_o.max():.3f}]')
    print(f'[lava] loading fingerprints (variant={args.variant})')
    train_vecs, test_vecs, keys = load_fingerprints(include_circular=args.variant == 'with_circular')
    test_sids = sorted(test_vecs.keys(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))[:args.n_test]
    print(f'  evaluating on {len(test_sids)} test seqs: {test_sids}')
    pt_gate = SpikingCohortGate(n_in=cfg['n_in'], n_hidden=cfg['n_h'], n_output=cfg['n_o'], alpha=cfg['alpha'], alpha_trace=cfg['alpha_trace'], theta0=cfg['theta0'], dtheta=cfg['dtheta'], theta_decay=cfg['theta_decay'], eta_plus=cfg['eta_plus'], eta_minus=cfg['eta_minus'], w_max=cfg['w_max'], seed=args.seed, update_mode=cfg.get('update_mode', 'online'))
    pt_gate.load_state_dict(sd)
    subset_vecs = {sid: test_vecs[sid] for sid in test_sids}
    pt_assign = assign_cohorts(pt_gate, subset_vecs, T=args.T, seed=args.seed)
    per_sample = []
    agreement = 0
    top2_shared = 0
    hidden_raster = None
    raster_sid = None
    n_draws = args.n_draws
    for i, sid in enumerate(test_sids):
        vec = np.asarray(subset_vecs[sid], dtype=np.float32)
        lava_votes = []
        lava_out_counts_agg = np.zeros(cfg['n_o'])
        for d in range(n_draws):
            draw_seed = args.seed + stable_hash(sid, 1000) + d
            input_spikes = build_poisson_input(vec, args.T, draw_seed)
            capture = i == 0 and d == 0
            out_spikes, hid_spikes = run_lava_sample(W_ih=W_ih, W_ho=W_ho, vth_out=theta_o, input_spikes=input_spikes, T=args.T, capture_hidden=capture)
            out_counts = out_spikes.sum(axis=1)
            lava_out_counts_agg += out_counts
            lava_votes.append(int(np.argmax(out_counts)) if out_counts.sum() > 0 else -1)
            if capture:
                hidden_raster = hid_spikes
                raster_sid = sid
                print(f'  [raster] captured hidden spikes for seq {sid}: {int(hid_spikes.sum())} total spikes')
        u, c = np.unique(lava_votes, return_counts=True)
        lava_cohort = int(u[c.argmax()])
        sorted_idx = np.argsort(-c)
        lava_top2 = set((int(u[k]) for k in sorted_idx[:2]))
        pt_cohort = pt_assign[sid]['cohort']
        pt_vote_items = sorted(pt_assign[sid]['votes'].items(), key=lambda kv: -int(kv[1]))
        pt_top2 = set((int(k) for k, _ in pt_vote_items[:2]))
        agree = int(lava_cohort == pt_cohort)
        agreement += agree
        top2_share = int(len(lava_top2 & pt_top2) >= 1)
        top2_shared += top2_share
        per_sample.append({'sid': sid, 'lava_cohort': lava_cohort, 'pt_cohort': pt_cohort, 'lava_top2': sorted(lava_top2), 'pt_top2': sorted(pt_top2), 'agree': bool(agree), 'top2_shared': bool(top2_share), 'lava_votes': [int(v) for v in lava_votes], 'lava_out_counts_mean': (lava_out_counts_agg / n_draws).tolist(), 'pt_votes': pt_assign[sid]['votes']})
        print(f'  seq {sid}: lava={lava_cohort}  pt={pt_cohort}  {('AGREE' if agree else 'MISMATCH')}  top2_share={('YES' if top2_share else 'no')}  (votes lava={lava_votes}  pt={pt_assign[sid]['votes']})')
    agreement_rate = agreement / len(test_sids) if test_sids else 0.0
    top2_rate = top2_shared / len(test_sids) if test_sids else 0.0
    print(f'\n[lava] argmax agreement:        {agreement}/{len(test_sids)} = {agreement_rate:.1%}')
    print(f'[lava] top-2 overlap (>=1 shared): {top2_shared}/{len(test_sids)} = {top2_rate:.1%}')
    fig_paths = []
    if hidden_raster is not None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        neurons, times = np.where(hidden_raster > 0)
        ax.scatter(times, neurons, s=4, c='k', marker='|')
        ax.set_xlim(0, args.T)
        ax.set_ylim(-0.5, hidden_raster.shape[0] - 0.5)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Hidden neuron')
        ax.set_title(f'Lava hidden-layer raster (seq {raster_sid}, T={args.T})  [{int(hidden_raster.sum())} spikes, WTA-gated]')
        ax.grid(alpha=0.2, linestyle=':')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            out_path = f'{args.fig_out}.{ext}'
            fig.savefig(out_path, dpi=150)
            fig_paths.append(out_path)
        plt.close(fig)
        print(f'[lava] saved raster: {fig_paths}')
    total_out_spikes = sum((sum(s['lava_out_counts_mean']) for s in per_sample))
    total_hid_spikes = float(hidden_raster.sum()) if hidden_raster is not None else None
    energy_proxy = {'note': 'Lava floating-point CPU sim; no silicon energy counter. Spike counts here are a proxy; Davies 2018 reports ~23 pJ/synaptic op on Loihi 1. Net is 384 plastic synapses (+ ~272 effective input) so per-sample energy bound ~ (total_spikes * fan_out * 23 pJ) << 1 uJ.', 'davies2018_pJ_per_synaptic_op': 23.0, 'total_out_spikes_subset': float(total_out_spikes), 'total_hidden_spikes_one_sample': total_hid_spikes, 'n_samples': len(test_sids)}
    out = {'lava_version': '0.10.0', 'python_version': sys.version.split()[0], 'numpy_version': np.__version__, 'checkpoint': args.ckpt, 'variant': args.variant, 'T': args.T, 'seed': args.seed, 'n_draws': n_draws, 'W_INH_HIDDEN': W_INH_HIDDEN, 'n_test': len(test_sids), 'agreement': agreement, 'agreement_rate': agreement_rate, 'top2_shared': top2_shared, 'top2_rate': top2_rate, 'per_sample': per_sample, 'energy_proxy': energy_proxy, 'figure_paths': fig_paths, 'notes': ['Inference only; STDP rules not wired up (trained weights imported from PyTorch).', 'Homeostatic threshold theta_o frozen per-neuron in LIF vth.', f'Hidden WTA: soft lateral inhibition with W={W_INH_HIDDEN:.2f} off-diagonal (Lava has 1-step feedback delay so hard WTA not natively possible).', 'du=1.0, dv=0.1 to match PyTorch V := alpha*V + I (alpha=0.9); hidden-layer per-neuron spike counts match PT EXACTLY without WTA (L1 diff=0) — see port report.', 'Poisson input spikes pre-generated using torch.Generator to match PT reference bit-exactly (torch.rand < rate semantics).', "Built with Lava 0.10.0, floating-point CPU ProcessModel (Loihi2SimCfg tag='floating_pt').", "Cohort argmax agreement metric depends on Poisson-noise floor: PT's own plurality-5 agreement with itself across different seed schemes is ~44-66%, so argmax agreement has an inherent ceiling well below 100%. Top-2-overlap is the informative semantic metric."]}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f'[lava] wrote {args.out_json}')
if __name__ == '__main__':
    main()
