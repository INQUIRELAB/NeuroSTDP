#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _stable_hash import stable_hash
P3 = Path(os.environ.get("NEUROSTDP_ROOT", str(Path(__file__).resolve().parents[2])))
LABEL_FREE_KEYS = [
    "ev_med", "ev_p90", "density_med",
    "cx_std", "cy_med", "cy_std", "cy_p10", "cy_p90",
    "area_med", "area_p10", "area_p90",
    "agree_mean", "agree_p10",
    "motion_range", "density_variance",
    "multi_drone_unsup", "stationary_fraction",
]
CIRCULAR_KEYS = ["iou_v127d_v47", "area_ratio_v127d_v47", "drop_fraction"]

def load_fingerprints(include_circular: bool):
    fp = json.loads((P3 / "docs/fingerprints_fred.json").read_text())
    zp = json.loads((P3 / "docs/fingerprint_zscore_params.json").read_text())
    keys = LABEL_FREE_KEYS + (CIRCULAR_KEYS if include_circular else [])
    def vectorise(entry):
        vec = []
        for k in keys:
            v = entry["features"].get(k, entry["features_circular"].get(k))
            if k == "density_variance":
                v = float(np.log1p(v))
                v = (v - np.log1p(zp[k]["mean"])) / 2.0
            else:
                v = (v - zp[k]["mean"]) / zp[k]["std"]
            vec.append(float(np.clip(v, -5.0, 5.0)))
        return vec
    train = {sid: vectorise(e) for sid, e in fp["canonical_train"].items()}
    test = {sid: vectorise(e) for sid, e in fp["canonical_test"].items()}
    return train, test, keys

def to_rate_tensor(vec, max_rate_per_step=0.08):
    x = torch.as_tensor(vec, dtype=torch.float32)
    return torch.sigmoid(x / 2.0) * max_rate_per_step

class SpikingCohortGate:
    def __init__(self, n_in=17, n_hidden=32, n_output=12,
                 alpha=0.9, alpha_trace=0.95,
                 theta0=1.0, dtheta=0.01, theta_decay=0.999,
                 eta_plus=0.002, eta_minus=0.002, w_max=0.8,
                 w_norm_target=3.0, seed=0, update_mode="online",
                 trace_order_fix=False, plastic_input=False,
                 disable_renorm=False):
        g = torch.Generator().manual_seed(seed)
        self.n_in = n_in; self.n_h = n_hidden; self.n_o = n_output
        self.alpha = alpha
        self.alpha_trace = alpha_trace
        self.theta0 = theta0
        self.dtheta = dtheta
        self.theta_decay = theta_decay
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.w_max = w_max
        self.w_norm_target = w_norm_target
        assert update_mode in ("online", "batched"), update_mode
        self.update_mode = update_mode
        self.trace_order_fix = trace_order_fix
        self.plastic_input = plastic_input
        self.disable_renorm = disable_renorm
        self.carry_state = False
        self.output_wta = False
        self.k_wta_hidden = 1
        self.reward_modulated = False
        self._reward_scalar = 1.0
        self.kmeans_alpha = 0.0
        self._V_h_persist = None
        self._V_o_persist = None
        self._pre_trace_persist = None
        self._post_trace_persist = None
        self._in_trace_persist = None
        W_ih = torch.rand(n_in, n_hidden, generator=g) * 0.4
        mask = torch.rand(n_in, n_hidden, generator=g) > 0.5
        self._W_ih_mask = mask.float()
        W_ih = W_ih * self._W_ih_mask
        self.W_ih = W_ih
        self._w_ih_norm_target = 1.5
        self.W_ho = torch.rand(n_hidden, n_output, generator=g) * 0.3
        self._renormalize()
        self.theta_o = torch.ones(n_output) * theta0
    def _renormalize(self):
        col_sum = self.W_ho.sum(dim=0, keepdim=True).clamp_min(1e-6)
        self.W_ho = self.W_ho * (self.w_norm_target / col_sum)
        self.W_ho = torch.clamp(self.W_ho, 0.0, self.w_max)
    def simulate(self, vec, T=350, train=True, rng=None, reward=None):
        if reward is not None:
            self._reward_scalar = float(reward)
        else:
            self._reward_scalar = 1.0
        if rng is None:
            rng = torch.Generator().manual_seed(int(time.time() * 1e6) % 2**32)
        rates = to_rate_tensor(vec)
        gain = self._reward_scalar if self.reward_modulated else 1.0
        eta_plus_eff = self.eta_plus * gain
        eta_minus_eff = self.eta_minus * gain
        if self.carry_state and self._V_h_persist is not None:
            V_h = self._V_h_persist.clone()
            V_o = self._V_o_persist.clone()
            in_trace = self._in_trace_persist.clone()
            pre_trace = self._pre_trace_persist.clone()
            post_trace = self._post_trace_persist.clone()
        else:
            V_h = torch.zeros(self.n_h)
            V_o = torch.zeros(self.n_o)
            in_trace = torch.zeros(self.n_in)
            pre_trace = torch.zeros(self.n_h)
            post_trace = torch.zeros(self.n_o)
        out_spikes = torch.zeros(self.n_o)
        hidden_spikes = torch.zeros(self.n_h)
        dW = torch.zeros_like(self.W_ho) if (train and self.update_mode == "batched") else None
        for _ in range(T):
            S_in = (torch.rand(self.n_in, generator=rng) < rates).float()
            I_h = S_in @ self.W_ih
            V_h = self.alpha * V_h + I_h
            s_h = (V_h > 1.0).float()
            if s_h.sum() > self.k_wta_hidden:
                topk = torch.topk(V_h, self.k_wta_hidden).indices
                s_h_wta = torch.zeros_like(s_h)
                s_h_wta[topk] = 1.0
                s_h = s_h_wta
            V_h = V_h * (1.0 - s_h)
            hidden_spikes += s_h
            I_o = s_h @ self.W_ho
            V_o = self.alpha * V_o + I_o
            s_o = (V_o > self.theta_o).float()
            if self.output_wta and s_o.sum() > 1:
                winner_o = V_o.argmax()
                s_o_wta = torch.zeros_like(s_o)
                s_o_wta[winner_o] = 1.0
                s_o = s_o_wta
            V_o = V_o * (1.0 - s_o)
            out_spikes += s_o
            if train and self.trace_order_fix:
                if self.update_mode == "online":
                    if s_o.sum() > 0:
                        self.W_ho = torch.clamp(
                            self.W_ho + eta_plus_eff * torch.outer(pre_trace, s_o),
                            0.0, self.w_max,
                        )
                    if s_h.sum() > 0:
                        self.W_ho = torch.clamp(
                            self.W_ho - eta_minus_eff * torch.outer(s_h, post_trace),
                            0.0, self.w_max,
                        )
                    if self.plastic_input and s_h.sum() > 0:
                        self.W_ih = torch.clamp(
                            self.W_ih + eta_plus_eff * torch.outer(in_trace, s_h),
                            0.0, self.w_max,
                        ) * self._W_ih_mask
                    if self.plastic_input and S_in.sum() > 0:
                        self.W_ih = torch.clamp(
                            self.W_ih - eta_minus_eff * torch.outer(S_in, pre_trace),
                            0.0, self.w_max,
                        ) * self._W_ih_mask
            in_trace = self.alpha_trace * in_trace + S_in
            pre_trace = self.alpha_trace * pre_trace + s_h
            post_trace = self.alpha_trace * post_trace + s_o
            if train and not self.trace_order_fix:
                if self.update_mode == "online":
                    if s_o.sum() > 0:
                        self.W_ho = torch.clamp(
                            self.W_ho + eta_plus_eff * torch.outer(pre_trace, s_o),
                            0.0, self.w_max,
                        )
                    if s_h.sum() > 0:
                        self.W_ho = torch.clamp(
                            self.W_ho - eta_minus_eff * torch.outer(s_h, post_trace),
                            0.0, self.w_max,
                        )
                    if self.plastic_input and s_h.sum() > 0:
                        self.W_ih = torch.clamp(
                            self.W_ih + eta_plus_eff * torch.outer(in_trace, s_h),
                            0.0, self.w_max,
                        ) * self._W_ih_mask
                    if self.plastic_input and S_in.sum() > 0:
                        self.W_ih = torch.clamp(
                            self.W_ih - eta_minus_eff * torch.outer(S_in, pre_trace),
                            0.0, self.w_max,
                        ) * self._W_ih_mask
                else:
                    if s_o.sum() > 0:
                        dW += eta_plus_eff * torch.outer(pre_trace, s_o)
                    if s_h.sum() > 0:
                        dW -= eta_minus_eff * torch.outer(s_h, post_trace)
            if train:
                self.theta_o = self.theta_o + self.dtheta * s_o
                self.theta_o = (self.theta_o * self.theta_decay
                                + self.theta0 * (1.0 - self.theta_decay))
        if train:
            if self.update_mode == "batched":
                self.W_ho = torch.clamp(self.W_ho + dW, 0.0, self.w_max)
            if not self.disable_renorm:
                self._renormalize()
        if self.carry_state:
            self._V_h_persist = V_h.clone()
            self._V_o_persist = V_o.clone()
            self._in_trace_persist = in_trace.clone()
            self._pre_trace_persist = pre_trace.clone()
            self._post_trace_persist = post_trace.clone()
        return out_spikes, hidden_spikes
    def apply_kmeans_update(self, hidden_spikes, out_spikes):
        if self.kmeans_alpha <= 0:
            return
        c = int(out_spikes.argmax())
        h = hidden_spikes / (hidden_spikes.sum().clamp_min(1e-9)) * self.w_norm_target
        self.W_ho[:, c] = self.W_ho[:, c] + self.kmeans_alpha * (h - self.W_ho[:, c])
        self.W_ho = torch.clamp(self.W_ho, 0.0, self.w_max)
        if not self.disable_renorm:
            self._renormalize()
    def predict(self, vec, T=350, seed=0):
        rng = torch.Generator().manual_seed(seed)
        out, _ = self.simulate(vec, T=T, train=False, rng=rng)
        return int(out.argmax()), out.numpy().tolist()
    def state_dict(self):
        return {
            "W_ih": self.W_ih,
            "W_ho": self.W_ho,
            "theta_o": self.theta_o,
            "config": {
                "n_in": self.n_in, "n_h": self.n_h, "n_o": self.n_o,
                "alpha": self.alpha, "alpha_trace": self.alpha_trace,
                "theta0": self.theta0, "dtheta": self.dtheta,
                "theta_decay": self.theta_decay,
                "eta_plus": self.eta_plus, "eta_minus": self.eta_minus,
                "w_max": self.w_max, "update_mode": self.update_mode,
            },
        }
    def load_state_dict(self, sd):
        self.W_ih = sd["W_ih"]
        self.W_ho = sd["W_ho"]
        self.theta_o = sd["theta_o"]
        for k, v in sd["config"].items():
            setattr(self, k, v)

def train_gate(train_vecs, epochs=30, T=350, n_output=12, seed=0,
               revive_dead=True, log_path=None, update_mode="online",
               n_hidden=32, n_presentations_per_stim=1, k_wta_hidden=1,
               reward_fn=None, reward_modulated=False):
    gate = SpikingCohortGate(
        n_in=len(next(iter(train_vecs.values()))),
        n_hidden=n_hidden, n_output=n_output, seed=seed, update_mode=update_mode,
    )
    gate.k_wta_hidden = k_wta_hidden
    gate.reward_modulated = bool(reward_modulated)
    def _sort_key(s):
        try:
            return (0, int(s))
        except (ValueError, TypeError):
            return (1, s)
    sids = sorted(train_vecs.keys(), key=_sort_key)
    rng_master = torch.Generator().manual_seed(seed)
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = -1
    for ep in range(epochs):
        order = torch.randperm(len(sids), generator=rng_master).tolist()
        out_spike_total = torch.zeros(n_output)
        assignments = {}
        for idx in order:
            sid = sids[idx]
            last_out = None
            stim_reward = reward_fn(sid) if (reward_fn is not None) else None
            for _pres in range(n_presentations_per_stim):
                out, _ = gate.simulate(
                    train_vecs[sid], T=T, train=True,
                    rng=torch.Generator().manual_seed(seed + ep * 1000 + idx + _pres * 7919),
                    reward=stim_reward,
                )
                last_out = out
            assignments[sid] = int(last_out.argmax())
            out_spike_total += last_out
        hist = np.bincount(list(assignments.values()), minlength=n_output)
        entropy = float(-sum((p / len(assignments)) * np.log(p / len(assignments) + 1e-12)
                             for p in hist if p > 0))
        active = int((hist > 0).sum())
        if revive_dead:
            for i in range(n_output):
                if out_spike_total[i] < 1.0:
                    gate.theta_o[i] *= 0.9
                    gate.theta_o[i] = max(float(gate.theta_o[i]), 0.3)
        score = active * entropy
        is_best = score > best_score
        if is_best:
            best_score = score
            best_state = {k: v.clone() if torch.is_tensor(v) else v
                          for k, v in gate.state_dict().items()
                          if k != "config"}
            best_state["config"] = dict(gate.state_dict()["config"])
            best_epoch = ep
        history.append({
            "epoch": ep,
            "assignment_hist": hist.tolist(),
            "entropy": entropy,
            "active_cohorts": active,
            "score": float(score),
            "is_best_so_far": bool(is_best),
            "theta_o_mean": float(gate.theta_o.mean()),
            "theta_o_min": float(gate.theta_o.min()),
            "theta_o_max": float(gate.theta_o.max()),
            "W_ho_mean": float(gate.W_ho.mean()),
            "W_ho_sparsity_below_0_1": float((gate.W_ho < 0.1).float().mean()),
        })
        print(f"ep {ep:02d} | active={active:2d}/{n_output} "
              f"H={entropy:.2f} score={score:.2f}"
              f"{' *' if is_best else '  '} | "
              f"θ∈[{history[-1]['theta_o_min']:.2f},{history[-1]['theta_o_max']:.2f}] "
              f"| W̄={history[-1]['W_ho_mean']:.3f} | hist={hist.tolist()}")
    if best_state is not None:
        gate.load_state_dict(best_state)
        print(f"→ restored best epoch {best_epoch} (score={best_score:.2f})")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "history": history,
            "best_epoch": best_epoch,
            "best_score": float(best_score),
        }, indent=2))
    return gate, history

def assign_cohorts(gate, vecs, T=350, n_eval_draws=5, seed=0):
    out = {}
    for sid, v in vecs.items():
        votes = []
        for d in range(n_eval_draws):
            idx, _ = gate.predict(v, T=T, seed=seed + stable_hash(sid, 1000) + d)
            votes.append(idx)
        u, c = np.unique(votes, return_counts=True)
        out[sid] = {"cohort": int(u[c.argmax()]),
                    "votes": {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["label_free", "with_circular"], default="label_free")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--T", type=int, default=350)
    ap.add_argument("--n-output", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--update-mode", choices=["online", "batched"], default="online",
                    help="online = event-time STDP (default); batched = legacy accumulated dW")
    ap.add_argument("--save-dir", default=str(P3 / "runs/snn_experiments"))
    args = ap.parse_args()
    train_vecs, test_vecs, keys = load_fingerprints(include_circular=(args.variant == "with_circular"))
    print(f"Loaded {len(train_vecs)} train + {len(test_vecs)} test "
          f"fingerprints ({len(keys)} features, variant={args.variant})")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = P3 / f"docs/snn_training_log_{args.variant}.json"
    gate, history = train_gate(
        train_vecs, epochs=args.epochs, T=args.T, n_output=args.n_output,
        seed=args.seed, log_path=log_path, update_mode=args.update_mode,
    )
    ckpt = save_dir / f"snn_{args.variant}.pt"
    torch.save(gate.state_dict(), ckpt)
    print(f"Saved {ckpt}")
    train_assign = assign_cohorts(gate, train_vecs, T=args.T, seed=args.seed)
    test_assign = assign_cohorts(gate, test_vecs, T=args.T, seed=args.seed)
    assign_path = P3 / f"docs/snn_cohort_assignments_{args.variant}.json"
    assign_path.write_text(json.dumps({
        "variant": args.variant, "epochs": args.epochs, "T": args.T,
        "n_output": args.n_output, "feature_keys": keys,
        "canonical_train": train_assign,
        "canonical_test": test_assign,
    }, indent=2))
    print(f"Saved {assign_path}")
    tr_hist = np.bincount([v["cohort"] for v in train_assign.values()],
                          minlength=args.n_output)
    te_hist = np.bincount([v["cohort"] for v in test_assign.values()],
                          minlength=args.n_output)
    print(f"Train cohort histogram: {tr_hist.tolist()}")
    print(f"Test  cohort histogram: {te_hist.tolist()}")
    print(f"Active cohorts: train={int((tr_hist>0).sum())}  "
          f"test={int((te_hist>0).sum())}")
if __name__ == "__main__":
    main()
