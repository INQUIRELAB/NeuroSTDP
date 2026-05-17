from __future__ import annotations
import numpy as np

def _soft(x: float, thr: float, k: float) -> float:
    return float(1.0 / (1.0 + np.exp(-k * (x - thr))))

def score_fingerprint_per_candidate(cluster_coords: np.ndarray, cluster_feats: np.ndarray, bbox: tuple[float, float, float, float], density_min: float=0.3, tb_active_min: float=14.0, tb_entropy_min: float=3.5, polarity_balance_tol: float=0.2, n_time_bins: int=16) -> float:
    n = int(cluster_coords.shape[0]) if cluster_coords is not None else 0
    if n < 3:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = max((x2 - x1) * (y2 - y1), 1.0)
    density = n / area
    s_density = _soft(density, density_min, k=5.0)
    tb_col = cluster_coords[:, 0].astype(np.int64)
    tb_col = np.clip(tb_col, 0, n_time_bins - 1)
    tb_counts = np.bincount(tb_col, minlength=n_time_bins)
    tb_used = int((tb_counts > 0).sum())
    s_tb_active = _soft(float(tb_used), tb_active_min, k=2.0)
    total = float(tb_counts.sum())
    if total > 0:
        p = tb_counts.astype(np.float64) / total
        nz = p > 0
        tb_entropy = float(-np.sum(p[nz] * np.log2(p[nz])))
    else:
        tb_entropy = 0.0
    s_tb_entropy = _soft(tb_entropy, tb_entropy_min, k=2.0)
    scores = [s_density, s_tb_active, s_tb_entropy]
    if cluster_feats is not None and cluster_feats.ndim == 2 and (cluster_feats.shape[1] >= 2):
        on_mask = cluster_feats[:, 0::2].sum(axis=1) > 0
        off_mask = cluster_feats[:, 1::2].sum(axis=1) > 0
        n_on = int(on_mask.sum())
        n_off = int(off_mask.sum())
        n_pol = n_on + n_off
        if n_pol > 0:
            on_fraction = n_on / float(n_pol)
            imbalance = abs(on_fraction - 0.5)
            s_polarity = _soft(-imbalance, -polarity_balance_tol, k=25.0)
            scores.append(s_polarity)
    return float(np.mean(scores))
