from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)
SENSOR_H, SENSOR_W = 720, 1280
TARGET_H, TARGET_W = 640, 640
SCALE_X = TARGET_W / SENSOR_W
SCALE_Y = TARGET_H / SENSOR_H
FRAME_DURATION_US = 33_333
DEFAULT_WINDOW_US = 250_000
DEFAULT_BIN_US = 1_000
DEFAULT_MIN_EVENTS = 30
ROTOR_F_LO = 80.0
ROTOR_F_HI = 600.0
FUND_F_LO = 80.0
FUND_F_HI = 300.0
MAINS_HZ = sorted(
    [50 * k for k in range(1, 13)] + [60 * k for k in range(1, 11)]
)
MAINS_TOL_HZ = 3.0

def load_events_raw(hdf5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import h5py
    with h5py.File(str(hdf5_path), "r") as f:
        ev = f["CD"]["events"]
        t = np.asarray(ev["t"][:], dtype=np.int64)
        x = np.asarray(ev["x"][:], dtype=np.int32)
        y = np.asarray(ev["y"][:], dtype=np.int32)
        p = np.asarray(ev["p"][:], dtype=np.int8)
    return t, x, y, p

def rescale_xy_640(x_raw: np.ndarray, y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.clip((x_raw.astype(np.float32) * SCALE_X).astype(np.int32), 0, TARGET_W - 1)
    y = np.clip((y_raw.astype(np.float32) * SCALE_Y).astype(np.int32), 0, TARGET_H - 1)
    return x, y

@dataclass

class PixelSpec:
    yx: Tuple[int, int]
    peak_hz: float
    peak_power: float
    peak_psr: float
    harm_hz: float
    harm_power: float
    harm_psr: float
    n_events: int

def _periodogram(
    t_rel_us: np.ndarray, window_us: int, bin_us: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = window_us // bin_us
    ts = np.zeros(n_bins, dtype=np.float32)
    idx = (t_rel_us // bin_us).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(ts, idx, 1.0)
    ts -= ts.mean()
    hann = np.hanning(n_bins).astype(np.float32)
    ts = ts * hann
    spec = np.abs(np.fft.rfft(ts))
    fs = 1e6 / bin_us
    freqs = np.fft.rfftfreq(n_bins, bin_us * 1e-6)
    return freqs, spec

def _near_mains(f: float) -> bool:
    return any(abs(f - m) < MAINS_TOL_HZ for m in MAINS_HZ)

def _sidelobe_median(spec: np.ndarray, freqs: np.ndarray, f0: float, half_bw: float = 20.0) -> float:
    mask = (freqs >= f0 - half_bw) & (freqs <= f0 + half_bw)
    if mask.sum() < 3:
        return float(np.median(spec)) if spec.size else 1e-9
    peak_idx = int(np.argmin(np.abs(freqs - f0)))
    mm = mask.copy()
    mm[peak_idx] = False
    if not mm.any():
        return float(np.median(spec[mask]))
    return float(np.median(spec[mm]))

def _psr_db(peak_amp: float, sidelobe_amp: float) -> float:
    if sidelobe_amp <= 1e-12:
        return 60.0
    return 20.0 * np.log10(peak_amp / sidelobe_amp)

def analyze_pixel(
    t_rel_us: np.ndarray,
    window_us: int = DEFAULT_WINDOW_US,
    bin_us: int = DEFAULT_BIN_US,
    require_harmonic: bool = True,
    psr_threshold_db: float = 10.0,
) -> Optional[PixelSpec]:
    n_events = int(t_rel_us.size)
    freqs, spec = _periodogram(t_rel_us, window_us, bin_us)
    fund_mask = (freqs >= FUND_F_LO) & (freqs <= FUND_F_HI)
    if not fund_mask.any():
        return None
    fund_sub_spec = spec.copy()
    fund_sub_spec[~fund_mask] = 0.0
    peak_idx = int(np.argmax(fund_sub_spec))
    peak_hz = float(freqs[peak_idx])
    peak_pow = float(spec[peak_idx])
    if _near_mains(peak_hz):
        return None
    sidelobe = _sidelobe_median(spec, freqs, peak_hz, half_bw=20.0)
    psr = _psr_db(peak_pow, sidelobe)
    if psr < psr_threshold_db:
        return None
    harm_f_target = 2.0 * peak_hz
    if require_harmonic and harm_f_target <= ROTOR_F_HI:
        harm_mask = (freqs >= harm_f_target - 15.0) & (freqs <= harm_f_target + 15.0)
        if not harm_mask.any():
            return None
        harm_sub = spec.copy()
        harm_sub[~harm_mask] = 0.0
        harm_idx = int(np.argmax(harm_sub))
        harm_hz = float(freqs[harm_idx])
        harm_pow = float(spec[harm_idx])
        harm_sidelobe = _sidelobe_median(spec, freqs, harm_hz, half_bw=20.0)
        harm_psr = _psr_db(harm_pow, harm_sidelobe)
        if _near_mains(harm_hz):
            return None
        if harm_psr < psr_threshold_db:
            return None
    else:
        harm_hz = float("nan")
        harm_pow = 0.0
        harm_psr = 0.0
    return PixelSpec(
        yx=(-1, -1),
        peak_hz=peak_hz,
        peak_power=peak_pow,
        peak_psr=psr,
        harm_hz=harm_hz,
        harm_power=harm_pow,
        harm_psr=harm_psr,
        n_events=n_events,
    )

@dataclass

class WindowResult:
    window_idx: int
    t_start_us: int
    t_end_us: int
    total_events: int
    n_pixels_ge_min: int
    n_rotor_pixels: int
    rotor_pixels: List[Tuple[int, int]] = field(default_factory=list)

def process_window(
    t_us: np.ndarray,
    x640: np.ndarray,
    y640: np.ndarray,
    t0: int,
    t1: int,
    window_idx: int,
    min_events: int = DEFAULT_MIN_EVENTS,
    max_events: int = 2000,
    bin_us: int = DEFAULT_BIN_US,
    require_harmonic: bool = True,
    psr_threshold_db: float = 10.0,
) -> Tuple[WindowResult, Dict[Tuple[int, int], PixelSpec]]:
    window_us = t1 - t0
    n_ev = int(t_us.size)
    n_bins = window_us // bin_us
    hann = np.hanning(n_bins).astype(np.float32)
    freqs = np.fft.rfftfreq(n_bins, bin_us * 1e-6).astype(np.float32)
    if n_ev == 0:
        return WindowResult(window_idx, t0, t1, 0, 0, 0, []), {}
    pix_flat = (y640.astype(np.int64) * TARGET_W) + x640.astype(np.int64)
    t_rel = (t_us - t0).astype(np.int64)
    t_bin = (t_rel // bin_us).astype(np.int64)
    np.clip(t_bin, 0, n_bins - 1, out=t_bin)
    unique_pix, inv, counts = np.unique(pix_flat, return_inverse=True, return_counts=True)
    active_mask = (counts >= min_events) & (counts <= max_events)
    n_active = int(active_mask.sum())
    n_pixels_ge_min = int((counts >= min_events).sum())
    if n_active == 0:
        return WindowResult(window_idx, t0, t1, n_ev, n_pixels_ge_min, 0, []), {}
    old_to_new = -np.ones(unique_pix.size, dtype=np.int64)
    active_pix_old_ids = np.where(active_mask)[0]
    old_to_new[active_pix_old_ids] = np.arange(n_active)
    remap = old_to_new[inv]
    keep = remap >= 0
    p_new = remap[keep]
    tb = t_bin[keep]
    mat_bytes = n_active * n_bins * 4
    if mat_bytes > 2_000_000_000:
        CHUNK = max(100, int(500_000_000 // (n_bins * 4)))
    else:
        CHUNK = n_active
    rotor_pixels: List[Tuple[int, int]] = []
    specs: Dict[Tuple[int, int], PixelSpec] = {}
    fund_mask = (freqs >= FUND_F_LO) & (freqs <= FUND_F_HI)
    mains_reject = np.zeros(freqs.size, dtype=bool)
    for m in MAINS_HZ:
        mains_reject |= np.abs(freqs - m) < MAINS_TOL_HZ
    for c_start in range(0, n_active, CHUNK):
        c_end = min(n_active, c_start + CHUNK)
        c_len = c_end - c_start
        ev_mask = (p_new >= c_start) & (p_new < c_end)
        p_c = p_new[ev_mask] - c_start
        tb_c = tb[ev_mask]
        flat_idx = p_c * n_bins + tb_c
        cnt_flat = np.bincount(flat_idx, minlength=c_len * n_bins)
        ts_mat = cnt_flat.reshape(c_len, n_bins).astype(np.float32)
        ts_mat -= ts_mat.mean(axis=1, keepdims=True)
        ts_mat *= hann[None, :]
        spec_mat = np.abs(np.fft.rfft(ts_mat, axis=1))
        fund_valid = fund_mask & (~mains_reject)
        fund_spec = np.where(fund_valid[None, :], spec_mat, -1.0)
        peak_idx = np.argmax(fund_spec, axis=1)
        peak_amp = spec_mat[np.arange(c_len), peak_idx]
        peak_hz = freqs[peak_idx]
        line_guard = 10.0
        annulus_outer = 40.0
        side_mask = ((np.abs(freqs[None, :] - peak_hz[:, None]) >= line_guard)
                     & (np.abs(freqs[None, :] - peak_hz[:, None]) <= annulus_outer))
        spec_for_side = np.where(side_mask, spec_mat, np.nan)
        with np.errstate(invalid="ignore"):
            sidelobe = np.nanmedian(spec_for_side, axis=1)
        sidelobe = np.where(np.isnan(sidelobe), 1e-9, sidelobe)
        sidelobe = np.clip(sidelobe, 1e-9, None)
        psr_db = 20.0 * np.log10(np.clip(peak_amp, 1e-9, None) / sidelobe)
        keep_fund = psr_db >= psr_threshold_db
        if require_harmonic:
            harm_target = 2.0 * peak_hz
            harm_range = ((np.abs(freqs[None, :] - harm_target[:, None]) <= 15.0)
                          & (freqs[None, :] <= ROTOR_F_HI))
            harm_spec = np.where(harm_range & (~mains_reject[None, :]), spec_mat, -1.0)
            harm_idx = np.argmax(harm_spec, axis=1)
            harm_valid = harm_spec[np.arange(c_len), harm_idx] > 0
            harm_amp = np.where(
                harm_valid, spec_mat[np.arange(c_len), harm_idx], 0.0
            )
            harm_hz = np.where(harm_valid, freqs[harm_idx], np.nan)
            harm_side_mask = ((np.abs(freqs[None, :] - harm_hz[:, None]) >= line_guard)
                              & (np.abs(freqs[None, :] - harm_hz[:, None]) <= annulus_outer))
            harm_spec_side = np.where(harm_side_mask, spec_mat, np.nan)
            with np.errstate(invalid="ignore"):
                harm_sidelobe = np.nanmedian(harm_spec_side, axis=1)
            harm_sidelobe = np.where(np.isnan(harm_sidelobe), 1e-9, harm_sidelobe)
            harm_sidelobe = np.clip(harm_sidelobe, 1e-9, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                harm_psr_db = 20.0 * np.log10(
                    np.clip(harm_amp, 1e-9, None) / harm_sidelobe
                )
            harm_psr_db = np.where(harm_valid, harm_psr_db, -1e9)
            keep_harm = harm_valid & (harm_psr_db >= psr_threshold_db)
        else:
            harm_hz = np.full(c_len, np.nan)
            harm_amp = np.zeros(c_len)
            harm_psr_db = np.zeros(c_len)
            keep_harm = np.ones(c_len, dtype=bool)
        keep_fund_strong = psr_db >= (psr_threshold_db * 1.5)
        keep_all = keep_fund & (keep_harm | keep_fund_strong)
        kept_local = np.where(keep_all)[0]
        for ki in kept_local:
            old_idx = active_pix_old_ids[c_start + ki]
            pix = int(unique_pix[old_idx])
            py, px = divmod(pix, TARGET_W)
            specs[(py, px)] = PixelSpec(
                yx=(py, px),
                peak_hz=float(peak_hz[ki]),
                peak_power=float(peak_amp[ki]),
                peak_psr=float(psr_db[ki]),
                harm_hz=float(harm_hz[ki]),
                harm_power=float(harm_amp[ki]),
                harm_psr=float(harm_psr_db[ki]),
                n_events=int(counts[old_idx]),
            )
            rotor_pixels.append((py, px))
    return (
        WindowResult(
            window_idx=window_idx,
            t_start_us=t0,
            t_end_us=t1,
            total_events=n_ev,
            n_pixels_ge_min=n_pixels_ge_min,
            n_rotor_pixels=len(rotor_pixels),
            rotor_pixels=rotor_pixels,
        ),
        specs,
    )

def dbscan_pixels(
    pixels: List[Tuple[int, int]], eps: float = 8.0, min_samples: int = 5
) -> List[List[int]]:
    n = len(pixels)
    if n == 0:
        return []
    pts = np.asarray(pixels, dtype=np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    neighbors = [np.where(d2[i] <= eps * eps)[0] for i in range(n)]
    labels = -np.ones(n, dtype=np.int64)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        neigh = neighbors[i]
        if len(neigh) < min_samples:
            continue
        labels[i] = cluster_id
        queue = list(neigh)
        visited = set([i])
        while queue:
            j = queue.pop(0)
            if j in visited:
                continue
            visited.add(j)
            if labels[j] == -1:
                labels[j] = cluster_id
            j_neigh = neighbors[j]
            if len(j_neigh) >= min_samples:
                for k in j_neigh:
                    if k not in visited:
                        queue.append(int(k))
        cluster_id += 1
    clusters: List[List[int]] = []
    for cid in range(cluster_id):
        mem = np.where(labels == cid)[0].tolist()
        if mem:
            clusters.append(mem)
    return clusters

@dataclass

class RotorSeqStats:
    seq_id: str
    n_events: int
    t_span_s: float
    n_windows: int
    total_active_pixels: int
    total_rotor_pixels: int
    total_persistent_pixels: int
    per_frame_boxes: Dict[int, List[Tuple[float, float, float, float, float]]] = field(default_factory=dict)

def _box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def _cluster_bbox(
    cluster_idxs: List[int],
    pix_list: List[Tuple[int, int]],
    events_yx: Optional[Tuple[np.ndarray, np.ndarray]],
    pad_px: int,
) -> Optional[Tuple[float, float, float, float]]:
    cl_pix = np.array([pix_list[i] for i in cluster_idxs], dtype=np.float32)
    y_lo, y_hi = cl_pix[:, 0].min(), cl_pix[:, 0].max()
    x_lo, x_hi = cl_pix[:, 1].min(), cl_pix[:, 1].max()
    tight = (y_lo, y_hi, x_lo, x_hi)
    refined = tight
    if events_yx is not None:
        ey, ex = events_yx
        r = 15
        m = (ey >= y_lo - r) & (ey <= y_hi + r) & (ex >= x_lo - r) & (ex <= x_hi + r)
        if m.sum() > 20:
            yy = ey[m]; xx = ex[m]
            y_med, x_med = np.median(yy), np.median(xx)
            y_mad = max(1.5, np.median(np.abs(yy - y_med)))
            x_mad = max(1.5, np.median(np.abs(xx - x_med)))
            refined = (
                y_med - 3 * y_mad,
                y_med + 3 * y_mad,
                x_med - 3 * x_mad,
                x_med + 3 * x_mad,
            )
    y_lo2 = min(tight[0], refined[0])
    y_hi2 = max(tight[1], refined[1])
    x_lo2 = min(tight[2], refined[2])
    x_hi2 = max(tight[3], refined[3])
    x1 = max(0.0, x_lo2 - pad_px)
    y1 = max(0.0, y_lo2 - pad_px)
    x2 = min(float(TARGET_W - 1), x_hi2 + pad_px)
    y2 = min(float(TARGET_H - 1), y_hi2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))

def persist_and_cluster(
    window_results: List[WindowResult],
    window_events_yx: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_persist: int = 3,
    dbscan_eps: float = 8.0,
    dbscan_min_samples: int = 5,
    pad_px: int = 5,
    track_iou_thresh: float = 0.1,
) -> Tuple[Dict[int, List[Tuple[float, float, float, float, float]]], int]:
    window_candidates: Dict[int, List[Tuple[float, float, float, float, int]]] = {}
    for wr in window_results:
        w = wr.window_idx
        if not wr.rotor_pixels:
            window_candidates[w] = []
            continue
        clusters = dbscan_pixels(
            wr.rotor_pixels, eps=dbscan_eps, min_samples=dbscan_min_samples
        )
        evx = window_events_yx.get(w)
        boxes: List[Tuple[float, float, float, float, int]] = []
        for cluster in clusters:
            bb = _cluster_bbox(cluster, wr.rotor_pixels, evx, pad_px)
            if bb is None:
                continue
            boxes.append((bb[0], bb[1], bb[2], bb[3], len(cluster)))
        window_candidates[w] = boxes
    win_idx_sorted = sorted(window_candidates.keys())
    track_len: Dict[Tuple[int, int], int] = {}
    prev_boxes: List[Tuple[float, float, float, float, int]] = []
    prev_w: Optional[int] = None
    prev_runs: List[int] = []
    for w in win_idx_sorted:
        cur_boxes = window_candidates[w]
        adjacent = (prev_w is not None) and (w == prev_w + 1)
        cur_runs: List[int] = []
        if not cur_boxes:
            prev_boxes = []
            prev_runs = []
            prev_w = w
            continue
        if not adjacent or not prev_boxes:
            cur_runs = [1] * len(cur_boxes)
        else:
            for cb in cur_boxes:
                best_iou = 0.0
                best_pi = -1
                for pi, pb in enumerate(prev_boxes):
                    iou = _box_iou(cb[:4], pb[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi
                if best_iou >= track_iou_thresh and best_pi >= 0:
                    cur_runs.append(prev_runs[best_pi] + 1)
                else:
                    cur_runs.append(1)
        for bi, r in enumerate(cur_runs):
            track_len[(w, bi)] = r
        prev_boxes = cur_boxes
        prev_runs = cur_runs
        prev_w = w
    window_boxes: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    total_persistent = 0
    for w in win_idx_sorted:
        cur_boxes = window_candidates[w]
        kept: List[Tuple[float, float, float, float, float]] = []
        for bi, cb in enumerate(cur_boxes):
            run = track_len.get((w, bi), 0)
            if run >= n_persist:
                score = float(cb[4]) * (1.0 + 0.1 * run)
                kept.append((cb[0], cb[1], cb[2], cb[3], score))
                total_persistent += 1
        window_boxes[w] = kept
    return window_boxes, total_persistent
