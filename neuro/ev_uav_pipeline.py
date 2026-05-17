#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label, maximum_filter
from sklearn.cluster import KMeans
P3 = Path(os.environ.get('NEUROSTDP_ROOT', '.'))
sys.path.insert(0, str(P3 / 'code'))
sys.path.insert(0, str(P3 / 'code' / 'neuro'))
from event_tubes import refine_detections_with_tubes
from hot_pixel_filter import compute_hot_pixel_mask, filter_events_by_hot_mask
EV_UAV_ROOT = Path(os.environ.get('EV_UAV_ROOT', 'data/datasets/ev_uav/EV-UAV-dataset'))
SENSOR_H, SENSOR_W = (260, 346)
FRAME_DT_MS = 20.0

def load_npz(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    ev = d['ev']
    return {'x': ev['x'].astype(np.int32), 'y': ev['y'].astype(np.int32), 't': ev['t'].astype(np.float64), 'p': ev['p'].astype(np.int32), 'label': ev['label'].astype(np.int32), 'name': ev['name'].astype(np.int32)}

def derive_per_frame_gt(ev: Dict, dt_ms: float=FRAME_DT_MS) -> Dict[int, List[Tuple]]:
    t_min, t_max = (float(ev['t'].min()), float(ev['t'].max()))
    n_frames = int(np.ceil((t_max - t_min) / dt_ms)) + 1
    gt = {}
    for f in range(n_frames):
        t0 = t_min + f * dt_ms
        t1 = t0 + dt_ms
        mask = (ev['t'] >= t0) & (ev['t'] < t1) & (ev['label'] == 1)
        if mask.sum() < 3:
            continue
        for n_id in np.unique(ev['name'][mask]):
            m = mask & (ev['name'] == n_id)
            if m.sum() < 3:
                continue
            xs, ys = (ev['x'][m], ev['y'][m])
            x0, y0 = (int(xs.min()), int(ys.min()))
            x1, y1 = (int(xs.max()) + 1, int(ys.max()) + 1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            gt.setdefault(f, []).append((int(n_id), x0, y0, x1, y1, int(m.sum())))
    return gt

def build_frame_histogram(ev: Dict, frame_idx: int, dt_ms: float=FRAME_DT_MS) -> np.ndarray:
    t_min = float(ev['t'].min())
    t0 = t_min + frame_idx * dt_ms
    t1 = t0 + dt_ms
    mask = (ev['t'] >= t0) & (ev['t'] < t1)
    if mask.sum() == 0:
        return np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
    xs = ev['x'][mask]
    ys = ev['y'][mask]
    ok = (xs >= 0) & (xs < SENSOR_W) & (ys >= 0) & (ys < SENSOR_H)
    xs, ys = (xs[ok], ys[ok])
    hist = np.bincount(ys * SENSOR_W + xs, minlength=SENSOR_H * SENSOR_W)
    return hist.reshape(SENSOR_H, SENSOR_W).astype(np.float32)

def build_frame_polarity(ev: Dict, frame_idx: int, dt_ms: float=FRAME_DT_MS) -> Tuple[np.ndarray, np.ndarray]:
    t_min = float(ev['t'].min())
    t0 = t_min + frame_idx * dt_ms
    t1 = t0 + dt_ms
    mask = (ev['t'] >= t0) & (ev['t'] < t1)
    if mask.sum() == 0:
        z = np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
        return (z, z)
    xs = ev['x'][mask]
    ys = ev['y'][mask]
    ps = ev['p'][mask]
    ok = (xs >= 0) & (xs < SENSOR_W) & (ys >= 0) & (ys < SENSOR_H)
    xs, ys, ps = (xs[ok], ys[ok], ps[ok])
    on = ps == 1
    off = ~on
    on_hist = np.bincount(ys[on] * SENSOR_W + xs[on], minlength=SENSOR_H * SENSOR_W).reshape(SENSOR_H, SENSOR_W).astype(np.float32)
    off_hist = np.bincount(ys[off] * SENSOR_W + xs[off], minlength=SENSOR_H * SENSOR_W).reshape(SENSOR_H, SENSOR_W).astype(np.float32)
    return (on_hist, off_hist)

def channel_density_watershed(hist: np.ndarray, sigma: float=0.8, top_k: int=5) -> List[Dict]:
    if hist.sum() < 3:
        return []
    smooth = gaussian_filter(hist, sigma=sigma)
    thr_v = max(smooth.mean() + 0.5 * smooth.std(), smooth.max() * 0.25)
    mask = smooth > thr_v
    if mask.sum() < 3:
        return []
    local_max = maximum_filter(smooth, size=3)
    peak_mask = (smooth == local_max) & mask
    ys, xs = np.nonzero(peak_mask)
    scores = smooth[ys, xs]
    if len(ys) == 0:
        return []
    order = np.argsort(-scores)
    keep_idx = order[:top_k]
    dets = []
    for i in keep_idx:
        py, px = (int(ys[i]), int(xs[i]))
        labeled, _ = cc_label(mask)
        comp = labeled == labeled[py, px]
        if comp.sum() < 3:
            continue
        ev_mask = comp & (hist > 0)
        if ev_mask.sum() < 2:
            continue
        eys, exs = np.nonzero(ev_mask)
        x0, y0 = (int(exs.min()), int(eys.min()))
        x1, y1 = (int(exs.max()) + 1, int(eys.max()) + 1)
        w = x1 - x0
        h = y1 - y0
        if w < 2 or h < 2 or w > 40 or (h > 40):
            continue
        dets.append({'bbox': (x0, y0, x1, y1), 'score': float(scores[i]), 'centroid': ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    return dets

def channel_kmeans(hist: np.ndarray, k: int=3, sigma: float=0.8) -> List[Dict]:
    if hist.sum() < 5:
        return []
    smooth = gaussian_filter(hist, sigma=sigma)
    thr_v = max(smooth.mean() + 0.3 * smooth.std(), smooth.max() * 0.2)
    mask = smooth > thr_v
    ys, xs = np.nonzero(mask)
    if len(ys) < k:
        return []
    pts = np.stack([ys, xs], axis=1).astype(np.float32)
    w = hist[ys, xs]
    if w.sum() == 0:
        return []
    n_clusters = min(k, len(ys))
    try:
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
        km.fit(pts, sample_weight=w)
    except Exception:
        return []
    dets = []
    for c in range(n_clusters):
        member = km.labels_ == c
        if member.sum() < 2:
            continue
        ypts, xpts = (pts[member, 0], pts[member, 1])
        wc = w[member]
        x0, y0 = (int(xpts.min()), int(ypts.min()))
        x1, y1 = (int(xpts.max()) + 1, int(ypts.max()) + 1)
        bw, bh = (x1 - x0, y1 - y0)
        if bw < 2 or bh < 2 or bw > 40 or (bh > 40):
            continue
        score = float(wc.sum())
        dets.append({'bbox': (x0, y0, x1, y1), 'score': score, 'centroid': ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    return dets

def channel_polarity_asymmetry(on: np.ndarray, off: np.ndarray, sigma: float=0.8) -> List[Dict]:
    total = on + off
    if total.sum() < 5:
        return []
    asym = np.abs(on - off) / (total + 1.0)
    signal = asym * gaussian_filter(total, sigma=sigma)
    s_smooth = gaussian_filter(signal, sigma=sigma)
    thr_v = max(s_smooth.mean() + 0.7 * s_smooth.std(), s_smooth.max() * 0.25)
    mask = s_smooth > thr_v
    if mask.sum() < 3:
        return []
    labeled, n_cc = cc_label(mask)
    dets = []
    for i in range(1, n_cc + 1):
        comp = labeled == i
        ev_mask = comp & (total > 0)
        if ev_mask.sum() < 2:
            continue
        eys, exs = np.nonzero(ev_mask)
        x0, y0 = (int(exs.min()), int(eys.min()))
        x1, y1 = (int(exs.max()) + 1, int(eys.max()) + 1)
        w, h = (x1 - x0, y1 - y0)
        if w < 2 or h < 2 or w > 40 or (h > 40):
            continue
        score = float(signal[comp].mean())
        dets.append({'bbox': (x0, y0, x1, y1), 'score': score, 'centroid': ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    return dets

def channel_multi_frame(ev: Dict, frame_idx: int, dt_ms: float=FRAME_DT_MS, window: int=2, sigma: float=0.8) -> List[Dict]:
    t_min = float(ev['t'].min())
    t0 = t_min + (frame_idx - window) * dt_ms
    t1 = t_min + (frame_idx + window + 1) * dt_ms
    mask = (ev['t'] >= t0) & (ev['t'] < t1)
    if mask.sum() == 0:
        return []
    xs, ys = (ev['x'][mask], ev['y'][mask])
    ok = (xs >= 0) & (xs < SENSOR_W) & (ys >= 0) & (ys < SENSOR_H)
    xs, ys = (xs[ok], ys[ok])
    pooled = np.bincount(ys * SENSOR_W + xs, minlength=SENSOR_H * SENSOR_W).reshape(SENSOR_H, SENSOR_W).astype(np.float32)
    return channel_density_watershed(pooled, sigma=sigma)

def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = a_area + b_area - inter
    return inter / u if u > 0 else 0.0

def _with_channel_metadata(dets: List[Dict], channel_id: str) -> List[Dict]:
    if not dets:
        return []
    scores = np.asarray([float(d.get('score', 0.0)) for d in dets], dtype=np.float64)
    order = np.argsort(-scores)
    denom = max(len(dets) - 1, 1)
    score_min = float(scores.min())
    score_ptp = float(scores.max() - score_min)
    out = []
    for rank, idx in enumerate(order):
        d = dict(dets[int(idx)])
        d['channel_id'] = channel_id
        d['channel_rank'] = float(1.0 - rank / denom) if len(dets) > 1 else 1.0
        d['channel_score_norm'] = float((scores[int(idx)] - score_min) / score_ptp) if score_ptp > 1e-09 else 1.0
        out.append(d)
    return out

def _annotate_fusion_support(det: Dict, all_dets: List[Dict], iou_thr: float) -> Dict:
    support = [d for d in all_dets if box_iou(det['bbox'], d['bbox']) >= iou_thr]
    channels = sorted({str(d.get('channel_id', 'unknown')) for d in support if d.get('channel_id') is not None})
    out = dict(det)
    out['fusion_support'] = int(len(support))
    out['fusion_channel_count'] = int(len(channels))
    out['fusion_channels'] = channels
    if support:
        out['fusion_mean_rank'] = float(np.mean([float(d.get('channel_rank', 0.0)) for d in support]))
        out['fusion_mean_score_norm'] = float(np.mean([float(d.get('channel_score_norm', 0.0)) for d in support]))
    else:
        out['fusion_mean_rank'] = float(det.get('channel_rank', 0.0))
        out['fusion_mean_score_norm'] = float(det.get('channel_score_norm', 0.0))
    return out

def nms(dets: List[Dict], iou_thr: float=0.3) -> List[Dict]:
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i]['score'])
    keep = []
    suppressed = [False] * len(dets)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(_annotate_fusion_support(dets[i], dets, iou_thr))
        for j in order:
            if suppressed[j] or i == j:
                continue
            if box_iou(dets[i]['bbox'], dets[j]['bbox']) > iou_thr:
                suppressed[j] = True
    return keep

def union_fusion(channel_outputs: List[List[Dict]], max_dets: int=10) -> List[Dict]:
    all_dets = []
    for ch in channel_outputs:
        all_dets.extend(ch)
    if not all_dets:
        return []
    merged = nms(all_dets, iou_thr=0.3)
    merged.sort(key=lambda d: -d['score'])
    return merged[:max_dets]

def top1_fusion(channel_outputs: List[List[Dict]]) -> List[Dict]:
    all_dets = []
    for ch in channel_outputs:
        all_dets.extend(ch)
    if not all_dets:
        return []
    all_dets.sort(key=lambda d: -d['score'])
    return [all_dets[0]]

def topk_fusion(channel_outputs: List[List[Dict]], k: int=2, iou_thr: float=0.3) -> List[Dict]:
    all_dets = []
    for ch in channel_outputs:
        all_dets.extend(ch)
    if not all_dets:
        return []
    merged = nms(all_dets, iou_thr=iou_thr)
    merged.sort(key=lambda d: -d['score'])
    return merged[:k]

def agreement_filter(channel_outputs: List[List[Dict]], iou_thr: float=0.3, min_other_corroborate: int=1) -> List[List[Dict]]:
    n_chans = len(channel_outputs)
    if n_chans <= 1:
        return channel_outputs
    out = [[] for _ in range(n_chans)]
    for i, ch_i in enumerate(channel_outputs):
        for d in ch_i:
            n_corroborate = 0
            for j, ch_j in enumerate(channel_outputs):
                if i == j:
                    continue
                if any((box_iou(d['bbox'], dj['bbox']) >= iou_thr for dj in ch_j)):
                    n_corroborate += 1
                    if n_corroborate >= min_other_corroborate:
                        break
            if n_corroborate >= min_other_corroborate:
                out[i].append(d)
    return out

def shrink_boxes(dets: List[Dict], factor: float) -> List[Dict]:
    if abs(factor - 1.0) < 1e-06:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * factor
        h = (y2 - y1) * factor
        nb = (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))
        if nb[2] <= nb[0] or nb[3] <= nb[1]:
            nb = (nb[0], nb[1], nb[0] + 1, nb[1] + 1)
        nd = dict(d)
        nd['bbox'] = nb
        nd['centroid'] = (cx, cy)
        out.append(nd)
    return out

def detect_sequence(npz_path: Path, use_channels=('density', 'kmeans', 'polarity', 'mf'), fusion_mode: str='top1', hot_pixel_filter: bool=False, hot_threshold_frac: float=0.2, topk_k: int=None, box_shrink: float=1.0, agreement_gate: bool=False, agreement_iou: float=0.3, agreement_min_others: int=1, tube_refine: bool=False, tube_min_len: int=3, tube_max_gap: int=2, tube_max_dist: float=12.0, tube_support_pad: float=3.0, tube_min_events: int=6, tube_filter_short: bool=False, tube_motion_comp: bool=False, tube_box_mode: str='none', tube_score_weight: float=0.0, tube_stdp_gate: bool=False, tube_stdp_profile: str='balanced', tube_stdp_threshold: float=0.5) -> Dict:
    ev_orig = load_npz(npz_path)
    gt = derive_per_frame_gt(ev_orig)
    if hot_pixel_filter:
        hot_mask, n_hot, _n_frames_hot = compute_hot_pixel_mask(ev_orig, threshold_frac=hot_threshold_frac)
        ev = filter_events_by_hot_mask(ev_orig, hot_mask)
    else:
        ev = ev_orig
        n_hot = 0
    t_min, t_max = (float(ev_orig['t'].min()), float(ev_orig['t'].max()))
    n_frames = int(np.ceil((t_max - t_min) / FRAME_DT_MS)) + 1
    per_frame_dets = {}
    per_frame_dets_union = {}
    for f in range(n_frames):
        chans = []
        if 'density' in use_channels:
            hist = build_frame_histogram(ev, f)
            chans.append(_with_channel_metadata(channel_density_watershed(hist), 'density'))
        if 'kmeans' in use_channels:
            if 'density' not in use_channels:
                hist = build_frame_histogram(ev, f)
            chans.append(_with_channel_metadata(channel_kmeans(hist), 'kmeans'))
        if 'polarity' in use_channels:
            on, off = build_frame_polarity(ev, f)
            chans.append(_with_channel_metadata(channel_polarity_asymmetry(on, off), 'polarity'))
        if 'mf' in use_channels:
            chans.append(_with_channel_metadata(channel_multi_frame(ev, f), 'mf'))
        if agreement_gate:
            chans = agreement_filter(chans, iou_thr=agreement_iou, min_other_corroborate=agreement_min_others)
        if fusion_mode == 'top1':
            merged = top1_fusion(chans)
        elif fusion_mode == 'top2':
            merged = topk_fusion(chans, k=2)
        elif fusion_mode == 'top3':
            merged = topk_fusion(chans, k=3)
        elif fusion_mode == 'top5':
            merged = topk_fusion(chans, k=5)
        elif fusion_mode == 'top10':
            merged = topk_fusion(chans, k=10)
        elif fusion_mode == 'topk':
            merged = topk_fusion(chans, k=topk_k if topk_k else 5)
        else:
            merged = union_fusion(chans)
        if merged:
            if abs(box_shrink - 1.0) > 1e-06:
                merged = shrink_boxes(merged, box_shrink)
            per_frame_dets[f] = merged
        u = union_fusion(chans)
        if u:
            per_frame_dets_union[f] = u
    if tube_refine:
        per_frame_dets, tube_summary = refine_detections_with_tubes(per_frame_dets, ev, sensor_shape=(SENSOR_H, SENSOR_W), dt_ms=FRAME_DT_MS, min_tube_len=tube_min_len, max_gap=tube_max_gap, max_link_dist=tube_max_dist, support_pad=tube_support_pad, min_support_events=tube_min_events, motion_compensate=tube_motion_comp, box_refine_mode=tube_box_mode, filter_short=tube_filter_short, score_weight=tube_score_weight, stdp_gate=tube_stdp_gate, stdp_profile=tube_stdp_profile, stdp_threshold=tube_stdp_threshold)
    else:
        tube_summary = {'n_tubes': 0, 'n_tube_dets': sum((len(v) for v in per_frame_dets.values())), 'n_refined': 0, 'n_long_tubes': 0, 'mean_tube_len': 0.0, 'mean_reliability': 0.0}
    ev_per_frame = []
    density_stds = []
    pol_imbs = []
    for f in range(0, n_frames, max(1, n_frames // 50)):
        hist = build_frame_histogram(ev, f)
        ev_per_frame.append(float(hist.sum()))
        density_stds.append(float(hist.std()))
        on, off = build_frame_polarity(ev, f)
        ts = on.sum() + off.sum()
        pol_imbs.append(abs(on.sum() - off.sum()) / (ts + 1.0))
    ev_per_frame = np.asarray(ev_per_frame)
    cxs = []
    cys = []
    areas = []
    for f in per_frame_dets_union:
        for d in per_frame_dets_union[f]:
            cx, cy = d['centroid']
            x1, y1, x2, y2 = d['bbox']
            cxs.append(cx / SENSOR_W)
            cys.append(cy / SENSOR_H)
            areas.append((x2 - x1) * (y2 - y1) / (SENSOR_W * SENSOR_H))
    if not cxs:
        cxs = [0.5]
        cys = [0.5]
        areas = [0.01]
    fp = {'ev_med': float(np.median(ev_per_frame)) if len(ev_per_frame) else 0.0, 'ev_std': float(np.std(ev_per_frame)) if len(ev_per_frame) else 0.0, 'density_std': float(np.mean(density_stds)) if density_stds else 0.0, 'pol_imbalance': float(np.mean(pol_imbs)) if pol_imbs else 0.0, 'temporal_var': float(np.var(ev_per_frame)) if len(ev_per_frame) > 1 else 0.0, 'cx_med': float(np.median(cxs)), 'cx_std': float(np.std(cxs)), 'cy_med': float(np.median(cys)), 'cy_std': float(np.std(cys)), 'area_med': float(np.median(areas))}
    n_gt_frames = len(gt)
    n_gt_boxes = sum((len(v) for v in gt.values()))
    return {'seq_id': npz_path.stem, 'n_frames': n_frames, 'n_gt_frames': n_gt_frames, 'n_gt_boxes': n_gt_boxes, 'n_hot_pixels': n_hot, 'per_frame_dets': {str(k): v for k, v in per_frame_dets.items()}, 'per_frame_gt': {str(k): v for k, v in gt.items()}, 'fingerprint': fp, 'tube_summary': tube_summary}

def compute_map_at_iou(all_dets, all_gt, iou_thr):
    sorted_dets = sorted(all_dets, key=lambda d: -d[3])
    n_gt_total = sum((len(v) for v in all_gt.values()))
    if n_gt_total == 0:
        return 0.0
    tp = np.zeros(len(sorted_dets), dtype=np.float32)
    fp = np.zeros(len(sorted_dets), dtype=np.float32)
    matched = {k: set() for k in all_gt}
    for i, (sid, f, bb, sc) in enumerate(sorted_dets):
        k = (sid, f)
        gts = all_gt.get(k, [])
        if not gts:
            fp[i] = 1
            continue
        best_iou = 0.0
        best_idx = -1
        for j, gt_bb in enumerate(gts):
            if j in matched[k]:
                continue
            iou = box_iou(bb, gt_bb)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            tp[i] = 1
            matched[k].add(best_idx)
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt_total
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        valid = recall >= r
        p_r = precision[valid].max() if valid.any() else 0.0
        ap += p_r / 101
    return float(ap)

def evaluate_sequences(seq_results, pred_key='per_frame_dets', iou_thresholds=(0.3, 0.5)):
    all_dets = []
    all_gt = {}
    for res in seq_results:
        sid = res['seq_id']
        for f_str, dets in res[pred_key].items():
            f = int(f_str)
            for d in dets:
                bb = d['bbox']
                sc = d['score']
                all_dets.append((sid, f, bb, sc))
        for f_str, gts in res['per_frame_gt'].items():
            f = int(f_str)
            key = (sid, f)
            all_gt[key] = [(g[1], g[2], g[3], g[4]) for g in gts]
    results = {}
    for thr in iou_thresholds:
        results[f'mAP@{int(thr * 100)}'] = compute_map_at_iou(all_dets, all_gt, thr)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    ap.add_argument('--n-seq', type=int, default=-1, help='-1 for all')
    ap.add_argument('--channels', default='density,kmeans,polarity,mf')
    ap.add_argument('--fusion', choices=['top1', 'top2', 'top3', 'top5', 'top10', 'topk', 'union'], default='top1')
    ap.add_argument('--topk-k', type=int, default=5, help='K for --fusion topk')
    ap.add_argument('--hot-pixel-filter', action='store_true', help='Apply per-sequence hot-pixel mask before channel processing (label-free).')
    ap.add_argument('--hot-threshold-frac', type=float, default=0.2, help='Fraction of frames a pixel must fire in to be classified as hot. Default 0.2 (20%%).')
    ap.add_argument('--box-shrink', type=float, default=1.0, help="Multiply each predicted box's w,h by this factor (centred). Default 1.0 (no-op). <1 shrinks; >1 pads.")
    ap.add_argument('--tube-refine', action='store_true', help='Link frame detections into label-free event tubes and refine boxes from tube support.')
    ap.add_argument('--tube-min-len', type=int, default=3, help='Minimum linked detections before a tube can refine boxes.')
    ap.add_argument('--tube-max-gap', type=int, default=2, help='Maximum missing-frame gap allowed while linking a tube.')
    ap.add_argument('--tube-max-dist', type=float, default=12.0, help='Maximum centroid-link distance in pixels before gap expansion.')
    ap.add_argument('--tube-support-pad', type=float, default=3.0, help='Padding around candidate boxes when collecting support events.')
    ap.add_argument('--tube-min-events', type=int, default=6, help='Minimum support events required before replacing a raw box.')
    ap.add_argument('--tube-filter-short', action='store_true', help='Drop detections not belonging to a tube of --tube-min-len or longer.')
    ap.add_argument('--tube-motion-comp', action='store_true', help='Warp support events by tube velocity before box refinement. Experimental; off by default.')
    ap.add_argument('--tube-box-mode', choices=['none', 'support', 'motion', 'auto'], default='none', help='Box refinement mode. Default none keeps raw boxes and uses tubes for filtering/scoring.')
    ap.add_argument('--tube-score-weight', type=float, default=0.0, help='Blend weight for tube reliability into candidate score.')
    ap.add_argument('--tube-stdp-gate', action='store_true', help='Use bounded tube-level STDP reliability instead of fixed length for filtering/scoring.')
    ap.add_argument('--tube-stdp-profile', choices=['balanced', 'precision'], default='balanced')
    ap.add_argument('--tube-stdp-threshold', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    split_dir = EV_UAV_ROOT / args.split
    seq_files = sorted(split_dir.glob('*.npz'))
    if args.n_seq > 0:
        seq_files = seq_files[:args.n_seq]
    print(f'[{args.split}] {len(seq_files)} sequences', flush=True)
    use_channels = tuple(args.channels.split(','))
    print(f'  channels: {use_channels}', flush=True)
    t0 = time.time()
    results = []
    for i, p in enumerate(seq_files):
        r = detect_sequence(p, use_channels=use_channels, fusion_mode=args.fusion, hot_pixel_filter=args.hot_pixel_filter, hot_threshold_frac=args.hot_threshold_frac, topk_k=args.topk_k, box_shrink=args.box_shrink, tube_refine=args.tube_refine, tube_min_len=args.tube_min_len, tube_max_gap=args.tube_max_gap, tube_max_dist=args.tube_max_dist, tube_support_pad=args.tube_support_pad, tube_min_events=args.tube_min_events, tube_filter_short=args.tube_filter_short, tube_motion_comp=args.tube_motion_comp, tube_box_mode=args.tube_box_mode, tube_score_weight=args.tube_score_weight, tube_stdp_gate=args.tube_stdp_gate, tube_stdp_profile=args.tube_stdp_profile, tube_stdp_threshold=args.tube_stdp_threshold)
        results.append(r)
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(seq_files) - i - 1)
            print(f'  [{i + 1}/{len(seq_files)}] {p.stem} n_frames={r['n_frames']} n_dets={sum((len(v) for v in r['per_frame_dets'].values()))} n_gt={r['n_gt_boxes']} | elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)
    metrics = evaluate_sequences(results)
    per_seq_metrics = {r['seq_id']: {'n_gt_boxes': r['n_gt_boxes'], 'n_dets_total': sum((len(v) for v in r['per_frame_dets'].values())), 'tube_summary': r.get('tube_summary', {})} for r in results}
    out = {'split': args.split, 'n_sequences': len(results), 'channels': list(use_channels), 'fusion': args.fusion, 'tube_refine': args.tube_refine, 'tube_params': {'min_len': args.tube_min_len, 'max_gap': args.tube_max_gap, 'max_dist': args.tube_max_dist, 'support_pad': args.tube_support_pad, 'min_events': args.tube_min_events, 'filter_short': args.tube_filter_short, 'motion_compensate': args.tube_motion_comp, 'box_mode': args.tube_box_mode, 'score_weight': args.tube_score_weight, 'stdp_gate': args.tube_stdp_gate, 'stdp_profile': args.tube_stdp_profile, 'stdp_threshold': args.tube_stdp_threshold}, 'mAP': metrics, 'per_seq_meta': per_seq_metrics, 'runtime_sec': time.time() - t0}
    suffix = f'_{args.fusion}' if args.fusion != 'union' else ''
    out_path = Path(args.out) if args.out else P3 / f'docs/ev_uav_{args.split}{suffix}_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    full_path = P3 / f'docs/ev_uav_{args.split}_full.json'
    compact = []
    for r in results:
        compact.append({'seq_id': r['seq_id'], 'n_frames': r['n_frames'], 'n_gt_frames': r['n_gt_frames'], 'n_gt_boxes': r['n_gt_boxes'], 'fingerprint': r['fingerprint'], 'tube_summary': r.get('tube_summary', {})})
    full_path.write_text(json.dumps(compact, indent=2))
    print(f'\n=== RESULTS ({args.split}) ===')
    print(f'N sequences: {len(results)}')
    for k, v in metrics.items():
        print(f'  {k}: {v * 100:.2f}%')
    print(f'Runtime: {time.time() - t0:.1f}s')
    print(f'Saved: {out_path}')
if __name__ == '__main__':
    main()
