#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
P3 = Path(os.environ.get('NEUROSTDP_ROOT', '.'))
sys.path.insert(0, str(P3 / 'code/neuro'))
from ev_uav_pipeline import EV_UAV_ROOT, SENSOR_H, SENSOR_W, FRAME_DT_MS, load_npz, derive_per_frame_gt, build_frame_histogram, build_frame_polarity, channel_density_watershed, channel_kmeans, channel_polarity_asymmetry, channel_multi_frame, union_fusion, top1_fusion, topk_fusion, shrink_boxes, box_iou, agreement_filter, _with_channel_metadata, compute_map_at_iou
from event_tubes import refine_detections_with_tubes
from hot_pixel_filter import compute_hot_pixel_mask, filter_events_by_hot_mask

def compute_event_level_metrics(ev: dict, per_frame_dets: dict) -> dict:
    t_min = float(ev['t'].min())
    t = ev['t']
    label = ev['label']
    x = ev['x']
    y = ev['y']
    n = len(t)
    f_idx = ((t - t_min) / FRAME_DT_MS).astype(np.int64)
    inside_any_pred = np.zeros(n, dtype=bool)
    for f, dets in per_frame_dets.items():
        fi = int(f)
        evm = f_idx == fi
        if evm.sum() == 0:
            continue
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            hit = evm & (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
            inside_any_pred |= hit
    target = label == 1
    tp = int(np.sum(inside_any_pred & target))
    fp = int(np.sum(inside_any_pred & ~target))
    fn = int(np.sum(~inside_any_pred & target))
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 0.0
    pd = tp / (tp + fn) if tp + fn > 0 else 0.0
    fa = fp / n if n > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'N': int(n), 'iou': float(iou), 'pd': float(pd), 'fa': float(fa)}

def write_yolo_labels(per_frame_dets: dict, out_dir: Path, n_frames: int, img_w: int=SENSOR_W, img_h: int=SENSOR_H):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in range(n_frames):
        fname = out_dir / f'{f:06d}.txt'
        dets = per_frame_dets.get(f, [])
        if not dets:
            fname.write_text('')
            continue
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0001, min(1.0, w))
            h = max(0.0001, min(1.0, h))
            score = float(d.get('score', 1.0))
            lines.append(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}')
        fname.write_text('\n'.join(lines) + '\n')

def process_sequence(npz_path: Path, pseudo_label_root: Path, use_channels=('density', 'kmeans', 'polarity', 'mf'), fusion_max: int=10, fusion_mode: str='union', hot_pixel_filter: bool=False, hot_threshold_frac: float=0.2, box_shrink: float=1.0, agreement_gate: bool=False, agreement_iou: float=0.3, agreement_min_others: int=1, tube_refine: bool=False, tube_min_len: int=3, tube_max_gap: int=2, tube_max_dist: float=12.0, tube_support_pad: float=3.0, tube_min_events: int=6, tube_filter_short: bool=False, tube_motion_comp: bool=False, tube_box_mode: str='none', tube_score_weight: float=0.0, tube_stdp_gate: bool=False, tube_stdp_profile: str='balanced', tube_stdp_threshold: float=0.5):
    ev_orig = load_npz(npz_path)
    gt = derive_per_frame_gt(ev_orig)
    if hot_pixel_filter:
        hot_mask, n_hot, _nf = compute_hot_pixel_mask(ev_orig, threshold_frac=hot_threshold_frac)
        ev = filter_events_by_hot_mask(ev_orig, hot_mask)
    else:
        ev = ev_orig
        n_hot = 0
    t_min, t_max = (float(ev_orig['t'].min()), float(ev_orig['t'].max()))
    n_frames = int(np.ceil((t_max - t_min) / FRAME_DT_MS)) + 1
    per_frame_dets = {}
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
        if fusion_mode == 'union':
            merged = union_fusion(chans, max_dets=fusion_max)
        elif fusion_mode == 'top1':
            merged = top1_fusion(chans)
        elif fusion_mode == 'top3':
            merged = topk_fusion(chans, k=3)
        elif fusion_mode == 'top5':
            merged = topk_fusion(chans, k=5)
        elif fusion_mode == 'top10':
            merged = topk_fusion(chans, k=10)
        else:
            merged = union_fusion(chans, max_dets=fusion_max)
        if merged:
            if abs(box_shrink - 1.0) > 1e-06:
                merged = shrink_boxes(merged, box_shrink)
            per_frame_dets[f] = merged
    if tube_refine:
        per_frame_dets, tube_summary = refine_detections_with_tubes(per_frame_dets, ev, sensor_shape=(SENSOR_H, SENSOR_W), dt_ms=FRAME_DT_MS, min_tube_len=tube_min_len, max_gap=tube_max_gap, max_link_dist=tube_max_dist, support_pad=tube_support_pad, min_support_events=tube_min_events, motion_compensate=tube_motion_comp, box_refine_mode=tube_box_mode, filter_short=tube_filter_short, score_weight=tube_score_weight, stdp_gate=tube_stdp_gate, stdp_profile=tube_stdp_profile, stdp_threshold=tube_stdp_threshold)
    else:
        tube_summary = {'n_tubes': 0, 'n_tube_dets': sum((len(v) for v in per_frame_dets.values())), 'n_refined': 0, 'n_long_tubes': 0, 'mean_tube_len': 0.0, 'mean_reliability': 0.0}
    seq_dir = pseudo_label_root / npz_path.stem
    write_yolo_labels(per_frame_dets, seq_dir, n_frames)
    evmetrics = compute_event_level_metrics(ev_orig, per_frame_dets)
    evmetrics['n_hot_pixels'] = n_hot
    ev_per_frame, density_stds, pol_imbs = ([], [], [])
    step = max(1, n_frames // 50)
    for f in range(0, n_frames, step):
        hist = build_frame_histogram(ev, f)
        ev_per_frame.append(float(hist.sum()))
        density_stds.append(float(hist.std()))
        on, off = build_frame_polarity(ev, f)
        ts = on.sum() + off.sum()
        pol_imbs.append(abs(on.sum() - off.sum()) / (ts + 1.0))
    ev_per_frame = np.asarray(ev_per_frame)
    cxs, cys, areas = ([], [], [])
    for f, dets in per_frame_dets.items():
        for d in dets:
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
    return {'seq_id': npz_path.stem, 'n_frames': n_frames, 'n_gt_frames': len(gt), 'n_gt_boxes': sum((len(v) for v in gt.values())), 'per_frame_dets': {str(k): v for k, v in per_frame_dets.items()}, 'per_frame_gt': {str(k): v for k, v in gt.items()}, 'fingerprint': fp, 'event_metrics': evmetrics, 'tube_summary': tube_summary}

def aggregate_split(results: list) -> dict:
    all_dets = []
    all_gt = {}
    for r in results:
        sid = r['seq_id']
        for fs, dets in r['per_frame_dets'].items():
            f = int(fs)
            for d in dets:
                all_dets.append((sid, f, d['bbox'], d['score']))
        for fs, gts in r['per_frame_gt'].items():
            f = int(fs)
            all_gt[sid, f] = [(g[1], g[2], g[3], g[4]) for g in gts]
    mAP30 = compute_map_at_iou(all_dets, all_gt, 0.3)
    mAP50 = compute_map_at_iou(all_dets, all_gt, 0.5)
    tp = sum((r['event_metrics']['tp'] for r in results))
    fp = sum((r['event_metrics']['fp'] for r in results))
    fn = sum((r['event_metrics']['fn'] for r in results))
    N = sum((r['event_metrics']['N'] for r in results))
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 0.0
    pd = tp / (tp + fn) if tp + fn > 0 else 0.0
    fa = fp / N if N > 0 else 0.0
    per_seq = {}
    for r in results:
        em = r['event_metrics']
        per_seq[r['seq_id']] = {'n_frames': r['n_frames'], 'n_gt_boxes': r['n_gt_boxes'], 'n_dets_total': sum((len(v) for v in r['per_frame_dets'].values())), 'iou_percent': em['iou'] * 100, 'pd_percent': em['pd'] * 100, 'fa_scaled_1e-4': em['fa'] * 10000.0, 'tp': em['tp'], 'fp': em['fp'], 'fn': em['fn'], 'N': em['N'], 'tube_summary': r.get('tube_summary', {})}
    return {'mAP30': mAP30, 'mAP50': mAP50, 'event_level': {'iou_percent': iou * 100, 'pd_percent': pd * 100, 'fa_scaled_1e-4': fa * 10000.0, 'tp': tp, 'fp': fp, 'fn': fn, 'N_events': N}, 'per_seq': per_seq}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    ap.add_argument('--n-seq', type=int, default=-1)
    ap.add_argument('--fusion-max', type=int, default=10)
    ap.add_argument('--fusion-mode', choices=['union', 'top1', 'top3', 'top5', 'top10'], default='union', help="Fusion mode. 'union' = NMS union of all channel dets.")
    ap.add_argument('--hot-pixel-filter', action='store_true', help='Apply per-sequence hot-pixel mask before channel processing (label-free).')
    ap.add_argument('--agreement-gate', action='store_true', help='Cross-channel-agreement gate: drop detections with no IoU corroboration from another channel.')
    ap.add_argument('--agreement-iou', type=float, default=0.3, help='IoU threshold for cross-channel agreement gate.')
    ap.add_argument('--agreement-min-others', type=int, default=1, help='Min number of OTHER channels that must corroborate a detection.')
    ap.add_argument('--hot-threshold-frac', type=float, default=0.2, help='Pixel firing in > THIS fraction of frames is classified as hot. Default 0.2.')
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
    ap.add_argument('--tube-stdp-profile', choices=['balanced', 'precision'], default='balanced', help='STDP gate profile. Precision raises the no-evidence threshold.')
    ap.add_argument('--tube-stdp-threshold', type=float, default=0.5, help='Minimum STDP reliability for keeping a tube when --tube-filter-short is set.')
    ap.add_argument('--out-suffix', default='', help='Suffix appended to output filename and pseudo_labels folder.')
    args = ap.parse_args()
    split_dir = EV_UAV_ROOT / args.split
    seq_files = sorted(split_dir.glob('*.npz'))
    if args.n_seq > 0:
        seq_files = seq_files[:args.n_seq]
    suffix = f'_{args.out_suffix}' if args.out_suffix else ''
    pseudo_root = P3 / f'pseudo_labels/ev_uav_{args.split}{suffix}/labels'
    print(f'[{args.split}] {len(seq_files)} sequences, pseudo-labels -> {pseudo_root}', flush=True)
    print(f'  fusion_mode={args.fusion_mode}, hot_filter={args.hot_pixel_filter} (thr={args.hot_threshold_frac})', flush=True)
    if args.tube_refine:
        print(f'  tube_refine=True min_len={args.tube_min_len} max_gap={args.tube_max_gap} max_dist={args.tube_max_dist} filter_short={args.tube_filter_short} box_mode={args.tube_box_mode} stdp={args.tube_stdp_gate}', flush=True)
    t0 = time.time()
    results = []
    for i, p in enumerate(seq_files):
        r = process_sequence(p, pseudo_root, fusion_max=args.fusion_max, fusion_mode=args.fusion_mode, hot_pixel_filter=args.hot_pixel_filter, hot_threshold_frac=args.hot_threshold_frac, box_shrink=args.box_shrink, agreement_gate=args.agreement_gate, agreement_iou=args.agreement_iou, agreement_min_others=args.agreement_min_others, tube_refine=args.tube_refine, tube_min_len=args.tube_min_len, tube_max_gap=args.tube_max_gap, tube_max_dist=args.tube_max_dist, tube_support_pad=args.tube_support_pad, tube_min_events=args.tube_min_events, tube_filter_short=args.tube_filter_short, tube_motion_comp=args.tube_motion_comp, tube_box_mode=args.tube_box_mode, tube_score_weight=args.tube_score_weight, tube_stdp_gate=args.tube_stdp_gate, tube_stdp_profile=args.tube_stdp_profile, tube_stdp_threshold=args.tube_stdp_threshold)
        results.append(r)
        if (i + 1) % 5 == 0 or i == 0 or i == len(seq_files) - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (len(seq_files) - i - 1)
            em = r['event_metrics']
            print(f'  [{i + 1}/{len(seq_files)}] {p.stem} n_fr={r['n_frames']} n_gt={r['n_gt_boxes']} IoU={em['iou'] * 100:.1f} Pd={em['pd'] * 100:.1f} Fa={em['fa'] * 10000.0:.0f} | elapsed={el:.0f}s eta={eta:.0f}s', flush=True)
    agg = aggregate_split(results)
    compact_results = []
    for r in results:
        compact_results.append({'seq_id': r['seq_id'], 'n_frames': r['n_frames'], 'n_gt_frames': r['n_gt_frames'], 'n_gt_boxes': r['n_gt_boxes'], 'n_dets_total': sum((len(v) for v in r['per_frame_dets'].values())), 'fingerprint': r['fingerprint'], 'event_metrics': r['event_metrics'], 'tube_summary': r.get('tube_summary', {})})
    out = {'split': args.split, 'n_sequences': len(results), 'fusion': args.fusion_mode, 'fusion_max': args.fusion_max, 'hot_pixel_filter': args.hot_pixel_filter, 'hot_threshold_frac': args.hot_threshold_frac, 'box_shrink': args.box_shrink, 'tube_refine': args.tube_refine, 'tube_params': {'min_len': args.tube_min_len, 'max_gap': args.tube_max_gap, 'max_dist': args.tube_max_dist, 'support_pad': args.tube_support_pad, 'min_events': args.tube_min_events, 'filter_short': args.tube_filter_short, 'motion_compensate': args.tube_motion_comp, 'box_mode': args.tube_box_mode, 'score_weight': args.tube_score_weight, 'stdp_gate': args.tube_stdp_gate, 'stdp_profile': args.tube_stdp_profile, 'stdp_threshold': args.tube_stdp_threshold}, 'channels': ['density', 'kmeans', 'polarity', 'mf'], 'mAP30': agg['mAP30'], 'mAP50': agg['mAP50'], 'event_level': agg['event_level'], 'per_seq': agg['per_seq'], 'sequence_summaries': compact_results, 'runtime_sec': time.time() - t0}
    suffix = f'_{args.out_suffix}' if args.out_suffix else ''
    out_path = P3 / f'docs/ev_uav_full_{args.split}{suffix}_results.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n=== {args.split.upper()} (n={len(results)}) ===')
    print(f'  mAP@30: {agg['mAP30'] * 100:.2f}%')
    print(f'  mAP@50: {agg['mAP50'] * 100:.2f}%')
    e = agg['event_level']
    print(f'  IoU:   {e['iou_percent']:.2f}%')
    print(f'  Pd:    {e['pd_percent']:.2f}%')
    print(f'  Fa:    {e['fa_scaled_1e-4']:.2f} x1e-4')
    print(f'  runtime: {out['runtime_sec']:.1f}s  -> {out_path}')
if __name__ == '__main__':
    main()
