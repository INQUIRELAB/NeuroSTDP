#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
import numpy as np
P3 = Path(os.environ.get('NEUROSTDP_ROOT', str(Path(__file__).resolve().parents[2])))
PL = P3 / 'pseudo_labels'
LEVER_TO_EVAL_KEY = {'no_op': 'v127d', 'shadow_rescore': 'v125c', 'mf3_hybrid': 'v121', 'raw_v47': 'v47'}
LEVER_TO_LABEL_DIR = {'no_op': 'discovery_v127d_max/labels', 'shadow_rescore': 'discovery_v125c_shadow_rescore/labels', 'mf3_hybrid': 'discovery_v121_mf3_k2/labels', 'raw_v47': 'discovery_v47_full/labels'}

def map_label_efficient(cohort_assign, eval_data):
    cohort_ids = sorted({v['cohort'] for v in cohort_assign.values()})
    out = {}
    for c in cohort_ids:
        sids_in_c = [sid for sid, v in cohort_assign.items() if v['cohort'] == c]
        if not sids_in_c:
            continue
        per_lever = {}
        for lever, eval_key in LEVER_TO_EVAL_KEY.items():
            scores = []
            for sid in sids_in_c:
                row = eval_data.get(eval_key, {}).get(sid)
                if row is not None and row.get('mAP_30') is not None:
                    scores.append(row['mAP_30'])
            if scores:
                per_lever[lever] = {'mean_map30': float(np.mean(scores)), 'median_map30': float(np.median(scores)), 'n_seqs': len(scores)}
        if not per_lever:
            continue
        best_lever = max(per_lever.items(), key=lambda kv: (kv[1]['mean_map30'], kv[0] == 'no_op'))[0]
        delta_vs_noop = per_lever[best_lever]['mean_map30'] - per_lever.get('no_op', {'mean_map30': 0.0})['mean_map30']
        out[int(c)] = {'n_seqs': len(sids_in_c), 'member_sids': sids_in_c, 'per_lever': per_lever, 'winner_lever': best_lever, 'winner_mean_map30': per_lever[best_lever]['mean_map30'], 'delta_vs_no_op': float(delta_vs_noop)}
    return out

def load_yolo_top1(path: Path, W=640, H=640):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
    return None

def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = (max(a[0], b[0]), max(a[1], b[1]))
    ix2, iy2 = (min(a[2], b[2]), min(a[3], b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def mean_cross_channel_agreement(sid: str, lever_top1_dir: Path, reference_dirs: list[Path], sample_every: int=5):
    frames = sorted(lever_top1_dir.glob(f'{sid}/{sid}_frame_*.txt'))
    if not frames:
        return None
    ious = []
    for idx, f in enumerate(frames):
        if idx % sample_every != 0:
            continue
        lever_box = load_yolo_top1(f)
        if lever_box is None:
            continue
        per_ref = []
        for rd in reference_dirs:
            ref_box = load_yolo_top1(rd / sid / f.name)
            if ref_box is not None:
                per_ref.append(box_iou(lever_box, ref_box))
        if per_ref:
            ious.append(float(np.mean(per_ref)))
    return float(np.mean(ious)) if ious else None
REFERENCE_CHANNELS = {'v47': PL / 'discovery_v47_full/labels', 'v66': PL / 'discovery_v66_full/labels', 'mf3': PL / 'multiframe_v3_full/labels', 'rotor': PL / 'rotor_fft_full/labels', 'polarity': PL / 'polarity_v72_full/labels'}
LEVER_SELF_REFERENCE = {'raw_v47': 'v47', 'no_op': None, 'shadow_rescore': None, 'mf3_hybrid': None}

def map_label_free(cohort_assign):
    cohort_ids = sorted({v['cohort'] for v in cohort_assign.values()})
    out = {}
    for c in cohort_ids:
        sids_in_c = [sid for sid, v in cohort_assign.items() if v['cohort'] == c]
        if not sids_in_c:
            continue
        per_lever = {}
        for lever, rel in LEVER_TO_LABEL_DIR.items():
            lever_dir = PL / rel.replace('/labels', '') / 'labels'
            self_name = LEVER_SELF_REFERENCE.get(lever)
            refs = [d for n, d in REFERENCE_CHANNELS.items() if n != self_name]
            agree_vals = []
            for sid in sids_in_c:
                a = mean_cross_channel_agreement(sid, lever_dir, refs)
                if a is not None:
                    agree_vals.append(a)
            if agree_vals:
                per_lever[lever] = {'mean_agreement': float(np.mean(agree_vals)), 'n_seqs': len(agree_vals), 'n_references': len(refs)}
        if not per_lever:
            continue
        best_lever = max(per_lever.items(), key=lambda kv: (kv[1]['mean_agreement'], kv[0] == 'no_op'))[0]
        out[int(c)] = {'n_seqs': len(sids_in_c), 'per_lever': per_lever, 'winner_lever': best_lever, 'winner_mean_agreement': per_lever[best_lever]['mean_agreement']}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assignments', default=str(P3 / 'docs/snn_cohort_assignments_label_free.json'))
    ap.add_argument('--eval-json', default=str(P3 / 'docs/eval_v2_multipred_multidrone.json'))
    ap.add_argument('--variant', choices=['label_efficient', 'label_free', 'both'], default='both')
    ap.add_argument('--out-prefix', default=str(P3 / 'docs/cohort_to_lever'))
    args = ap.parse_args()
    assign = json.loads(Path(args.assignments).read_text())
    train_assign = assign['canonical_train']
    eval_data = json.loads(Path(args.eval_json).read_text())
    print(f'Loaded {len(train_assign)} train assignments across {len(set((v['cohort'] for v in train_assign.values())))} cohorts')
    if args.variant in ('label_efficient', 'both'):
        mapping = map_label_efficient(train_assign, eval_data)
        out = {'variant': 'label_efficient', 'supervision_footprint_bits': len(mapping) * np.log2(len(LEVER_TO_EVAL_KEY)), 'cohorts': mapping}
        out_path = Path(f'{args.out_prefix}_label_efficient.json')
        out_path.write_text(json.dumps(out, indent=2))
        print(f'\nLabel-efficient mapping → {out_path}')
        for c, info in mapping.items():
            print(f'  cohort {c:>2d}  (n={info['n_seqs']:>3d})  winner={info['winner_lever']:<16s}  mean mAP@30={info['winner_mean_map30']:.4f}  Δ_vs_noop={info['delta_vs_no_op']:+.4f}')
    if args.variant in ('label_free', 'both'):
        print('\nComputing label-free cross-channel agreement (sampled every 5 frames) ...')
        mapping = map_label_free(train_assign)
        out = {'variant': 'label_free', 'cohorts': mapping}
        out_path = Path(f'{args.out_prefix}_label_free.json')
        out_path.write_text(json.dumps(out, indent=2))
        print(f'\nLabel-free mapping → {out_path}')
        for c, info in mapping.items():
            print(f'  cohort {c:>2d}  (n={info['n_seqs']:>3d})  winner={info['winner_lever']:<16s}  mean_agreement={info['winner_mean_agreement']:.4f}')
if __name__ == '__main__':
    main()
