#!/usr/bin/env python3
import sys
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
import spconv.pytorch as spconv
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent
_PAPER3_ROOT = _CODE_DIR.parent
_PROJECT_ROOT = _PAPER3_ROOT.parent
_PAPER1_ROOT = _PROJECT_ROOT / 'Paper1_SparseVoxelDet'
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PAPER1_ROOT))
from V2.models.sparse_voxel_det_v82 import SparseVoxelDet
from sparse_fcos_v1.scripts.metrics import MAPCalculator
CHECKPOINT = _PAPER3_ROOT / 'runs/zero_label/self_train_v1/round1/best.pt'
SPARSE_SPLIT_ROOT = _PROJECT_ROOT / 'data/datasets/fred_paper_parity_v82_640/sparse'
LABEL_SPLIT_ROOT = _PROJECT_ROOT / 'data/datasets/fred_paper_parity/labels'
DEFAULT_SPLIT = 'canonical_test'
VALID_SPLITS = ('canonical_test', 'canonical_train')
DENSITY_TIERS: List[Tuple[str, float, float]] = [('sparse', 0.0, 3000.0), ('medium', 3000.0, 10000.0), ('dense', 10000.0, 20000.0), ('very_dense', 20000.0, float('inf'))]
DENSITY_SAMPLE_FRAMES = 20
TIME_BINS = 16
IN_CHANNELS = 6
INPUT_SIZE = (640, 640)
SCORE_THRESH = 0.001
NMS_THRESH = 0.5
MAX_DETECTIONS = 10
BATCH_SIZE = 4
NUM_WORKERS = 4

class TestDataset(Dataset):
    def __init__(self, sparse_dir: Path, label_dir: Path, sequence_id: Optional[str]=None, time_bins: int=TIME_BINS, in_channels: int=IN_CHANNELS, target_size: Tuple[int, int]=INPUT_SIZE):
        self.label_dir = label_dir
        self.time_bins = time_bins
        self.in_channels = in_channels
        self.target_size = target_size
        self.spatial_shape = [time_bins, target_size[0], target_size[1]]
        if not sparse_dir.exists():
            raise RuntimeError(f'Sparse dir not found: {sparse_dir}')
        if not label_dir.exists():
            raise RuntimeError(f'Label dir not found: {label_dir}')
        self.samples = []
        seq_dirs = sorted(sparse_dir.iterdir())
        for seq_dir in seq_dirs:
            if not seq_dir.is_dir():
                continue
            if sequence_id is not None and seq_dir.name != sequence_id:
                continue
            seq = seq_dir.name
            for frame_file in sorted(seq_dir.glob('frame_*.npz')):
                frame_idx = int(frame_file.stem.split('_')[1])
                label_name = f'{seq}_frame_{frame_idx:06d}.txt'
                label_path = label_dir / label_name
                if not label_path.exists():
                    label_name_alt = f'{seq}_frame_{frame_idx}.txt'
                    label_path_alt = label_dir / label_name_alt
                    if label_path_alt.exists():
                        label_path = label_path_alt
                    else:
                        label_path = None
                self.samples.append((frame_file, label_path, seq))
    def __len__(self) -> int:
        return len(self.samples)
    def _load_sparse(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        if 'coords' not in data or 'feats' not in data:
            raise ValueError(f'Missing coords/feats in {path}')
        coords = data['coords'].astype(np.int32)
        feats = data['feats'].astype(np.float32)
        T, H, W = self.spatial_shape
        if 'time_bins' in data:
            data_T = int(data['time_bins'])
        else:
            data_T = int(coords[:, 0].max()) + 1 if len(coords) > 0 else T
        if data_T != T and len(coords) > 0:
            coords[:, 0] = coords[:, 0] * T // data_T
            keys = coords[:, 0].astype(np.int64) * (H * W) + coords[:, 1].astype(np.int64) * W + coords[:, 2].astype(np.int64)
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            if len(unique_keys) < len(coords):
                n_unique = len(unique_keys)
                new_feats = np.zeros((n_unique, feats.shape[1]), dtype=feats.dtype)
                np.maximum.at(new_feats, inverse, feats)
                new_coords = np.zeros((n_unique, 3), dtype=coords.dtype)
                new_coords[:, 0] = (unique_keys // (H * W)).astype(coords.dtype)
                new_coords[:, 1] = (unique_keys % (H * W) // W).astype(coords.dtype)
                new_coords[:, 2] = (unique_keys % W).astype(coords.dtype)
                coords = new_coords
                feats = new_feats
        if len(coords) > 0:
            coords[:, 0] = np.clip(coords[:, 0], 0, T - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
            coords[:, 2] = np.clip(coords[:, 2], 0, W - 1)
        if len(feats) > 0 and (not np.all(np.isfinite(feats))):
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if feats.shape[1] > self.in_channels:
            feats = feats[:, :self.in_channels].copy()
        return (coords, feats)
    def _load_gt_labels(self, path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
        if path is None or not path.exists():
            return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        H, W = self.target_size
        boxes_list = []
        classes_list = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                boxes_list.append([x1, y1, x2, y2])
                classes_list.append(cls)
        if not boxes_list:
            return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        return (np.array(boxes_list, dtype=np.float32), np.array(classes_list, dtype=np.int64))
    def __getitem__(self, idx: int) -> Dict:
        sparse_path, label_path, seq_id = self.samples[idx]
        frame_idx = int(sparse_path.stem.split('_')[1])
        coords, feats = self._load_sparse(sparse_path)
        gt_boxes, gt_classes = self._load_gt_labels(label_path)
        if len(coords) == 0:
            coords = np.zeros((1, 3), dtype=np.int32)
            feats = np.zeros((1, self.in_channels), dtype=np.float32)
        return {'coords': torch.from_numpy(coords), 'feats': torch.from_numpy(feats), 'gt_boxes': torch.from_numpy(gt_boxes), 'gt_classes': torch.from_numpy(gt_classes), 'seq_id': seq_id, 'frame_idx': frame_idx}

def collate_fn(batch: List[Dict]) -> Dict:
    all_coords = []
    all_feats = []
    gt_boxes = []
    gt_labels_yolo = []
    seq_ids = []
    frame_idxs = []
    H, W = INPUT_SIZE
    for b_idx, sample in enumerate(batch):
        coords = sample['coords'].clone()
        batch_col = torch.full((coords.shape[0], 1), b_idx, dtype=torch.int32)
        coords_4d = torch.cat([batch_col, coords], dim=1)
        all_coords.append(coords_4d)
        all_feats.append(sample['feats'])
        gt_boxes.append(sample['gt_boxes'])
        boxes_b = sample['gt_boxes']
        labels_b = sample['gt_classes']
        if len(boxes_b) > 0:
            cx = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0 / W
            cy = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0 / H
            bw = (boxes_b[:, 2] - boxes_b[:, 0]) / W
            bh = (boxes_b[:, 3] - boxes_b[:, 1]) / H
            yolo = torch.stack([labels_b.float(), cx, cy, bw, bh], dim=1)
        else:
            yolo = torch.zeros((0, 5))
        gt_labels_yolo.append(yolo)
        seq_ids.append(sample['seq_id'])
        frame_idxs.append(sample['frame_idx'])
    return {'coords': torch.cat(all_coords, dim=0), 'feats': torch.cat(all_feats, dim=0), 'spatial_shape': [TIME_BINS, H, W], 'batch_size': len(batch), 'gt_boxes': gt_boxes, 'gt_labels_yolo': gt_labels_yolo, 'seq_ids': seq_ids, 'frame_idxs': frame_idxs}

def load_model(checkpoint_path: Path, device: torch.device) -> SparseVoxelDet:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = SparseVoxelDet(in_channels=IN_CHANNELS, num_classes=1, backbone_size='nano_deep', fpn_channels=128, head_convs=2, input_size=INPUT_SIZE, time_bins=TIME_BINS, prior_prob=0.01, score_thresh=SCORE_THRESH, nms_thresh=NMS_THRESH, max_detections=MAX_DETECTIONS)
    if 'ema_state_dict' in ckpt:
        ema_obj = ckpt['ema_state_dict']
        if isinstance(ema_obj, dict) and 'shadow' in ema_obj:
            state_dict = ema_obj['shadow']
            print("  Using ema_state_dict['shadow']")
        else:
            state_dict = ema_obj
            print('  Using ema_state_dict (direct)')
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print('  WARNING: ema_state_dict not found, using model_state_dict')
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print('  WARNING: using state_dict key')
    else:
        state_dict = ckpt
        print('  WARNING: checkpoint has no recognized key, treating as raw state_dict')
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f'  WARNING: missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
    model = model.to(device)
    model.eval()
    print(f'  Model loaded: {sum((p.numel() for p in model.parameters())):,} params')
    if 'epoch' in ckpt:
        print(f'  Checkpoint epoch: {ckpt['epoch']}')
    if 'best_map50' in ckpt:
        print(f'  Checkpoint best_map50: {ckpt['best_map50']:.4f}')
    elif 'val_map50' in ckpt:
        print(f'  Checkpoint val_map50: {ckpt['val_map50']:.4f}')
    return model

def estimate_seq_density(seq_dir: Path, n_sample: int=DENSITY_SAMPLE_FRAMES) -> float:
    frame_paths = sorted(seq_dir.glob('frame_*.npz'))
    if not frame_paths:
        return 0.0
    n = min(n_sample, len(frame_paths))
    idxs = np.linspace(0, len(frame_paths) - 1, n, dtype=int)
    totals = []
    for i in idxs:
        try:
            d = np.load(frame_paths[int(i)])
            if 'n_events' in d.files:
                totals.append(float(int(d['n_events'])))
            else:
                totals.append(float(d['coords'].shape[0]))
        except Exception:
            continue
    if not totals:
        return 0.0
    return float(np.mean(totals))

def density_tier_for(events_per_frame: float) -> str:
    for name, lo, hi in DENSITY_TIERS:
        if lo <= events_per_frame < hi:
            return name
    return DENSITY_TIERS[-1][0]

def density_tier_summary(results: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for tier_name, _lo, _hi in DENSITY_TIERS:
        vals = [r['mAP_50'] for r in results if r.get('density_tier') == tier_name and (not np.isnan(r['mAP_50']))]
        if not vals:
            summary[tier_name] = {'count': 0, 'mean': None, 'median': None, 'min': None, 'p10': None, 'p90': None, 'max': None}
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[tier_name] = {'count': int(arr.size), 'mean': float(arr.mean()), 'median': float(np.median(arr)), 'min': float(arr.min()), 'p10': float(np.percentile(arr, 10)), 'p90': float(np.percentile(arr, 90)), 'max': float(arr.max())}
    return summary

@torch.no_grad()

def evaluate_sequence(model: SparseVoxelDet, seq_id: str, device: torch.device, sparse_root: Path, label_dir: Path, use_amp: bool=True) -> Dict:
    ds = TestDataset(sparse_root, label_dir, sequence_id=seq_id)
    if len(ds) == 0:
        print(f'  [{seq_id}] No samples found — skipping')
        return {'seq_id': seq_id, 'mAP_50': float('nan'), 'n_frames': 0}
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=device.type == 'cuda', persistent_workers=False)
    calculator = MAPCalculator(num_classes=1, img_size=INPUT_SIZE, conf_threshold=0.001, max_predictions_per_image=MAX_DETECTIONS)
    model.set_decode_params(score_thresh=SCORE_THRESH, nms_thresh=NMS_THRESH, max_detections=MAX_DETECTIONS)
    for batch in loader:
        try:
            sparse_input = spconv.SparseConvTensor(features=batch['feats'].to(device), indices=batch['coords'].to(device), spatial_shape=batch['spatial_shape'], batch_size=batch['batch_size'])
            with autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                outputs = model(sparse_input, batch_size=batch['batch_size'])
            detections = outputs['detections']
            bs = batch['batch_size']
            pred_list = []
            for b in range(bs):
                dets = detections[b]
                mask = dets[:, 4] > 0
                pred_list.append(dets[mask])
            calculator.update(pred_list, batch['gt_labels_yolo'])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                print(f'  [{seq_id}] OOM — skipping batch')
                continue
            raise
    metrics = calculator.compute()
    return {'seq_id': seq_id, 'mAP_50': metrics.mAP_50 * 100.0, 'mAP_50_95': metrics.mAP_50_95 * 100.0, 'precision': metrics.precision * 100.0, 'recall': metrics.recall * 100.0, 'f1': metrics.f1, 'n_frames': len(ds), 'n_preds': metrics.total_predictions, 'n_gts': metrics.total_ground_truths}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Per-sequence mAP@50 evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: V1 best)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--split', type=str, default=DEFAULT_SPLIT, choices=list(VALID_SPLITS), help=f'Dataset split (default: {DEFAULT_SPLIT}). canonical_train=184 seqs, canonical_test=47 seqs.')
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT
    sparse_root = SPARSE_SPLIT_ROOT / args.split
    label_dir = LABEL_SPLIT_ROOT / args.split
    print('=' * 70)
    print(f'Per-Sequence mAP@50 Evaluation  |  split={args.split}')
    print('=' * 70)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(args.gpu)}')
    if not checkpoint_path.exists():
        raise RuntimeError(f'Checkpoint not found: {checkpoint_path}')
    if not sparse_root.exists():
        raise RuntimeError(f'Sparse data not found: {sparse_root}')
    if not label_dir.exists():
        raise RuntimeError(f'Label dir not found: {label_dir}')
    print(f'\n  Checkpoint : {checkpoint_path}')
    print(f'  Sparse root: {sparse_root}')
    print(f'  Labels     : {label_dir}')
    print('\n[1] Loading model...')
    model = load_model(checkpoint_path, device)
    def _seq_sort_key(name: str):
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)
    seq_ids = sorted([d.name for d in sparse_root.iterdir() if d.is_dir()], key=_seq_sort_key)
    print(f"\n[2] Found {len(seq_ids)} sequences in split '{args.split}'.")
    print(f'\n[3] Evaluating {len(seq_ids)} sequences (batch_size={BATCH_SIZE})...')
    print('-' * 70)
    results = []
    for i, seq_id in enumerate(seq_ids):
        r = evaluate_sequence(model, seq_id, device, sparse_root, label_dir, use_amp=True)
        seq_dir = sparse_root / seq_id
        epf = estimate_seq_density(seq_dir, n_sample=DENSITY_SAMPLE_FRAMES)
        r['avg_events_per_frame'] = round(epf, 1)
        r['density_tier'] = density_tier_for(epf)
        results.append(r)
        status = f'{r['mAP_50']:.2f}%' if not np.isnan(r['mAP_50']) else 'N/A'
        print(f'  [{i + 1:3d}/{len(seq_ids)}] seq={seq_id:>6s}  mAP@50={status:>8s}  frames={r['n_frames']:5d}  preds={r.get('n_preds', 0):6d}  gts={r.get('n_gts', 0):6d}  P={r.get('precision', 0):.1f}%  R={r.get('recall', 0):.1f}%  epf={epf:>7.0f}  tier={r['density_tier']}')
    valid = [r for r in results if not np.isnan(r['mAP_50'])]
    valid_sorted = sorted(valid, key=lambda x: x['mAP_50'])
    total_frames = sum((r['n_frames'] for r in valid))
    if total_frames > 0:
        weighted_map = sum((r['mAP_50'] * r['n_frames'] for r in valid)) / total_frames
    else:
        weighted_map = 0.0
    simple_mean = float(np.mean([r['mAP_50'] for r in valid])) if valid else 0.0
    print('\n' + '=' * 70)
    print('RESULTS — Per-Sequence mAP@50 (sorted worst → best)')
    print('=' * 70)
    print(f'  {'Rank':<5}  {'Seq':<8}  {'mAP@50':>8}  {'Frames':>7}  {'Preds':>7}  {'GTs':>6}  {'Recall':>7}  {'Prec':>7}')
    print(f'  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 6}  {'-' * 7}  {'-' * 7}')
    for rank, r in enumerate(valid_sorted, 1):
        print(f'  {rank:<5d}  {r['seq_id']:<8s}  {r['mAP_50']:>7.2f}%  {r['n_frames']:>7d}  {r.get('n_preds', 0):>7d}  {r.get('n_gts', 0):>6d}  {r.get('recall', 0):>6.1f}%  {r.get('precision', 0):>6.1f}%')
    print(f'\n  Overall mean mAP@50  (simple)  : {simple_mean:.2f}%')
    print(f'  Overall mean mAP@50  (weighted): {weighted_map:.2f}%')
    print('\n' + '=' * 70)
    print('10 WORST sequences:')
    print('=' * 70)
    for r in valid_sorted[:10]:
        print(f'  seq={r['seq_id']:>6s}  mAP@50={r['mAP_50']:>7.2f}%  frames={r['n_frames']:4d}  recall={r.get('recall', 0):.1f}%  prec={r.get('precision', 0):.1f}%')
    print('\n' + '=' * 70)
    print('10 BEST sequences:')
    print('=' * 70)
    for r in valid_sorted[-10:][::-1]:
        print(f'  seq={r['seq_id']:>6s}  mAP@50={r['mAP_50']:>7.2f}%  frames={r['n_frames']:4d}  recall={r.get('recall', 0):.1f}%  prec={r.get('precision', 0):.1f}%')
    tier_summary = density_tier_summary(valid_sorted)
    print('\n' + '=' * 70)
    print('DENSITY-TIER SUMMARY (mAP@50 aggregated per tier)')
    print('=' * 70)
    print(f'  {'Tier':<12}  {'Count':>6}  {'Mean':>8}  {'Median':>8}  {'Min':>8}  {'p10':>8}  {'p90':>8}  {'Max':>8}')
    print(f'  {'-' * 12}  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}')
    for tier_name, _lo, _hi in DENSITY_TIERS:
        s = tier_summary[tier_name]
        if s['count'] == 0:
            print(f'  {tier_name:<12}  {s['count']:>6}  (empty)')
            continue
        print(f'  {tier_name:<12}  {s['count']:>6d}  {s['mean']:>7.2f}%  {s['median']:>7.2f}%  {s['min']:>7.2f}%  {s['p10']:>7.2f}%  {s['p90']:>7.2f}%  {s['max']:>7.2f}%')
    import json
    ckpt_stem = checkpoint_path.stem
    if args.split == DEFAULT_SPLIT:
        out_name = f'eval_per_sequence_{ckpt_stem}.json'
    else:
        out_name = f'eval_per_sequence_{ckpt_stem}_{args.split}.json'
    out_path = checkpoint_path.parent / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'checkpoint': str(checkpoint_path), 'split': args.split, 'n_sequences': len(valid_sorted), 'overall_mean_mAP50': simple_mean, 'overall_weighted_mAP50': weighted_map, 'density_tier_summary': tier_summary, 'sequences': valid_sorted}, f, indent=2)
    print(f'\n  Results saved to: {out_path}')
    print('Done.')
if __name__ == '__main__':
    main()
