#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, Iterable, List, Tuple
import numpy as np
from stdp_reliability_gate import evaluate_tube_sequence
BBox = Tuple[int, int, int, int]

@dataclass

class TubeCandidate:
    frame: int
    det_index: int
    bbox: BBox
    score: float
    centroid: Tuple[float, float]
    area: float
    channel_id: str = 'unknown'
    channel_rank: float = 0.0
    fusion_support: int = 0
    fusion_channel_count: int = 0
    fusion_mean_rank: float = 0.0

@dataclass

class EventTube:
    tube_id: int
    candidates: List[TubeCandidate] = field(default_factory=list)
    gap_count: int = 0
    reliability: float = 0.0
    mode: str = 'uncertain'
    velocity: Tuple[float, float] = (0.0, 0.0)
    mean_residual: float = 0.0
    @property
    def last(self) -> TubeCandidate:
        return self.candidates[-1]
    @property
    def length(self) -> int:
        return len(self.candidates)
    @property
    def span(self) -> int:
        if not self.candidates:
            return 0
        return self.candidates[-1].frame - self.candidates[0].frame + 1

def _clamp_box(box: BBox, sensor_w: int, sensor_h: int, min_size: int=1) -> BBox:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(sensor_w - min_size, x1)))
    y1 = int(max(0, min(sensor_h - min_size, y1)))
    x2 = int(max(x1 + min_size, min(sensor_w, x2)))
    y2 = int(max(y1 + min_size, min(sensor_h, y2)))
    return (x1, y1, x2, y2)

def _pad_box(box: BBox, pad: float, sensor_w: int, sensor_h: int) -> BBox:
    x1, y1, x2, y2 = box
    return _clamp_box((int(np.floor(x1 - pad)), int(np.floor(y1 - pad)), int(np.ceil(x2 + pad)), int(np.ceil(y2 + pad))), sensor_w, sensor_h)

def _box_area(box: BBox) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

def _centroid(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    denom = _box_area(a) + _box_area(b) - inter
    return inter / denom if denom > 0 else 0.0

def _to_candidates(per_frame_dets: Dict[int, List[dict]]) -> Dict[int, List[TubeCandidate]]:
    out: Dict[int, List[TubeCandidate]] = {}
    for frame, dets in per_frame_dets.items():
        frame_i = int(frame)
        converted = []
        for idx, det in enumerate(dets):
            bbox = tuple((int(v) for v in det['bbox']))
            converted.append(TubeCandidate(frame=frame_i, det_index=idx, bbox=bbox, score=float(det.get('score', 1.0)), centroid=tuple((float(v) for v in det.get('centroid', _centroid(bbox)))), area=max(_box_area(bbox), 1.0), channel_id=str(det.get('channel_id', 'unknown')), channel_rank=float(det.get('channel_rank', 0.0)), fusion_support=int(det.get('fusion_support', 0)), fusion_channel_count=int(det.get('fusion_channel_count', 0)), fusion_mean_rank=float(det.get('fusion_mean_rank', det.get('channel_rank', 0.0)))))
        if converted:
            out[frame_i] = converted
    return out

def link_event_tubes(per_frame_dets: Dict[int, List[dict]], max_gap: int=2, max_link_dist: float=12.0, area_ratio_limit: float=5.0) -> List[EventTube]:
    candidates_by_frame = _to_candidates(per_frame_dets)
    active: List[EventTube] = []
    finished: List[EventTube] = []
    next_id = 0
    for frame in sorted(candidates_by_frame):
        frame_candidates = sorted(candidates_by_frame[frame], key=lambda c: -c.score)
        assigned_cands = set()
        assigned_tubes = set()
        proposals = []
        for ti, tube in enumerate(active):
            dt = max(1, frame - tube.last.frame)
            if dt > max_gap + 1:
                continue
            pred_x = tube.last.centroid[0] + tube.velocity[0] * dt
            pred_y = tube.last.centroid[1] + tube.velocity[1] * dt
            for ci, cand in enumerate(frame_candidates):
                dx = cand.centroid[0] - pred_x
                dy = cand.centroid[1] - pred_y
                dist = float(np.hypot(dx, dy))
                if dist > max_link_dist + 2.0 * (dt - 1):
                    continue
                ratio = max(cand.area, tube.last.area) / max(1.0, min(cand.area, tube.last.area))
                if ratio > area_ratio_limit:
                    continue
                overlap_bonus = 1.0 - _iou(tube.last.bbox, cand.bbox)
                cost = dist + 3.0 * max(0.0, np.log(ratio)) + 2.0 * overlap_bonus + 2.5 * (dt - 1)
                proposals.append((cost, ti, ci))
        for _cost, ti, ci in sorted(proposals, key=lambda x: x[0]):
            if ti in assigned_tubes or ci in assigned_cands:
                continue
            tube = active[ti]
            cand = frame_candidates[ci]
            prev = tube.last
            dt = max(1, cand.frame - prev.frame)
            vx = (cand.centroid[0] - prev.centroid[0]) / dt
            vy = (cand.centroid[1] - prev.centroid[1]) / dt
            tube.velocity = (0.65 * tube.velocity[0] + 0.35 * vx, 0.65 * tube.velocity[1] + 0.35 * vy)
            tube.candidates.append(cand)
            tube.gap_count += max(0, dt - 1)
            assigned_tubes.add(ti)
            assigned_cands.add(ci)
        still_active = []
        for ti, tube in enumerate(active):
            if ti in assigned_tubes:
                still_active.append(tube)
                continue
            if frame - tube.last.frame <= max_gap:
                still_active.append(tube)
            else:
                finished.append(tube)
        active = still_active
        for ci, cand in enumerate(frame_candidates):
            if ci in assigned_cands:
                continue
            active.append(EventTube(tube_id=next_id, candidates=[cand]))
            next_id += 1
    finished.extend(active)
    _score_tubes(finished)
    return finished

def _score_tubes(tubes: Iterable[EventTube]) -> None:
    for tube in tubes:
        if tube.length <= 1:
            tube.reliability = 0.2
            tube.mode = 'single'
            tube.mean_residual = 0.0
            continue
        frames = np.asarray([c.frame for c in tube.candidates], dtype=np.float64)
        cxs = np.asarray([c.centroid[0] for c in tube.candidates], dtype=np.float64)
        cys = np.asarray([c.centroid[1] for c in tube.candidates], dtype=np.float64)
        areas = np.asarray([c.area for c in tube.candidates], dtype=np.float64)
        if len(np.unique(frames)) >= 2:
            vx, x0 = np.polyfit(frames, cxs, 1)
            vy, y0 = np.polyfit(frames, cys, 1)
            pred_x = vx * frames + x0
            pred_y = vy * frames + y0
            residual = np.sqrt((cxs - pred_x) ** 2 + (cys - pred_y) ** 2)
            mean_residual = float(np.mean(residual))
            speed = float(np.hypot(vx, vy))
        else:
            vx = vy = speed = mean_residual = 0.0
        tube.velocity = (float(vx), float(vy))
        tube.mean_residual = mean_residual
        if speed < 0.25:
            tube.mode = 'stationary'
        elif mean_residual < 4.0:
            tube.mode = 'constant_velocity'
        else:
            tube.mode = 'uncertain'
        span = max(tube.span, 1)
        continuity = tube.length / span
        age_score = min(1.0, tube.length / 5.0)
        residual_score = float(np.exp(-mean_residual / 8.0))
        area_cv = float(np.std(areas) / max(np.mean(areas), 1.0))
        area_score = float(np.exp(-area_cv))
        score_cv = float(np.std([c.score for c in tube.candidates]) / (np.mean([c.score for c in tube.candidates]) + 1e-06))
        score_stability = float(np.exp(-score_cv))
        tube.reliability = float(np.clip(0.3 * age_score + 0.25 * continuity + 0.2 * residual_score + 0.15 * area_score + 0.1 * score_stability, 0.0, 1.0))

def _build_frame_event_cache(ev: dict, t_min: float, dt_ms: float) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    frames = np.floor((ev['t'] - t_min) / dt_ms).astype(np.int64)
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for frame in np.unique(frames):
        mask = frames == frame
        cache[int(frame)] = (ev['x'][mask].astype(np.float64), ev['y'][mask].astype(np.float64))
    return cache

def _events_in_frame_box(frame_cache: Dict[int, Tuple[np.ndarray, np.ndarray]], frame: int, bbox: BBox) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    if frame not in frame_cache:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    xs, ys = frame_cache[frame]
    mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return (xs[mask], ys[mask])

def _quantile_box(xs: np.ndarray, ys: np.ndarray, sensor_w: int, sensor_h: int, q_low: float, q_high: float, pad: float, min_size: int) -> BBox | None:
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = np.quantile(xs, q_low) - pad
    y1 = np.quantile(ys, q_low) - pad
    x2 = np.quantile(xs, q_high) + pad + 1.0
    y2 = np.quantile(ys, q_high) + pad + 1.0
    return _clamp_box((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))), sensor_w, sensor_h, min_size)

def _refine_candidate_box(tube: EventTube, cand: TubeCandidate, frame_cache: Dict[int, Tuple[np.ndarray, np.ndarray]], sensor_w: int, sensor_h: int, support_pad: float, q_low: float, q_high: float, min_support_events: int, min_box_size: int, motion_compensate: bool) -> BBox:
    xs_all = []
    ys_all = []
    for other in tube.candidates:
        search_box = _pad_box(other.bbox, support_pad, sensor_w, sensor_h)
        xs, ys = _events_in_frame_box(frame_cache, other.frame, search_box)
        if len(xs) == 0:
            continue
        if motion_compensate and tube.length >= 3:
            dt = cand.frame - other.frame
            xs = xs + tube.velocity[0] * dt
            ys = ys + tube.velocity[1] * dt
        xs_all.append(xs)
        ys_all.append(ys)
    if not xs_all:
        return cand.bbox
    xs_cat = np.concatenate(xs_all)
    ys_cat = np.concatenate(ys_all)
    if len(xs_cat) < min_support_events:
        return cand.bbox
    refined = _quantile_box(xs_cat, ys_cat, sensor_w, sensor_h, q_low, q_high, pad=1.0, min_size=min_box_size)
    if refined is None:
        return cand.bbox
    raw_area = max(_box_area(cand.bbox), 1.0)
    ref_area = max(_box_area(refined), 1.0)
    area_ratio = ref_area / raw_area
    if area_ratio < 0.2 or area_ratio > 3.5:
        return cand.bbox
    cx_raw, cy_raw = cand.centroid
    cx_ref, cy_ref = _centroid(refined)
    if np.hypot(cx_ref - cx_raw, cy_ref - cy_raw) > max(10.0, np.sqrt(raw_area) * 1.5):
        return cand.bbox
    return refined

def _tube_support_stats(tube: EventTube, frame_cache: Dict[int, Tuple[np.ndarray, np.ndarray]], sensor_w: int, sensor_h: int, support_pad: float) -> Tuple[float, float]:
    counts = []
    densities = []
    for cand in tube.candidates:
        search_box = _pad_box(cand.bbox, support_pad, sensor_w, sensor_h)
        xs, _ys = _events_in_frame_box(frame_cache, cand.frame, search_box)
        area = max(_box_area(search_box), 1.0)
        counts.append(float(len(xs)))
        densities.append(float(len(xs)) / area)
    if not counts:
        return (0.0, 0.0)
    return (float(np.mean(counts)), float(np.mean(densities)))

def _tube_feature_row(tube: EventTube, frame_cache: Dict[int, Tuple[np.ndarray, np.ndarray]], sensor_w: int, sensor_h: int, support_pad: float) -> dict:
    areas = np.asarray([c.area for c in tube.candidates], dtype=np.float64)
    scores = np.asarray([c.score for c in tube.candidates], dtype=np.float64)
    fusion_channel_counts = np.asarray([c.fusion_channel_count for c in tube.candidates], dtype=np.float64)
    fusion_support = np.asarray([c.fusion_support for c in tube.candidates], dtype=np.float64)
    fusion_ranks = np.asarray([c.fusion_mean_rank for c in tube.candidates], dtype=np.float64)
    speed = float(np.hypot(tube.velocity[0], tube.velocity[1]))
    span = max(tube.span, 1)
    gap_fraction = max(0.0, (span - tube.length) / span)
    continuity = tube.length / span
    area_cv = float(np.std(areas) / max(np.mean(areas), 1.0)) if len(areas) else 1.0
    score_cv = float(np.std(scores) / (np.mean(scores) + 1e-06)) if len(scores) else 1.0
    mean_support, support_density = _tube_support_stats(tube, frame_cache, sensor_w, sensor_h, support_pad)
    median_area_frac = float(np.median(areas) / max(float(sensor_w * sensor_h), 1.0)) if len(areas) else 1.0
    known_channel = fusion_channel_counts > 0
    if np.any(known_channel):
        channel_agreement = float(np.mean(np.clip((fusion_channel_counts[known_channel] - 1.0) / 3.0, 0.0, 1.0)))
        isolated = float(np.mean(fusion_channel_counts[known_channel] <= 1.0))
        fusion_support_score = float(np.clip(np.log1p(np.mean(fusion_support[known_channel])) / np.log1p(4.0), 0.0, 1.0))
        rank_support = float(np.clip(np.mean(fusion_ranks[known_channel]), 0.0, 1.0))
    else:
        channel_agreement = 0.0
        isolated = 0.0
        fusion_support_score = 0.0
        rank_support = 0.0
    return {'age': float(np.clip(tube.length / 5.0, 0.0, 1.0)), 'continuity': float(np.clip(continuity, 0.0, 1.0)), 'smooth_motion': float(np.exp(-tube.mean_residual / 8.0)), 'stable_area': float(np.exp(-area_cv)), 'stable_score': float(np.exp(-score_cv)), 'support': float(np.clip(np.log1p(mean_support) / np.log1p(80.0), 0.0, 1.0)), 'support_density': float(np.clip(support_density / 0.2, 0.0, 1.0)), 'channel_agreement': channel_agreement, 'fusion_support': fusion_support_score, 'rank_support': rank_support, 'isolated': isolated, 'moving': float(np.clip(speed / 4.0, 0.0, 1.0)), 'stationary': float(np.clip(1.0 - speed / 1.25, 0.0, 1.0)), 'gap_free': float(np.clip(1.0 - gap_fraction, 0.0, 1.0)), 'small_area': float(np.clip(1.0 - median_area_frac / 0.015, 0.0, 1.0)), 'large_area': float(np.clip((median_area_frac - 0.015) / 0.05, 0.0, 1.0)), 'high_score': float(np.clip(np.mean(scores), 0.0, 1.0)) if len(scores) else 0.0}

def refine_detections_with_tubes(per_frame_dets: Dict[int, List[dict]], ev: dict, sensor_shape: Tuple[int, int], dt_ms: float, min_tube_len: int=3, max_gap: int=2, max_link_dist: float=12.0, support_pad: float=3.0, q_low: float=0.05, q_high: float=0.95, min_support_events: int=6, min_box_size: int=2, motion_compensate: bool=False, box_refine_mode: str='support', filter_short: bool=False, score_weight: float=0.0, stdp_gate: bool=False, stdp_profile: str='balanced', stdp_threshold: float=0.5) -> Tuple[Dict[int, List[dict]], dict]:
    if not per_frame_dets:
        return (per_frame_dets, {'n_tubes': 0, 'n_tube_dets': 0, 'n_refined': 0})
    sensor_h, sensor_w = sensor_shape
    int_keyed = {int(k): list(v) for k, v in per_frame_dets.items()}
    tubes = link_event_tubes(int_keyed, max_gap=max_gap, max_link_dist=max_link_dist)
    t_min = float(ev['t'].min())
    frame_cache = _build_frame_event_cache(ev, t_min, dt_ms)
    out: Dict[int, List[dict]] = {}
    n_refined = 0
    n_kept = 0
    if box_refine_mode not in {'none', 'support', 'motion', 'auto'}:
        raise ValueError(f'unknown box_refine_mode={box_refine_mode!r}')
    if stdp_profile not in {'balanced', 'precision'}:
        raise ValueError(f'unknown stdp_profile={stdp_profile!r}')
    tube_gate: Dict[int, dict] = {}
    if stdp_gate:
        feature_rows = [_tube_feature_row(tube, frame_cache, sensor_w, sensor_h, support_pad) for tube in tubes]
        gate_results = evaluate_tube_sequence(feature_rows, profile=stdp_profile, learn=True)
        for tube, row, gate in zip(tubes, feature_rows, gate_results):
            keep = gate.reliability >= stdp_threshold and gate.state in {'real_moving', 'real_stationary', 'ambiguous'}
            if tube.length < min_tube_len:
                keep = False
            tube_gate[tube.tube_id] = {'features': row, 'state': gate.state, 'reliability': gate.reliability, 'output_spikes': gate.output_spikes, 'positive_evidence': gate.positive_evidence, 'negative_evidence': gate.negative_evidence, 'keep': keep}
    for tube in tubes:
        use_refinement = tube.length >= min_tube_len
        tube_frames = [c.frame for c in tube.candidates]
        tube_span = max(tube.span, 1)
        tube_gap_fraction = max(0.0, (tube_span - tube.length) / tube_span)
        if stdp_gate:
            gate_info = tube_gate[tube.tube_id]
            if filter_short and (not gate_info['keep']):
                continue
            reliability = float(gate_info['reliability'])
        else:
            if filter_short and (not use_refinement):
                continue
            reliability = tube.reliability if use_refinement else min(tube.reliability, 0.35)
        score_multiplier = 1.0 - score_weight + score_weight * max(0.05, reliability)
        for cand in tube.candidates:
            det = dict(int_keyed[cand.frame][cand.det_index])
            original_box = tuple((int(v) for v in det['bbox']))
            original_score = float(det.get('score', 1.0))
            new_box = original_box
            if use_refinement and box_refine_mode != 'none':
                use_motion = motion_compensate or box_refine_mode == 'motion'
                if box_refine_mode == 'support':
                    use_motion = False
                elif box_refine_mode == 'auto':
                    use_motion = tube.mode == 'constant_velocity' and tube.length >= max(5, min_tube_len) and (tube.reliability >= 0.75) and (tube.mean_residual <= 3.0)
                new_box = _refine_candidate_box(tube, cand, frame_cache, sensor_w, sensor_h, support_pad, q_low, q_high, min_support_events, min_box_size, use_motion)
            if new_box != original_box:
                n_refined += 1
            det['bbox'] = new_box
            det['centroid'] = _centroid(new_box)
            det['score_original'] = original_score
            det['score'] = original_score * score_multiplier
            det['tube_id'] = tube.tube_id
            det['tube_len'] = tube.length
            det['tube_span'] = tube_span
            det['tube_start_frame'] = int(min(tube_frames)) if tube_frames else int(cand.frame)
            det['tube_end_frame'] = int(max(tube_frames)) if tube_frames else int(cand.frame)
            det['tube_gap_fraction'] = float(tube_gap_fraction)
            det['tube_mode'] = tube.mode
            det['tube_reliability'] = reliability
            det['tube_velocity_x'] = float(tube.velocity[0])
            det['tube_velocity_y'] = float(tube.velocity[1])
            det['tube_mean_residual'] = float(tube.mean_residual)
            if stdp_gate:
                gate_info = tube_gate[tube.tube_id]
                det['tube_stdp_state'] = gate_info['state']
                det['tube_stdp_spikes'] = gate_info['output_spikes']
                det['tube_stdp_positive_evidence'] = gate_info['positive_evidence']
                det['tube_stdp_negative_evidence'] = gate_info['negative_evidence']
            out.setdefault(cand.frame, []).append(det)
            n_kept += 1
    for frame, dets in out.items():
        dets.sort(key=lambda d: -float(d.get('score', 1.0)))
    summary = {'n_tubes': len(tubes), 'n_tube_dets': n_kept, 'n_refined': n_refined, 'n_long_tubes': sum((1 for t in tubes if t.length >= min_tube_len)), 'mean_tube_len': float(np.mean([t.length for t in tubes])) if tubes else 0.0, 'mean_reliability': float(np.mean([t.reliability for t in tubes])) if tubes else 0.0, 'stdp_gate': bool(stdp_gate)}
    if stdp_gate:
        states = Counter((info['state'] for info in tube_gate.values()))
        summary.update({'stdp_profile': stdp_profile, 'stdp_threshold': stdp_threshold, 'stdp_state_counts': dict(states), 'stdp_kept_tubes': sum((1 for info in tube_gate.values() if info['keep'])), 'stdp_mean_reliability': float(np.mean([info['reliability'] for info in tube_gate.values()])) if tube_gate else 0.0})
    return (out, summary)
