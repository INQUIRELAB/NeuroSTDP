from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
DEFAULT_SENSOR_H = 260
DEFAULT_SENSOR_W = 346
DEFAULT_FRAME_DT_MS = 20.0
DEFAULT_HOT_THRESHOLD_FRAC = 0.2

def compute_hot_pixel_mask(ev: Dict[str, np.ndarray], threshold_frac: float=DEFAULT_HOT_THRESHOLD_FRAC, sensor_h: int=DEFAULT_SENSOR_H, sensor_w: int=DEFAULT_SENSOR_W, dt_ms: float=DEFAULT_FRAME_DT_MS) -> Tuple[np.ndarray, int, int]:
    xs, ys, ts = (ev['x'], ev['y'], ev['t'])
    ok = (xs >= 0) & (xs < sensor_w) & (ys >= 0) & (ys < sensor_h)
    xs, ys, ts = (xs[ok], ys[ok], ts[ok])
    t_min, t_max = (float(ts.min()), float(ts.max()))
    n_frames = int(np.ceil((t_max - t_min) / dt_ms)) + 1
    frame_idx = ((ts - t_min) / dt_ms).astype(np.int64)
    pixel_id = ys.astype(np.int64) * sensor_w + xs.astype(np.int64)
    pair = frame_idx * (sensor_h * sensor_w) + pixel_id
    u_pair = np.unique(pair)
    u_pixel = u_pair % (sensor_h * sensor_w)
    counts = np.bincount(u_pixel, minlength=sensor_h * sensor_w)
    counts_2d = counts.reshape(sensor_h, sensor_w)
    hot = counts_2d > threshold_frac * n_frames
    return (hot, int(hot.sum()), n_frames)

def filter_events_by_hot_mask(ev: Dict[str, np.ndarray], hot_mask: np.ndarray) -> Dict[str, np.ndarray]:
    xs, ys = (ev['x'], ev['y'])
    sensor_h, sensor_w = hot_mask.shape
    in_bounds = (xs >= 0) & (xs < sensor_w) & (ys >= 0) & (ys < sensor_h)
    keep = np.ones(len(xs), dtype=bool)
    ib_idx = np.nonzero(in_bounds)[0]
    is_hot = hot_mask[ys[ib_idx], xs[ib_idx]]
    keep[ib_idx] = ~is_hot
    return {k: v[keep] if hasattr(v, '__len__') and len(v) == len(xs) else v for k, v in ev.items()}
