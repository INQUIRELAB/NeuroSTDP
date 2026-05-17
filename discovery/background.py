from __future__ import annotations
import numpy as np

def build_background_mask(frame_loader, frame_paths: list, spatial_shape: tuple[int, int], n_samples: int=80, temporal_threshold: float=0.25, spatial_radius: int=1, spatial_count_threshold: int=3, require_scattered: bool=False, max_gap_frames: int=30) -> tuple[np.ndarray, dict]:
    H, W = spatial_shape
    n_samples = min(n_samples, len(frame_paths))
    sample_indices = np.linspace(0, len(frame_paths) - 1, n_samples, dtype=int)
    occupancy = np.zeros((H, W), dtype=np.int32)
    activity_stack = None
    if require_scattered:
        activity_stack = np.zeros((n_samples, H, W), dtype=np.bool_)
    for si, idx in enumerate(sample_indices):
        coords, feats, n_events, _ = frame_loader(frame_paths[idx])
        if coords.shape[0] == 0:
            continue
        ys = coords[:, 1].astype(np.int32)
        xs = coords[:, 2].astype(np.int32)
        frame_occ = np.zeros((H, W), dtype=np.bool_)
        frame_occ[ys, xs] = True
        occupancy += frame_occ.astype(np.int32)
        if require_scattered:
            activity_stack[si] = frame_occ
    threshold_count = int(n_samples * temporal_threshold)
    persistent = occupancy > threshold_count
    n_persistent = int(persistent.sum())
    n_scattered_filtered = 0
    if require_scattered and activity_stack is not None and (n_persistent > 0):
        ys_p, xs_p = np.where(persistent)
        pix_activity = activity_stack[:, ys_p, xs_p]
        inactive = ~pix_activity
        n_pix = pix_activity.shape[1]
        max_gap = np.zeros(n_pix, dtype=np.int32)
        for j in range(n_pix):
            col = inactive[:, j]
            if not col.any():
                max_gap[j] = 0
                continue
            padded = np.concatenate(([0], col.astype(np.int8), [0]))
            diff = np.diff(padded)
            run_starts = np.where(diff == 1)[0]
            run_ends = np.where(diff == -1)[0]
            if len(run_starts) == 0:
                max_gap[j] = 0
            else:
                max_gap[j] = (run_ends - run_starts).max()
        keep_bg = max_gap < max_gap_frames
        n_scattered_filtered = int((~keep_bg).sum())
        new_persistent = np.zeros_like(persistent)
        spared_ys = ys_p[keep_bg]
        spared_xs = xs_p[keep_bg]
        new_persistent[spared_ys, spared_xs] = True
        persistent = new_persistent
    if spatial_radius > 0 and spatial_count_threshold > 0:
        from scipy.ndimage import uniform_filter
        kernel_size = 2 * spatial_radius + 1
        neighbor_count = uniform_filter(persistent.astype(np.float64), size=kernel_size, mode='constant') * kernel_size ** 2
        spatial_consistent = neighbor_count >= spatial_count_threshold
        mask = persistent | spatial_consistent
    else:
        mask = persistent
    stats = {'n_samples': int(n_samples), 'threshold_count': threshold_count, 'n_persistent': n_persistent, 'n_scattered_spared': n_scattered_filtered, 'n_final': int(mask.sum())}
    return (mask, stats)
