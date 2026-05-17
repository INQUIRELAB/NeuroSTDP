from __future__ import annotations
import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label, maximum_filter, zoom
from skimage.segmentation import watershed

def events_to_histogram(coords: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'coords must be [N, 3] (time_bin, y, x), got shape {coords.shape}')
    H, W = spatial_shape
    if H <= 0 or W <= 0:
        raise ValueError(f'spatial_shape must be positive, got ({H}, {W})')
    if coords.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float64)
    y = coords[:, 1]
    x = coords[:, 2]
    if np.any(y < 0) or np.any(y >= H) or np.any(x < 0) or np.any(x >= W):
        raise ValueError(f'coords contain out-of-bounds values for spatial_shape ({H}, {W}): y in [{y.min()}, {y.max()}], x in [{x.min()}, {x.max()}]')
    linear = y.astype(np.int64) * W + x.astype(np.int64)
    counts = np.bincount(linear, minlength=H * W)
    return counts.reshape(H, W).astype(np.float64)

def multi_scale_peaks(density_maps: dict[float, np.ndarray], suppress_radius: float | None=None) -> list[tuple[int, int, float, float]]:
    if not density_maps:
        raise ValueError('density_maps must not be empty')
    sigmas = sorted(density_maps.keys())
    shapes = {s: d.shape for s, d in density_maps.items()}
    ref_shape = shapes[sigmas[0]]
    for s, shape in shapes.items():
        if shape != ref_shape:
            raise ValueError(f'Inconsistent density map shapes: sigma={sigmas[0]} has {ref_shape}, sigma={s} has {shape}')
    if suppress_radius is None:
        suppress_radius = 2.0 * max(sigmas)
    all_peaks: list[tuple[int, int, float, float]] = []
    for sigma in sigmas:
        D = density_maps[sigma]
        nonzero_vals = D[D > 0]
        if len(nonzero_vals) == 0:
            continue
        p95 = np.percentile(nonzero_vals, 95)
        d_max = float(D.max())
        tau = max(p95 * 0.1, d_max * 0.01)
        kernel_size = max(3, int(2 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        D_f32 = D.astype(np.float32) if D.dtype != np.float32 else D
        local_max = maximum_filter(D_f32, size=kernel_size)
        peak_mask = (D_f32 == local_max) & (D_f32 >= tau)
        peak_ys, peak_xs = np.nonzero(peak_mask)
        peak_scores = D[peak_ys, peak_xs]
        for py, px, sc in zip(peak_ys, peak_xs, peak_scores):
            all_peaks.append((int(py), int(px), float(sc), sigma))
    if not all_peaks:
        return []
    all_peaks.sort(key=lambda p: -p[2])
    n = len(all_peaks)
    if n == 1:
        return all_peaks
    coords_arr = np.array([(p[0], p[1]) for p in all_peaks], dtype=np.float64)
    r2 = suppress_radius * suppress_radius
    kept: list[tuple[int, int, float, float]] = []
    suppressed = np.zeros(n, dtype=bool)
    for i in range(n):
        if suppressed[i]:
            continue
        kept.append(all_peaks[i])
        if i + 1 < n:
            remaining = ~suppressed[i + 1:]
            if remaining.any():
                dy = coords_arr[i + 1:, 0] - coords_arr[i, 0]
                dx = coords_arr[i + 1:, 1] - coords_arr[i, 1]
                dist2 = dy * dy + dx * dx
                suppress_mask = (dist2 <= r2) & remaining
                suppressed[i + 1:] |= suppress_mask
    return kept
_DOWNSAMPLE_SIGMA_THRESHOLD = 12.0

def _compute_density_maps(histogram: np.ndarray, sigmas: list[float]) -> dict[float, np.ndarray]:
    H, W = histogram.shape
    hist_f32 = histogram.astype(np.float32)
    need_half = any((s >= _DOWNSAMPLE_SIGMA_THRESHOLD for s in sigmas))
    hist_half = zoom(hist_f32, 0.5, order=1) if need_half else None
    density_maps: dict[float, np.ndarray] = {}
    for sigma in sigmas:
        if sigma >= _DOWNSAMPLE_SIGMA_THRESHOLD and hist_half is not None:
            d_half = gaussian_filter(hist_half, sigma=sigma / 2.0, truncate=3.0)
            d_full = zoom(d_half, (H / d_half.shape[0], W / d_half.shape[1]), order=1)
            density_maps[sigma] = d_full.astype(np.float64)
        else:
            density_maps[sigma] = gaussian_filter(hist_f32, sigma=sigma, truncate=3.0).astype(np.float64)
    return density_maps

def _watershed_segment(density_combined: np.ndarray, peaks: list[tuple[int, int, float, float]], watershed_threshold_frac: float=0.0, watershed_expand_frac: float=0.0) -> np.ndarray:
    H, W = density_combined.shape
    markers = np.zeros((H, W), dtype=np.int32)
    for i, (py, px, _, _) in enumerate(peaks):
        markers[py, px] = i + 1
    inverted = -density_combined.astype(np.float32)
    if watershed_threshold_frac > 0:
        d_max = float(density_combined.max())
        threshold = d_max * watershed_threshold_frac
        mask = density_combined > threshold
        if watershed_expand_frac > 0:
            threshold_low = d_max * watershed_expand_frac
            mask_low = density_combined > threshold_low
            labeled_low, _ = label(mask_low)
            yy, xx = np.ogrid[:H, :W]
            for i, (py, px, _, sigma) in enumerate(peaks):
                if labeled_low[py, px] > 0:
                    comp_mask = labeled_low == labeled_low[py, px]
                    max_r = max(4.0 * sigma, 40.0)
                    near_peak = (yy - py) ** 2 + (xx - px) ** 2 <= max_r ** 2
                    mask |= comp_mask & near_peak
    else:
        mask = density_combined > 0
    labels = watershed(inverted, markers=markers, mask=mask)
    return labels

def _extract_boxes_from_segments(labels: np.ndarray, histogram: np.ndarray, peaks: list[tuple[int, int, float, float]], min_box: int, max_box: int, min_density: float) -> list[dict]:
    detections: list[dict] = []
    n_labels = len(peaks)
    slices = find_objects(labels)
    for i in range(n_labels):
        if i >= len(slices) or slices[i] is None:
            continue
        sy, sx = slices[i]
        seg_crop = labels[sy, sx] == i + 1
        hist_crop = histogram[sy, sx]
        event_mask = seg_crop & (hist_crop > 0)
        event_ys_local, event_xs_local = np.nonzero(event_mask)
        if len(event_ys_local) == 0:
            continue
        y1 = sy.start + int(event_ys_local.min())
        y2 = sy.start + int(event_ys_local.max()) + 1
        x1 = sx.start + int(event_xs_local.min())
        x2 = sx.start + int(event_xs_local.max()) + 1
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < min_box or box_h < min_box:
            continue
        if box_w > max_box or box_h > max_box:
            continue
        area = box_w * box_h
        n_events = int(hist_crop[seg_crop].sum())
        density = n_events / area
        if density < min_density:
            continue
        score = peaks[i][2]
        peak_xy = (int(peaks[i][1]), int(peaks[i][0]))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        event_weights = hist_crop[event_ys_local, event_xs_local]
        total_weight = float(event_weights.sum())
        if total_weight > 0:
            wcx = sx.start + float(np.average(event_xs_local.astype(np.float64), weights=event_weights)) + 0.5
            wcy = sy.start + float(np.average(event_ys_local.astype(np.float64), weights=event_weights)) + 0.5
        else:
            wcx, wcy = (cx, cy)
        detections.append({'bbox': (x1, y1, x2, y2), 'score': float(score), 'centroid': (cx, cy), 'peak_xy': peak_xy, 'weighted_centroid': (wcx, wcy), 'area': area, 'n_events': n_events})
    detections.sort(key=lambda d: -d['score'])
    return detections

def _connected_component_boxes(density_combined: np.ndarray, histogram: np.ndarray, peaks: list[tuple[int, int, float, float]], min_box: int, max_box: int, min_density: float, threshold_fraction: float=0.3) -> list[dict]:
    detections: list[dict] = []
    for i, (py, px, peak_score, peak_sigma) in enumerate(peaks):
        thresh = peak_score * threshold_fraction
        binary = density_combined >= thresh
        labeled, n_components = label(binary)
        if labeled[py, px] == 0:
            continue
        comp_label = labeled[py, px]
        comp_mask = labeled == comp_label
        event_mask = comp_mask & (histogram > 0)
        event_ys, event_xs = np.nonzero(event_mask)
        if len(event_ys) == 0:
            continue
        x1 = int(event_xs.min())
        x2 = int(event_xs.max()) + 1
        y1 = int(event_ys.min())
        y2 = int(event_ys.max()) + 1
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < min_box or box_h < min_box:
            continue
        if box_w > max_box or box_h > max_box:
            continue
        area = box_w * box_h
        n_events = int(histogram[comp_mask].sum())
        density = n_events / area
        if density < min_density:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        event_weights = histogram[event_ys, event_xs]
        total_weight = float(event_weights.sum())
        if total_weight > 0:
            wcx = float(np.average(event_xs.astype(np.float64), weights=event_weights)) + 0.5
            wcy = float(np.average(event_ys.astype(np.float64), weights=event_weights)) + 0.5
        else:
            wcx, wcy = (cx, cy)
        detections.append({'bbox': (x1, y1, x2, y2), 'score': float(peak_score), 'centroid': (cx, cy), 'peak_xy': (int(px), int(py)), 'weighted_centroid': (wcx, wcy), 'area': area, 'n_events': n_events})
    detections.sort(key=lambda d: -d['score'])
    return detections

def _gaussian_fit_boxes(histogram: np.ndarray, peaks: list[tuple[int, int, float, float]], min_box: int, max_box: int, min_density: float, sigma_mult: float=2.0) -> list[dict]:
    H, W = histogram.shape
    detections: list[dict] = []
    for py, px, peak_score, peak_sigma in peaks:
        radius = int(2.0 * peak_sigma) + 1
        y_lo = max(0, py - radius)
        y_hi = min(H, py + radius + 1)
        x_lo = max(0, px - radius)
        x_hi = min(W, px + radius + 1)
        crop = histogram[y_lo:y_hi, x_lo:x_hi]
        ey_local, ex_local = np.nonzero(crop > 0)
        if len(ey_local) < 5:
            continue
        ey = ey_local + y_lo
        ex = ex_local + x_lo
        weights = crop[ey_local, ex_local].astype(np.float64)
        total_w = weights.sum()
        if total_w == 0:
            continue
        mu_x = np.average(ex.astype(np.float64), weights=weights)
        mu_y = np.average(ey.astype(np.float64), weights=weights)
        dx = ex.astype(np.float64) - mu_x
        dy = ey.astype(np.float64) - mu_y
        cov_xx = np.average(dx * dx, weights=weights)
        cov_yy = np.average(dy * dy, weights=weights)
        std_x = max(np.sqrt(cov_xx), 1.0)
        std_y = max(np.sqrt(cov_yy), 1.0)
        half_w = sigma_mult * std_x
        half_h = sigma_mult * std_y
        x1 = max(0, int(mu_x - half_w))
        y1 = max(0, int(mu_y - half_h))
        x2 = min(W, int(mu_x + half_w) + 1)
        y2 = min(H, int(mu_y + half_h) + 1)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < min_box or box_h < min_box:
            continue
        if box_w > max_box or box_h > max_box:
            continue
        area = box_w * box_h
        n_events = int(histogram[y1:y2, x1:x2].sum())
        density = n_events / area if area > 0 else 0
        if density < min_density:
            continue
        cx = mu_x + 0.5
        cy = mu_y + 0.5
        detections.append({'bbox': (x1, y1, x2, y2), 'score': float(peak_score), 'centroid': (cx, cy), 'peak_xy': (int(px), int(py)), 'weighted_centroid': (cx, cy), 'area': area, 'n_events': n_events})
    detections.sort(key=lambda d: -d['score'])
    return detections

def detect_clusters(event_histogram: np.ndarray, sigmas: list[float] | None=None, min_box: int=4, max_box: int=200, min_density: float=0.05, method: str='watershed', cc_threshold_fraction: float=0.3, gauss_sigma_mult: float=2.0, watershed_threshold_frac: float=0.0, watershed_expand_frac: float=0.0) -> list[dict]:
    if sigmas is None:
        sigmas = [4.0, 8.0, 16.0, 32.0]
    if event_histogram.ndim != 2:
        raise ValueError(f'event_histogram must be 2D, got shape {event_histogram.shape}')
    if np.any(event_histogram < 0):
        raise ValueError('event_histogram must be non-negative')
    histogram = event_histogram.astype(np.float64)
    if histogram.sum() == 0:
        return []
    density_maps = _compute_density_maps(histogram, sigmas)
    peaks = multi_scale_peaks(density_maps, suppress_radius=2.0 * max(sigmas))
    if not peaks:
        return []
    density_combined = np.stack(list(density_maps.values())).max(axis=0)
    if method == 'connected_components':
        detections = _connected_component_boxes(density_combined, histogram, peaks, min_box, max_box, min_density, threshold_fraction=cc_threshold_fraction)
    elif method == 'gaussian_fit':
        detections = _gaussian_fit_boxes(histogram, peaks, min_box, max_box, min_density, sigma_mult=gauss_sigma_mult)
    else:
        labels = _watershed_segment(density_combined, peaks, watershed_threshold_frac=watershed_threshold_frac, watershed_expand_frac=watershed_expand_frac)
        detections = _extract_boxes_from_segments(labels, histogram, peaks, min_box, max_box, min_density)
    return detections

def detect_single_object(coords: np.ndarray, feats: np.ndarray, spatial_shape: tuple[int, int]=(640, 640), method: str='watershed', cc_threshold_fraction: float=0.3, gauss_sigma_mult: float=2.0, sigmas: list[float] | None=None, search_center: tuple[float, float] | None=None, search_radius: float=80.0, watershed_threshold_frac: float=0.0, watershed_expand_frac: float=0.0, background_map: np.ndarray | None=None) -> dict | None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'coords must be [N, 3], got shape {coords.shape}')
    if feats.ndim != 2:
        raise ValueError(f'feats must be [N, C], got shape {feats.shape}')
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f'coords ({coords.shape[0]}) and feats ({feats.shape[0]}) must have same number of voxels')
    if coords.shape[0] == 0:
        return None
    H, W = spatial_shape
    y = coords[:, 1]
    x = coords[:, 2]
    if np.any(y < 0) or np.any(y >= H) or np.any(x < 0) or np.any(x >= W):
        raise ValueError(f'coords contain out-of-bounds values for spatial_shape ({H}, {W}): y in [{y.min()}, {y.max()}], x in [{x.min()}, {x.max()}]')
    if search_center is not None:
        sc_x, sc_y = search_center
        dist_sq = (x.astype(np.float64) - sc_x) ** 2 + (y.astype(np.float64) - sc_y) ** 2
        within = dist_sq <= search_radius ** 2
        if within.sum() == 0:
            return None
        coords = coords[within]
        feats = feats[within]
        y = coords[:, 1]
        x = coords[:, 2]
    feat_weight = np.log1p(np.linalg.norm(feats, axis=1))
    t = coords[:, 0].astype(np.float64)
    t_max = t.max()
    if t_max > 0:
        time_weight = 1.0 + t / t_max
    else:
        time_weight = np.ones_like(t)
    feat_weight = feat_weight * time_weight
    linear = y.astype(np.int64) * W + x.astype(np.int64)
    histogram = np.zeros(H * W, dtype=np.float64)
    np.add.at(histogram, linear, feat_weight)
    histogram = histogram.reshape(H, W)
    if background_map is not None:
        histogram = np.maximum(histogram - background_map, 0.0)
    detections = detect_clusters(histogram, sigmas=sigmas, method=method, cc_threshold_fraction=cc_threshold_fraction, gauss_sigma_mult=gauss_sigma_mult, watershed_threshold_frac=watershed_threshold_frac, watershed_expand_frac=watershed_expand_frac)
    if not detections:
        return None
    return detections[0]

def detect_multiple_objects(coords: np.ndarray, feats: np.ndarray, spatial_shape: tuple[int, int]=(640, 640), method: str='watershed', cc_threshold_fraction: float=0.3, gauss_sigma_mult: float=2.0, sigmas: list[float] | None=None, search_center: tuple[float, float] | None=None, search_radius: float=80.0, watershed_threshold_frac: float=0.0, watershed_expand_frac: float=0.0, background_map: np.ndarray | None=None, top_k: int=5) -> list[dict]:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'coords must be [N, 3], got shape {coords.shape}')
    if feats.ndim != 2:
        raise ValueError(f'feats must be [N, C], got shape {feats.shape}')
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f'coords ({coords.shape[0]}) and feats ({feats.shape[0]}) must have same number of voxels')
    if coords.shape[0] == 0:
        return []
    H, W = spatial_shape
    y = coords[:, 1]
    x = coords[:, 2]
    if np.any(y < 0) or np.any(y >= H) or np.any(x < 0) or np.any(x >= W):
        raise ValueError(f'coords out-of-bounds for spatial_shape ({H}, {W}): y in [{y.min()}, {y.max()}], x in [{x.min()}, {x.max()}]')
    if search_center is not None:
        sc_x, sc_y = search_center
        dist_sq = (x.astype(np.float64) - sc_x) ** 2 + (y.astype(np.float64) - sc_y) ** 2
        within = dist_sq <= search_radius ** 2
        if within.sum() == 0:
            return []
        coords = coords[within]
        feats = feats[within]
        y = coords[:, 1]
        x = coords[:, 2]
    feat_weight = np.log1p(np.linalg.norm(feats, axis=1))
    t = coords[:, 0].astype(np.float64)
    t_max = t.max()
    if t_max > 0:
        time_weight = 1.0 + t / t_max
    else:
        time_weight = np.ones_like(t)
    feat_weight = feat_weight * time_weight
    linear = y.astype(np.int64) * W + x.astype(np.int64)
    histogram = np.zeros(H * W, dtype=np.float64)
    np.add.at(histogram, linear, feat_weight)
    histogram = histogram.reshape(H, W)
    if background_map is not None:
        histogram = np.maximum(histogram - background_map, 0.0)
    detections = detect_clusters(histogram, sigmas=sigmas, method=method, cc_threshold_fraction=cc_threshold_fraction, gauss_sigma_mult=gauss_sigma_mult, watershed_threshold_frac=watershed_threshold_frac, watershed_expand_frac=watershed_expand_frac)
    if not detections:
        return []
    return detections[:max(1, int(top_k))]
