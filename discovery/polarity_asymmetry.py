from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from scipy import ndimage as ndi
    from scipy.ndimage import label as cc_label
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
logger = logging.getLogger(__name__)
TARGET_H, TARGET_W = (640, 640)

def load_onoff_from_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    coords = z['coords']
    feats = z['feats']
    if coords.size == 0:
        return (np.zeros((TARGET_H, TARGET_W), dtype=np.float32), np.zeros((TARGET_H, TARGET_W), dtype=np.float32))
    y = coords[:, 1].astype(np.int64)
    x = coords[:, 2].astype(np.int64)
    on_lin = np.expm1(feats[:, 0].astype(np.float32)).clip(min=0.0)
    off_lin = np.expm1(feats[:, 1].astype(np.float32)).clip(min=0.0)
    on_img = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
    off_img = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
    np.add.at(on_img, (y, x), on_lin)
    np.add.at(off_img, (y, x), off_lin)
    return (on_img, off_img)

def polarity_ratio(on: np.ndarray, off: np.ndarray, eps: float=1.0) -> np.ndarray:
    return (on - off) / (on + off + eps)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
_SOBEL_Y = _SOBEL_X.T.copy()

def sobel_gradient(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _HAVE_SCIPY:
        gx = ndi.convolve(r, _SOBEL_X, mode='nearest')
        gy = ndi.convolve(r, _SOBEL_Y, mode='nearest')
    else:
        gx = _conv2_manual(r, _SOBEL_X)
        gy = _conv2_manual(r, _SOBEL_Y)
    return (gx, gy)

def _conv2_manual(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    pad = 1
    pimg = np.pad(img, pad, mode='edge')
    H, W = img.shape
    out = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * pimg[dy:dy + H, dx:dx + W]
    return out

def patch_coherence(gx: np.ndarray, gy: np.ndarray, patch: int=5) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.sqrt(gx * gx + gy * gy)
    if _HAVE_SCIPY:
        kernel = np.ones((patch, patch), dtype=np.float32)
        sum_gx = ndi.convolve(gx, kernel, mode='constant', cval=0.0)
        sum_gy = ndi.convolve(gy, kernel, mode='constant', cval=0.0)
        sum_mag = ndi.convolve(mag, kernel, mode='constant', cval=0.0)
    else:
        sum_gx = _sum_filter(gx, patch)
        sum_gy = _sum_filter(gy, patch)
        sum_mag = _sum_filter(mag, patch)
    resultant_len = np.sqrt(sum_gx * sum_gx + sum_gy * sum_gy)
    coh = np.zeros_like(mag)
    valid = sum_mag > 1e-06
    coh[valid] = resultant_len[valid] / sum_mag[valid]
    mean_mag = sum_mag / float(patch * patch)
    return (mean_mag, coh)

def _sum_filter(img: np.ndarray, patch: int) -> np.ndarray:
    p = patch // 2
    H, W = img.shape
    pimg = np.pad(img, p, mode='constant', constant_values=0.0)
    cs = np.cumsum(np.cumsum(pimg, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode='constant', constant_values=0.0)
    H2 = H
    W2 = W
    y1 = np.arange(H2)
    y2 = y1 + patch
    x1 = np.arange(W2)
    x2 = x1 + patch
    Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
    Y2, X2 = np.meshgrid(y2, x2, indexing='ij')
    out = cs[Y2, X2] - cs[Y1, X2] - cs[Y2, X1] + cs[Y1, X1]
    return out

@dataclass

class FrameBlob:
    y1: float
    x1: float
    y2: float
    x2: float
    area: int
    mean_mag: float
    mean_coh: float
    score: float

def detect_polarity_blobs(on: np.ndarray, off: np.ndarray, tau_mag: float=0.005, tau_coh: float=0.5, patch: int=5, area_min: int=100, area_max: int=4000, aspect_min: float=0.2, aspect_max: float=5.0, min_events_per_pixel: float=1.0, high_pass: bool=True, high_pass_sigma: float=12.0, morph_close: int=3, min_event_mask: bool=True) -> List[FrameBlob]:
    total = on + off
    r = polarity_ratio(on, off)
    if min_events_per_pixel > 0:
        r = np.where(total >= min_events_per_pixel, r, 0.0)
    if high_pass and _HAVE_SCIPY:
        r_lp = ndi.gaussian_filter(r, sigma=high_pass_sigma, mode='nearest')
        r_hp = r - r_lp
    else:
        r_hp = r
    gx, gy = sobel_gradient(r_hp)
    mean_mag, coh = patch_coherence(gx, gy, patch=patch)
    mask = (mean_mag >= tau_mag) & (coh >= tau_coh)
    if min_event_mask:
        if _HAVE_SCIPY:
            local_events = ndi.uniform_filter(total.astype(np.float32), size=patch) * (patch * patch)
            mask &= local_events >= 1.0
    if not mask.any():
        return []
    if morph_close > 0 and _HAVE_SCIPY:
        struct = np.ones((morph_close, morph_close), dtype=bool)
        mask = ndi.binary_closing(mask, structure=struct, iterations=1)
    if _HAVE_SCIPY:
        labels, n_cc = cc_label(mask, structure=np.ones((3, 3), dtype=bool))
    else:
        labels, n_cc = _flood_fill_cc(mask)
    if n_cc == 0:
        return []
    blobs: List[FrameBlob] = []
    if _HAVE_SCIPY:
        slices = ndi.find_objects(labels)
        areas = np.bincount(labels.ravel(), minlength=n_cc + 1)[1:]
        sum_mag = np.bincount(labels.ravel(), weights=mean_mag.ravel(), minlength=n_cc + 1)[1:]
        sum_coh = np.bincount(labels.ravel(), weights=coh.ravel(), minlength=n_cc + 1)[1:]
        for cid in range(n_cc):
            sl = slices[cid]
            if sl is None:
                continue
            area = int(areas[cid])
            if area < area_min or area > area_max:
                continue
            y1 = float(sl[0].start)
            y2 = float(sl[0].stop)
            x1 = float(sl[1].start)
            x2 = float(sl[1].stop)
            w = x2 - x1
            h = y2 - y1
            if h < 1 or w < 1:
                continue
            aspect = w / h
            if aspect < aspect_min or aspect > aspect_max:
                continue
            mmag = float(sum_mag[cid] / max(area, 1))
            mcoh = float(sum_coh[cid] / max(area, 1))
            score = float(area) * mmag * mcoh
            blobs.append(FrameBlob(y1=y1, x1=x1, y2=y2, x2=x2, area=area, mean_mag=mmag, mean_coh=mcoh, score=score))
    else:
        for cid in range(1, n_cc + 1):
            ys, xs = np.where(labels == cid)
            area = len(ys)
            if area < area_min or area > area_max:
                continue
            y1, y2 = (float(ys.min()), float(ys.max() + 1))
            x1, x2 = (float(xs.min()), float(xs.max() + 1))
            w = x2 - x1
            h = y2 - y1
            if h < 1 or w < 1:
                continue
            aspect = w / h
            if aspect < aspect_min or aspect > aspect_max:
                continue
            mmag = float(mean_mag[ys, xs].mean())
            mcoh = float(coh[ys, xs].mean())
            score = float(area) * mmag * mcoh
            blobs.append(FrameBlob(y1=y1, x1=x1, y2=y2, x2=x2, area=area, mean_mag=mmag, mean_coh=mcoh, score=score))
    blobs.sort(key=lambda b: b.score, reverse=True)
    return blobs

def _flood_fill_cc(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    nxt = 0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            nxt += 1
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= H or cx < 0 or (cx >= W):
                    continue
                if not mask[cy, cx] or labels[cy, cx] != 0:
                    continue
                labels[cy, cx] = nxt
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        stack.append((cy + dy, cx + dx))
    return (labels, nxt)

def persistent_tracks(per_frame_blobs: Dict[int, List[FrameBlob]], frame_idxs: List[int], max_center_dist: float=40.0, min_persist: int=3) -> Dict[int, List[FrameBlob]]:
    tracklets: List[List[Tuple[int, int]]] = []
    prev_frame = None
    prev_blobs: List[FrameBlob] = []
    prev_track_id: List[int] = []
    for k in frame_idxs:
        blobs = per_frame_blobs.get(k, [])
        cur_track_id: List[int] = [-1] * len(blobs)
        is_adjacent = prev_frame is not None and k == prev_frame + 1
        if is_adjacent and prev_blobs and blobs:
            pcy = np.array([(b.y1 + b.y2) / 2.0 for b in prev_blobs])
            pcx = np.array([(b.x1 + b.x2) / 2.0 for b in prev_blobs])
            ccy = np.array([(b.y1 + b.y2) / 2.0 for b in blobs])
            ccx = np.array([(b.x1 + b.x2) / 2.0 for b in blobs])
            d = np.sqrt((ccy[:, None] - pcy[None, :]) ** 2 + (ccx[:, None] - pcx[None, :]) ** 2)
            used_prev = set()
            for ci in range(len(blobs)):
                candidates = np.argsort(d[ci])
                picked = -1
                for pi in candidates:
                    if pi in used_prev:
                        continue
                    if d[ci, pi] > max_center_dist:
                        break
                    picked = int(pi)
                    break
                if picked >= 0:
                    used_prev.add(picked)
                    cur_track_id[ci] = prev_track_id[picked]
        for ci, tid in enumerate(cur_track_id):
            if tid < 0:
                new_id = len(tracklets)
                tracklets.append([])
                cur_track_id[ci] = new_id
            tracklets[cur_track_id[ci]].append((k, ci))
        prev_blobs = blobs
        prev_track_id = cur_track_id
        prev_frame = k
    kept_blobs_per_frame: Dict[int, List[FrameBlob]] = {k: [] for k in frame_idxs}
    n_total_blobs = sum((len(v) for v in per_frame_blobs.values()))
    n_kept = 0
    for tr in tracklets:
        if len(tr) < min_persist:
            continue
        tr_len = len(tr)
        for k, bi in tr:
            blob = per_frame_blobs[k][bi]
            boosted = FrameBlob(y1=blob.y1, x1=blob.x1, y2=blob.y2, x2=blob.x2, area=blob.area, mean_mag=blob.mean_mag, mean_coh=blob.mean_coh, score=blob.score * (1.0 + 0.1 * tr_len))
            kept_blobs_per_frame[k].append(boosted)
            n_kept += 1
    logger.info('Persistence filter: %d -> %d blobs kept (min_persist=%d)', n_total_blobs, n_kept, min_persist)
    return kept_blobs_per_frame
