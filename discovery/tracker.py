from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if boxes_a.ndim != 2 or boxes_a.shape[1] != 4:
        raise ValueError(f'boxes_a must be (M, 4), got {boxes_a.shape}')
    if boxes_b.ndim != 2 or boxes_b.shape[1] != 4:
        raise ValueError(f'boxes_b must be (N, 4), got {boxes_b.shape}')
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou

def boxes_to_pseudo_labels(tracklets: list[Tracklet], num_frames: int, class_id: int=0) -> dict[int, list[np.ndarray]]:
    if num_frames <= 0:
        raise ValueError(f'num_frames must be positive, got {num_frames}')
    labels: dict[int, list[np.ndarray]] = {}
    for t in tracklets:
        refined = t.refine()
        for fidx, box in zip(t.frame_indices, refined):
            if fidx < 0 or fidx >= num_frames:
                continue
            label = np.empty(5, dtype=np.float64)
            label[:4] = box
            label[4] = class_id
            labels.setdefault(fidx, []).append(label)
    return labels

class KalmanBoxTracker:
    _next_id: int = 0
    def __init__(self, bbox: np.ndarray) -> None:
        bbox = np.asarray(bbox, dtype=np.float64).ravel()
        if bbox.shape[0] != 4:
            raise ValueError(f'bbox must have 4 elements, got {bbox.shape[0]}')
        self._id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self._x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        self._F = np.eye(6, dtype=np.float64)
        self._F[0, 4] = 1.0
        self._F[1, 5] = 1.0
        self._H = np.zeros((4, 6), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0
        self._Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
        self._R = np.diag([1.0, 1.0, 5.0, 5.0])
        self._hits: int = 1
        self._misses: int = 0
        self._age: int = 1
        self._predicted_this_step: bool = False
    @property
    def id(self) -> int:
        return self._id
    @property
    def hits(self) -> int:
        return self._hits
    @property
    def misses(self) -> int:
        return self._misses
    @property
    def age(self) -> int:
        return self._age
    def predict(self) -> np.ndarray:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 1.0)
        self._age += 1
        self._predicted_this_step = True
        return self._state_to_bbox()
    def update(self, bbox: np.ndarray) -> None:
        bbox = np.asarray(bbox, dtype=np.float64).ravel()
        if bbox.shape[0] != 4:
            raise ValueError(f'bbox must have 4 elements, got {bbox.shape[0]}')
        z = self._bbox_to_obs(bbox)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        I_KH = np.eye(6) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 1.0)
        self._hits += 1
        self._misses = 0
    def mark_missed(self) -> None:
        self._misses += 1
    def get_bbox(self) -> np.ndarray:
        return self._state_to_bbox()
    def _state_to_bbox(self) -> np.ndarray:
        cx, cy, w, h = (self._x[0], self._x[1], self._x[2], self._x[3])
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)
    @staticmethod
    def _bbox_to_obs(bbox: np.ndarray) -> np.ndarray:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array([cx, cy, w, h], dtype=np.float64)
    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 0

@dataclass

class Tracklet:
    track_id: int
    boxes: list[np.ndarray] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    def __len__(self) -> int:
        return len(self.boxes)
    @property
    def num_hits(self) -> int:
        return sum((1 for s in self.scores if not np.isnan(s)))
    def refine(self, smooth_window: int=11, smooth_order: int=2, min_size: float=8.0, max_size: float=150.0) -> list[np.ndarray]:
        n = len(self.boxes)
        if n == 0:
            raise ValueError('Cannot refine an empty tracklet')
        if smooth_window < 3:
            raise ValueError(f'smooth_window must be >= 3, got {smooth_window}')
        if smooth_order < 1:
            raise ValueError(f'smooth_order must be >= 1, got {smooth_order}')
        all_boxes = np.array(self.boxes, dtype=np.float64)
        cx = (all_boxes[:, 0] + all_boxes[:, 2]) / 2.0
        cy = (all_boxes[:, 1] + all_boxes[:, 3]) / 2.0
        ws = all_boxes[:, 2] - all_boxes[:, 0]
        hs = all_boxes[:, 3] - all_boxes[:, 1]
        med_w = float(np.clip(np.median(ws), min_size, max_size))
        med_h = float(np.clip(np.median(hs), min_size, max_size))
        if n >= 3:
            win = min(smooth_window, n)
            if win % 2 == 0:
                win -= 1
            win = max(win, 3)
            order = min(smooth_order, win - 1)
            cx_smooth = savgol_filter(cx, window_length=win, polyorder=order)
            cy_smooth = savgol_filter(cy, window_length=win, polyorder=order)
        else:
            cx_smooth = cx.copy()
            cy_smooth = cy.copy()
        refined: list[np.ndarray] = []
        half_w = med_w / 2.0
        half_h = med_h / 2.0
        for i in range(n):
            refined.append(np.array([cx_smooth[i] - half_w, cy_smooth[i] - half_h, cx_smooth[i] + half_w, cy_smooth[i] + half_h], dtype=np.float64))
        return refined

class MultiFrameTracker:
    def __init__(self, iou_threshold: float=0.1, max_misses: int=5, min_tracklet: int=10, max_center_dist: float=0.0) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f'iou_threshold must be in (0, 1], got {iou_threshold}')
        if max_misses < 0:
            raise ValueError(f'max_misses must be >= 0, got {max_misses}')
        if min_tracklet < 1:
            raise ValueError(f'min_tracklet must be >= 1, got {min_tracklet}')
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_tracklet = min_tracklet
        self.max_center_dist = max_center_dist
        self._trackers: list[KalmanBoxTracker] = []
        self._histories: dict[int, Tracklet] = {}
        self._frame_idx: int = -1
    def update(self, detections: list[dict]) -> list[dict]:
        if not isinstance(detections, list):
            raise TypeError(f'detections must be a list, got {type(detections).__name__}')
        self._frame_idx += 1
        det_boxes: list[np.ndarray] = []
        det_scores: list[float] = []
        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                raise TypeError(f'Each detection must be a dict, got {type(det).__name__} at index {i}')
            if 'bbox' not in det:
                raise KeyError(f"Detection at index {i} missing 'bbox' key")
            bbox = np.asarray(det['bbox'], dtype=np.float64).ravel()
            if bbox.shape[0] != 4:
                raise ValueError(f'Detection bbox must have 4 elements, got {bbox.shape[0]} at index {i}')
            det_boxes.append(bbox)
            det_scores.append(float(det.get('score', 1.0)))
        n_det = len(det_boxes)
        n_trk = len(self._trackers)
        trk_boxes = np.empty((n_trk, 4), dtype=np.float64)
        for i, trk in enumerate(self._trackers):
            trk_boxes[i] = trk.predict()
        matched_det_idx: set[int] = set()
        matched_trk_idx: set[int] = set()
        if n_det > 0 and n_trk > 0:
            det_arr = np.array(det_boxes, dtype=np.float64)
            iou_matrix = compute_iou(det_arr, trk_boxes)
            cost = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    if self.max_center_dist > 0:
                        d_cx = (det_boxes[r][0] + det_boxes[r][2]) / 2
                        d_cy = (det_boxes[r][1] + det_boxes[r][3]) / 2
                        t_cx = (trk_boxes[c][0] + trk_boxes[c][2]) / 2
                        t_cy = (trk_boxes[c][1] + trk_boxes[c][3]) / 2
                        cdist = np.sqrt((d_cx - t_cx) ** 2 + (d_cy - t_cy) ** 2)
                        if cdist > self.max_center_dist:
                            continue
                    matched_det_idx.add(r)
                    matched_trk_idx.add(c)
                    self._trackers[c].update(det_boxes[r])
                    tid = self._trackers[c].id
                    self._histories[tid].boxes.append(self._trackers[c].get_bbox().copy())
                    self._histories[tid].frame_indices.append(self._frame_idx)
                    self._histories[tid].scores.append(det_scores[r])
        for i in range(n_trk):
            if i not in matched_trk_idx:
                self._trackers[i].mark_missed()
                tid = self._trackers[i].id
                self._histories[tid].boxes.append(self._trackers[i].get_bbox().copy())
                self._histories[tid].frame_indices.append(self._frame_idx)
                self._histories[tid].scores.append(float('nan'))
        for i in range(n_det):
            if i not in matched_det_idx:
                new_trk = KalmanBoxTracker(det_boxes[i])
                self._trackers.append(new_trk)
                self._histories[new_trk.id] = Tracklet(track_id=new_trk.id, boxes=[new_trk.get_bbox().copy()], frame_indices=[self._frame_idx], scores=[det_scores[i]])
        self._trackers = [t for t in self._trackers if t.misses <= self.max_misses]
        result: list[dict] = []
        for trk in self._trackers:
            result.append({'bbox': trk.get_bbox().copy(), 'track_id': trk.id, 'hits': trk.hits, 'misses': trk.misses})
        return result
    def get_tracklets(self) -> list[Tracklet]:
        confirmed = [t for t in self._histories.values() if t.num_hits >= self.min_tracklet]
        confirmed.sort(key=lambda t: t.track_id)
        return confirmed
    def get_all_tracklets(self) -> list[Tracklet]:
        result = list(self._histories.values())
        result.sort(key=lambda t: t.track_id)
        return result
    @property
    def frame_index(self) -> int:
        return self._frame_idx
    @property
    def num_active_tracks(self) -> int:
        return len(self._trackers)

def kalman_rts_smooth(raw_detections: dict[int, np.ndarray], n_frames: int, Q_scale: float=1.0, R_scale: float=1.0, outlier_gate: float=12.0, min_box: float=8.0, max_box: float=150.0) -> dict:
    if not raw_detections:
        raise ValueError('raw_detections must not be empty')
    if n_frames < 1:
        raise ValueError(f'n_frames must be >= 1, got {n_frames}')
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]) * Q_scale
    R = np.diag([1.0, 1.0, 5.0, 5.0]) * R_scale
    F = np.eye(6, dtype=np.float64)
    F[0, 4] = 1.0
    F[1, 5] = 1.0
    H = np.zeros((4, 6), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    I6 = np.eye(6)
    sorted_det_frames = sorted(raw_detections.keys())
    first_frame = sorted_det_frames[0]
    last_frame = sorted_det_frames[-1]
    bbox0 = np.asarray(raw_detections[first_frame], dtype=np.float64)
    x_init = np.array([(bbox0[0] + bbox0[2]) / 2.0, (bbox0[1] + bbox0[3]) / 2.0, bbox0[2] - bbox0[0], bbox0[3] - bbox0[1], 0.0, 0.0])
    P_init = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
    x_filt = {}
    P_filt = {}
    x_pred = {}
    P_pred = {}
    innovations = {}
    is_detected = {}
    is_outlier = {}
    n_outliers = 0
    x = x_init.copy()
    P = P_init.copy()
    for t in range(first_frame, last_frame + 1):
        if t == first_frame:
            x_p = x.copy()
            P_p = P.copy()
        else:
            x_p = F @ x
            P_p = F @ P @ F.T + Q
        x_pred[t] = x_p.copy()
        P_pred[t] = P_p.copy()
        if t in raw_detections:
            bbox = np.asarray(raw_detections[t], dtype=np.float64)
            z = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, bbox[2] - bbox[0], bbox[3] - bbox[1]])
            y = z - H @ x_p
            S = H @ P_p @ H.T + R
            innov_norm = float(np.sqrt(y @ y) / max(np.sqrt(np.trace(S)), 1e-06))
            innovations[t] = innov_norm
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            mahal_sq = float(y @ S_inv @ y)
            if mahal_sq < outlier_gate:
                K = P_p @ H.T @ S_inv
                x = x_p + K @ y
                I_KH = I6 - K @ H
                P = I_KH @ P_p @ I_KH.T + K @ R @ K.T
                is_outlier[t] = False
            else:
                x = x_p.copy()
                P = P_p.copy()
                is_outlier[t] = True
                n_outliers += 1
            is_detected[t] = True
        else:
            x = x_p.copy()
            P = P_p.copy()
            is_detected[t] = False
            is_outlier[t] = False
        x[2] = max(x[2], 1.0)
        x[3] = max(x[3], 1.0)
        x_filt[t] = x.copy()
        P_filt[t] = P.copy()
    x_smooth = {}
    P_smooth = {}
    x_smooth[last_frame] = x_filt[last_frame].copy()
    P_smooth[last_frame] = P_filt[last_frame].copy()
    for t in range(last_frame - 1, first_frame - 1, -1):
        P_pred_next = P_pred[t + 1]
        try:
            P_pred_inv = np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred_next)
        G = P_filt[t] @ F.T @ P_pred_inv
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred_next) @ G.T
    boxes = {}
    uncertainties = {}
    for t in range(first_frame, last_frame + 1):
        xs = x_smooth[t]
        cx, cy = (xs[0], xs[1])
        w = float(np.clip(xs[2], min_box, max_box))
        h = float(np.clip(xs[3], min_box, max_box))
        boxes[t] = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)
        uncertainties[t] = float(np.trace(P_smooth[t]))
    return {'boxes': boxes, 'innovations': innovations, 'uncertainties': uncertainties, 'is_detected': is_detected, 'is_outlier': is_outlier, 'n_outliers': n_outliers}
