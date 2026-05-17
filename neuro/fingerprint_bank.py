#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
PROJECT = Path(os.environ.get("NEUROSTDP_ROOT", str(Path(__file__).resolve().parents[2])))
P3 = PROJECT
PL = P3 / "pseudo_labels"
RAW = PROJECT / "raw_data/FRED/train"
DATA_NPZ = {
    "canonical_train": PROJECT / "data/datasets/fred_paper_parity_v82_640/sparse/canonical_train",
    "canonical_test":  PROJECT / "data/datasets/fred_paper_parity_v82_640/sparse/canonical_test",
}
CHANNELS_BY_SPLIT = {
    "canonical_train": {
        "v47":      PL / "discovery_v47_full/labels",
        "v66":      PL / "discovery_v66_full/labels",
        "rotor":    PL / "rotor_fft_full/labels",
        "polarity": PL / "polarity_v72_full/labels",
        "tube":     PL / "tube_v73_full/labels",
        "mf3":      PL / "multiframe_v3_full/labels",
    },
    "canonical_test": {
        "v47":      PL / "discovery_v47_test/labels",
        "v66":      PL / "discovery_v66_test/labels",
        "rotor":    PL / "rotor_fft_test/labels",
        "polarity": PL / "polarity_v72_test/labels",
        "tube":     PL / "tube_v73_test/labels",
        "mf3":      PL / "multiframe_v3_test/labels",
    },
}
V47_BY_SPLIT = {
    "canonical_train": PL / "discovery_v47_full/labels",
    "canonical_test":  PL / "discovery_v47_test/labels",
}
V127D_BY_SPLIT = {
    "canonical_train": PL / "discovery_v127d_max/labels",
    "canonical_test":  PL / "discovery_v127d_max_test/labels",
}
W = H = 640
RAW_W, RAW_H = 1280, 720

def load_yolo_xyxy(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        out.append([(cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H])
    return out

def load_raw_gt_stats(sid: str):
    zp = RAW / f"{sid}.zip"
    if not zp.exists():
        return None
    sx = W / RAW_W
    sy = H / RAW_H
    areas, tids, types = [], set(), set()
    with zipfile.ZipFile(zp) as zf:
        try:
            with zf.open(f"{sid}/coordinates.txt") as f:
                for line in f:
                    line = line.decode("utf-8").strip()
                    parts = line.split(": ")
                    if len(parts) != 2:
                        continue
                    vals = parts[1].split(", ")
                    if len(vals) < 6:
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, vals[:4])
                        tid = int(vals[4])
                        dt = vals[5]
                    except ValueError:
                        continue
                    w = (x2 - x1) * sx
                    h = (y2 - y1) * sy
                    if w <= 0 or h <= 0:
                        continue
                    areas.append(w * h)
                    tids.add(tid)
                    types.add(dt)
        except KeyError:
            return None
    if not areas:
        return None
    return {
        "n_tids": len(tids),
        "types": sorted(types),
        "gt_area_med": float(np.median(areas)),
    }

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def box_area(b):
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

def box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def events_in_box(ys, xs, box):
    x1, y1, x2, y2 = box
    m = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return int(m.sum())

def compute_fingerprint(args):
    sid, split, v125c_map30, oracle_map30 = args
    split_root = DATA_NPZ[split]
    npz_files = sorted((split_root / sid).glob("*.npz"))
    if not npz_files:
        return sid, split, None
    channels = CHANNELS_BY_SPLIT[split]
    v47_dir = V47_BY_SPLIT[split]
    v127d_dir = V127D_BY_SPLIT[split]
    raw_stats = load_raw_gt_stats(sid) or {}
    ev_per_frame = []
    top1_cx, top1_cy, top1_area, top1_density = [], [], [], []
    agree_counts = []
    prev_center = None
    stationary_hits, motion_hits_total = 0, 0
    max_displacement = 0.0
    multi_drone_hits = 0
    per_frame_iou = []
    per_frame_area_ratio = []
    drop_hits = 0
    for nf in npz_files:
        fnum = nf.stem.replace("frame_", "")
        fname = f"{sid}_frame_{fnum}.txt"
        data = np.load(nf)
        coords = data["coords"]
        if coords.shape[0] > 0:
            ys = coords[:, 1].astype(np.int32)
            xs = coords[:, 2].astype(np.int32)
            m = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys, xs = ys[m], xs[m]
            n_ev = int(ys.size)
        else:
            ys = np.zeros(0, dtype=np.int32)
            xs = ys.copy()
            n_ev = 0
        ev_per_frame.append(n_ev)
        v47_boxes = load_yolo_xyxy(v47_dir / sid / fname)
        v127d_boxes = load_yolo_xyxy(v127d_dir / sid / fname)
        v47_top1 = v47_boxes[0] if v47_boxes else None
        v127d_top1 = v127d_boxes[0] if v127d_boxes else None
        if v47_top1 is not None:
            cx, cy = box_center(v47_top1)
            area = max(1.0, box_area(v47_top1))
            top1_cx.append(cx); top1_cy.append(cy); top1_area.append(area)
            top1_density.append(events_in_box(ys, xs, v47_top1) / area)
            a = 0
            for d in channels.values():
                cands = load_yolo_xyxy(d / sid / fname)[:3]
                for cb in cands:
                    ccx, ccy = box_center(cb)
                    if abs(ccx - cx) < 40 and abs(ccy - cy) < 40:
                        a += 1
                        break
            agree_counts.append(a)
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                disp = float(np.hypot(dx, dy))
                max_displacement = max(max_displacement, disp)
                motion_hits_total += 1
                if disp < 5.0:
                    stationary_hits += 1
            prev_center = (cx, cy)
            far_channels = 0
            for d in channels.values():
                cands = load_yolo_xyxy(d / sid / fname)
                if not cands:
                    continue
                ccx, ccy = box_center(cands[0])
                if np.hypot(ccx - cx, ccy - cy) > 80:
                    far_channels += 1
            if far_channels >= 2:
                multi_drone_hits += 1
        if v47_top1 is not None and v127d_top1 is not None:
            per_frame_iou.append(box_iou(v127d_top1, v47_top1))
            a47 = max(1.0, box_area(v47_top1))
            a127 = max(1.0, box_area(v127d_top1))
            per_frame_area_ratio.append(a127 / a47)
        if v47_top1 is not None and v127d_top1 is None:
            drop_hits += 1
        elif v47_top1 is not None and v127d_top1 is not None:
            a47 = max(1.0, box_area(v47_top1))
            a127 = max(1.0, box_area(v127d_top1))
            if a127 / a47 > 3.0:
                drop_hits += 1
    n_frames = len(npz_files)
    if not top1_cy:
        return sid, split, None
    ev_per_frame_arr = np.asarray(ev_per_frame, dtype=np.float64)
    features = {
        "ev_med": float(np.median(ev_per_frame_arr)),
        "ev_p90": float(np.percentile(ev_per_frame_arr, 90)),
        "density_med": float(np.median(top1_density)),
        "cx_std": float(np.std(top1_cx)),
        "cy_med": float(np.median(top1_cy)),
        "cy_std": float(np.std(top1_cy)),
        "cy_p10": float(np.percentile(top1_cy, 10)),
        "cy_p90": float(np.percentile(top1_cy, 90)),
        "area_med": float(np.median(top1_area)),
        "area_p10": float(np.percentile(top1_area, 10)),
        "area_p90": float(np.percentile(top1_area, 90)),
        "agree_mean": float(np.mean(agree_counts)) if agree_counts else 0.0,
        "agree_p10": float(np.percentile(agree_counts, 10)) if agree_counts else 0.0,
        "motion_range": float(max_displacement),
        "density_variance": float(np.var(ev_per_frame_arr)),
        "multi_drone_unsup": float(multi_drone_hits / n_frames),
        "stationary_fraction": float(stationary_hits / motion_hits_total) if motion_hits_total else 0.0,
    }
    features_circular = {
        "iou_v127d_v47": float(np.mean(per_frame_iou)) if per_frame_iou else 0.0,
        "area_ratio_v127d_v47": float(np.median(per_frame_area_ratio)) if per_frame_area_ratio else 1.0,
        "drop_fraction": float(drop_hits / n_frames),
    }
    features_deferred = {
        "polarity_imbalance": None,
    }
    metadata = {
        "n_frames": n_frames,
        "n_tids": raw_stats.get("n_tids"),
        "drone_types": raw_stats.get("types", []),
    }
    gt_derived = {
        "v125c_map30": v125c_map30 * 100 if v125c_map30 is not None else None,
        "oracle_map30": oracle_map30 * 100 if oracle_map30 is not None else None,
        "gap_to_oracle": (oracle_map30 - v125c_map30) * 100
            if (v125c_map30 is not None and oracle_map30 is not None) else None,
        "gt_area_med": raw_stats.get("gt_area_med"),
        "box_size_ratio": (np.median(top1_area) / raw_stats["gt_area_med"])
            if raw_stats.get("gt_area_med") else None,
    }
    return sid, split, {
        "sid": sid,
        "split": split,
        "features": features,
        "features_circular": features_circular,
        "features_deferred": features_deferred,
        "metadata": metadata,
        "gt_derived": gt_derived,
    }

def load_eval_maps():
    v125c_by_split = {"canonical_train": {}, "canonical_test": {}}
    oracle_by_split = {"canonical_train": {}, "canonical_test": {}}
    train_eval = P3 / "docs/eval_v2_multipred_multidrone.json"
    if train_eval.exists():
        d = json.loads(train_eval.read_text())
        for sid, row in d.get("v125c", {}).items():
            v125c_by_split["canonical_train"][sid] = row.get("mAP_30")
    test_eval = P3 / "docs/eval_v2_multipred_multidrone_TEST.json"
    if test_eval.exists():
        d = json.loads(test_eval.read_text())
        for sid, row in d.get("v125c", {}).items():
            v125c_by_split["canonical_test"][sid] = row.get("mAP_30")
    train_oracle = P3 / "docs/phase_a1_oracle_6ch.json"
    if train_oracle.exists():
        lst = json.loads(train_oracle.read_text())
        for row in lst if isinstance(lst, list) else []:
            oracle_by_split["canonical_train"][row["sid"]] = row.get("recall_6ch")
    test_oracle = P3 / "docs/phase_a1_oracle_6ch_test.json"
    if test_oracle.exists():
        lst = json.loads(test_oracle.read_text())
        for row in lst if isinstance(lst, list) else []:
            oracle_by_split["canonical_test"][row["sid"]] = row.get("recall_6ch")
    return v125c_by_split, oracle_by_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["canonical_train", "canonical_test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(P3 / "docs/fingerprints_fred.json"))
    ap.add_argument("--zscore-out", default=str(P3 / "docs/fingerprint_zscore_params.json"))
    args = ap.parse_args()
    v125c_maps, oracle_maps = load_eval_maps()
    tasks = []
    for split in args.splits:
        root = DATA_NPZ[split]
        if not root.exists():
            print(f"[skip] {split}: {root} missing")
            continue
        sids = sorted([d.name for d in root.iterdir() if d.is_dir()], key=lambda x: int(x))
        for sid in sids:
            tasks.append((sid, split,
                          v125c_maps[split].get(sid),
                          oracle_maps[split].get(sid)))
    print(f"Computing fingerprints for {len(tasks)} seq×split entries "
          f"({args.workers} workers)")
    with Pool(args.workers) as pool:
        results = pool.map(compute_fingerprint, tasks)
    entries = {}
    for sid, split, data in results:
        if data is None:
            continue
        entries.setdefault(split, {})[sid] = data
    out_path = Path(args.out)
    out_path.write_text(json.dumps(entries, indent=2))
    total = sum(len(v) for v in entries.values())
    print(f"Wrote {out_path}  ({total} entries across {len(entries)} splits)")
    train_entries = entries.get("canonical_train", {})
    if train_entries:
        keys = (list(next(iter(train_entries.values()))["features"].keys())
                + list(next(iter(train_entries.values()))["features_circular"].keys()))
        params = {}
        for k in keys:
            vals = []
            for e in train_entries.values():
                if k in e["features"]:
                    vals.append(e["features"][k])
                elif k in e["features_circular"]:
                    vals.append(e["features_circular"][k])
            a = np.asarray(vals, dtype=np.float64)
            params[k] = {"mean": float(a.mean()), "std": float(a.std() + 1e-8)}
        zp = Path(args.zscore_out)
        zp.write_text(json.dumps(params, indent=2))
        print(f"Wrote {zp}  (z-score on train only, {len(params)} features)")
if __name__ == "__main__":
    main()
