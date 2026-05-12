"""3D tracking benchmark for the BBQ scene-graph pipeline on IsaacSim sequences.

Reuses ``main_with_edges.py``'s integration loop and computes MOTA, MOTP,
ID Consistency, ID Switches, T-SR and T-mIoU via ``metrics.tracking_metrics``.

Usage:
    python benchmark_tracking.py \
        --config_path examples/configs/isaac/franka_cab_dex_more.yaml \
        --iou_threshold 0.25 \
        --matcher hungarian \
        --output_dir benchmark_results
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor
from metrics.tracking_metrics import (
    FrameRecord,
    GTInstance,
    MetricsAccumulator,
    PredInstance,
    match_mode_factory,
    print_summary,
    save_metrics,
)

warnings.filterwarnings("ignore")

_FRAME_RE = re.compile(r"frame(\d+)\.(?:jpg|png)$")
_DEFAULT_SKIP_LABELS: Set[str] = {"wall", "floor", "ground", "ceiling", "background"}


def set_seed(seed: int = 18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------- GT reader
# Mirrors data_loaders/isaacsim.py.get_gt_instances() without its missing deps.

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_bbox2d_tight(bbox_data: dict) -> Dict[int, dict]:
    boxes = bbox_data.get("bboxes", {}).get("bbox_2d_tight", {}).get("boxes", [])
    out: Dict[int, dict] = {}
    for b in boxes:
        bid = b.get("bbox_2d_id", b.get("bbox_id"))
        if bid is not None:
            out[int(bid)] = b
    return out


def _parse_bbox3d(bbox_data: dict) -> List[dict]:
    return [
        b for b in bbox_data.get("bboxes", {}).get("bbox_3d", {}).get("boxes", [])
        if "track_id" in b
    ]


def _parse_instance_color_map(seg_info: dict) -> Dict[int, Tuple[int, int, int]]:
    out: Dict[int, Tuple[int, int, int]] = {}
    for k, v in seg_info.items():
        if not str(k).isdigit():
            continue
        c = v.get("color_bgr")
        if c and len(c) == 3:
            out[int(k)] = (int(c[0]), int(c[1]), int(c[2]))
    return out


def _infer_class_name(b3d: dict, b2d: Optional[dict]) -> str:
    if isinstance(b3d.get("label"), str) and b3d["label"]:
        return b3d["label"]
    if b2d is not None:
        lab = b2d.get("label")
        if isinstance(lab, dict) and lab:
            return str(next(iter(lab.values())))
    return "unknown"


def _extract_bbox_xyzxyz(b3d: dict) -> Optional[Tuple[float, ...]]:
    aabb = b3d.get("aabb_xyzmin_xyzmax")
    if isinstance(aabb, (list, tuple)) and len(aabb) == 6:
        return tuple(float(v) for v in aabb)
    aabb = b3d.get("aabb")
    if isinstance(aabb, dict):
        mn, mx = aabb.get("min"), aabb.get("max")
        if (
            isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple))
            and len(mn) == 3 and len(mx) == 3
        ):
            return tuple(float(v) for v in list(mn) + list(mx))
    return None


def load_gt_for_frame(
    scene_dir: Path,
    frame_num: int,
    skip_labels: Set[str],
    load_masks: bool,
) -> Optional[List[GTInstance]]:
    fs = f"{frame_num:06d}"
    bbox_path = scene_dir / "bbox" / f"bboxes{fs}_info.json"
    seg_png_path = scene_dir / "seg" / f"semantic{fs}.png"
    seg_info_path = scene_dir / "seg" / f"semantic{fs}_info.json"

    if not bbox_path.exists():
        return None
    bbox_data = _read_json(bbox_path)
    if bbox_data is None:
        return None

    seg_info = _read_json(seg_info_path) if seg_info_path.exists() else {}
    seg_img = None
    if load_masks and seg_png_path.exists():
        seg_img = cv2.imread(str(seg_png_path), cv2.IMREAD_COLOR)

    bbox2d_by_id = _parse_bbox2d_tight(bbox_data)
    bbox3d_list = _parse_bbox3d(bbox_data)
    color_by_inst = _parse_instance_color_map(seg_info)

    instances: List[GTInstance] = []
    for b3d in bbox3d_list:
        track_id = int(b3d["track_id"])
        inst_seg_id = int(b3d.get("instance_seg_id", -1))
        bbox_2d_id = int(b3d.get("bbox_2d_id", -1))
        bbox_3d_id = int(b3d.get("bbox_3d_id", -1))
        if bbox_3d_id < 0 or bbox_2d_id < 0 or inst_seg_id < 0:
            continue

        b2d = bbox2d_by_id.get(bbox_2d_id)
        class_name = _infer_class_name(b3d, b2d)
        cls_lower = class_name.lower()
        if any(skip in cls_lower for skip in skip_labels):
            continue

        bbox_xyxy = None
        if b2d is not None:
            xyxy = b2d.get("xyxy")
            if xyxy and len(xyxy) == 4:
                bbox_xyxy = tuple(float(v) for v in xyxy)

        mask = None
        if seg_img is not None:
            color = color_by_inst.get(inst_seg_id)
            if color is not None:
                mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=2)

        bbox_xyzxyz = _extract_bbox_xyzxyz(b3d)
        if bbox_xyzxyz is None:
            continue

        instances.append(GTInstance(
            track_id=track_id,
            class_name=class_name,
            mask=mask,
            bbox_xyxy=bbox_xyxy,
            bbox_xyzxyz=bbox_xyzxyz,
        ))
    return instances


# ---------------------------------------------------------------- helpers

def bbox_to_xyzxyz(bbox) -> Tuple[float, float, float, float, float, float]:
    """Convert an open3d AxisAlignedBoundingBox or OrientedBoundingBox to AABB tuple."""
    pts = np.asarray(bbox.get_box_points())
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mn[2]),
            float(mx[0]), float(mx[1]), float(mx[2]))


def frame_num_from_path(path: str) -> Optional[int]:
    m = _FRAME_RE.search(path)
    return int(m.group(1)) if m else None


class StableTrackIDs:
    """Maps each scene-object dict to a stable monotonic track ID.

    Uses ``id()`` of the dict, and keeps a strong reference to every dict
    we've ever seen so that GC cannot recycle an address after a merge
    drops a track.
    """

    def __init__(self) -> None:
        self._tid: Dict[int, int] = {}
        self._refs: List[dict] = []
        self._next = 0

    def get(self, obj: dict) -> int:
        key = id(obj)
        tid = self._tid.get(key)
        if tid is None:
            tid = self._next
            self._next += 1
            self._tid[key] = tid
            self._refs.append(obj)
        return tid


# ---------------------------------------------------------------- main

def main(args: argparse.Namespace) -> None:
    with open(args.config_path) as f:
        config = yaml.full_load(f)

    if config["dataset"].get("relative_pose", False):
        raise RuntimeError(
            "config has relative_pose=True; predictions would be in a "
            "different frame from GT. Set relative_pose: False to benchmark."
        )

    nodes_constructor = NodesConstructor(config["nodes_constructor"])
    rgbd_dataset = get_dataset(config["dataset"])

    scene_dir = Path(config["dataset"]["base_dir"]) / config["dataset"]["sequence"]
    if not (scene_dir / "bbox").exists():
        raise RuntimeError(f"GT directory not found: {scene_dir / 'bbox'}")

    matcher = match_mode_factory(args.matcher)
    accumulator = MetricsAccumulator()
    tracker_ids = StableTrackIDs()

    n_frames = len(rgbd_dataset)
    if args.limit is not None:
        n_frames = min(n_frames, args.limit)

    n_with_gt = 0
    logger.info(f"Benchmarking {n_frames} frames from {scene_dir}.")
    logger.info(f"Matcher={args.matcher}, IoU threshold={args.iou_threshold}, mode=bbox3d.")

    for step_idx in tqdm(range(n_frames)):
        frame = rgbd_dataset[step_idx]
        nodes_constructor.integrate(step_idx, frame, save_path=None)
        torch.cuda.empty_cache()

        frame_num = frame_num_from_path(rgbd_dataset.color_paths[step_idx])
        if frame_num is None:
            logger.warning(f"step {step_idx}: could not parse frame number; skipping GT.")
            continue

        gt_instances = load_gt_for_frame(
            scene_dir, frame_num,
            skip_labels=_DEFAULT_SKIP_LABELS,
            load_masks=False,  # 3D-only eval
        )
        if not gt_instances:
            continue
        n_with_gt += 1

        # Predictions: scene objects with a detection this frame.
        pred_instances: List[PredInstance] = []
        for obj in nodes_constructor.objects:
            ids: set = obj.get("id", set())
            if step_idx not in ids:
                continue
            bbox = obj.get("bbox")
            if bbox is None:
                continue
            try:
                bb = bbox_to_xyzxyz(bbox)
            except Exception as e:
                logger.debug(f"frame {step_idx}: bbox parse failed: {e}")
                continue
            pred_instances.append(PredInstance(
                pred_id=tracker_ids.get(obj),
                class_name=None,  # captioning happens after the loop in BBQ
                bbox_xyzxyz=bb,
            ))

        mapping, ious = matcher(
            gt_instances, pred_instances,
            iou_threshold=args.iou_threshold,
            match_mode="bbox3d",
        )
        accumulator.add_frame(FrameRecord(
            frame_idx=step_idx,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        ))

    if n_with_gt == 0:
        raise RuntimeError(
            f"No frames produced GT. Check that {scene_dir} contains "
            f"bbox/bboxes######_info.json files."
        )

    results = accumulator.compute()
    results["benchmark_config"] = {
        "scene": config["dataset"]["sequence"],
        "iou_threshold": args.iou_threshold,
        "matcher": args.matcher,
        "match_mode": "bbox3d",
        "frames_with_gt": n_with_gt,
        "total_frames_processed": n_frames,
    }

    print_summary(results, title=f"3D TRACKING - {config['dataset']['sequence']}")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics(results, out_dir, scene_name=config["dataset"]["sequence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D tracking benchmark for the BBQ pipeline on IsaacSim.")
    parser.add_argument(
        "--config_path",
        default="examples/configs/isaac/franka_cab_dex_more.yaml",
        help="BBQ pipeline config (same one used by main_with_edges.py).")
    parser.add_argument(
        "--output_dir", default="benchmark_results",
        help="Where per-run metric JSONs are written.")
    parser.add_argument(
        "--iou_threshold", type=float, default=0.25,
        help="3D bbox IoU threshold for a successful GT<->pred match.")
    parser.add_argument(
        "--matcher", choices=["greedy", "hungarian"], default="hungarian")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many frames (for quick smoke tests).")
    parser.add_argument("--logger_level", default="INFO")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), level=args.logger_level, colorize=True)

    set_seed()
    main(args)
