"""Visualize the BBQ 3D tracker as per-scene MP4s.

For each scene under ``--dataset_root`` (or the single scene named in the
config), runs the BBQ integration loop frame-by-frame and renders an MP4
where every predicted track is drawn as:

  * a coloured, alpha-blended 2D mask (colour is a deterministic function
    of the predicted track ID, so it only changes when the track ID changes),
  * a black-haloed track ID at the mask centroid.

The 2D mask is obtained by projecting the scene object's accumulated 3D
point cloud into the current camera view (inverse pose + intrinsics K),
then dilating the sparse pixel set into a solid blob.
"""
from __future__ import annotations

import argparse
import copy
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor
from bbq.objects_map.utils import MapObjectList
from benchmark_tracking import (
    StableTrackIDs,
    discover_scenes,
    set_seed,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- colour map

def color_for_tid(tid: int) -> Tuple[int, int, int]:
    """Deterministic RGB colour for *tid*. Same tid → same colour, forever."""
    # Golden-ratio-like hue step gives well-spread colours for small/large tids.
    h = int((tid * 47 + 17) % 180)            # OpenCV H range is [0, 179]
    hsv = np.uint8([[[h, 220, 240]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# ---------------------------------------------------------------- projection

def project_pcd_to_mask(
    pts_world: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    H: int, W: int,
    dilate_kernel: int,
    dilate_iters: int,
) -> np.ndarray:
    """Project world-coord points to an HxW binary mask under camera *pose_c2w*."""
    if pts_world.size == 0:
        return np.zeros((H, W), dtype=bool)

    T_w2c = np.linalg.inv(pose_c2w)
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_c = (T_w2c @ pts_h.T).T[:, :3]

    in_front = pts_c[:, 2] > 0.05
    if not in_front.any():
        return np.zeros((H, W), dtype=bool)
    pts_c = pts_c[in_front]

    uv = (K @ pts_c.T).T
    u = (uv[:, 0] / uv[:, 2]).astype(np.int32)
    v = (uv[:, 1] / uv[:, 2]).astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_img], v[in_img]
    if u.size == 0:
        return np.zeros((H, W), dtype=bool)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v, u] = 1
    if dilate_iters > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


# ---------------------------------------------------------------- rendering

def render_frame(
    bgr: np.ndarray,
    masks_and_ids: List[Tuple[np.ndarray, int]],
    alpha: float,
    min_mask_pixels: int,
) -> np.ndarray:
    """Composite all masks (alpha-blended) and draw track IDs at centroids."""
    out = bgr.astype(np.float32)
    # 1) alpha-blend each mask
    for mask, tid in masks_and_ids:
        if mask.sum() < min_mask_pixels:
            continue
        r, g, b = color_for_tid(tid)
        color = np.array([b, g, r], dtype=np.float32)   # BGR for OpenCV
        m3 = mask[..., None]
        out = np.where(m3, alpha * color + (1.0 - alpha) * out, out)
    out = out.astype(np.uint8)

    # 2) draw text on top
    for mask, tid in masks_and_ids:
        if mask.sum() < min_mask_pixels:
            continue
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        txt = str(tid)
        cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- per-scene driver

def visualize_scene(
    nodes_constructor: NodesConstructor,
    scene_dir: Path,
    dataset_cfg_template: dict,
    out_mp4: Path,
    args: argparse.Namespace,
) -> None:
    cfg = copy.deepcopy(dataset_cfg_template)
    cfg["base_dir"] = str(scene_dir.parent)
    cfg["sequence"] = scene_dir.name
    rgbd_dataset = get_dataset(cfg)

    nodes_constructor.objects = MapObjectList()
    tracker_ids = StableTrackIDs()

    n_frames = len(rgbd_dataset)
    if args.limit is not None:
        n_frames = min(n_frames, args.limit)
    if n_frames == 0:
        logger.warning(f"[{scene_dir.name}] no frames; skipping.")
        return

    # Probe first frame for output dimensions; intrinsics K is constant.
    color0, _, intrinsics_44, _ = rgbd_dataset[0]
    H, W = int(color0.shape[0]), int(color0.shape[1])
    K = intrinsics_44.cpu().numpy()[:3, :3]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, args.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_mp4}.")

    logger.info(f"[{scene_dir.name}] {n_frames} frames -> {out_mp4} @ {args.fps} fps")

    try:
        for step_idx in tqdm(range(n_frames), desc=scene_dir.name, leave=False):
            frame = rgbd_dataset[step_idx]
            color_t, _, _, pose_t = frame

            nodes_constructor.integrate(step_idx, frame, save_path=None)
            torch.cuda.empty_cache()

            rgb = color_t.cpu().numpy().astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pose_c2w = pose_t.cpu().numpy()

            masks_and_ids: List[Tuple[np.ndarray, int]] = []
            for obj in nodes_constructor.objects:
                if step_idx not in obj.get("id", set()):
                    continue
                pcd = obj.get("pcd")
                if pcd is None or len(pcd.points) == 0:
                    continue
                pts = np.asarray(pcd.points)
                mask = project_pcd_to_mask(
                    pts, pose_c2w, K, H, W,
                    dilate_kernel=args.dilate_kernel,
                    dilate_iters=args.dilate_iters,
                )
                if mask.any():
                    masks_and_ids.append((mask, tracker_ids.get(obj)))

            rendered = render_frame(
                bgr, masks_and_ids,
                alpha=args.alpha,
                min_mask_pixels=args.min_mask_pixels,
            )
            writer.write(rendered)
    finally:
        writer.release()
    logger.info(f"[{scene_dir.name}] saved {out_mp4}.")


# ---------------------------------------------------------------- main

def main(args: argparse.Namespace) -> None:
    with open(args.config_path) as f:
        config = yaml.full_load(f)
    if config["dataset"].get("relative_pose", False):
        raise RuntimeError(
            "config has relative_pose=True. Visualization projects 3D pcds "
            "(world coords) back to view using the dataset poses, so need "
            "relative_pose: False for consistent rendering."
        )

    if args.dataset_root:
        scenes = discover_scenes(Path(args.dataset_root))
        logger.info(f"Discovered {len(scenes)} scenes under {args.dataset_root}.")
    else:
        scenes = [Path(config["dataset"]["base_dir"]) / config["dataset"]["sequence"]]

    nodes_constructor = NodesConstructor(config["nodes_constructor"])
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    failures = {}
    for scene_dir in scenes:
        out_mp4 = run_dir / f"{scene_dir.name}.mp4"
        try:
            visualize_scene(nodes_constructor, scene_dir, config["dataset"], out_mp4, args)
        except Exception as e:
            logger.exception(f"[{scene_dir.name}] failed: {e}")
            failures[scene_dir.name] = str(e)

    if failures:
        logger.warning(f"{len(failures)} scene(s) failed: {sorted(failures)}")
    logger.info(f"All MP4s under {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the BBQ 3D tracker as per-scene MP4s.")
    parser.add_argument("--config_path",
        default="examples/configs/isaac/franka_cab_dex_more.yaml",
        help="BBQ pipeline config (same one used by main_with_edges.py).")
    parser.add_argument("--dataset_root", default=None,
        help="Root with one IsaacSim scene per subfolder. If unset, uses the config's single scene.")
    parser.add_argument("--output_dir", default="viz_results")
    parser.add_argument("--fps", type=int, default=10,
        help="Output MP4 frame rate.")
    parser.add_argument("--alpha", type=float, default=0.5,
        help="Mask overlay transparency in [0, 1]. 0 = invisible, 1 = solid.")
    parser.add_argument("--dilate_kernel", type=int, default=5,
        help="Morphological kernel size used to thicken sparse pcd projections into a mask.")
    parser.add_argument("--dilate_iters", type=int, default=2,
        help="Dilation iterations; raise if masks look too thin.")
    parser.add_argument("--min_mask_pixels", type=int, default=100,
        help="Skip drawing tracks whose projected mask is smaller than this.")
    parser.add_argument("--limit", type=int, default=None,
        help="Per scene: render at most this many frames (smoke test).")
    parser.add_argument("--logger_level", default="INFO")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), level=args.logger_level, colorize=True)

    set_seed()
    main(args)
