# 3D Tracking Benchmark for the BBQ Pipeline

This document explains how [`benchmark_tracking.py`](benchmark_tracking.py) turns the BBQ scene-graph builder into a 3D multi-object tracker that can be scored with **MOTA, MOTP, ID Consistency, ID Switches, T-SR and T-mIoU**.

Metric definitions live in [`metrics/tracking_metrics.py`](metrics/tracking_metrics.py); IsaacSim GT layout follows the loader in [`data_loaders/isaacsim.py`](data_loaders/isaacsim.py).

---

## 1. The pipeline already performs 3D MOT

The BBQ scene-graph builder is, structurally, a 3D multi-object tracker. Every persistent entry in `NodesConstructor.objects` is a track, and `ObjectsAssociator` is the data-association step. Per frame, [`NodesConstructor.integrate`](bbq/objects_map/nodes_constructor.py#L29-L59) does:

1. Generate class-agnostic masks (MobileSAM) and DINO features for the current RGBD frame.
2. Build per-detection point clouds in **world coordinates** via `pose`-transform ([detections_assembler.py:95](bbq/objects_map/detections_assembler.py#L95)).
3. Associate each detection to an existing scene object using **spatial + visual similarity**, or spawn a new scene object ([objects_associator.py:30-47](bbq/objects_map/objects_associator.py#L30-L47)).
4. Every `merge_interval` frames, collapse overlapping scene objects via `merge_objects`.

Each scene object accumulates an `id` set holding **every frame index it has been detected in** ([detections_assembler.py:132](bbq/objects_map/detections_assembler.py#L132)). This is the hook the benchmark uses to emit per-frame predictions.

---

## 2. Extracting per-frame predictions

The benchmark wraps `integrate` rather than modifying it. After each frame:

```python
nodes_constructor.integrate(step_idx, frame, save_path=None)
for obj in nodes_constructor.objects:
    if step_idx in obj["id"]:                       # detected this frame
        pred_instances.append(PredInstance(
            pred_id=tracker_ids.get(obj),           # stable track ID
            bbox_xyzxyz=bbox_to_xyzxyz(obj["bbox"]),
        ))
```

Implementation choices behind that snippet:

- **"Detected this frame" filter.** Only scene objects whose accumulated `id` set contains `step_idx` are emitted. That matches GT visibility per frame and avoids inflating FPs with tracks that exist in the map but weren't seen this frame.
- **Bbox in world coords.** `obj["bbox"]` is either an `open3d.AxisAlignedBoundingBox` or `OrientedBoundingBox`. [`bbox_to_xyzxyz`](benchmark_tracking.py#L195-L201) converts both to `(xmin,ymin,zmin,xmax,ymax,zmax)` by taking min/max of `get_box_points()`, so downstream IoU only needs the axis-aligned form.
- **No captioning / VL-SAT.** Those run *after* the integration loop in `main_with_edges.py` and don't influence tracking — the benchmark stops at the integrate stage.

---

## 3. Stable predicted track IDs

ID-Consistency, IDF1-style metrics, and ID-switch counts only make sense if predicted IDs are stable across frames. BBQ's `node_id` is only assigned in `describe()` (post-hoc), so we need a track identifier that is **available at integration time and survives merges**.

Approach: use the **Python `id()` of the scene-object dict** as the stable handle, mapped to a monotonic counter. Implementation: [`StableTrackIDs`](benchmark_tracking.py#L222-L242).

```python
class StableTrackIDs:
    def get(self, obj: dict) -> int:
        key = id(obj)
        tid = self._tid.get(key)
        if tid is None:
            tid = self._next
            self._next += 1
            self._tid[key] = tid
            self._refs.append(obj)   # keep alive — prevents id() reuse
        return tid
```

Why this works with merges:

- `merge_obj2_into_obj1(obj1, obj2, ...)` mutates `obj1` **in place** and returns it ([utils/objects.py:195-231](bbq/objects_map/utils/objects.py#L195-L231)). The surviving dict's `id()` is unchanged → its track ID is preserved.
- `merge_overlap_objects` rebuilds the object list but only filters dicts ([utils/objects.py:190-191](bbq/objects_map/utils/objects.py#L190-L191)); surviving dicts keep Python identity.
- Dropped (merged-away) dicts would normally be garbage-collected, freeing their `id()` for reuse. The `_refs` list holds a strong reference to every dict we've ever seen, so `id()` collisions are impossible for the lifetime of the run.

Why not just add a `track_id` field to the dict? [`merge_obj2_into_obj1`](bbq/objects_map/utils/objects.py#L201-L207) auto-merges non-special keys with `obj1[k] += obj2[k]`, which would silently corrupt an integer ID. The side-channel mapping avoids touching pipeline internals.

---

## 4. Ground-truth loading

IsaacSim emits per-frame JSON/PNG bundles ([data_loaders/isaacsim.py](data_loaders/isaacsim.py)). The benchmark inlines a slim GT reader because the provided loader imports `core.types` / `depth_providers.*` modules that don't exist in this repo, and for 3D tracking we only need bboxes — not depth or the full loader machinery.

For each frame number `N`, [`load_gt_for_frame`](benchmark_tracking.py#L122-L188) reads:

| File | Used for |
|---|---|
| `bbox/bboxes000N_info.json` → `bboxes.bbox_3d.boxes[]` | 3D AABB (`aabb_xyzmin_xyzmax`), `track_id`, label |
| `bbox/bboxes000N_info.json` → `bboxes.bbox_2d_tight.boxes[]` | optional 2D bbox + label fallback |
| `seg/semantic000N_info.json` | colour↔`instance_seg_id` map (only if `load_masks=True`) |
| `seg/semantic000N.png` | per-pixel instance mask (only if `load_masks=True`) |

For each box the loader emits a `GTInstance(track_id, class_name, bbox_xyzxyz, …)`. Structural classes (`wall`, `floor`, `ground`, `ceiling`, `background`) are dropped. The benchmark currently runs in **3D-only mode** (`load_masks=False`); masks are only needed for `mask2d` matching.

---

## 5. Frame-index alignment

The BBQ `IsaacSimDataset` applies `start` / `stride` / `end` to the raw `rgb/` listing, so `step_idx` is **not** the IsaacSim frame number. The benchmark reads the 1-based frame number directly from the filename of `rgbd_dataset.color_paths[step_idx]` using `_FRAME_RE` ([benchmark_tracking.py:204-206](benchmark_tracking.py#L204-L206)). This means `stride > 1` configs are handled correctly — GT is fetched for the actual sampled frame, not the strided index.

---

## 6. Coordinate-frame alignment

BBQ stores predicted 3D bboxes in **world coordinates** (point clouds are transformed by `pose` in [detections_assembler.py:95](bbq/objects_map/detections_assembler.py#L95)). IsaacSim GT 3D AABBs are also world-space. For these two to be comparable, the BBQ dataset must use raw camera poses, **not** relativised ones.

The benchmark asserts `config["dataset"]["relative_pose"] == False` ([benchmark_tracking.py:341-345](benchmark_tracking.py#L341-L345)). If you flip that flag, predictions are anchored to frame-0 and GT isn't — every IoU goes to zero and metrics silently look terrible.

---

## 7. Matching strategy

Per frame:

```python
mapping, ious = matcher(
    gt_instances, pred_instances,
    iou_threshold=args.iou_threshold,   # default 0.25
    match_mode="bbox3d",
)
```

- `--matcher hungarian` (default) calls `match_hungarian` for the global IoU-optimal assignment. `--matcher greedy` picks the matcher used in [ConceptGraphs](https://github.com/concept-graphs/concept-graphs)-style evals.
- `match_mode="bbox3d"` uses `bbox_iou_3d` on axis-aligned `(xmin,ymin,zmin,xmax,ymax,zmax)` tuples ([tracking_metrics.py:95-117](metrics/tracking_metrics.py#L95-L117)).
- IoU threshold of 0.25 is conventional for 3D MOT (KITTI / nuScenes / ScanNet); raise to 0.5 for stricter eval.

The resulting `FrameRecord` (gt list, pred list, `mapping` gt→pred, `ious` gt→IoU) goes into a [`MetricsAccumulator`](metrics/tracking_metrics.py#L341-L489) which computes the headline metrics at the end:

| Metric | What it measures |
|---|---|
| **T-mIoU** | Mean 3D IoU per GT track over frames in which it was matched. |
| **T-SR** | Per GT track: fraction of frames in which it was successfully matched. |
| **ID Consistency** | Per GT track: fraction of matched frames where the predicted ID equals the modal predicted ID. |
| **ID Switches** | Total transitions in the predicted ID assigned to a GT track. |
| **MOTA** | `1 − (FN + FP + IDSW) / GT_total`. Reported alongside the FN / FP / IDSW ratios so you can see what's driving it. |
| **MOTP** | Mean IoU over **all** matched `(gt, pred)` pairs (pooled across frames). |

---

## 8. Multi-scene benchmarking

With `--dataset_root /path/to/IsaacSimData`:

1. `discover_scenes(root)` returns every subfolder with both `rgb/` and `bbox/`.
2. The `NodesConstructor` (and its MobileSAM + DINO weights) is **constructed once** and reused across scenes. Only `nodes_constructor.objects = MapObjectList()` and `StableTrackIDs` are reset between scenes.
3. The YAML config's `dataset.base_dir` and `dataset.sequence` are overridden per scene; every other field (intrinsics, thresholds, output paths) is left alone — so the same config is the source of truth.
4. Per-scene metrics are printed and dumped to `benchmark_results/<timestamp>/<scene>_metrics.json`.
5. After all scenes, [`aggregate_macro`](benchmark_tracking.py#L300-L313) computes a **macro average** (equal weight per scene) of T-mIoU, T-SR, ID Consistency, MOTA, MOTP and the FN/FP/IDSW ratios, plus pooled **sums** of counts (frames, GT/Pred instances, ID switches). Output: `_macro_avg_all_scenes_metrics.json`.
6. Per-scene exceptions are caught, logged, and recorded in `_failures.json` — one bad scene doesn't kill the run.

**Macro vs micro.** The aggregate weights every scene equally regardless of length. A pooled (micro-average) variant would re-feed every `FrameRecord` into a single global `MetricsAccumulator`; add this if you want frame-weighted numbers.

---

## 9. Known limitations

- **Class labels are not propagated.** `PredInstance.class_name=None`, so the per-class T-mIoU breakdown is keyed only by GT class. To get per-class predicted accuracy, run `nodes_constructor.describe(...)` after the loop and back-fill `class_name` on cached `PredInstance`s.
- **Detection bbox vs accumulated bbox.** The benchmark uses the *accumulated* bbox at the time of frame `t` (i.e. fused over all frames up to `t`). This is faithful to how BBQ "predicts", but a per-frame eval that uses the raw detection bbox of frame `t` would be more comparable to standard MOT trackers. Easy swap if needed — replace `obj["bbox"]` with the corresponding detection bbox cached in `ObjectsAssociator`.
- **Axis-aligned 3D IoU only.** Both BBQ's `OrientedBoundingBox` and IsaacSim's 3D boxes are reduced to AABBs. Oriented IoU would be more accurate for rotated objects but requires a different metric implementation.
- **Tracker is open-loop on GT.** The matcher computes GT↔pred assignment *for metrics*; the BBQ tracker itself has no access to GT. So this is a faithful "deploy as-is and score" benchmark.
