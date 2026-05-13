# 3D Tracking in the BBQ Pipeline

This document explains how the BBQ scene-graph builder performs 3D multi-object tracking. Every persistent entry in `NodesConstructor.objects` is a track; the per-frame `integrate(...)` call is the tracker's update step.

Reference code:
- [bbq/objects_map/nodes_constructor.py](bbq/objects_map/nodes_constructor.py)
- [bbq/objects_map/detections_assembler.py](bbq/objects_map/detections_assembler.py)
- [bbq/objects_map/objects_associator.py](bbq/objects_map/objects_associator.py)
- [bbq/objects_map/utils/similarities.py](bbq/objects_map/utils/similarities.py)
- [bbq/objects_map/utils/objects.py](bbq/objects_map/utils/objects.py)

---

## 1. Tracker state

A single Python list, `NodesConstructor.objects` (`MapObjectList`), holds every active track. Each track is a `dict` with these load-bearing fields:

| Key | Type | Meaning |
|---|---|---|
| `pcd` | `open3d.PointCloud` | Accumulated 3D points in **world coordinates**. |
| `bbox` | `open3d.AxisAlignedBoundingBox` / `OrientedBoundingBox` | Bounding box derived from `pcd`. |
| `descriptor` | `torch.Tensor[1, D]` | Running DINO appearance descriptor (averaged across merges). |
| `num_detections` | `int` | How many detections have been folded into this track. |
| `id` | `set[int]` | Set of **frame indices** in which this track was observed — this is the per-frame "I detected this track" log. |

The track's identity is the Python dict itself. There is no explicit `track_id` field stored during tracking; downstream tools (e.g. `node_id`, the benchmark's `StableTrackIDs`) attach one later.

---

## 2. Per-frame update: `NodesConstructor.integrate`

Each frame consists of `(color, depth, intrinsics, pose)` where `pose` is the camera-to-world transform. The update step ([nodes_constructor.py:29-59](bbq/objects_map/nodes_constructor.py#L29-L59)) does five things:

```
        ┌─────────────────────────┐
RGB ──► │ MobileSAM masks         │ ──► xyxy boxes + binary masks + scores
        └─────────────────────────┘
        ┌─────────────────────────┐
RGB ──► │ DINO features           │ ──► per-pixel descriptor map
        └─────────────────────────┘
                                              │
                                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ DetectionsAssembler                                         │
        │   filter masks → unproject to pcd → DBSCAN → bbox + desc.   │
        └─────────────────────────────────────────────────────────────┘
                                              │  detected_objects
                                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ ObjectsAssociator                                           │
        │   spatial_sim (3D IoU) → visual_sim (DINO cos) → greedy     │
        │   match → spawn new tracks or merge into existing tracks.   │
        └─────────────────────────────────────────────────────────────┘
                                              │  updated scene_objects
                                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ merge_objects (every merge_interval frames)                 │
        │   periodic global cleanup: overlap matrix → collapse pairs  │
        └─────────────────────────────────────────────────────────────┘
```

If this is the very first frame (no existing tracks), every detection becomes a new track without going through association ([nodes_constructor.py:42-46](bbq/objects_map/nodes_constructor.py#L42-L46)).

---

## 3. Detection construction (`DetectionsAssembler`)

Per frame, `DetectionsAssembler` turns 2D mask proposals into 3D detections ([detections_assembler.py:53-136](bbq/objects_map/detections_assembler.py#L53-L136)). For each MobileSAM mask it sequentially applies:

1. **Confidence gate** (`mask_conf_threshold`, default 0.95).
2. **Mask-area gate** (`mask_area_threshold`, default 500 px).
3. **Containment subtraction** ([detections_assembler.py:138-183](bbq/objects_map/detections_assembler.py#L138-L183)) — if mask A contains mask B (IoB1 < 0.7, IoB2 > 0.8), subtract B from A and clean the edge with erode + dilate. Stops a bookshelf's mask from including all its books.
4. **2D bbox-area gate**: drop boxes that cover more than `max_bbox_area_ratio` (default 75%) of the image — they're usually backgrounds (wall, floor) masquerading as objects.
5. **Unproject to point cloud**: back-project masked depth pixels via `K` into camera coords, then transform to world coords with `pose`:
   ```python
   x = (u - cx) * depth / fx;  y = (v - cy) * depth / fy;  z = depth
   global_object_pcd = camera_object_pcd.transform(pose)
   ```
6. **Min-points gate** (`min_points_threshold`, default 150).
7. **DBSCAN denoising** (`dbscan_eps=0.05`, `min_points=10`); drop if the dominant cluster holds < 90% of points.
8. **Volume gate**: drop bboxes with volume < 1e-6 m³.
9. **DINO descriptor**: interpolate the mask down to feature-map resolution, then average DINO features inside the mask.

The surviving detection is packaged as:

```python
{
  'pcd': global_object_pcd,
  'bbox': pcd_bbox,
  'descriptor': loc_descriptor,
  'num_detections': 1,
  'id': {step_idx},      # this frame's index
}
```

The set of all surviving detections for the frame is a `DetectionList` of size M.

---

## 4. Data association (`ObjectsAssociator`)

Given M detections and N existing tracks, the associator decides for each detection: **merge into an existing track** or **spawn a new one** ([objects_associator.py:15-47](bbq/objects_map/objects_associator.py#L15-L47)).

### 4.1 Two-stage similarity

**Stage A — spatial similarity (3D IoU).** Compute the MxN matrix of axis-aligned-friendly 3D IoU between every detection bbox and every track bbox ([similarities.py:6-20](bbq/objects_map/utils/similarities.py#L6-L20)). Entries at or below `merge_det_obj_spatial_sim_thresh` (default 0.01) are set to `-inf`.

**Stage B — visual similarity (DINO cosine).** Only compute cosine similarity between detection and track descriptors where `spatial_sim > -inf` ([similarities.py:22-42](bbq/objects_map/utils/similarities.py#L22-L42)). This is the spatial gate keeping the visual computation cheap. Entries below `merge_det_obj_visual_sim_thresh` (default 0.5) are set to `-inf`.

So a detection can match a track only if it's **spatially close** *and* **visually similar**.

### 4.2 Greedy assignment

`merge_detections_to_objects` iterates detections in order ([objects_associator.py:30-47](bbq/objects_map/objects_associator.py#L30-L47)):

```python
for i in range(M):
    if visual_sim[i].max() == -inf:
        scene_objects.append(detected_objects[i])     # spawn new track
    else:
        j = visual_sim[i].argmax()
        scene_objects[j] = merge_obj2_into_obj1(
            scene_objects[j], detected_objects[i], ...)  # merge into existing
```

This is per-detection greedy — not Hungarian. Two detections can in principle match the same track; the first wins the dict-mutation race and the second updates the now-mutated track. In practice this is rare due to the containment-subtraction step.

### 4.3 Merging a detection into a track

[`merge_obj2_into_obj1`](bbq/objects_map/utils/objects.py#L195-L231) updates the surviving track **in place** (Python dict identity preserved):

| Field | Merge rule |
|---|---|
| `pcd` | Union the points, then voxel-downsample. |
| `bbox` | Recompute from the merged pcd. |
| `descriptor` | Average detection and track descriptors (with bias toward the track when `are_objects=False`, used in association). |
| `id` | Set union — the track's `id` set now includes the new frame index. |
| any other int / list | Summed (e.g. `num_detections += 1`). |

The detection dict is discarded; only the track dict survives. Detection point-cloud color is randomised per object at unproject time, so colours in the open3d viz are per-track, not per-class.

---

## 5. Periodic global merge

Every `merge_interval` frames (default 20), [`merge_objects`](bbq/objects_map/utils/objects.py#L233-L243) is run:

1. Compute an NxN **overlap matrix** between all tracks (`compute_overlap_matrix`, voxel-grid intersection counts).
2. Iterate pairs `(i, j)` in **decreasing overlap order**.
3. If `overlap_ratio > merge_objects_overlap_thresh` (default 0.7) **and** DINO cosine `> merge_objects_visual_sim_thresh` (default 0.4), merge `i` into `j` and drop `i`.

This is BBQ's main cleanup mechanism: two tracks that drifted out of sync but represent the same physical object (e.g. one chair seen from two views in the first few frames) get collapsed into one. From a MOT perspective every such collapse is an **ID switch**: the dropped track is "dead" and any future detection that would have matched it now matches the survivor instead. Tune `merge_interval` and the two thresholds carefully — they trade FP rate against ID-switch rate.

A second cleanup pass (overlap thresholds + minimum-detection / minimum-point gates) runs in `postprocessing()` after the loop completes ([bbq/objects_map/utils/postprocessing.py](bbq/objects_map/utils/postprocessing.py)), but that's a post-processing step, not a per-frame tracking step.

---

## 6. Track creation, continuation, and death

- **Birth**: a detection whose row in `visual_sim` is entirely `-inf` becomes a new track. Its `id` set is `{step_idx}`.
- **Continuation**: a detection's row has `argmax` j → it is merged into track j; track j's `id` set gains `step_idx`. The track's `pcd` grows, its `bbox` is recomputed, its `descriptor` is moved toward the new view.
- **Skip**: a detection that fails any `DetectionsAssembler` filter never enters association; no track gains an `id` for this frame.
- **Death**: tracks are not "killed" by missed detections. The only sources of death are:
  - **periodic merge** — track `i` is folded into `j`, `i`'s dict is dropped from `scene_objects`.
  - **postprocessing** — small / under-detected tracks pruned at the end.

So a long-lived track that simply leaves the camera view stays in `scene_objects` indefinitely; it just stops gaining new frame indices in its `id` set.

---

## 7. Coordinate frame

Every track's `pcd` and `bbox` are in **world coordinates**, because `DetectionsAssembler.create_object_pcd` transforms each per-frame point cloud via `pose.cpu().numpy()` ([detections_assembler.py:95](bbq/objects_map/detections_assembler.py#L95)). This is what lets BBQ accumulate views across the trajectory.

For this to be consistent with the input dataset poses, the BBQ dataset must use raw `traj.txt` poses, **not** relativised ones — i.e. `config["dataset"]["relative_pose"]` must be `False`. The benchmark and visualizer both assert this.

---

## 8. What "track ID" means in BBQ

There is **no explicit per-track integer ID** during tracking. The track *is* the dict; downstream tools assign IDs in their own way:

- **`describe()`** assigns `node_id` after the integration loop is complete ([nodes_constructor.py:73-85](bbq/objects_map/nodes_constructor.py#L73-L85)). This is the canonical "final" ID for grounding / edge prediction / output.
- **`benchmark_tracking.py`** keeps a `StableTrackIDs` map (`id(dict)` → monotonic int) so per-frame predicted IDs can be reported and `ID switches / consistency` measured. See [benchmark.md](benchmark.md) §3.
- **`visualize_tracking.py`** uses the same `StableTrackIDs` to colour masks; a colour change in the video corresponds 1-to-1 to a tracker ID change (typically caused by `merge_objects` collapsing a track).

The key invariant exploited by both: `merge_obj2_into_obj1` and `merge_overlap_objects` are **identity-preserving** for the surviving dict, so dict identity is a stable handle for as long as a track lives.

---

## 9. Hyperparameter cheat-sheet

| Param (config path) | Default | Effect on tracking |
|---|---|---|
| `detections_assembler.mask_conf_threshold` | 0.95 | High = fewer noisy detections, more FN. |
| `detections_assembler.mask_area_threshold` | 500 px | High = filters tiny objects. |
| `detections_assembler.max_bbox_area_ratio` | 0.75 | Drops near-full-frame masks (wall/floor). |
| `detections_assembler.min_points_threshold` | 150 | Min 3D points for a detection to survive. |
| `detections_assembler.dbscan_eps` | 0.05 m | DBSCAN neighbourhood for noise denoising. |
| `objects_associator.merge_det_obj_spatial_sim_thresh` | 0.01 | 3D IoU gate before computing visual sim. Lower = more candidate pairs. |
| `objects_associator.merge_det_obj_visual_sim_thresh` | 0.5 | DINO cosine threshold for matching a detection. Higher = more spawns, fewer merges. |
| `objects_associator.merge_interval` | 20 | How often `merge_objects` runs. Lower = aggressive ID collapsing (more ID switches). |
| `objects_associator.merge_objects_overlap_thresh` | 0.7 | Min overlap ratio for two tracks to be merged globally. |
| `objects_associator.merge_objects_visual_sim_thresh` | 0.4 | Min DINO cosine for a global track-track merge. |

Tuning intuitions:
- **High FN, missing tracks?** Lower `mask_conf_threshold`, lower `min_points_threshold`, lower `merge_det_obj_spatial_sim_thresh`.
- **High FP, ghost tracks?** Raise `merge_det_obj_visual_sim_thresh`, raise `min_points_threshold`, raise `mask_area_threshold`.
- **High ID switches?** Increase `merge_interval`, raise `merge_objects_overlap_thresh` (less aggressive merging).
- **Same physical object getting two tracks?** Lower `merge_objects_overlap_thresh`, lower `merge_objects_visual_sim_thresh` (more aggressive merging).
