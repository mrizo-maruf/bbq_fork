import json
import numpy as np


def add_detection_noise(
    objects,
    center_noise_ratio=0.03,   # 3% of object size
    extent_noise_ratio=0.08,   # 8% size noise
    rot_noise_deg=5.0,         # rotation noise
    drop_prob=0.1              # 10% missed detections
):
    noisy_objects = []

    for obj in objects:
        # Simulate missed detection
        if np.random.rand() < drop_prob:
            continue

        center = np.array(obj["bbox_center"], dtype=float)
        extent = np.array(obj["bbox_extent"], dtype=float)

        # --- Center noise (scaled by object size)
        center_sigma = center_noise_ratio * extent
        center_noise = np.random.normal(0, center_sigma)
        new_center = center + center_noise

        # --- Extent noise (multiplicative)
        scale_noise = np.random.normal(1.0, extent_noise_ratio, size=3)
        new_extent = extent * scale_noise
        new_extent = np.clip(new_extent, 1e-3, None)  # avoid negative sizes

        # --- Rotation noise
        rot = obj.get("bbox_rotation")
        if rot is None:
            R = np.eye(3)
        else:
            R = np.array(rot, dtype=float)

        # small random rotation
        angle = np.deg2rad(np.random.normal(0, rot_noise_deg))
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R_noise = (
            np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K)
        )

        new_R = R_noise @ R

        # Save noisy object
        new_obj = obj.copy()
        new_obj["bbox_center"] = new_center.tolist()
        new_obj["bbox_extent"] = new_extent.tolist()
        new_obj["bbox_rotation"] = new_R.tolist()

        noisy_objects.append(new_obj)

    return noisy_objects


def add_noise_to_json(input_json, output_json):
    with open(input_json, "r") as f:
        objects = json.load(f)

    noisy = add_detection_noise(objects)

    with open(output_json, "w") as f:
        json.dump(noisy, f, indent=2)

    print(f"Saved {len(noisy)} noisy objects to {output_json}")


if __name__ == "__main__":
    add_noise_to_json("/home/docker_user/BeyondBareQueries/output/scenes_franka_cab_dex_more_nodes_edges.json", "/home/docker_user/BeyondBareQueries/output/scenes_franka_cab_dex_more_objects_noisy.json")
