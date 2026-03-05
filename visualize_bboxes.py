import json
import numpy as np
import open3d as o3d


def create_bbox(center, extent, rotation=None):
    """
    Create an Open3D bounding box.
    center: [x, y, z]
    extent: [dx, dy, dz]
    rotation: 3x3 matrix or None
    """
    center = np.array(center, dtype=float)
    extent = np.array(extent, dtype=float)

    # If rotation is None -> identity
    if rotation is None:
        R = np.eye(3)
    else:
        R = np.array(rotation, dtype=float)

    bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    bbox.color = (1, 0, 0)  # red
    return bbox


def load_and_visualize(json_path):
    with open(json_path, "r") as f:
        objects = json.load(f)

    geometries = []

    obj_i = 0
    for obj in objects:
        obj_i = obj_i + 1
        center = obj.get("bbox_center")
        extent = obj.get("bbox_extent")
        rotation = obj.get("bbox_rotation")

        # Skip if missing data
        if center is None or extent is None:
            continue

        # Handle null rotation from JSON
        if rotation is None:
            R = None
        else:
            R = rotation

        bbox = create_bbox(center, extent, R)
        geometries.append(bbox)

        print(obj_i)

    # Add coordinate frame for reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    json_path = "/home/docker_user/BeyondBareQueries/output/scenes_franka_cab_dex_more_objects_noisy.json"  # change path
    load_and_visualize(json_path)
