import json
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def create_aabb_from_center_extent(center, extent, color=(0.0, 1.0, 0.0)):
    """
    Create an Open3D AxisAlignedBoundingBox from center and full extent.
    extent = [dx, dy, dz]
    """
    center = np.asarray(center, dtype=float)
    extent = np.asarray(extent, dtype=float)

    min_bound = center - extent / 2.0
    max_bound = center + extent / 2.0

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    aabb.color = color
    return aabb


def load_relations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def collect_unique_objects(relations):
    """
    Collect unique objects by object id from both source and target.
    """
    objects = {}

    for rel in relations:
        src = rel["source"]
        tgt = rel["target"]

        sid = src["source_id"]
        tid = tgt["target_id"]

        if sid not in objects:
            objects[sid] = {
                "id": sid,
                "center": np.array(src["source_center"], dtype=float),
                "extent": np.array(src["source_extent"], dtype=float),
            }

        if tid not in objects:
            objects[tid] = {
                "id": tid,
                "center": np.array(tgt["target_center"], dtype=float),
                "extent": np.array(tgt["target_extent"], dtype=float),
            }

    return objects


def build_relation_lines(relations):
    """
    Build one LineSet for all relations.
    Also returns label positions and strings.
    """
    points = []
    lines = []
    colors = []
    labels = []

    for i, rel in enumerate(relations):
        src_center = np.array(rel["source"]["source_center"], dtype=float)
        tgt_center = np.array(rel["target"]["target_center"], dtype=float)
        relation_text = rel["relation"]

        start_idx = len(points)
        points.append(src_center)
        points.append(tgt_center)
        lines.append([start_idx, start_idx + 1])

        # line color: red
        colors.append([1.0, 0.0, 0.0])

        midpoint = (src_center + tgt_center) / 2.0

        # small offset so label is not exactly on the line
        offset = np.array([0.0, 0.0, 0.03 * (i % 3 + 1)])
        label_pos = midpoint + offset

        labels.append((label_pos, relation_text))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    return line_set, labels


def create_center_spheres(objects, radius=0.02):
    """
    Small spheres to show object centers.
    """
    geometries = []
    for obj_id, obj in objects.items():
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(obj["center"])
        mesh.paint_uniform_color([0.0, 0.0, 1.0])  # blue
        mesh.compute_vertex_normals()
        geometries.append((f"center_{obj_id}", mesh))
    return geometries


def main(json_path):
    relations = load_relations(json_path)
    objects = collect_unique_objects(relations)

    # Initialize Open3D GUI
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("3D Relations Viewer", 1400, 1000)
    vis.show_ground = True
    vis.show_skybox(False)

    # Add world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    world_frame.compute_vertex_normals()
    vis.add_geometry("world_frame", world_frame)

    # Add boxes
    for obj_id, obj in objects.items():
        bbox = create_aabb_from_center_extent(obj["center"], obj["extent"], color=(0.0, 1.0, 0.0))
        vis.add_geometry(f"bbox_{obj_id}", bbox)

        # Add object id label slightly above box center
        label_pos = obj["center"] + np.array([0.0, 0.0, obj["extent"][2] / 2.0 + 0.03])
        vis.add_3d_label(label_pos, f"id={obj_id}")

    # Add center spheres
    for name, geom in create_center_spheres(objects):
        vis.add_geometry(name, geom)

    # Add relation lines
    relation_lines, labels = build_relation_lines(relations)
    vis.add_geometry("relation_lines", relation_lines)

    # Add relation labels
    for pos, text in labels:
        vis.add_3d_label(pos, text)

    # Camera setup
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    # Change this to your JSON file path
    json_path = "/home/yehia/rizo/code/bbq_fork/scene_0.json"
    main(json_path)