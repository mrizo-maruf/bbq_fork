import os
import json
import argparse
from tqdm import tqdm
from itertools import permutations
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras
from bbq.sceneverse_heuristics.relationships.proximity import cal_proximity_relationships
from bbq.sceneverse_heuristics.relationships.support import cal_support_relations
from bbq.sceneverse_heuristics.relationships.hanging import cal_hanging_relationships
from bbq.sceneverse_heuristics.relationships.proximity import cal_proximity_relationships
from bbq.sceneverse_heuristics.relationships.multi_objs import find_aligned_furniture
from bbq.sceneverse_heuristics.relationships.multi_objs import find_middle_furniture
from bbq.sceneverse_heuristics.relationships.ObjNode import ObjNode
import networkx as nx
import bbq.sceneverse_heuristics.relationships.ssg_utils as utils
import gzip
import pickle
from src.utils.config import Config
from src.model.model import MMGNet
from src.utils import op_utils
from itertools import product
import open3d as o3d
from collections import Counter

# --- Helper functions required by the BBQ Heuristic ---
def egoview_project(target, anchor, center):
    anchor_obj_loc = torch.tensor(anchor['obj_loc']).unsqueeze(0).float()
    cam_pos = torch.tensor(center).unsqueeze(0).float()
    R, T = look_at_view_transform(eye=cam_pos, at=anchor_obj_loc, up=((0.0, 0.0, 1.0),))
    camera = FoVOrthographicCameras(device='cpu', R=R, T=T)

    pos = torch.tensor(target['obj_loc']).unsqueeze(0).float()
    target_pos_2d = camera.transform_points_screen(pos, image_size=(512, 2048))
    anchor_pose_2d = camera.transform_points_screen(anchor_obj_loc, image_size=(512, 2048))

    return target_pos_2d, anchor_pose_2d

def get_semantic_edge(target, anchor, center_point):
    t = {"obj_loc": target}
    a = {"obj_loc": anchor}
    target_pos_2d, anchor_pose_2d = egoview_project(t, a, center_point)

    relations = []
    if target_pos_2d[0, 0] < anchor_pose_2d[0, 0]:
        relations.append("left")
    if target_pos_2d[0, 0] > anchor_pose_2d[0, 0]:
        relations.append("right")
    if target_pos_2d[0, 2] < anchor_pose_2d[0, 2]:
        relations.append("front")
    if target_pos_2d[0, 2] > anchor_pose_2d[0, 2]:
        relations.append("back")
    if target_pos_2d[0, 1] < anchor_pose_2d[0, 1]:
        relations.append("above")
    if target_pos_2d[0, 1] > anchor_pose_2d[0, 1]:
        relations.append("below")

    return relations

def convert_numpy_to_list(data):
    """
    Recursively converts numpy arrays in a dictionary or list to Python lists.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def remove_pcd_from_nodes(nodes):
    """
    Creates a new list of nodes with the 'pcd_np' key removed from each.
    """
    cleaned_nodes = []
    keys_to_remove = ['pcd_np', 'pcd_color_np', 'descriptor']

    for node in nodes:
        # Create a copy of the node dictionary without the 'pcd_np' key
        node_copy = {key: value for key, value in node.items() if key not in keys_to_remove}
        cleaned_nodes.append(node_copy)
    return cleaned_nodes

# --- Engine 1: The AI-based Predictor ---
class VLSAT_Predictor:
    def __init__(self, model_path, config_path, rel_list_path, device="cuda"):
        """
        Initializes the VL-SAT model by loading the trained checkpoint.
        """
        print(f"Loading VL-SAT model from {model_path}...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # --- This logic is ported from vl_sat_inference.py ---
        # TODO: Replace with your actual VL-SAT file loading logic
        self.config = Config(config_path)
        self.config.exp = model_path
        self.config.MODE = "eval"
        self.padding = 0.2
        self.model = MMGNet(self.config)
        
        if torch.cuda.is_available() and len(self.config.GPU) > 0:
            self.config.DEVICE = torch.device("cuda")
        else:
            self.config.DEVICE = torch.device("cpu")
        self.model.load(best=True)
        self.model.model.eval()
        
        # Load the list of possible relationship names
        with open(rel_list_path, "r") as f:
            self.relationships_list = f.readlines()
        
        self.rel_id_to_rel_name = {
            i: name.strip()
            for i, name in enumerate(self.relationships_list[1:])
        }
        print(f"VL-SAT Predictor initialized for device '{self.device}'.")

    
    def _bbox_center_from_bbox_np(self, bbox_np):
        """
        bbox_np: (N,3) numpy array of bounding box corner points (e.g. 8 corners).
        Returns center (3,)
        """
        pts = np.asarray(bbox_np)
        return pts.mean(axis=0)


    def pcd_denoise_dbscan(self, _pcd, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
        ### Remove noise via clustering
        # print("Type", '**'*30, type(_pcd))
        pcd_clusters = _pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
        )
        
        # Convert to numpy arrays
        obj_points = np.asarray(_pcd.points)
        #obj_colors = np.asarray(pcd.colors)
        pcd_clusters = np.array(pcd_clusters)

        # Count all labels in the cluster
        counter = Counter(pcd_clusters)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            
            # Create mask for points in the largest cluster
            largest_mask = pcd_clusters == most_common_label

            # Apply mask
            largest_cluster_points = obj_points[largest_mask]
            #largest_cluster_colors = obj_colors[largest_mask]
            
            # If the largest cluster is too small, return the original point cloud
            if len(largest_cluster_points) < 5:
                return _pcd

            # Create a new PointCloud object
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            
            _pcd = largest_cluster_pcd
            
        return _pcd

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point
    
    def _preprocess_objects(self, bbq_objects):
        """
        Converts a list of BBQ objects into the tensor format required by the VL-SAT model.
        """
        num_objects = len(bbq_objects)
        print(f"Num objects: {num_objects}")

        pcds = {}
        for obj_id, obj in enumerate(bbq_objects):
            pcds[obj_id] = {}
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(obj['pcd_np'])
            # print(f"Type: {'*'*30} {type(pcd_o3d)}")
            pointcloud_ = self.pcd_denoise_dbscan(pcd_o3d)
            
            # TO-DO: Add pose information if available
            center = self._bbox_center_from_bbox_np(obj['bbox_np'])
            obj['pose'] = center.tolist() 
            pose = obj["pose"]
            
            # random pose
            # pose = [random.randint(10, 50) for _ in range(3)]
            

            pcd_array = np.array(pointcloud_.points)
            
            size = [0.3, 0.3, 0.15]
            nx, ny, nz = (16, 16, 16)
            
            x = np.linspace(pose[0] - size[0]/2, pose[0] + size[0]/2, nx)
            y = np.linspace(pose[1] - size[1]/2, pose[1] + size[1]/2, ny)
            z = np.linspace(pose[2] - size[2]/2, pose[2] + size[2]/2, nz)
            xv, yv, zv = np.meshgrid(x, y, z)
            
            # print(x.shape, y.shape, z.shape)
            # print(xv)
            # print(xv.shape, yv.shape, zv.shape)
            #print(xv.flatten())
            
            grid_pc = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=1)
            pcds[obj_id]['point_cloud'] = grid_pc
            
            pcds[obj_id]['position'] = [
                    np.round(np.mean(pcds[obj_id]['point_cloud'][:, 0]),2),
                    np.round(np.mean(pcds[obj_id]['point_cloud'][:, 1]),2),
                    np.round(np.mean(pcds[obj_id]['point_cloud'][:, 2]),2),
                    ]
        return pcds

    def preprocess_poinclouds(self, points, num_points):
        assert len(points) > 1, "Number of objects should be at least 2"
        print(f"Num of points: {num_points}")
        print(f"Shape: {points[0].shape}")
        
        edge_indices = list(product(list(range(len(points))), list(range(len(points)))))
        edge_indices = [i for i in edge_indices if i[0]!=i[1]]

        num_objects = len(points)
        dim_point = points[0].shape[-1]

        instances_box = dict()
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        obj_2d_feats = np.zeros([num_objects, 512])

        for i, pcd in enumerate(points):
            # get node point
            min_box = np.min(pcd, 0) - self.padding
            max_box = np.max(pcd, 0) + self.padding
            instances_box[i] = (min_box, max_box)
            choice = np.random.choice(len(pcd), num_points, replace=True)
            pcd = pcd[choice, :]
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(pcd))
            pcd = torch.from_numpy(pcd.astype(np.float32))
            pcd = self.zero_mean(pcd)
            obj_points[i] = pcd

        edge_indices = torch.tensor(edge_indices, dtype=torch.long).permute(1, 0)
        obj_2d_feats = torch.from_numpy(obj_2d_feats.astype(np.float32))    
        obj_points = obj_points.permute(0, 2, 1)
        batch_ids = torch.zeros((num_objects, 1))
        return obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids

    def predict_relations(self, obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids):
        obj_points = obj_points.to(self.config.DEVICE)
        obj_2d_feats = obj_2d_feats.to(self.config.DEVICE)
        edge_indices = edge_indices.to(self.config.DEVICE)
        descriptor = descriptor.to(self.config.DEVICE)
        batch_ids = batch_ids.to(self.config.DEVICE)
        with torch.no_grad():
            rel_cls_3d = self.model.model(
                obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids=batch_ids
            )
        return rel_cls_3d

    def save_relations(self, tracking_ids, timestamps, class_names, predicted_relations, edge_indices):
        saved_relations = []
        for k in range(predicted_relations.shape[0]):
            idx_1 = edge_indices[0][k].item()
            idx_2 = edge_indices[1][k].item()

            id_1 = tracking_ids[idx_1]
            id_2 = tracking_ids[idx_2]

            timestamp_1 = timestamps[idx_1]
            timestamp_2 = timestamps[idx_2]

            class_name_1 = class_names[idx_1]
            class_name_2 = class_names[idx_2]

            rel_id = torch.argmax(predicted_relations, dim=1)[k].item()
            rel_name = self.rel_id_to_rel_name[rel_id]

            rel_dict = {
                #"id_1": id_1,
                #"timestamp_1": timestamp_1,
                "class_name_1": class_name_1,
                "rel_name": rel_name,
                #"id_2": id_2,
                #"timestamp_2": timestamp_2,
                "class_name_2": class_name_2,
                #"rel_id": rel_id,
                
            }
            saved_relations.append(rel_dict)

        return saved_relations

    def predict(self, nodes):
        """
        The main prediction method. It takes BBQ objects, preprocesses them,
        runs inference, and formats the output.
        """
        print("Predicting edges with VL-SAT...")

        if len(nodes) < 2:
            print("Not enough nodes to predict relationships.")
            return []
            
        # 1. Preprocess the nodes into point clouds and other required formats
        preprocessed_data = self._preprocess_objects(nodes)
        pcds_list = [_pcd['point_cloud'] for _pcd in preprocessed_data.values()]
        
        print(f"pcds_list len: {len(pcds_list)}, {type(pcds_list[0])}")
        print(f"pcds_list shapes: {[_pcd.shape for _pcd in pcds_list]}")
        
        # 2. Preprocess the point clouds for the model
        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = self.preprocess_poinclouds(
            pcds_list,
            self.config.dataset.num_points
        )
        
        # 3. Predict
        predicted_relations = self.predict_relations(obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids)
        topk_values, topk_indices = torch.topk(predicted_relations, 5, dim=1,  largest=True)
        # print(topk_indices, topk_values)
        
        # 4. Save the relations in a standardized format
        tracking_ids = [str(i) for i in range(len(pcds_list))]
        timestamps = ["001539" for i in range(len(pcds_list))]
        class_names = [f"class {i}" for i in range(len(pcds_list))]
        
        saved_relations = self.save_relations(tracking_ids, timestamps, class_names, predicted_relations, edge_indices)
        
        with open("output/scenegraphs/saved_relations.json", "w") as f:
            json.dump(saved_relations, f, indent=4)

        print("Saved to output/scenegraphs/saved_relations.json")
        
        return saved_relations
    
# --- Engine 2: The Advanced Heuristic Predictor ---
class SceneVerse_Predictor:
    def __init__(self):
        """
        Initializes the rule-based SceneVerse heuristic predictor.
        """
        print("Initializing SceneVerse heuristic relationship functions...")
        # This function sets a fixed virtual camera viewpoint for consistency.
        self.camera_view = [0, -1, 0]
        self.camera_pos = [0, 0, 0]
        self.camera_view = self.camera_view / np.linalg.norm(self.camera_view)

        if self.camera_view[0] < 0:
            self.camera_angle = -utils.get_theta(self.camera_view, [0, 1, 0])
        else:
            self.camera_angle = utils.get_theta(self.camera_view, [0, 1, 0])

    def _convert_bbq_to_sceneverse(self, bbq_node):
        """
        Helper function to convert a BBQ object dict to a SceneVerse ObjNode.
        """
        return ObjNode(
            id=bbq_node['id'],
            label=bbq_node['description'],
            position=np.array(bbq_node['bbox_center']),
            size=np.array(bbq_node['bbox_extent'])
        )

    def predict(self, nodes):
        """
        Takes a list of BBQ object nodes and calculates their relationships
        using the suite of SceneVerse heuristic functions.
        """
        print("Predicting edges with SceneVerse Heuristics...")

        if len(nodes) < 2:
            print("Not enough nodes to predict relationships.")
            return []

        # 1. Use the adapter to convert all BBQ nodes to SceneVerse ObjNodes
        ObjNode_dict = {node['id']: self._convert_bbq_to_sceneverse(node) for node in nodes}

        # Calculate scene boundaries, which some heuristics need
        all_centers = np.array([node['bbox_center'] for node in nodes])
        all_extents = np.array([node['bbox_extent'] for node in nodes])
        min_coords = np.min(all_centers - all_extents / 2, axis=0)
        max_coords = np.max(all_centers + all_extents / 2, axis=0)
        scene_high = max_coords[2] - min_coords[2]

        # 2. Call the "Specialist" functions in sequence, mimicking ssg_main.py
        
        # A. Support and Embedded relationships (The "Gravity" Specialist)
        support_relations, embedded_relations, hanging_objs_dict = cal_support_relations(ObjNode_dict, self.camera_angle)
        
        print(f"support rel: {support_relations}, embedded_relations: {embedded_relations}, hanging_objs_dict: {hanging_objs_dict}")
        
        # B. Build a temporary graph to find neighbors for proximity checks
        G = nx.DiGraph()
        for node_id in ObjNode_dict:
            G.add_node(node_id)
        for rel in support_relations:
            G.add_edge(rel[0], rel[1]) # Add support edges for neighborhood context

        # C. Hanging relationships (The "Ceiling" Specialist)
        hanging_relationships = cal_hanging_relationships(ObjNode_dict, hanging_objs_dict, self.camera_angle, scene_high)

        # D. Proximity relationships (The "Navigator" Specialist)
        proximity_relations = []
        for node_id in G:
            # Find immediate neighbors using the temporary graph
            successors = list(G.successors(node_id))
            predecessors = list(G.predecessors(node_id))
            neighbor_ids = list(set(successors + predecessors))
            if len(neighbor_ids) > 1:
                proximity = cal_proximity_relationships(neighbor_ids, self.camera_angle, ObjNode_dict, scene_high)
                print(f"proximity: {proximity}")
                proximity_relations.extend(proximity)
        
        # Calculate proximity for all pairs as a fallback (optional but robust)
        # Note: You can enable this if the neighbor-based approach is too sparse.
        all_ids = list(ObjNode_dict.keys())
        proximity_relations.extend(cal_proximity_relationships(all_ids, self.camera_angle, ObjNode_dict, scene_high))


        # E. Multi-Object relationships (The "Interior Designer" Specialist)
        all_ids = list(ObjNode_dict.keys())
        aligned_furniture = find_aligned_furniture(all_ids, ObjNode_dict, 0.065)
        middle_relationships = find_middle_furniture(proximity_relations, ObjNode_dict)
        print(f"aligned_furniture: {aligned_furniture}, middle_relationships: {middle_relationships}")

        # 3. Collect all relationship lists into one master list
        # Note: multi-object relations have a different format, so we handle them separately.
        all_relations = support_relations + embedded_relations + hanging_relationships + proximity_relations
        
        # 4. Convert all found relations to the standardized output format
        edges = []
        for rel in all_relations:
            # Standard format is [source_id, target_id, relation_name]
            if len(rel) == 3:
                source_id, target_id, relation_name = rel
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": str(relation_name).strip()
                })

        # Process multi-object relations
        for rel in aligned_furniture:
            # Format: [[id1, id2, id3], "Aligned"]
            obj_ids, relation_name = rel
            # Create pairwise "aligned with" relations
            for id1, id2 in combinations(obj_ids, 2):
                edges.append({"source": id1, "target": id2, "relation": "aligned with"})

        for rel in middle_relationships:
            # Format: [middle_obj_id, [obj1_id, obj2_id], "in the middle of"]
            middle_id, (id1, id2), relation_name = rel
            edges.append({"source": middle_id, "target": id1, "relation": "in middle of"})
            edges.append({"source": middle_id, "target": id2, "relation": "in middle of"})
            
        print(f"SceneVerse predicted {len(edges)} edges.")
        return edges

# --- Engine 3: The Baseline Heuristic Predictor ---
class BBQ_Predictor:
    def __init__(self):
        print("Using default BBQ heuristic predictor.")
        self.scene_center_point = None

    def predict(self, nodes):
        print("Predicting edges with BBQ Heuristic...")
        
        # Calculate the center point of the scene from all object centers
        all_centers = [node['bbox_center'] for node in nodes]
        self.scene_center_point = np.mean(all_centers, axis=0)
        
        edges = []
        # Iterate through every possible pair of objects (e.g., A->B and B->A)
        for i, j in tqdm(permutations(range(len(nodes)), 2), total=len(nodes)*(len(nodes)-1)):
            source_node = nodes[i]
            target_node = nodes[j]

            relations = get_semantic_edge(
                target=source_node['bbox_center'],
                anchor=target_node['bbox_center'],
                center_point=self.scene_center_point
            )
            
            # For each relation found (left, above, etc.), create a standard edge dict
            for rel in relations:
                edges.append({
                    "source": source_node['id'],
                    "target": target_node['id'],
                    "relation": rel
                })
                
        print(f"BBQ Heuristic predicted {len(edges)} edges.")
        return edges

# --- Main Orchestration Logic ---
def build_graph(input_nodes_path, predictor_type):
    output_graph_path = f"./output/scenegraphs/{predictor_type}_graph.json"
    print(f"Loading nodes from {input_nodes_path}...")
    
    if predictor_type in ['bbq', 'sceneverse']:
        # These predictors only need the JSON file with bounding boxes.
        if not input_nodes_path.endswith('.json'):
            raise ValueError("For 'bbq' or 'sceneverse' predictors, the --input file must be the ...nodes.json file.")
        with open(input_nodes_path, 'r') as f:
            nodes = json.load(f)
            
    elif predictor_type == 'vlsat':
        # The VL-SAT predictor needs the PKL file with full point cloud data.
        if not input_nodes_path.endswith('.pkl.gz'):
            raise ValueError("For the 'vlsat' predictor, the --input file must be the frame_last_objects.pkl.gz file.")
        with gzip.open(input_nodes_path, "rb") as f:
            data = pickle.load(f)
            # The object list is nested under the 'objects' key in the pickle file
            nodes = data['objects']
    else:
        raise ValueError("Invalid predictor type.")

    print(f"Loaded {len(nodes)} object nodes.")

    
    # --- INITIALIZE PREDICTOR (This part remains the same) ---
    if predictor_type == 'vlsat':
        predictor = VLSAT_Predictor(
            model_path="./3dssg_best_ckpt",
            config_path="./config/mmgnet.json",
            rel_list_path="./config/relations.txt"
        )
    elif predictor_type == 'sceneverse':
        predictor = SceneVerse_Predictor()
    elif predictor_type == 'bbq':
        predictor = BBQ_Predictor()
        
    edges = predictor.predict(nodes)
    # scene_graph = {"nodes": nodes, "edges": edges}

    # nodes_without_pcd = remove_pcd_from_nodes(nodes)

    # scene_graph = {"nodes": nodes_without_pcd, "edges": edges}
    scene_graph = {"edges": edges}
    
    scene_graph_serializable = convert_numpy_to_list(scene_graph)

    print(f"Saving complete scene graph to {output_graph_path}...")
    with open(output_graph_path, 'w') as f:
        # Save the translated, serializable version of the graph
        json.dump(scene_graph_serializable, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a complete 3D scene graph with nodes and edges.")
    parser.add_argument(
        "--predictor", 
        required=True, 
        choices=['vlsat', 'sceneverse', 'bbq'],
        help="The engine to use for edge prediction."
    )
    
    args = parser.parse_args()
    input_file = "/home/docker_user/BeyondBareQueries/output/frame_last_objects.pkl.gz"
    
    build_graph(
        input_nodes_path=input_file,
        predictor_type=args.predictor
    )