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

# --- Engine 1: The AI-based Predictor ---
class VLSAT_Predictor:
    def __init__(self, model_path, config_path, rel_list_path):
        print(f"Loading VL-SAT model from {model_path}...")
        # TODO: Load your trained VL-SAT model checkpoint and configs here.
        pass

    def predict(self, nodes):
        print("Predicting edges with VL-SAT...")
        # TODO: Implement the VL-SAT prediction logic.
        edges = []
        print(f"VL-SAT predicted {len(edges)} edges.")
        return edges

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
    output_graph_path = f"output/scenegraphs/{predictor_type}_graph.json"
    print(f"Loading nodes from {input_nodes_path}...")
    with open(input_nodes_path, 'r') as f:
        nodes = json.load(f)
    print(f"Loaded {len(nodes)} object nodes.")

    if predictor_type == 'vlsat':
        # TODO: Update these paths
        predictor = VLSAT_Predictor(
            model_path="path/to/your/3dssg_best.ckpt",
            config_path="path/to/your/mmgnet.json",
            rel_list_path="path/to/your/relations.txt"
        )
    elif predictor_type == 'sceneverse':
        predictor = SceneVerse_Predictor()
    elif predictor_type == 'bbq':
        predictor = BBQ_Predictor()
    else:
        raise ValueError(f"Unknown predictor type: '{predictor_type}'.")

    edges = predictor.predict(nodes)
    scene_graph = {"nodes": nodes, "edges": edges}

    print(f"Saving complete scene graph to {output_graph_path}...")
    with open(output_graph_path, 'w') as f:
        json.dump(scene_graph, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a complete 3D scene graph with nodes and edges.")
    parser.add_argument("--input", required=True, help="Path to the nodes-only JSON file generated by main.py.")
    parser.add_argument(
        "--predictor", 
        required=True, 
        choices=['vlsat', 'sceneverse', 'bbq'],
        help="The engine to use for edge prediction."
    )
    
    args = parser.parse_args()
    build_graph(
        input_nodes_path=args.input,
        predictor_type=args.predictor
    )