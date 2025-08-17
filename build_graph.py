import os
import json
import argparse
from tqdm import tqdm
from itertools import permutations
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras


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
        print("Loading SceneVerse heuristic relationship functions...")
        # TODO: Import or define the necessary functions from the SceneVerse script.
        pass

    def predict(self, nodes):
        print("Predicting edges with SceneVerse Heuristics...")
        # TODO: Implement the SceneVerse prediction logic.
        edges = []
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
def build_graph(input_nodes_path, output_graph_path, predictor_type):
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
    parser.add_argument("--output", required=True, help="Path to save the new, complete scene graph JSON file.")
    parser.add_argument(
        "--predictor", 
        required=True, 
        choices=['vlsat', 'sceneverse', 'bbq'],
        help="The engine to use for edge prediction."
    )
    
    args = parser.parse_args()
    build_graph(
        input_nodes_path=args.input, 
        output_graph_path=args.output, 
        predictor_type=args.predictor
    )