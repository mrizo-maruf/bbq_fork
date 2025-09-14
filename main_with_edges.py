import os
import gzip
import pickle
import argparse
import warnings
import json
import yaml
import random
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor
from build_graph import VLSAT_Predictor, SceneVerse_Predictor

warnings.filterwarnings('ignore')

# Disable existing loggers to avoid conflicts
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


class TqdmLoggingHandler:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message, **kwargs):
        if message.strip() != "":
            tqdm.write(message, end="")

    def flush(self):
        pass

def set_seed(seed: int = 18) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")


def create_sceneverse_nodes(objects):
    """Convert BBQ objects to SceneVerse compatible format."""
    nodes_sceneverse = []
    for obj in objects:
        node = {
            'id': obj['node_id'],
            'bbox_center': np.asarray(obj['bbox'].center),
            'bbox_extent': np.asarray(obj['bbox'].extent),
            'description': obj['description']
        }
        nodes_sceneverse.append(node)
    return nodes_sceneverse


def save_objects(config, timestamp, objects, suffix=""):
    """Save objects to compressed pickle file."""
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    
    filename = timestamp.strftime("%m.%d.%Y_%H:%M:%S") + suffix + config["nodes_constructor"]["output_name_objects"]
    filepath = os.path.join(output_path, filename)
    
    results = {'objects': objects.to_serializable()}
    with gzip.open(filepath, 'wb') as file:
        pickle.dump(results, file)


def save_nodes_json(config, timestamp, nodes, suffix=""):
    """Save nodes to JSON file."""
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    
    filename = timestamp.strftime("%m.%d.%Y_%H:%M:%S") + suffix + config["nodes_constructor"]["output_name_nodes"]
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        json.dump(nodes, f)


def save_edges_json(config, edges, filename="sceneverse_edges.json"):
    """Save edges to JSON file."""
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    
    filepath = os.path.join(output_path, filename)
    with open(filepath, 'w') as f:
        json.dump(edges, f)

def main(args):
    """Main function to build 3D scene graph with edges."""
    timestamp = datetime.now()
    
    # Load configuration
    with open(args.config_path) as file:
        config = yaml.full_load(file)
    
    # Save metadata if save_path is provided
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        with gzip.open(os.path.join(args.save_path, "meta.pkl.gz"), "wb") as file:
            pickle.dump({"config": config}, file)
    
    logger.info(f"Parsed arguments. Utilizing config from {args.config_path}.")

    # Initialize components
    nodes_constructor = NodesConstructor(config["nodes_constructor"])
    rgbd_dataset = get_dataset(config["dataset"])

    # Section 3.1: Process RGBD sequence to accumulate 3D objects
    logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
    for step_idx in tqdm(range(len(rgbd_dataset))):
        frame = rgbd_dataset[step_idx]
        nodes_constructor.integrate(step_idx, frame, args.save_path)
        torch.cuda.empty_cache()
    
    nodes_constructor.postprocessing()
    torch.cuda.empty_cache()
    
    # Save intermediate results if save_path is provided
    if args.save_path:
        results = {'objects': nodes_constructor.objects.to_serializable()}
        with gzip.open(os.path.join(args.save_path, "frame_last_objects.pkl.gz"), "wb") as f:
            pickle.dump(results, f)

    # Section 3.2: Find 2D view to caption 3D objects
    logger.info('Finding 2D view to caption 3D objects.')
    nodes_constructor.project(
        poses=rgbd_dataset.poses,
        intrinsics=rgbd_dataset.get_cam_K()
    )
    torch.cuda.empty_cache()

    # Section 3.3: Caption 3D objects
    logger.info('Captioning 3D objects.')
    nodes = nodes_constructor.describe(colors=rgbd_dataset.color_paths)
    torch.cuda.empty_cache()

    # Section 3.4: Predict edges using VL-SAT
    logger.info('Predicting VL-SAT based edges')
    predictor_vl_sat = VLSAT_Predictor(
        model_path="/home/docker_user/BeyondBareQueries/3dssg_best_ckpt",
        config_path="config/mmgnet.json",
        rel_list_path="/home/docker_user/BeyondBareQueries/config/relations.txt"
    )
    
    vl_sat_edges, edge_feat_3d = predictor_vl_sat.predict(nodes_constructor.objects)
    nodes_constructor.add_edges_vl_sat(vl_sat_edges, edge_feat_3d)

    # Section 3.5: Predict edges using SceneVerse heuristics
    logger.info('Predicting SceneVerse based edges')
    predict_sceneverse = SceneVerse_Predictor()
    nodes_sceneverse = create_sceneverse_nodes(nodes_constructor.objects)
    sceneverse_edges = predict_sceneverse.predict(nodes_sceneverse)
    nodes_constructor.add_edges_sv(sceneverse_edges)
    
    # Save SceneVerse edges
    save_edges_json(config, sceneverse_edges, "sceneverse_edges.json")

    # Section 3.6: Save final results
    logger.info('Saving final objects and nodes.')
    save_objects(config, timestamp, nodes_constructor.objects, "_edge")
    save_nodes_json(config, timestamp, nodes, "_edge")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 3D scene object map with edge prediction. "
                   "For more information see Sec. 3.1 - 3.6.")
    parser.add_argument(
        "--config_path", 
        default="examples/configs/isaac/eval.yaml",
        help="Path to configuration file")
    parser.add_argument(
        "--logger_level", 
        default="INFO",
        help="Logging level")
    parser.add_argument(
        "--save_path", 
        default=None,
        help="Folder to save intermediate steps for visualization")
    
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(TqdmLoggingHandler(), level=args.logger_level, colorize=True)

    # Run main pipeline
    set_seed()
    main(args)
