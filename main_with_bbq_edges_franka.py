import os
import gzip
import pickle
import argparse
import warnings
import json
import yaml
import random
from datetime import datetime
from time import perf_counter

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor
from build_graph import BBQ_Predictor

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

def save_objects_to_json(config, objects, json_name):

    def to_list(x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    output = []

    for obj in objects:
        for edge_key in ["edges_vl_sat", "edges_sv"]:
            if edge_key in obj and obj[edge_key] is not None:
                for edge in obj[edge_key]:
                    edge.pop("3d_feat", None)

    for obj in objects:
        output.append({
            "node_id": obj.get("node_id"),
            "bbox_center": to_list(obj.get("bbox_center")),
            "bbox_extent": to_list(obj.get("bbox_extent")),
            "bbox_rotation": to_list(obj.get("bbox_rotation")),
            "edges_vl_sat": to_list(obj.get("edges_vl_sat")),
            "edges_sv": to_list(obj.get("edges_sv")),
        })

    print(f'DEBUG: SAving at {config["nodes_constructor"]["output_path"] + "_" + json_name}')
    with open(config["nodes_constructor"]["output_path"] + "_" + json_name, "w") as f:
        json.dump(output, f, indent=2)

def save_nodes_json(config, timestamp, nodes, suffix=""):
    """Save nodes to JSON file."""
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    
    filename = timestamp.strftime("%m.%d.%Y_%H:%M:%S") + suffix + config["nodes_constructor"]["output_name_nodes"]
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        json.dump(nodes, f)

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_sets(v) for v in obj]
    return obj
    
def save_edges_json(config, edges, filename="sceneverse_edges.json"):
    """Save edges to JSON file."""
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    
    filename = f'{config["dataset"]["sequence"]}.json'
    filepath = os.path.join(output_path, filename)
    with open(filepath, 'w') as f:
        json.dump(convert_sets(edges), f)

from pathlib import Path

def get_subfolders(path):
    p = Path(path)
    return [f.name for f in p.iterdir() if f.is_dir()]


def log_section_time(section_name, start_time):
    elapsed_ms = (perf_counter() - start_time) * 1000
    logger.info(f"{section_name} completed in {elapsed_ms:.2f} ms")
    return elapsed_ms

def main(config_path, dataset_sequences):
    """Main function to build 3D scene graph with edges."""
    timestamp = datetime.now()
    
    # Load configuration
    with open(config_path) as file:
        config = yaml.full_load(file)
    
    for sequence_name in dataset_sequences:
        config["dataset"]["sequence"] = sequence_name
        print("======= WORKING ON SEQUENCE:", config["dataset"]["sequence"], "=======")

        # Initialize components
        nodes_constructor = NodesConstructor(config["nodes_constructor"])
        rgbd_dataset = get_dataset(config["dataset"])
        section_timings_ms = {}

        # Section 3.1: Process RGBD sequence to accumulate 3D objects
        section_start = perf_counter()
        logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
        for step_idx in tqdm(range(len(rgbd_dataset))):
            frame = rgbd_dataset[step_idx]
            nodes_constructor.integrate(step_idx, frame)
            torch.cuda.empty_cache()
        
        nodes_constructor.postprocessing()
        torch.cuda.empty_cache()
        section_timings_ms["3.1"] = log_section_time("Section 3.1", section_start)

        # Section 3.2: Find 2D view to caption 3D objects
        section_start = perf_counter()
        logger.info('Finding 2D view to caption 3D objects.')
        nodes_constructor.project(
            poses=rgbd_dataset.poses,
            intrinsics=rgbd_dataset.get_cam_K()
        )
        torch.cuda.empty_cache()
        section_timings_ms["3.2"] = log_section_time("Section 3.2", section_start)

        # Section 3.3: Caption 3D objects
        section_start = perf_counter()
        logger.info('Captioning 3D objects.')
        nodes = nodes_constructor.describe(colors=rgbd_dataset.color_paths)
        torch.cuda.empty_cache()
        section_timings_ms["3.3"] = log_section_time("Section 3.3", section_start)

        # Section 3.4: Predict edges using BBQ
        section_start = perf_counter()
        logger.info('Predicting BBQ based edges')
        bbq_edge_predictor = BBQ_Predictor()
        bbq_graph = bbq_edge_predictor.predict(nodes_constructor.objects)
        section_timings_ms["3.4"] = log_section_time("Section 3.4", section_start)

        # Section 3.5: Save BBQ edges
        section_start = perf_counter()
        save_edges_json(config, bbq_graph)
        section_timings_ms["3.5"] = log_section_time("Section 3.5", section_start)

        torch.cuda.empty_cache()
        logger.info(
            "Section timings (ms): "
            + ", ".join(
                f"{section}={duration:.2f}"
                for section, duration in section_timings_ms.items()
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 3D scene object map with edge prediction. "
                   "For more information see Sec. 3.1 - 3.6.")
    parser.add_argument(
        "--config_path", 
        default="examples/configs/isaac/kg_nav_IsaacSimData.yaml",
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
    config_path = "/home/docker_user/BeyondBareQueries/examples/configs/isaac/kg_nav_IsaacSimData.yaml"
    folders = get_subfolders("/home/docker_user/BeyondBareQueries/IsaacSimData/kg_nav_IsaacSimData")

    print(f"SCENES TO WORK WITH {folders}")
    main(config_path, folders)
