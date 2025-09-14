import os
import gzip
import pickle

import torch
torch.set_grad_enabled(False)
from loguru import logger

from bbq.objects_map import ObjectsAssociator, DetectionsAssembler, \
    create_object_masks, describe_objects
from bbq.objects_map.utils import MapObjectList, merge_objects, postprocessing
from bbq.models import DINOFeaturesExtractor, ClassAgnosticMaskGenerator


class NodesConstructor:
    def __init__(self, config):
        self.config = config
        self.objects = MapObjectList()

        self.mask_generator = ClassAgnosticMaskGenerator(
            **config["mask_generator"])
        self.features_generator = DINOFeaturesExtractor(
            **config["dino_generator"])
        self.detections_assembler = DetectionsAssembler(
            **config["detections_assembler"])
        self.objects_mapper = ObjectsAssociator(
            **config["objects_associator"])

    def integrate(self, step_idx, frame, save_path=False):
        color, depth, intrinsics, pose = frame
        # generate class-agnostic masks
        masks_result = self.mask_generator(color)
        # generate DINO features
        descriptors = self.features_generator(color)
        # aggregate information about detected objects
        detected_objects = self.detections_assembler(
            step_idx, color, depth, intrinsics, pose, masks_result, descriptors)
        
        if len(detected_objects) == 0 and len(self.objects) != 0:
            logger.debug("no detected objects")
            return
        if len(self.objects) == 0:
            # add all detections to the map
            for i in range(len(detected_objects)):
                self.objects.append(detected_objects[i])
            logger.debug(f"Initialize {len(detected_objects)} detections as objects")

        # objects accumulation
        self.objects = self.objects_mapper(detected_objects, self.objects)
        if step_idx > 0 and step_idx % self.config["objects_associator"]["merge_interval"] == 0:
            self.objects = merge_objects(self.objects,
                self.config["objects_associator"]["merge_objects_overlap_thresh"],
                self.config["objects_associator"]["merge_objects_visual_sim_thresh"],
                self.config["detections_assembler"]["downsample_voxel_size"])
        
        if save_path:
            results = {'objects': self.objects.to_serializable()}
            with gzip.open(os.path.join(save_path, f"frame_{step_idx}_objects.pkl.gz"), "wb") as f:
                pickle.dump(results, f)
            
    def postprocessing(self):
        self.objects = postprocessing(self.objects, self.config)

    def project(self, poses, intrinsics):
        self.objects = create_object_masks(
            self.objects, poses, intrinsics,
            self.config["projector"]["num_views"],
            self.config["projector"]["top_k"],
            (self.config["projector"]["desired_height"], 
             self.config["projector"]["desired_width"])
        )

    def describe(self, colors):
        templates = describe_objects(self.objects, colors, isLLava=False)
        
        for temp, obj in zip(templates, self.objects):
            obj["clip_descriptor"] = temp["clip_descriptor"]
            obj["node_id"] = temp["id"]
            obj['description'] = temp['description']
            # obj["bbox_extent"] = temp["bbox_extent"]
            # obj["bbox_center"] = temp["bbox_extent"]
            # if obj['id'] == temp['id']:
            # else:
            #     print(f"IDs did not match, obj id: {obj['id']}, temp id: {temp['id']}")
            del temp["clip_descriptor"]
        # print(f"objects: {list(self.objects[0].keys())}")
        # print(f"template: {list(templates[0].keys())}")
        
        return templates
    
    def add_edges_vl_sat(self, predicted_edges, edge_feat_3d):
        """
        Adds relational edges to each object (node) in the map.

        Each object in self.objects will be updated with a new key 'edges',
        which is a list of all relations it participates in (either as 
        subject or object).

        Args:
            predicted_edges (list): A list of dictionaries, where each dictionary
                                    represents a relational edge between two objects.
                                    This is the output of your `save_relations` function.
        """
        # Step 1: Initialize an empty 'edges' list for every object.
        # This ensures the key exists even for nodes with no connections.
        for obj in self.objects:
            obj['edges_vl_sat'] = []
            
        # Step 2: Create a mapping from node_id to the object dictionary.
        # This is for efficient lookup (O(1) on average) and avoids slow,
        # repetitive searching through the self.objects list in a loop.
        try:
            node_map = {obj['node_id']: obj for obj in self.objects}
        except KeyError:
            logger.error("Could not create node map. Make sure 'describe()' has been run and objects have 'node_id'.")
            return

        edge_counter = 0
        # Step 3: Iterate through each edge and add it to the relevant nodes.
        for edge in predicted_edges:
            subject_id = edge['id_1']
            object_id = edge['id_2']
            
            # Find the corresponding node objects using our fast lookup map
            subject_node = node_map.get(subject_id)
            # object_node = node_map.get(object_id)

            # Check if both nodes involved in the edge exist in our map
            if subject_node:
                # An edge is relevant to both nodes. Add the edge dictionary
                # to the lists of both the subject and the object.
                edge['3d_feat'] = edge_feat_3d[edge_counter]
                # edge['2d_feat'] = edge_feat_2d[edge_counter]
                subject_node['edges_vl_sat'].append(edge)
                edge_counter = edge_counter + 1
            else:
                # Log a warning if a node mentioned in an edge wasn't found.
                if not subject_node:
                    logger.warning(f"Node with ID {subject_id} from an edge was not found in self.objects.")
                    logger.warning(f"subject_id {type(subject_id)}, object node id: {type(self.objects[0]['node_id'])}")
                    logger.warning(f"subject_id {subject_id}, object node id: {self.objects[0]['node_id']}")

    def add_edges_sv(self, predicted_edges_sv):
        # Step 1: Initialize an empty 'edges' list for every object.
        # This ensures the key exists even for nodes with no connections.
        for obj in self.objects:
            obj['edges_sv'] = []
            
        # Step 2: Create a mapping from node_id to the object dictionary.
        # This is for efficient lookup (O(1) on average) and avoids slow,
        # repetitive searching through the self.objects list in a loop.
        try:
            node_map = {obj['node_id']: obj for obj in self.objects}
        except KeyError:
            logger.error("Could not create node map. Make sure 'describe()' has been run and objects have 'node_id'.")
            return

        edge_counter = 0
        # Step 3: Iterate through each edge and add it to the relevant nodes.
        for edge in predicted_edges_sv:
            subject_id = edge['source']
            object_id = edge['target']
            
            # Find the corresponding node objects using our fast lookup map
            subject_node = node_map.get(subject_id)
            # object_node = node_map.get(object_id)

            # Check if both nodes involved in the edge exist in our map
            if subject_node:
                # An edge is relevant to both nodes. Add the edge dictionary
                # to the lists of both the subject and the object.
                subject_node['edges_sv'].append(edge)
            else:
                # Log a warning if a node mentioned in an edge wasn't found.
                if not subject_node:
                    logger.warning(f"Node with ID {subject_id} from an edge was not found in self.objects.")
                    logger.warning(f"subject_id {type(subject_id)}, object node id: {type(self.objects[0]['node_id'])}")
                    logger.warning(f"subject_id {subject_id}, object node id: {self.objects[0]['node_id']}")

    
    # def add_edge_feat(self, edge_feat_3d, edge_feat_2d):
    #     for node in self.objects:
    #         node['edge_feat_3d'] = []
    #         node['edge_feat_2d'] = []
    #         for ef_3d, ef_2d in zip(edge_feat_3d, edge_feat_2d):
    #             node['edge_feat_3d'].append(ef_3d)
    #             node['edge_feat_2d'].append(ef_2d)