import copy
from collections.abc import Iterable

import torch
import matplotlib
import numpy as np
import open3d as o3d
from loguru import logger
import torch.nn.functional as F


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]

    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)

    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self

    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes

    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool=True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color

    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return

        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]


class MapObjectList(DetectionList):
    def compute_similarities(self, new_features):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor 
        new_features = to_tensor(new_features)

        # assuming cosine similarity for features
        features = self.get_stacked_values_torch('descriptor')

        similarities = F.cosine_similarity(new_features.unsqueeze(0), features)
        return similarities
    import numpy as np

    def bbox_center_and_extent(self, corners, oriented=True, half_extents=False):
        """
        corners: array-like of shape (8,3) OR flat shape (24,) OR a list of 8 (3,) points.
        oriented: if True compute extents in the box's principal axes (PCA); 
                if False compute axis-aligned extents (world axes).
        half_extents: if True return extents/2 (common convention in some libraries).
        Returns:
        center: (3,) ndarray
        extents: (3,) ndarray (full lengths unless half_extents=True)
        axes: (3,3) ndarray of axes as columns (only returned if oriented=True)
        """
        arr = np.asarray(corners)
        # Normalize shapes:
        if arr.ndim == 1 and arr.size == 24:
            arr = arr.reshape(8, 3)
        elif arr.shape == (8,):
            # allow list of 8 points, each a length-3 sequence
            try:
                arr = np.stack(arr)
            except Exception as e:
                raise ValueError("Input shape (8,) not stackable into (8,3).") from e

        if arr.shape != (8, 3):
            raise ValueError(f"Expected corners shape (8,3) or (24,), got {arr.shape}")

        center = arr.mean(axis=0)  # (3,)

        if not oriented:
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            extents = maxs - mins
            if half_extents:
                extents = extents / 2.0
            return center, extents

        # oriented: find orthonormal axes via PCA (covariance of centered points)
        pts = arr - center  # center the points
        cov = pts.T @ pts   # 3x3 covariance-like matrix (unnormalized)
        eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs columns correspond to eigenvalues
        # sort eigenvectors by descending variance
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]  # 3x3, columns = principal axes

        # project centered points into the PCA axes (local box coordinates)
        projected = pts @ axes       # shape (8,3)
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        extents = maxs - mins        # full lengths along each axis
        if half_extents:
            extents = extents / 2.0

        return center, extents


    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)

            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            s_obj_dict['bbox_center'] = np.asarray(s_obj_dict['bbox'].center)
            s_obj_dict['bbox_extent'] = np.asarray(s_obj_dict['bbox'].extent)
            s_obj_dict['bbox_rotation'] = np.asarray(s_obj_dict['bbox'].R)
            
            try:
                s_obj_dict['descriptor'] = to_numpy(s_obj_dict['descriptor'])
            except:
                logger.warning("can't load descriptor")
                
            try:
                # print(obj.keys())
                s_obj_dict['clip_descriptor'] = to_numpy(s_obj_dict['clip_descriptor'])
                # print(f"type: {type(s_obj_dict['clip_descriptor'])}")
                # print(f"clip_descriptor shape: {s_obj_dict['clip_descriptor'].shape}")
                # print(f"clip_descriptor: {s_obj_dict['clip_descriptor'][2]}")
            except Exception as e:
                logger.warning(f"""can't load clip_descriptor: {e}""")

            try:
                s_obj_dict['id'] = list(s_obj_dict['id'])
            except:
                logger.warning("can't load id")

            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)

            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])

            try:
                new_obj['descriptor'] = to_tensor(new_obj['descriptor'])
            except:
                logger.warning("can't load descriptor")

            try:
                new_obj['clip_descriptor'] = to_tensor(new_obj['clip_descriptor'])
            except:
                logger.warning("can't load clip_descriptor")

            try:
                new_obj['id'] = set(new_obj['id'])
            except:
                logger.warning("can't load id")
                
            try:
                new_obj['bbox_extent'] = to_tensor(new_obj['bbox_extent'])
            except Exception as e:
                logger.warning(f"can't load new_obj['bbox_extent']: {e}")

            try:
                new_obj['bbox_center'] = to_tensor(new_obj['bbox_center'])
            except Exception as e:
                logger.warning(f"can't load new_obj['bbox_center']: {e}")

            # why here being deleted?
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']

            self.append(new_obj)
