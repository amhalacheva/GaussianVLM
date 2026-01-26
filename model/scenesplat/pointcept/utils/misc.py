"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from scipy.spatial import cKDTree
from importlib import import_module

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def _majority_vote(neighbor_labels, ignore_label, num_classes):
    """
    neighbor_labels: (N, K) array, each row are the K neighbor labels for a point
    ignore_label: int, label to treat as "invalid" that we ignore in counting
    num_classes: total number of valid classes (>= ignore_label+1 if ignore is negative, 
                or just the max label + 1)
    Returns:
    out: (N,) array of majority‐voted labels
    """
# We define the numba‐accelerated function inside so it can capture arguments
    @numba.njit(parallel=True)
    def _vote(labels_2d, out):
        n_points = labels_2d.shape[0]
        k = labels_2d.shape[1]
        for i in numba.prange(n_points):
            counts = np.zeros(num_classes, dtype=np.int32)
            max_count = 0
            best_label = ignore_label
            # Count only valid labels
            for j in range(k):
                lbl = labels_2d[i, j]
                if lbl != ignore_label and 0 <= lbl < num_classes:
                    counts[lbl] += 1
            # Get majority
            for c in range(num_classes):
                if counts[c] > max_count:
                    max_count = counts[c]
                    best_label = c
            out[i] = best_label

    n_points = neighbor_labels.shape[0]
    out = np.full((n_points,), ignore_label, dtype=np.int32)
    _vote(neighbor_labels, out)
    return out

def neighbor_voting(coords, pred, vote_k, ignore_label, num_classes, valid_mask=None):
    """
    coords:       (N, 3) array of all points
    pred:         (N,)   array of predicted labels for each point
    vote_k:       int, number of neighbors to fetch for each point
    ignore_label: int, label for 'invalid' or 'ignored'
    num_classes:  int, total # of "real" classes (not counting ignore_label)
    valid_mask:   (N,) bool array, optional. 
                  If provided, we build the KD‐tree only on coords[valid_mask],
                  but we still query for neighbors for *all* N points.

    Returns:
      new_pred: (N,) array of updated predictions after neighbor voting.
                If the majority of neighbors are ignore_label, the result is ignore_label.
    """
    if valid_mask is not None:
        used_coords = coords[valid_mask]
        used_labels = pred[valid_mask]
        print(f"Using valid_mask {len(used_coords)}/{len(coords)} points for voting")
    else:
        used_coords = coords
        used_labels = pred

    if len(used_coords) == 0:
        return pred

    kd_tree = cKDTree(used_coords)
    # Query neighbors for ALL points (including those that fail valid_mask or are ignore_label)
    # nn_indices will be shape (N, vote_k)
    _, nn_indices = kd_tree.query(coords, k=vote_k)
    neighbor_labels = used_labels[nn_indices]

    new_pred = _majority_vote(
            neighbor_labels, ignore_label, num_classes
        )
    return new_pred

def clustering_voting(pred, instance_labels, ignore_index):
    """ 
    Args:
        pred (np.ndarray): Predicted semantic labels for each point, shape (N,)
        instance_labels (np.ndarray): Instance ID for each point, shape (N,)
        ignore_index (int): Instance ID value to ignore (e.g., -1 for background)
    Returns:
        np.ndarray: Updated semantic predictions with consistent labels per instance
    """
    # Ensure inputs have the same shape
    if pred.shape != instance_labels.shape:
        print("clustering_voting: prediction and instance arrays must have the same shape")
        return pred
    
    updated_pred = pred.copy()
    unique_instances = np.unique(instance_labels)
    valid_instances = unique_instances[unique_instances != ignore_index]
    
    for instance_id in valid_instances:
        instance_mask = instance_labels == instance_id
        instance_preds = pred[instance_mask]
        unique_classes, counts = np.unique(instance_preds, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        updated_pred[instance_mask] = majority_class
    
    return updated_pred


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    is_ddp_model = type(model).__name__.startswith('DistributedDataParallel')
    state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_ddp_model and not state_dict_has_module:
        # Add 'module.' prefix
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    elif not is_ddp_model and state_dict_has_module:
        # Remove 'module.' prefix
        new_state_dict = {k.replace('module.', ''): v for k in state_dict.keys()}
        model.load_state_dict(new_state_dict)
    else:
        # No adjustment needed
        model.load_state_dict(state_dict)