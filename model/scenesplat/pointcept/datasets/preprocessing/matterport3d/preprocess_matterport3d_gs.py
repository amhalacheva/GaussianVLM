import argparse
import json
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import h5py
import numpy as np
import os
from plyfile import PlyData
from sklearn.neighbors import KDTree

################################################################################
# I/O Utilities
################################################################################

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.ply']:
            return cls._read_ply(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)

################################################################################
# Gaussian Reading
################################################################################

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_attribute(vertex, attribute=["coord", "opacity", "scale", "quat", "color"]):
    """
    Reads a 'vertex' structure from a PlyData file and returns a dictionary
    with 'coord', 'opacity', 'scale', 'quat', 'color' (all optional).
    """
    data = {}

    # Coordinates (xyz)
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    data["coord"] = np.stack((x, y, z), axis=-1)  # [N, 3]

    # Opacity
    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity)  # range (0,1)
        data["opacity"] = opacity

    # Scale & Quaternion
    if "scale" in attribute and ("quat" in attribute or "euler" in attribute):
        scale_names = [
            p.name for p in vertex.properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data["coord"].shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        scales = np.exp(scales)  # exponentiate to get actual scale
        data["scale"] = scales

        rot_names = [
            p.name for p in vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        # Normalize the quaternion
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # Enforce positive real part
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots

    # Color (from spherical harmonics DC term) or direct color
    if "sh" in attribute or "color" in attribute:
        # DC terms
        features_dc = np.zeros((data["coord"].shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        # Move from SH DC to approximate color
        C0 = 0.28209479177387814  # Spherical Harmonics Y00
        feature_pc = (feature_pc * C0).astype(np.float32) + 0.5
        feature_pc = np.clip(feature_pc, 0, 1)
        # Store 0-255
        data["color"] = (feature_pc * 255).astype(np.uint8)

    return data

################################################################################
# Utility: find folder that ends with a certain suffix
################################################################################

def find_folder_with_suffix(root_dir, suffix):
    """
    Return the first folder under `root_dir` whose name ends with `suffix`.
    """
    root = Path(root_dir)
    matching_folders = [
        folder for folder in root.rglob("*")
        if folder.is_dir() and folder.name.endswith(suffix)
    ]
    if len(matching_folders) == 0:
        raise FileNotFoundError(f"No folder with suffix {suffix} found in {root_dir}")
    return matching_folders

################################################################################
# Main Parsing Logic
################################################################################

def parse_scene(
    scene_name,
    split,
    gs_root,
    pc_root,
    output_root,
    debug=False
):
    """
    scene_name: e.g., '17DRP5sb8fy_00'
    split: 'train', 'val', or 'test'
    gs_root: Path to your 3D Gaussians. Scenes are all in a single folder structure.
    pc_root: Path to point cloud subfolders, each subfolder has coord.npy, segment.npy, etc.
    output_root: Where we save the per-scene Gaussian data
    debug: If True, skip actually writing data
    """
    if debug:
        print("===================")
        print("DEBUG MODE, TURN OFF TO SAVE DATA")

    print(f"Parsing scene {scene_name} in {split} split")

    # 1) Find the folder in gs_root that ends with scene_name
    scene_path_candidates = find_folder_with_suffix(gs_root, scene_name)
    scene_path = scene_path_candidates[0]
    gs_path = scene_path / "ckpts" / "point_cloud_30000.ply"

    # 2) Load the GS file
    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print(f"Error loading {gs_path}: {e}")
        return  # skip this scene

    # 3) Parse the Gaussians
    vertex = gs["vertex"]
    gs_data = read_gaussian_attribute(
        vertex,
        attribute=["coord", "opacity", "scale", "quat", "color"]
    )

    coord = gs_data["coord"]
    color = gs_data["color"]
    opacity = gs_data["opacity"]
    scale = gs_data["scale"]
    quat = gs_data["quat"]

    # 4) Load the PC from pc_root for nearest neighbor labeling
    #    We'll look for pc_root/ split / scene_name
    scene_pc_dir = Path(pc_root) / split / scene_name
    pc_coord_path = scene_pc_dir / "coord.npy"
    pc_segment_path = scene_pc_dir / "segment.npy"
    pc_segment_nyu_path = scene_pc_dir / "segment_nyu_160.npy"

    if not pc_coord_path.exists() or not pc_segment_path.exists():
        print(f"Point cloud or segment file not found for scene {scene_name}. Skipping.")
        return

    pc_coord = np.load(pc_coord_path)  # (N, 3)
    pc_segment = np.load(pc_segment_path)  # (N,) or (N,1)
    # handle shape
    if pc_segment.ndim == 2 and pc_segment.shape[1] == 1:
        pc_segment = pc_segment.squeeze(1)

    # If segment_nyu_160.npy exists, read it
    pc_segment_nyu = None
    if pc_segment_nyu_path.exists():
        pc_segment_nyu = np.load(pc_segment_nyu_path)
        if pc_segment_nyu.ndim == 2 and pc_segment_nyu.shape[1] == 1:
            pc_segment_nyu = pc_segment_nyu.squeeze(1)

    # 5) Optional bounding-box-based pruning
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
    oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
    enlargement = 0.25  # in meters
    new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement
    oriented_bbox.extent = new_extent

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord)

    within_mask = oriented_bbox.get_point_indices_within_bounding_box(gs_o3d.points)
    print(f"Pruned {len(coord) - len(within_mask)} gaussians by bounding box.")
    coord = coord[within_mask]
    color = color[within_mask]
    opacity = opacity[within_mask]
    scale = scale[within_mask]
    quat = quat[within_mask]

    # 6) Nearest Neighbor to get the semantic label
    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    gs_segment = pc_segment[indices[:, 0]]

    # If we also have NYU 160 class labels
    gs_segment_nyu = None
    if pc_segment_nyu is not None:
        gs_segment_nyu = pc_segment_nyu[indices[:, 0]]

    # 7) Save results
    save_path = Path(output_root) / split / scene_name
    save_path.mkdir(parents=True, exist_ok=True)

    if not debug:
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "opacity.npy", opacity)
        np.save(save_path / "scale.npy", scale)
        np.save(save_path / "quat.npy", quat)
        # Save nearest-neighbor semantic data
        np.save(save_path / "segment.npy", gs_segment)
        if gs_segment_nyu is not None:
            np.save(save_path / "segment_nyu_160.npy", gs_segment_nyu)
    else:
        print("Debug mode: not saving data for scene:", scene_name)

################################################################################
# Main script
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_root", required=True,
                        help="Path to the Matterport3D Gaussian dataset.")
    parser.add_argument("--pc_root", required=True,
                        help="Path to the Matterport3D preprocessed point cloud dataset.")
    parser.add_argument("--output_root", required=True,
                        help="Output path where train/val/test folders will be located.")
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int,
                        help="Num workers for preprocessing.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, will not save data. For debugging.")
    config = parser.parse_args()

    config.gs_root = Path(config.gs_root)
    config.pc_root = Path(config.pc_root)
    config.output_root = Path(config.output_root)

    # Collect the scenes from train/val/test subfolders in pc_root
    def get_scenes_from_split(split_name):
        split_dir = config.pc_root / split_name
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist!")
        # Scenes are subfolders
        return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    train_scenes = get_scenes_from_split("train")
    val_scenes   = get_scenes_from_split("val")
    test_scenes  = get_scenes_from_split("test")

    # we have to filter scenes that does not exsit in gs_root
    train_scenes = [scene for scene in train_scenes if (config.gs_root / scene).exists()]
    val_scenes = [scene for scene in val_scenes if (config.gs_root / scene).exists()]
    test_scenes = [scene for scene in test_scenes if (config.gs_root / scene).exists()]
    data_list = train_scenes + val_scenes + test_scenes

    print("Num train scenes:", len(train_scenes))
    print("Num val scenes:", len(val_scenes))
    print("Num test scenes:", len(test_scenes))
    print("Total scenes:", len(data_list))

    # Combine them for easy iteration
    split_list = (
        ["train"] * len(train_scenes)
        + ["val"] * len(val_scenes)
        + ["test"] * len(test_scenes)
    )

    # Parallel processing
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    futures = []
    for scene_name, split_name in zip(data_list, split_list):
        futures.append(pool.submit(
            parse_scene,
            scene_name,
            split_name,
            config.gs_root,
            config.pc_root,
            config.output_root,
            config.debug
        ))
    _ = [f.result() for f in futures]
    pool.shutdown()

    print("Done preprocessing Matterport3D Gaussian data!")