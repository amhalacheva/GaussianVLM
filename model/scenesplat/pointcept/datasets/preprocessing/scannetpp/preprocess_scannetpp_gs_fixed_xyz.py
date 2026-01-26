"""
Preprocessing Script for ScanNet++ 3DGS with language features,
This is comatible for both MCMC and fixed_xyz version of 3DGS.
"""

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
import open3d
import os
import torch
from plyfile import PlyData

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.ply']:
            return cls._read_ply(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

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



def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_attribute(vertex, attribute=["coord", "opacity", "scale", "quat", "color"]):
    # print(vertex.data.dtype.names)
    # assert "coord" in attribute, "At least need xyz attribute" can free this one actually
    # record the attribute and the index to read it
    data = dict()

    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    # data = np.stack((x, y, z), axis=-1) # [n, 3]
    data["coord"] = np.stack((x, y, z), axis=-1) # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity) # sigmoid activation
        data["opacity"] = opacity

    if "scale" in attribute and ("quat" in attribute or "euler" in attribute ):
        scale_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data["coord"].shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        
        scales = np.exp(scales)  # scale activation
        data["scale"] = scales

        rot_names = [
            p.name for p in vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots
 
    if "sh" in attribute or "color" in attribute:
        # sphere homrincals to rgb
        features_dc = np.zeros((data["coord"].shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
  
        feature_pc = features_dc.reshape(-1, 3)
        if "color" in attribute:
            C0 = 0.28209479177387814
            feature_pc = (feature_pc*C0).astype(np.float32) + 0.5
            feature_pc = np.clip(feature_pc, 0, 1)
        # data = np.concatenate((data, feature_pc), axis=1)
        data["color"] = feature_pc * 255
        
    return data

def find_folder_with_suffix(root_dir, suffix):
    # Convert the root directory to a Path object
    root = Path(root_dir)
    
    # Search for directories whose names end with the given suffix
    matching_folders = [folder for folder in root.rglob("*") if folder.is_dir() and folder.name.endswith(suffix)]
    assert len(matching_folders) > 0, f"No folder with suffix {suffix} found in {root_dir}"
    
    return matching_folders


def parse_scene(
    name,
    split,
    dataset_root,
    gs_root,
    pc_root,
    output_root,
    label_mapping,
    class2idx,
    ignore_index=-1,
    feat_root=None,
    feat_only=False,
    skip_feat=False,
    debug=False
):
    if debug:
        print("===================")
        print("DEBUG MODE, TURN OFF TO SAVE DATA")
    print(f"Parsing scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    gs_root = Path(gs_root)
    pc_root = Path(pc_root)
    output_root = Path(output_root)
    
    scene_path = find_folder_with_suffix(gs_root, name)[0]
    print("scene_path", scene_path)
    data_path = dataset_root / "data" if split != "test" else dataset_root / "sem_test"
    mesh_path = data_path / str(name) / "scans" / "mesh_aligned_0.05.ply"
    
    gs_path = scene_path / "ckpts" /  "point_cloud_30000.ply"
    pc_path = pc_root / split / name / "coord.npy"
    pc_instance = pc_root / split / name / "instance.npy"
    pc_semantic = pc_root / split / name / "segment.npy"
    feat_path = feat_root / name / "langfeat.pth" if feat_root else None
    if skip_feat or feat_path is None:
        print("Skipping feature processing...")

    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print("Error in loading:")
        raise e
    
    ##### speeding up when doing feat_only
    if feat_only:
        gs_feat_np = None
        if feat_path is not None:
            gs_feat, _ = torch.load(feat_path, map_location="cpu")
            gs_feat_np = gs_feat.to(torch.float16).numpy()
            # check if the feature is all zero, save as int
            valid_feat_mask = np.any(gs_feat_np != 0., axis=1).astype(int)
            valid_percent = valid_feat_mask.sum() / valid_feat_mask.size
            if valid_percent < 0.5:
                print(f"Valid language feature percent too low {valid_percent}, skipping scene {name}...")
                return
        if gs_feat_np is not None:
            save_path = output_root / split / name
            save_path.mkdir(parents=True, exist_ok=True)
            # gs_feat_np = gs_feat_np[within_mask]
            # valid_feat_mask = valid_feat_mask[within_mask]
            np.save(save_path / "valid_feat_mask.npy", valid_feat_mask)
            np.save(save_path / "lang_feat.npy", gs_feat_np)
        print(f"Saved features to {save_path}")
        return

    vertex = gs["vertex"]
    gs_data = read_gaussian_attribute(vertex, attribute=["coord", "opacity", "scale", "quat", "color"])

    # create mesh from coord in gs_data
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals(normalized=True)
    normal = np.array(mesh.vertex_normals).astype(np.float32)

    # get gaussian instance and semantic labels from nearest pointcloud
    pc_coord = np.load(pc_path)
    if split != "test":
        pc_instance = np.load(pc_instance)
        pc_semantic = np.load(pc_semantic)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
    oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
    # extend the bbox with 0.2 meters
    enlargement = 0.25  # 0.1 meters
    new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement 
    oriented_bbox.extent = new_extent

    coord = gs_data["coord"].astype(np.float32)
    color = gs_data["color"].astype(np.uint8)
    opacity = gs_data["opacity"].astype(np.float32)
    scale = gs_data["scale"].astype(np.float32)
    quat = gs_data["quat"].astype(np.float32)

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord)

    gs_feat_np = None
    if feat_path is not None and not skip_feat:
         gs_feat, _ = torch.load(feat_path, map_location="cpu")
         gs_feat_np = gs_feat.to(torch.float16).numpy()
         # check if the feature is all zero, save as int
         valid_feat_mask = np.any(gs_feat_np != 0., axis=1).astype(int)
         assert len(coord) == len(gs_feat_np), f"coord and gs_feat_np not match {len(coord)} and {len(gs_feat_np)}"
    
    within_mask = oriented_bbox.get_point_indices_within_bounding_box(gs_o3d.points)
    print("Pruned", len(coord) - len(within_mask), "gaussians by init bounding box.")
    coord = coord[within_mask]
    color = color[within_mask]
    opacity = opacity[within_mask]
    scale = scale[within_mask]
    quat = quat[within_mask]

    save_path = output_root / split / name
    save_path.mkdir(parents=True, exist_ok=True)
    coord_length = len(coord)

    if gs_feat_np is not None:
        gs_feat_np = gs_feat_np[within_mask]
        valid_feat_mask = valid_feat_mask[within_mask]
        np.save(save_path / "valid_feat_mask.npy", valid_feat_mask)
        np.save(save_path / "lang_feat.npy", gs_feat_np)

    if not debug:
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "normal.npy", normal)
        np.save(save_path / "opacity.npy", opacity)
        np.save(save_path / "scale.npy", scale)
        np.save(save_path / "quat.npy", quat)

    if split == "test":
        print(f"Saved scene data to {save_path}")
        return


    # print("point cloud pc_instance", pc_instance.shape) Nx3
    # print("point cloud pc_semantic", pc_semantic.shape) Nx3

    # get the nearest pointcloud points for each gaussian point
    from sklearn.neighbors import KDTree
    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    gs_instance = pc_instance[indices][:,0] # Nx1x3
    gs_semantic = pc_semantic[indices][:,0]
    # distance between the gaussian and the nearest pointcloud point, for debugging
    print("coord: ", coord.shape)
    print("pc_coord[indices]: ", pc_coord[indices].shape)
    distance = np.abs(pc_coord[indices].reshape(-1, 3) - coord) # l1
    print("gs to pc NN distance: mean", distance.mean(), "std", distance.std(), "max", distance.max(), "min", distance.min())

    print("gs_instance", gs_instance.shape)
    print("gs_semantic", gs_semantic.shape)

    np.save(save_path / "instance.npy", gs_instance)
    np.save(save_path / "segment.npy", gs_semantic)

    print(f"Saved scene data to {save_path}")


def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
    if count_thresh > 0 and count_type in mapping.columns:
        mapping = mapping[mapping[count_type] >= count_thresh]
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet++ dataset containing data/metadata/splits.",
    )
    parser.add_argument(
        "--gs_root",
        required=True,
        help="Path to the ScanNet++ Gaussian dataset.",
    )
    parser.add_argument(
        "--pc_root",
        required=True,
        help="Path to the ScanNet++ Point Cloud dataset.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located.",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Default ignore index.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--feat_root",
        help="Path to language features.",
    )
    parser.add_argument("--feat_only", action="store_true", help="Only process features.")
    parser.add_argument("--skip_feat", action="store_true", help="Skip feature processing.")
    config = parser.parse_args()

    print("Loading meta data...")
    config.dataset_root = Path(config.dataset_root)
    config.output_root = Path(config.output_root)
    config.feat_root = Path(config.feat_root) if config.feat_root else None

    train_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_train.txt",
        dtype=str,
    )
    print("Num samples in training split:", len(train_list))

    val_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    print("Num samples in validation split:", len(val_list))

    test_list = np.loadtxt(
        config.dataset_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in testing split:", len(test_list))

    data_list = np.concatenate([train_list, val_list, test_list])
    split_list = np.concatenate(
        [
            np.full_like(train_list, "train"),
            np.full_like(val_list, "val"),
            np.full_like(test_list, "test"),
        ]
    )

    # Parsing label information and mapping
    segment_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in segment class list:", len(segment_class_names))

    instance_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100_instance.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in instance class list:", len(instance_class_names))

    label_mapping = pd.read_csv(
        config.dataset_root / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(
        label_mapping, count_thresh=0, count_type="count", mapping_type="semantic"
    )
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            parse_scene,
            data_list,
            split_list,
            repeat(config.dataset_root),
            repeat(config.gs_root),
            repeat(config.pc_root),
            repeat(config.output_root),
            repeat(label_mapping),
            repeat(class2idx),
            repeat(config.ignore_index),
            repeat(config.feat_root),
            repeat(config.feat_only),
            repeat(config.skip_feat),
        )
    )
    pool.shutdown()
    print("Preprocessing done!")
