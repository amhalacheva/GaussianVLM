"""
Preprocessing Script for ScanNet++
modified from official preprocess code.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
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
# import open3d
import os
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

    # TODO: remove this
    # for name in ["x", "opacity", "scale_0", "rot_0", "f_dc_0", "f_rest_0"]:
    #     param = vertex[name]
    #     print(f"{name} min {param.min()} max {param.max()}")
    #     print(f"{name} shape {param.shape}")

    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    # data = np.stack((x, y, z), axis=-1) # [n, 3]
    data["coord"] = np.stack((x, y, z), axis=-1) # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity)
        # opacity range from 0 to 1
        # data = np.concatenate((data, opacity), axis=-1)
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
        
        scales = np.exp(scales)  # scale normalization
        data["scale"] = scales

        rot_names = [
            p.name for p in vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data["coord"].shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        # normalize the quaternion to have unit length
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)

        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        data["quat"] = rots
        # data = np.concatenate((data, scales, rots), axis=-1)
 
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

    # # keep gs with opacity higher than 1 / 255
    # indices = np.where(data["opacity"] > 1 / 255)[0]
    # for key in data.keys():
    #     data[key] = data[key][indices]
        
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
    # find folder end with name
    scene_path = find_folder_with_suffix(gs_root, name)[0]
    print("scene_path", scene_path)
    # scene_path = gs_root / f"max_1500000_depth_True_mcmc_{name}"
    # mesh_path = scene_path / "mesh_aligned_0.05.ply"
    gs_path = scene_path / "ckpts" /  "point_cloud_30000.ply"
    pc_path = pc_root / split / name / "coord.npy"
    pc_instance = pc_root / split / name / "instance.npy"
    pc_semantic = pc_root / split / name / "segment.npy"

    # segs_path = scene_path / "segments.json"
    # anno_path = scene_path / "segments_anno.json"

    # load mesh vertices and colors
    # mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    try:
        gs = IO.get(gs_path)
    except Exception as e:
        print("Error in loading", gs_path, e)

    vertex = gs["vertex"]

    gs_data = read_gaussian_attribute(vertex, attribute=["coord", "opacity", "scale", "quat", "color"])

    # extract mesh information
    # mesh.compute_vertex_normals(normalized=True)
    # coord = np.array(mesh.vertices).astype(np.float32)
    # color = (np.array(mesh.vertex_colors) * 255).astype(np.uint8)
    # normal = np.array(mesh.vertex_normals).astype(np.float32)
    # get gaussian instance and semantic labels from nearest pointcloud
    pc_coord = np.load(pc_path)
    pc_instance = np.load(pc_instance)
    pc_semantic = np.load(pc_semantic)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
    oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
    # extend the bbox with 0.2 meters
    enlargement = 0.2  # 0.1 meters
    new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement 
    oriented_bbox.extent = new_extent

    coord = gs_data["coord"].astype(np.float32)
    color = gs_data["color"].astype(np.uint8)
    opacity = gs_data["opacity"].astype(np.float32)
    scale = gs_data["scale"].astype(np.float32)
    quat = gs_data["quat"].astype(np.float32)

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord)
    
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
    
    if not debug:
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        # np.save(save_path / "normal.npy", normal)
        np.save(save_path / "opacity.npy", opacity)
        np.save(save_path / "scale.npy", scale)
        np.save(save_path / "quat.npy", quat)


    if split == "test":
        return


    # print("point cloud pc_instance", pc_instance.shape) Nx3
    # print("point cloud pc_semantic", pc_semantic.shape) Nx3

    # get the nearest pointcloud points for each gaussian point
    from sklearn.neighbors import KDTree
    tree = KDTree(pc_coord)
    _, indices = tree.query(coord, k=1)
    gs_instance = pc_instance[indices][:,0] # Nx1x3
    gs_semantic = pc_semantic[indices][:,0]

    print("gs_instance", gs_instance.shape)
    print("gs_semantic", gs_semantic.shape)

    np.save(save_path / "instance.npy", gs_instance)
    np.save(save_path / "segment.npy", gs_semantic)

    if debug:
        try:
            # use coord.npy as xyz and color from semantic.npy
            # pcd = o3d.geometry.PointCloud()
            # print("coord", coord.shape)
            # pcd.points = o3d.utility.Vector3dVector(coord)
            import trimesh 
            from matplotlib import pyplot as plt
            # print("gs_semantic", gs_semantic.shape)
            gs_semantic_color = gs_semantic[:,0]
            # print("gs_semantic_color", gs_semantic_color.shape)
            gs_semantic_color = (gs_semantic_color - gs_semantic_color.min()) / (gs_semantic_color.max() - gs_semantic_color.min())
            colormap = plt.cm.viridis  
            gs_semantic_color = colormap(gs_semantic_color)[:,:3]
            print("gs_semantic_color normalized ", gs_semantic_color.shape)
            assert len(gs_semantic_color) == coord_length, f"gaussian ply and gaussian coord not match, {len(gs_semantic_color)}, and {coord_length}"
            gs_semantic_color = (gs_semantic_color * 255).astype(np.uint8)
            gs_semantic_color = gs_semantic_color / 255.
            gs_o3d = o3d.geometry.PointCloud()
            gs_o3d.points = o3d.utility.Vector3dVector(coord)
            gs_o3d.colors = o3d.utility.Vector3dVector(gs_semantic_color)
            o3d.io.write_point_cloud(save_path / "gs_semantic.ply", gs_o3d)
            # reload_o3d = o3d.io.read_point_cloud(str(save_path / "gs_semantic.ply"))
            # reload_coord = np.asarray(reload_o3d.points)
            # assert len(reload_coord) == len(coord), f"reload coord not match {len(reload_coord)} and {len(coord)}"
            # print("gs_semantic_color int", gs_semantic_color.shape)
            # gs_semantic_color = np.stack([gs_semantic_color, gs_semantic_color, gs_semantic_color], axis=-1)
            # print("gs_semantic_color stack", gs_semantic_color.shape)
            # colormap the gs_semantic_color
            # mesh = trimesh.Trimesh(vertices=coord, vertex_colors=gs_semantic_color)
            # mesh.export(save_path / "gs_semantic_prune.ply")

            # reload_mesh = trimesh.load(save_path / "gs_semantic_prune.ply")
            # length_reload = len(reload_mesh.vertices)
            # assert length_reload == len(coord), f"reload mesh vertices not match {length_reload} and {len(coord)}"

            # pcd.colors = o3d.utility.Vector3dVector(gs_semantic_color / 255 )
            # o3d.io.write_point_cloud(save_path / "gs_semantic.ply", pcd)
            pc_semantic_color = pc_semantic[:,0]
            pc_semantic_color = (pc_semantic_color - pc_semantic_color.min()) / (pc_semantic_color.max() - pc_semantic_color.min())
            pc_semantic_color = colormap(pc_semantic_color)[:,:3]

            pc_semantic_color = (pc_semantic_color * 255).astype(np.uint8)
            pc_semantic_color = pc_semantic_color / 255.
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
            pc_o3d.colors = o3d.utility.Vector3dVector(pc_semantic_color)
            o3d.io.write_point_cloud(save_path / "pc_semantic.ply", pc_o3d)

            # mesh = trimesh.Trimesh(vertices=pc_coord, vertex_colors=pc_semantic_color)
            # mesh.export(save_path / "pc_semantic.ply")


        except Exception as e:
            print("Error in writing", save_path / "gs_semantic.ply","error:", e)


    # # get label on vertices
    # # load segments = vertices per segment ID
    # with open(segs_path) as f:
    #     segments = json.load(f)
    # # load anno = (instance, groups of segments)
    # with open(anno_path) as f:
    #     anno = json.load(f)
    # seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
    # num_vertices = len(seg_indices)
    # assert num_vertices == len(coord)
    # semantic_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index
    # instance_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index

    # # number of labels are used per vertex. initially 0
    # # increment each time a new label is added
    # instance_size = np.ones((num_vertices, 3), dtype=np.int16) * np.inf

    # # keep track of the size of the instance (#vertices) assigned to each vertex
    # # later, keep the label of the smallest instance for major label of vertices
    # # store inf initially so that we can pick the smallest instance
    # labels_used = np.zeros(num_vertices, dtype=np.int16)

    # for idx, instance in enumerate(anno["segGroups"]):
    #     label = instance["label"]
    #     instance["label_orig"] = label
    #     # remap label
    #     instance["label"] = label_mapping.get(label, None)
    #     instance["label_index"] = class2idx.get(label, ignore_index)

    #     if instance["label_index"] == ignore_index:
    #         continue
    #     # get all the vertices with segment index in this instance
    #     # and max number of labels not yet applied
    #     mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
    #     size = mask.sum()
    #     if size == 0:
    #         continue

    #     # get the position to add the label - 0, 1, 2
    #     label_position = labels_used[mask]
    #     semantic_gt[mask, label_position] = instance["label_index"]
    #     # store all valid instance (include ignored instance)
    #     instance_gt[mask, label_position] = instance["objectId"]
    #     instance_size[mask, label_position] = size
    #     labels_used[mask] += 1

    # # major label is the label of smallest instance for each vertex
    # # use major label for single class segmentation
    # # shift major label to the first column
    # mask = labels_used > 1
    # if mask.sum() > 0:
    #     major_label_position = np.argmin(instance_size[mask], axis=1)

    #     major_semantic_label = semantic_gt[mask, major_label_position]
    #     semantic_gt[mask, major_label_position] = semantic_gt[:, 0][mask]
    #     semantic_gt[:, 0][mask] = major_semantic_label

    #     major_instance_label = instance_gt[mask, major_label_position]
    #     instance_gt[mask, major_label_position] = instance_gt[:, 0][mask]
    #     instance_gt[:, 0][mask] = major_instance_label

    # np.save(save_path / "segment.npy", semantic_gt)
    # np.save(save_path / "instance.npy", instance_gt)


def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
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
    config = parser.parse_args()

    print("Loading meta data...")
    config.dataset_root = Path(config.dataset_root)
    config.output_root = Path(config.output_root)

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
        )
    )
    pool.shutdown()
