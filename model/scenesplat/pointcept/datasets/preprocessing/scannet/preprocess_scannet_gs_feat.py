"""
Preprocessing Script for ScanNet 20/200 with Gaussian Splat
Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

Please cite our work if the code is helpful to you.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import glob
import json
import plyfile
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from sklearn.neighbors import KDTree

from IPython import embed
# Load external constants
from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1

###############################################################################
# 1) Utility Functions
###############################################################################

def read_plymesh(filepath):
    """
    Read the standard ScanNet mesh (with vertices and faces) 
    and return as (vertices, faces). Returns (None, None) if empty.
    """
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces
    return None, None

def face_normal(vertex_coords, face):
    """
    Compute face normals + face areas for the given mesh.
    Returns (nf, area) for each face.
    """
    v01 = vertex_coords[face[:, 1]] - vertex_coords[face[:, 0]]
    v02 = vertex_coords[face[:, 2]] - vertex_coords[face[:, 0]]
    vec = np.cross(v01, v02)  # [F, 3]
    area = 0.5 * np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
    length = np.maximum(1e-8, np.sqrt(np.sum(vec**2, axis=1, keepdims=True)))
    nf = vec / length
    return nf, area

def vertex_normal(vertex_coords, face):
    """
    Compute per-vertex normals by accumulating area-weighted face normals.
    """
    nf, area = face_normal(vertex_coords, face)
    nf_area = nf * area
    nv = np.zeros_like(vertex_coords)
    for i in range(face.shape[0]):
        inds = face[i]
        nv[inds[0]] += nf_area[i]
        nv[inds[1]] += nf_area[i]
        nv[inds[2]] += nf_area[i]
    lengths = np.maximum(1e-8, np.sqrt(np.sum(nv**2, axis=1, keepdims=True)))
    nv = nv / lengths
    return nv

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_ply(filepath):
    """
    Reads a Gaussian Splat .ply with fields such as:
      x, y, z, opacity, scale_0, scale_1, ..., rot_0, rot_1, ..., f_dc_0, f_dc_1, ...
    Returns a dict with keys: coord, color, opacity, scale, quat.
    """
    with open(filepath, "rb") as f:
        ply_data = plyfile.PlyData.read(f)
    vertex = ply_data["vertex"]
    N = vertex.count

    # Basic coordinates
    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Opacity (using a sigmoid)
    if "opacity" in vertex.data.dtype.names:
        opacity_raw = vertex["opacity"].astype(np.float32)
        opacity = np_sigmoid(opacity_raw)
    else:
        opacity = np.ones(N, dtype=np.float32)

    # Scale values (exponentiated)
    scale_cols = [c for c in vertex.data.dtype.names if c.startswith("scale_")]
    scale_cols = sorted(scale_cols, key=lambda c: int(c.split("_")[-1]))
    scale = [np.exp(vertex[c]) for c in scale_cols]
    scale = np.stack(scale, axis=-1).astype(np.float32) if scale else np.ones((N, 1), np.float32)

    # Quaternion (rotation)
    rot_cols = [c for c in vertex.data.dtype.names if c.startswith("rot_")]
    rot_cols = sorted(rot_cols, key=lambda c: int(c.split("_")[-1]))
    quat = [vertex[c] for c in rot_cols]
    if quat:
        quat = np.stack(quat, axis=-1).astype(np.float32)
        length = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9
        quat = quat / length
        sign_vector = np.sign(quat[:, 0])
        quat = quat * sign_vector[:, None]
    else:
        quat = np.ones((N, 4), dtype=np.float32)
    
    # Color from f_dc_0, f_dc_1, f_dc_2
    fdc_cols = [c for c in vertex.data.dtype.names if c.startswith("f_dc_")]
    fdc_cols = sorted(fdc_cols, key=lambda c: int(c.split("_")[-1]))
    if len(fdc_cols) >= 3:
        fdc_stack = [vertex[c] for c in fdc_cols]
        fdc_stack = np.stack(fdc_stack, axis=-1).astype(np.float32)
        C0 = 0.28209479177387814
        color = np.clip(fdc_stack * C0 + 0.5, 0, 1) * 255
        color = color.astype(np.uint8)
    else:
        color = np.full((N, 3), 128, dtype=np.uint8)

    return {"coord": coord, "color": color, "opacity": opacity, "scale": scale, "quat": quat}

def point_indices_from_group(seg_indices, group, labels_pd):
    """
    For a group in the aggregation, map the raw label to 20- and 200-class IDs.
    """
    group_segments = np.array(group["segments"])
    label = group["label"]
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if not label_id20.empty else 0
    label_id20 = CLASS_IDS20.index(label_id20) if label_id20 in CLASS_IDS20 else IGNORE_INDEX

    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if not label_id200.empty else 0
    label_id200 = CLASS_IDS200.index(label_id200) if label_id200 in CLASS_IDS200 else IGNORE_INDEX

    return group_segments, label_id20, label_id200


###############################################################################
# 2) Main Processing Function
###############################################################################

def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes, gs_root, feat_root=None, feat_only=False, skip_feat=False):
    """
    Process one scene.
      - For non-test splits: load the mesh and Gaussian splat data, compute normals, etc.
      - For test split: read the preprocessed point data (color.npy, coord.npy, normal.npy)
        from the given scene folder under preprocess_point/test/.
    """
    scene_id = os.path.basename(scene_path)
    print(f"Processing scene: {scene_id}")

    # dataset_root = Path(dataset_root)
    # pc_root = Path(pc_root)
    gs_root = Path(gs_root)
    feat_root = Path(feat_root) if feat_root else None

    if scene_id in train_scenes:
        output_path = os.path.join(output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_id in val_scenes:
        output_path = os.path.join(output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Standard mesh and, if applicable, segmentation/aggregation files from dataset_root.
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}")
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    feat_path = feat_root / scene_id / "langfeat.pth" if feat_root else None
    if skip_feat or feat_path is None:
        print("Skipping feature processing...")

    vertices, faces = read_plymesh(mesh_path)
    if vertices is None:
        print(f"[WARN] No vertices found for {scene_id} at {mesh_path}")
        return

    mesh_coords = vertices[:, :3]
    mesh_normals = vertex_normal(mesh_coords, faces)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(mesh_coords)
    oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
    enlargement = 0.25  # 0.1 meters
    new_extent = np.asarray(oriented_bbox.extent) + 2 * enlargement 
    oriented_bbox.extent = new_extent

    if split_name != "test":
        with open(segments_file) as f:
            segdata = json.load(f)
            seg_indices = np.array(segdata["segIndices"])
        with open(aggregations_file) as f:
            agg = json.load(f)
            seg_groups = agg["segGroups"]
    else:
        seg_indices = None
        seg_groups = []

    # Load Gaussian Splat data from gs_root
    gs_scene_dir = os.path.join(gs_root, scene_id, "ckpts")
    gs_candidates = glob.glob(os.path.join(gs_scene_dir, "*.ply"))
    print("GS scene directory:", gs_scene_dir)
    if len(gs_candidates) == 0:
        print(f"[WARN] No Gaussian .ply found for {scene_id}")
        return
    gs_path = gs_candidates[0]
    gs_data = read_gaussian_ply(gs_path)

    coord_gs = gs_data["coord"]
    color_gs = gs_data["color"]
    opacity_gs = gs_data["opacity"]
    scale_gs = gs_data["scale"]
    quat_gs = gs_data["quat"]
    N_gs = coord_gs.shape[0]

    gs_o3d = o3d.geometry.PointCloud()
    gs_o3d.points = o3d.utility.Vector3dVector(coord_gs)

    gs_feat_np = None
    if feat_path and not skip_feat:
         gs_feat, _ = torch.load(feat_path, map_location="cpu")
         gs_feat_np = gs_feat.to(torch.float16).numpy()
         # check if the feature is all zero, save as int
         valid_feat_mask = np.any(gs_feat_np != 0., axis=1).astype(int)
         assert len(coord_gs) == len(gs_feat_np), f"coord and gs_feat_np not match {len(coord_gs)} and {len(gs_feat_np)}"

    tree = KDTree(mesh_coords)
    _, nn_idx = tree.query(coord_gs, k=1)
    nn_idx = nn_idx[:, 0]
    # print the average distance of the nearest neighbor
    print("Average distance of nearest pc neighbor:", np.mean(np.linalg.norm(coord_gs - mesh_coords[nn_idx], axis=1)))
    normal_gs = mesh_normals[nn_idx, :]

    if seg_indices is not None:
        segIndex_gs = seg_indices[nn_idx]
    else:
        segIndex_gs = np.zeros_like(nn_idx)

    semantic20_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)
    semantic200_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)
    instance_gs = np.full(N_gs, IGNORE_INDEX, dtype=np.int16)

    if split_name != "test":
        for group in seg_groups:
            group_segments, label_id20, label_id200 = point_indices_from_group(
                seg_indices, group, labels_pd
            )
            mask = np.isin(segIndex_gs, group_segments)
            semantic20_gs[mask] = label_id20
            semantic200_gs[mask] = label_id200
            instance_gs[mask] = group["id"]
    
    # double check, prune gaussians outside the mesh bbox
    within_mask = oriented_bbox.get_point_indices_within_bounding_box(gs_o3d.points)
    print("Pruned", len(coord_gs) - len(within_mask), "gaussians by init bounding box.")

    if not skip_feat and gs_feat_np is not None:
        gs_feat_np = gs_feat_np[within_mask]
        valid_feat_mask = valid_feat_mask[within_mask]
        np.save(output_path / "valid_feat_mask.npy", valid_feat_mask)
        np.save(output_path / "lang_feat.npy", gs_feat_np)

    # Save outputs
    np.save(os.path.join(output_path, "coord.npy"), coord_gs[within_mask])
    np.save(os.path.join(output_path, "color.npy"), color_gs[within_mask])
    np.save(os.path.join(output_path, "opacity.npy"), opacity_gs[within_mask])
    np.save(os.path.join(output_path, "scale.npy"), scale_gs[within_mask])
    np.save(os.path.join(output_path, "quat.npy"), quat_gs[within_mask])
    np.save(os.path.join(output_path, "normal.npy"), normal_gs[within_mask])
    if split_name != "test":
        np.save(os.path.join(output_path, "segment20.npy"), semantic20_gs[within_mask])
        np.save(os.path.join(output_path, "segment200.npy"), semantic200_gs[within_mask])
        np.save(os.path.join(output_path, "instance.npy"), instance_gs[within_mask])

###############################################################################
# 3) Main
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the standard ScanNet dataset containing scene folders (ignored for test split)",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located",
    )
    parser.add_argument(
        "--gs_root",
        required=True,
        help="Folder containing Gaussian Splat .ply for each scene (ignored for test split)",
    )
    parser.add_argument(
        "--feat_root",
        help="Path to language features.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Number of workers for preprocessing.",
    )
    parser.add_argument("--feat_only", action="store_true", help="Only process features.")
    parser.add_argument("--skip_feat", action="store_true", help="Skip feature processing.")
    config = parser.parse_args()

    # For train/val, load scene paths from the dataset_root based on meta files.
    script_dir = Path(os.path.dirname(__file__))
    meta_root = script_dir / "meta_data"

    # Load label map
    labels_pd = pd.read_csv(
        meta_root / "scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(meta_root / "scannetv2_train.txt") as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / "scannetv2_val.txt") as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans*/scene*"))

    print(f"Found in total {len(scene_paths)} scenes under {config.dataset_root}")

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(labels_pd),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(config.gs_root),
            repeat(config.feat_root),
            repeat(config.feat_only),
            repeat(config.skip_feat),
        )
    )
    pool.shutdown()