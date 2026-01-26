import open3d as o3d
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


def calculate_min_rotated_box(pc_path):
    pc_path = os.path.join(pc_path, 'coord.npy')
    pc_coord = np.load(pc_path)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_coord)
    oriented_bbox = pc_o3d.get_minimal_oriented_bounding_box()
    # Extract bounding box properties
    center = oriented_bbox.center  # Center of the bounding box
    extent = oriented_bbox.extent  # Extent (lengths along principal axes)
    rotation_matrix = oriented_bbox.R  # 3x3 rotation matrix
    # Save to a dictionary (can be converted to NumPy or list)
    bbox_data = {
        "center": center,
        "extent": extent,
        "rotation_matrix": rotation_matrix
    }
    # Option 1: Save as a NumPy array
    bbox_np = np.array([center, extent, rotation_matrix.flatten()], dtype=object)
    save_path = pc_path.replace(".npy", "_oriented_bbox.npy")
    np.save(save_path, bbox_np)


data_root_dir = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetppgs_processed_mcmc_prune/train_grid1mm_chunk6x6_stride3x3'

scenes_id = os.listdir(data_root_dir)
scenes_id = [os.path.join(data_root_dir, scene_id) for scene_id in scenes_id]
scenes_id = sorted(scenes_id)

with Pool(4) as p:
    r = list(tqdm(p.imap(calculate_min_rotated_box, scenes_id), total=len(scenes_id)))