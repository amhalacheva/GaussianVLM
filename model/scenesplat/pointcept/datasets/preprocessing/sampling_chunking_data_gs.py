"""
Chunking Data

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import open3d as o3d

def chunking_scene(
    name,
    dataset_root,
    split,
    grid_size=None,
    chunk_range=(6, 6),
    chunk_stride=(3, 3),
    chunk_minimum_size=10000,
    debug=False,
):  
    if debug:
        print("=============================")
        print("DEBUG MODE, turn off to save chunks")
    print(f"Chunking scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    scene_path = dataset_root / split / name
    assets = os.listdir(scene_path)
    data_dict = dict()
    for asset in assets:
        if not asset.endswith(".npy"):
            continue
        data_dict[asset[:-4]] = np.load(scene_path / asset)
    # recenter the coordinates!!!
    coord = data_dict["coord"] - data_dict["coord"].min(axis=0)
    if debug:
        coord_length = len(coord)
    if grid_size is not None:
        grid_coord = np.floor(coord / grid_size).astype(int)
        _, idx = np.unique(grid_coord, axis=0, return_index=True)
        coord = coord[idx]
        for key in data_dict.keys():
            data_dict[key] = data_dict[key][idx]

    bev_range = coord.max(axis=0)[:2] 
    bev_start = coord.min(axis=0)[:2]  # debug
    # debug
    if debug:
        print("name", name)
        print("bev_range", bev_range) # max in x, y
        print("bev_start", bev_start) # min in x, y
    x, y = np.meshgrid(
        # note small rooms will be skipped here!
        # e.g, 2.5+3-6<0, there will be no chunk in this case
        np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        indexing="ij",
    )
    chunks = np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=-1)
    chunk_idx = 0

    if debug:
        gs_color_ply = scene_path / "gs_semantic_prune.ply"
        gs_color_ply = o3d.io.read_point_cloud(str(gs_color_ply))
        gs_color_ply_pts = np.asarray(gs_color_ply.points)
        gs_color_ply_color = np.asarray(gs_color_ply.colors)
        gs_color_ply_pts = gs_color_ply_pts - gs_color_ply_pts.min(axis=0)
        gs_color_ply_pts_grid = np.floor(gs_color_ply_pts / grid_size).astype(int)
        _, idx = np.unique(gs_color_ply_pts_grid, axis=0, return_index=True)
        gs_color_ply_pts = gs_color_ply_pts[idx]
        gs_color_ply_color = gs_color_ply_color[idx]

        pc_color_ply = scene_path / "pc_semantic.ply"
        pc_color_ply = o3d.io.read_point_cloud(str(pc_color_ply))
        pc_color_ply_pts = np.asarray(pc_color_ply.points)
        pc_color_ply_color = np.asarray(pc_color_ply.colors)
        pc_color_ply_pts = pc_color_ply_pts - pc_color_ply_pts.min(axis=0)
        pc_color_ply_pts_grid = np.floor(pc_color_ply_pts / grid_size).astype(int)
        _, idx = np.unique(pc_color_ply_pts_grid, axis=0, return_index=True)
        pc_color_ply_pts = pc_color_ply_pts[idx]
        pc_color_ply_color = pc_color_ply_color[idx]

    for chunk in chunks:
        if debug:
            print("name",name, "chunk", chunk, "chunk_range", chunk_range)
        mask = (
            (coord[:, 0] >= chunk[0])
            & (coord[:, 0] < chunk[0] + chunk_range[0])
            & (coord[:, 1] >= chunk[1])
            & (coord[:, 1] < chunk[1] + chunk_range[1])
        )
        if debug:
            print("mask", mask.shape, np.sum(mask))

        if np.sum(mask) < chunk_minimum_size:
            print(f"{name} Chunk size too small, skip,", "coords within the chunk: ", np.sum(mask))
            continue

        chunk_data_name = f"{name}_{chunk_idx}"
        if grid_size is not None:
            chunk_split_name = (
                f"{split}_"
                f"grid{grid_size * 100:.0f}mm_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )
        else:
            chunk_split_name = (
                f"{split}_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )

        chunk_save_path = dataset_root / chunk_split_name / chunk_data_name
        chunk_save_path.mkdir(parents=True, exist_ok=True)
        if not debug:
            for key in data_dict.keys():
                np.save(chunk_save_path / f"{key}.npy", data_dict[key][mask])

        if debug:
            gs_mask = (
            (gs_color_ply_pts[:, 0] >= chunk[0])
            & (gs_color_ply_pts[:, 0] < chunk[0] + chunk_range[0])
            & (gs_color_ply_pts[:, 1] >= chunk[1])
            & (gs_color_ply_pts[:, 1] < chunk[1] + chunk_range[1])
            )

            chunk_gs_color_pts = gs_color_ply_pts[gs_mask]
            chunk_gs_color_color = gs_color_ply_color[gs_mask]
            chunk_gs_color_ply = o3d.geometry.PointCloud()
            chunk_gs_color_ply.points = o3d.utility.Vector3dVector(chunk_gs_color_pts)
            chunk_gs_color_ply.colors = o3d.utility.Vector3dVector(chunk_gs_color_color)
            o3d.io.write_point_cloud(
                str(chunk_save_path / "gs_color.ply"), chunk_gs_color_ply
            )
            print("Write" + str(chunk_save_path / "gs_color.ply"))
            
            pc_mask = (
            (pc_color_ply_pts[:, 0] >= chunk[0])
            & (pc_color_ply_pts[:, 0] < chunk[0] + chunk_range[0])
            & (pc_color_ply_pts[:, 1] >= chunk[1])
            & (pc_color_ply_pts[:, 1] < chunk[1] + chunk_range[1])
            )
            chunk_pc_color_pts = pc_color_ply_pts[pc_mask]
            chunk_pc_color_color = pc_color_ply_color[pc_mask]
            chunk_pc_color_ply = o3d.geometry.PointCloud()
            chunk_pc_color_ply.points = o3d.utility.Vector3dVector(chunk_pc_color_pts)
            chunk_pc_color_ply.colors = o3d.utility.Vector3dVector(chunk_pc_color_color)
            o3d.io.write_point_cloud(
                str(chunk_save_path / "pc_color.ply"), chunk_pc_color_ply
            )  
            print("Write" + str(chunk_save_path / "pc_color.ply"))

        chunk_idx += 1
    
    print("in total valid chunks:", chunk_idx, "in", name)
    # if debug:
    #     if chunk_idx == 0:
    #         raise ValueError("No valid chunks found in the scene.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Pointcept processed ScanNet++ dataset.",
    )
    parser.add_argument(
        "--split",
        required=True,
        default="train",
        type=str,
        help="Split need to process.",
    )
    parser.add_argument(
        "--grid_size",
        default=None,
        type=float,
        help="Grid size for initial grid sampling",
    )
    parser.add_argument(
        "--chunk_range",
        default=[6, 6],
        type=int,
        nargs="+",
        help="Range of each chunk, e.g. --chunk_range 6 6",
    )
    parser.add_argument(
        "--chunk_stride",
        default=[3, 3],
        type=int,
        nargs="+",
        help="Stride of each chunk, e.g. --chunk_stride 3 3",
    )
    parser.add_argument(
        "--chunk_minimum_size",
        default=10000,
        type=int,
        help="Minimum number of points in each chunk",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )

    config = parser.parse_args()
    config.dataset_root = Path(config.dataset_root)
    data_list = os.listdir(config.dataset_root / config.split)

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            chunking_scene,
            data_list,
            repeat(config.dataset_root),
            repeat(config.split),
            repeat(config.grid_size),
            repeat(config.chunk_range),
            repeat(config.chunk_stride),
            repeat(config.chunk_minimum_size),
        )
    )
    pool.shutdown()
