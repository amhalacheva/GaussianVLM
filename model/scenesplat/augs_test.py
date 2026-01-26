#!/usr/bin/env python3
import os
import copy
import random
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

# =========================
# Transformation Functions
# =========================

from pointcept.datasets.transform import (
    RandomDropout,
    RandomRotate,
    RandomFlip,
    RandomRotateTargetAngle,
    RandomScale,
    CenterShift,
    NormalizeColor,
    NormalizeCoord,
    SphereCrop,
    RandomJitter,
    ElasticDistortion,
)

# =========================
# Ply I/O Helper Functions
# =========================


def load_ply(file_path):
    """
    Load a 3dgs ply file and return a data_dict with keys:
      - coord: (N,3)
      - color: (N,3)  [from f_dc_*]
      - opacity: (N,1) after applying sigmoid
      - scale: (N,3) after applying exp
      - quat: (N,4)
      - normal: (N,3) (if not present, zeros are used)
    Discards high order SH coefficients (f_rest_*).
    """
    plydata = PlyData.read(file_path)
    v = plydata["vertex"]

    # Coordinates
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)

    # Color from f_dc channels
    f_dc = np.stack(
        [np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])],
        axis=1,
    )

    f_pc = f_dc.reshape(-1, 3)
    C0 = 0.28209479177387814
    f_pc = (f_pc*C0).astype(np.float32) + 0.5 # 0.5 is the mean value of the SH basis
    f_pc = np.clip(f_pc, 0, 1)

    # Raw opacity then apply sigmoid
    raw_opacity = np.asarray(v["opacity"]).reshape(-1, 1)
    opacity = 1 / (1 + np.exp(-raw_opacity))

    # Scales: sort keys that start with "scale_"
    scale_keys = [key for key in v.data.dtype.names if key.startswith("scale_")]
    scale_keys = sorted(scale_keys, key=lambda x: int(x.split("_")[-1]))
    scales = np.stack([np.asarray(v[key]) for key in scale_keys], axis=1)
    scales = np.exp(scales)

    # Quaternions: sort keys that start with "rot"
    quat_keys = [key for key in v.data.dtype.names if key.startswith("rot")]
    quat_keys = sorted(quat_keys, key=lambda x: int(x.split("_")[-1]))
    quat = np.stack([np.asarray(v[key]) for key in quat_keys], axis=1)
    # normalize quaternions
    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / quat_norm

    # Normals: if available, else zeros.
    if all(k in v.data.dtype.names for k in ["nx", "ny", "nz"]):
        normals = np.stack(
            [np.asarray(v["nx"]), np.asarray(v["ny"]), np.asarray(v["nz"])], axis=1
        )
    else:
        normals = np.zeros_like(xyz)

    data_dict = {
        "coord": xyz,
        "color": f_dc,  # note in dataloading, we have a convert and read f_pc*255
        "opacity": opacity,
        "scale": scales,
        "quat": quat,
        "normal": normals,
    }
    return data_dict


def save_ply(data_dict, file_path, max_sh_degree=3):
    """
    Save a 3dgs ply file.
      - f_rest channels (extra SH coefficients) are set to zero.
      - Before saving, the opacity and scale values are converted back to their raw forms,
        i.e. the inverse of the sigmoid (logit) and the inverse of exp (log) respectively.
      - The attributes are ordered as:
          x, y, z, nx, ny, nz,
          f_dc_0, f_dc_1, ..., f_dc_{C-1},
          f_rest_0, ..., f_rest_{R-1},
          opacity,
          scale_0, scale_1, ..., scale_{S-1},
          rot_0, rot_1, ..., rot_{Q-1}
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    N = data_dict["coord"].shape[0]

    # Coordinates and normals
    xyz = data_dict["coord"]  # (N, 3)
    normals = data_dict.get("normal", np.zeros_like(xyz))  # (N, 3)

    # f_dc channels from "color" (assumed shape: (N, 3))
    f_dc = data_dict["color"]
    num_f_dc = f_dc.shape[1]

    # f_rest channels: set all to zero
    num_f_rest = 3 * (((max_sh_degree + 1) ** 2) - 1)  # For max_sh_degree=3, equals 45
    f_rest = np.zeros((N, num_f_rest), dtype=np.float32)

    # Inverse transform opacity:
    # data_dict["opacity"] was obtained via sigmoid: opacity = 1 / (1 + exp(-raw_opacity))
    # Inverse (logit): raw_opacity = ln(opacity/(1-opacity))
    opacity = data_dict["opacity"]
    if opacity.ndim == 1:
        opacity = opacity.reshape(-1, 1)
    eps = 1e-7  # to prevent division by zero
    opacity = np.clip(opacity, eps, 1 - eps)
    raw_opacity = np.log(opacity / (1 - opacity))

    # Inverse transform scales:
    # data_dict["scale"] was obtained via: scale = exp(raw_scale)
    # So, raw_scale = log(scale)
    scales = data_dict["scale"]
    raw_scales = np.log(scales)
    num_scale = scales.shape[1]

    # Rotation channels (quaternions)
    quat = data_dict["quat"]
    num_quat = quat.shape[1]

    # Build dtype list following the attribute order
    dtype_list = []
    # Coordinates and normals
    for attr in ["x", "y", "z", "nx", "ny", "nz"]:
        dtype_list.append((attr, "f4"))
    # f_dc channels
    for i in range(num_f_dc):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # f_rest channels
    for i in range(num_f_rest):
        dtype_list.append((f"f_rest_{i}", "f4"))
    # Opacity (raw value)
    dtype_list.append(("opacity", "f4"))
    # Scale channels (raw values)
    for i in range(num_scale):
        dtype_list.append((f"scale_{i}", "f4"))
    # Rotation channels (quaternions)
    for i in range(num_quat):
        dtype_list.append((f"rot_{i}", "f4"))

    # Create and fill the structured array
    vertex_all = np.empty(N, dtype=dtype_list)
    vertex_all["x"] = xyz[:, 0]
    vertex_all["y"] = xyz[:, 1]
    vertex_all["z"] = xyz[:, 2]
    vertex_all["nx"] = normals[:, 0]
    vertex_all["ny"] = normals[:, 1]
    vertex_all["nz"] = normals[:, 2]
    for i in range(num_f_dc):
        vertex_all[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(num_f_rest):
        vertex_all[f"f_rest_{i}"] = f_rest[:, i]
    vertex_all["opacity"] = raw_opacity[:, 0]
    for i in range(num_scale):
        vertex_all[f"scale_{i}"] = raw_scales[:, i]
    for i in range(num_quat):
        vertex_all[f"rot_{i}"] = quat[:, i]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(file_path)
    print(f"Saved {file_path}")


# =========================
# Main Test Script
# =========================


def main():
    # Input file path (update if needed)
    input_file = "/home/yue/projects/gaussian_world/GS_Transformer/036bce3393_3dgs.ply"
    data_dict = load_ply(input_file)
    print("Loaded input file.")

    # For testing, we want to force the augmentation to apply.
    transforms = {
        "RandomDropout": RandomDropout(
            dropout_ratio=0.2, dropout_application_ratio=1.0
        ),
        "RandomRotate": RandomRotate(
            angle=[-1, 1], center=None, axis="z", always_apply=True
        ),
        "RandomRotateTargetAngle": RandomRotateTargetAngle(
            angle=(0.5, 1, 1.5), center=None, axis="z", always_apply=True
        ),
        "RandomFlip": RandomFlip(p=1.0),
        "RandomScale": RandomScale(scale=(0.7, 0.8)),
        "CenterShift": CenterShift(apply_z=True),
        "NormalizeColor": NormalizeColor(),
        "NormalizeCoord": NormalizeCoord(),
        "SphereCrop": SphereCrop(point_max=204800, mode="random"),
        "RandomJitter": RandomJitter(sigma=0.005, clip=0.02),
        "ElasticDistortion": ElasticDistortion(
            distortion_params=[[0.2, 0.4], [0.8, 1.6]]
        ),
    }

    # Apply each transform and write out a new file.
    for name, transform in transforms.items():
        # Make a deep copy so each transform starts from the original data.
        transformed_data = transform(copy.deepcopy(data_dict))
        output_file = f"exp/augs_test/output_{name}.ply"
        save_ply(transformed_data, output_file)


if __name__ == "__main__":
    main()
