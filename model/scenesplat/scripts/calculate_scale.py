import open3d as o3d
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


def calculate_min_rotated_box(pc_path):
    pc_path = os.path.join(pc_path, 'scale.npy')
    scale_i = np.load(pc_path)
    # print("scale_i", scale_i.min(), scale_i.max())
    return scale_i.min(), scale_i.max(), scale_i.mean(), scale_i.std()


data_root_dir = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetppgs_processed_mcmc_prune/train_grid1mm_chunk6x6_stride3x3'

scenes_id = os.listdir(data_root_dir)
scenes_id = [os.path.join(data_root_dir, scene_id) for scene_id in scenes_id]
scenes_id = sorted(scenes_id)

# with Pool(4) as p:
#     r = list(tqdm(p.imap(calculate_min_rotated_box, scenes_id), total=len(scenes_id)))
min_list = []
max_list = []
mean_list = []
std_list = []
for i in tqdm(scenes_id):
    min_i, max_i,mean_i, std_i = calculate_min_rotated_box(i)
    min_list.append(min_i)
    max_list.append(max_i)
    mean_list.append(mean_i)
    std_list.append(std_i)
    

print("np.min(min_list)", np.min(min_list))
print("np.max(max_list)", np.max(max_list))

# print("np.min(mean_list)", np.min(mean_list))
# print("np.max(mean_list)", np.max(mean_list))
print("np.mean(mean_list)", np.mean(mean_list))

print("std_list", np.mean(std_list))