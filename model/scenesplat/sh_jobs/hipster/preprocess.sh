#!/bin/bash
#SBATCH --job-name=scanpp_skip  # Job name
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p performance      # performance for RTX 6000 Ada 48G, use 'capacity' for 24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1   # 1 for single gpu
#SBATCH --mem=45G           # max 90G for single gpu
#SBATCH --time=24:00:00     # hour

##SBATCH --nodelist=hipster-cn006

source /home/yli7/.bashrc
micromamba activate pointcept

echo "Job Start"
echo "running on $(hostname)"

# data preprocess
cd /home/yli7/projects/yue/GS_Transformer/pointcept/datasets/preprocessing/scannetpp
python -u preprocess_scannetpp_gs_fixed.py \
    --dataset_root /home/yli7/scratch2/datasets/scannetpp_v1 \
    --output_root /home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs \
    --num_workers 8 \
    --gs_root /home/yli7/scratch2/outputs/scannetpp_v1_default_fix_xyz_gs \
    --pc_root /home/yli7/scratch2/datasets/scannetpp_preprocessed \
    #--gs_feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_lang_feat

# python -u pointcept/datasets/preprocessing/sampling_chunking_data_gs_feat.py --dataset_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_3dgs_default_depth_true_clip_feat  --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 1

echo "Job End"
echo "Time: $(date)"