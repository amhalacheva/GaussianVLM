#!/bin/bash
#SBATCH --job-name=scanpp_skip  # Job name
#SBATCH --output=logs/scanpp_skip_%A.log  # Output log file
#SBATCH --error=logs/scanpp_skip_%A.log   # Error log file
#SBATCH -p capacity      # for RTX 6000 Ada 48G, use 'capacity' for 24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2 # 1 for single gpu
#SBATCH --gpus-per-node=2  # 1 for single gpu
#SBATCH --mem=180G          # 90G for single gpu
#SBATCH --time=96:00:00     # hour
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch



# srun --gres=gpu:1 --mem=80G  --cpus-per-task=30 --exclude=hipster-cn001 --time=7-0 -p performance -D `pwd` --pty bash
# srun --gres=gpu:1 --mem=80G  --cpus-per-task=30 --exclude=hipster-cn001 --time=7-0 -p capacity -D `pwd` --pty bash

# install 
# conda create -n pointcept python=3.9 -y
# conda activate pointcept
# export CONDA_OVERRIDE_CUDA=11.8
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH

# conda install ninja -y
# conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install h5py pyyaml -c anaconda -y
# conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
# conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
# pip install torch-geometric
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

source /home/yli7/.bashrc
micromamba activate pointcept

cd /home/yli7/projects/qimaqi/GS_Transformer/
# python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}

# data process
# cd /home/yli7/projects/qimaqi/GS_Transformer/pointcept/datasets/preprocessing/scannetpp
# python preprocess_scannetpp_gs_feat.py --dataset_root /home/yli7/scratch2/datasets/scannetpp --output_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_3dgs_default_depth_true_clip_feat --num_workers 1 --gs_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_3dgs_default --pc_root /home/yli7/scratch2/datasets/scannetpp_preprocessed --gs_feat_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_lang_feat


# python pointcept/datasets/preprocessing/sampling_chunking_data_gs_feat.py --dataset_root /home/yli7/scratch2/datasets/gaussian_world/scannetpp_3dgs_default_depth_true_clip_feat  --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 1

export PYTHONPATH=./
python tools/train.py --config-file configs/scannetpp/hisper/semseg-gs-v3m1-0-base-pretrain.py --options save_path=exp/scannetppgs_default/semseg-gs-v3m1-0-base-pretrain --num-gpus 1

# resume=True weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth


