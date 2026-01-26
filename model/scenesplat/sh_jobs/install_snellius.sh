#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --time=2:00:00

export TORCH_CUDA_ARCH_LIST="8.0 9.0"

module purge
module load 2023
module load CUDA/12.4.0

# env py3.10+cu12.4
conda create -n pointcept_new python=3.10 -y
conda activate pointcept_new
conda install ninja -y
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster -c pyg -y
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric spconv-cu120 flash-attn
pip install numpy==1.26.4 torchvision==0.19.1 # check the torchvison version mismatch

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Open3D (visualization, optional)
pip install open3d


# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept_new

cd /home/yli7/projects/gaussian_world/GS_Transformer
cd libs/pointops
python setup.py install