#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=3
#SBATCH --mem=480G
#SBATCH --time=80:00:00
##SBATCH --scratch-node

# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept_new


echo "Running on $(hostname)"
cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./

# python -u tools/train.py \
#         --config-file configs/scannet200/snellius/lang-pretrain-scannet-all-w-normal-contrastive-siglip2-voting.py \
#         --options save_path=exp/lang_pretrainer/base-scannet-fix-xyz-all-w-normal-contrastive-siglip2-voting \
#          weight=/home/yli7/projects/gaussian_world/GS_Transformer/exp/lang_pretrainer/base-scannet-fix-xyz-all-w-normal-contrastive-siglip2-voting/model/model_best.pth \
#         --num-gpus 4 \
#         --no_distri

python -u tools/train.py \
        --config-file configs/scannet200/snellius/lang-pretrain-scannet-all-w-normal-wo-contrastive-siglip2-voting.py \
        --options save_path=exp/lang_pretrainer/base-scannet-fix-xyz-all-w-normal-wo-contrastive-siglip2-voting \
        --num-gpus 3 \
        --no_distri

# python -u tools/train.py \
#         --config-file configs/scannet200/snellius/lang-pretrain-scannet-all-w-normal-all-contrastive-siglip2-voting.py \
#         --options save_path=exp/lang_pretrainer/base-scannet-fix-xyz-all-w-normal-all-contrastive-siglip2-voting \
#         --num-gpus 3 \
#         --no_distri