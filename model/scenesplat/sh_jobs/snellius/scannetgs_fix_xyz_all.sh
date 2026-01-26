#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=300G
#SBATCH --time=50:00:00
##SBATCH --constraint=scratch-node

# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept_new

echo "Running on $(hostname)"

gpu_num=2
batch_size=$((12*gpu_num))
batch_size_val=$((2*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((16*gpu_num))


cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./
# python -u tools/train.py \
#         --config-file configs/scannet200/snellius/semseg-gs-base-all-w-fix-xyz.py \
#         --options save_path=exp/scannetgs_default_fix_xyz/base-scannet200-all-w-fix-xyz \
#         batch_size=$batch_size batch_size_val=$batch_size_val \
#         batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
#         --num-gpus 2 \
#         --no_distri \

python -u tools/train.py \
        --config-file configs/scannet/snellius/semseg-gs-base-all-w-fix-xyz.py \
        --options save_path=exp/scannetgs_default_fix_xyz/base-scannet20-all-w-fix-xyz \
        batch_size=$batch_size batch_size_val=$batch_size_val \
        batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
        weight=exp/scannetgs_default_fix_xyz/base-scannet20-all-w-fix-xyz/model/model_best.pth  \
        --num-gpus 2 \
        --no_distri \

# weight=exp/lang_pretrainer/lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive/model/model_best.pth \
