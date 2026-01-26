#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=600G
#SBATCH --time=32:00:00
##SBATCH --scratch-node

# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept

# pointcept_new
# module purge
# module load 2023
# module load CUDA/12.4.0

cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./
srun python tools/train.py \
        --config-file configs/scannetpp/snellius/base-lang-pretrain-scannetppv1-color-normal-contrastive-siglip2-voting.py \
        --options save_path=exp/lang_pretrainer/base-color-normal-scannetppv1-fix-xyz-late75-contrastive-siglip2 batch_size=8\
        --num-gpus 4 \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth