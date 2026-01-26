#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=600G
#SBATCH --time=32:00:00
##SBATCH --scratch-node

# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept_new


echo "Running on $(hostname)"
cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./

python -u tools/train.py \
        --config-file configs/scannetpp/snellius/lang-pretrain-ppv2-scannet-fixed-all-w-normal-late-contrastive.py \
        --options save_path=exp/lang_pretrainer/lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive  \
                gpu_nums=1 num_worker=16 batch_size=3 batch_size_val=2 \
        --num-gpus 1 \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth