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
micromamba activate pointcept

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

echo "Running on $(hostname)"
# echo "Start copying data to $TMPDIR on $(date)"
# cp -r /home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs $TMPDIR
# echo "Data copied to $TMPDIR on $(date)"

cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./
# python -u tools/train.py \
#         --config-file configs/scannetpp/snellius/semseg-gs-v3m1-0-base-all-lang-pretrain.py \
#         --options save_path=exp/scannetppgs_default_fix_xyz/base-all-fix-xyz-lang-pretrain-scannetpp-v1 \
#          data.train.data_root=/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs   \
#         --num-gpus 4 \

python -u tools/train.py \
        --config-file configs/scannetpp/snellius/semseg-gs-v3m1-0-base-all-lang-pretrain-all-contrastive-siglip2-voting.py \
        --options save_path=exp/lang_pretrainer/base-scannetpp-v1-fix-xyz-all-contras-siglip2-voting25 batch_size=8\
        --num-gpus 4 \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth