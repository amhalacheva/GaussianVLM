#!/bin/bash
#SBATCH --job-name=submit  # Job name
#SBATCH --output=logs/scanpp_skip_%A.log  # Output log file
#SBATCH --error=logs/scanpp_skip_%A.log   # Error log file
#SBATCH -p performance      # for RTX 6000 Ada 48G, use 'capacity' for 24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4  # 1 for single gpu
#SBATCH --gpus-per-node=4   # 1 for single gpu
#SBATCH --mem=200G          # 90G for single gpu
#SBATCH --time=96:00:00     


export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

source /home/yli7/.bashrc
micromamba activate pointcept
cd /home/yli7/projects/yue/GS_Transformer

export PYTHONPATH=./
python -u tools/train.py \
        --config-file configs/scannetpp/hisper/semseg-gs-v3m1-0-base-all-w-fix-xyz.py \
        --options save_path=exp/scannetppgs_default/semseg-gs-v3m1-0-base-all-w-fix-xyz  \
        --num-gpus 8 \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth