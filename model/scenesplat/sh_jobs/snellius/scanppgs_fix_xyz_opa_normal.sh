#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=700G
#SBATCH --time=20:00:00
#SBATCH --constraint=scratch-node

# conda activation
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept

echo "Start copying data to $TMPDIR on $(date)"
cp -r /home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs $TMPDIR
echo "Data copied to $TMPDIR on $(date)"

cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./
python -u tools/train.py \
        --config-file configs/scannetpp/snellius/semseg-gs-v3m1-0-base-opa-normal-w-fix-xyz.py \
        --options save_path=exp/scannetppgs_default_fix_xyz/semseg-gs-v3m1-0-base-opa-normal-wo-sampling-tail-classes \
         data.train.data_root=$TMPDIR/scannetpp_v1_default_fix_xyz_gs   \
        --num-gpus 4 \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth