#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 2
#SBATCH --ntasks-per-node=4   # CHANGED: One task per GPU (4 GPUs per node)
#SBATCH --cpus-per-task=16    # CHANGED: Divided CPUs per GPU
#SBATCH --gpus-per-node=4
#SBATCH --mem=600G
#SBATCH --time=28:00:00
#SBATCH --exclude=gcn146

# conda activation (same as before)
export MAMBA_EXE='/gpfs/home3/yli7/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/gpfs/home3/yli7/local/micromamba';
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate pointcept_new

module purge
module load 2023
module load CUDA/12.4.0

echo "Running on $(hostname) | $(date)"

# Get the master node information
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=${MASTER_NODE}.local.snellius.surf.nl
MASTER_PORT=29501
echo "Master node: $MASTER_NODE"
echo "Master address: $MASTER_ADDR"

# NCCL configuration (keep your existing settings)
export NCCL_DEBUG=WARN       # INFO, for debugging
export NCCL_DEBUG_SUBSYS=INIT,COLL   # ALL, for debugging
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2np0
# export NCCL_SOCKET_IFNAME=ib-bond0  # Try use the bonded IB interface instead of ethernet
export NCCL_SOCKET_TIMEOUT=120

WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

gpu_num=$((4*$SLURM_NNODES))
batch_size=$((2*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((16*gpu_num))

cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./
srun python tools/train.py \
        --config-file configs/scannetpp/snellius/semseg-gs-v3m1-0-base-all-w-fix-xyz.py \
        --options save_path=exp/scannetppgs_default_fix_xyz/semseg-gs-v3m1-0-base-all-fix-xyz-load-vae16-pretrain-ckpt \
         data.train.data_root=/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v1_default_fix_xyz_gs   \
         weight=exp/ckpts/scannetpp_language_vae16_pretrain.pth \
         batch_size=$batch_size batch_size_val=$batch_size_val \
         batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
        # --no_distri \

# resume=True 
# weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-default-base-no-normal_rerun/model/model_last.pth