#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 2
#SBATCH --ntasks-per-node=4   # CHANGED: One task per GPU (4 GPUs per node)
#SBATCH --cpus-per-task=16    # CHANGED: Divided CPUs per GPU
#SBATCH --gpus-per-node=4
#SBATCH --mem=600G
#SBATCH --time=16:00:00

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
MASTER_PORT=29500
echo "Master node: $MASTER_NODE"
echo "Master address: $MASTER_ADDR"

# NCCL configuration (keep your existing settings)
export NCCL_DEBUG=WARN       # INFO, for debugging
export NCCL_DEBUG_SUBSYS=INIT,COLL   # ALL, for debugging
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2np0
export NCCL_SOCKET_TIMEOUT=120

WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

cd /home/yli7/projects/gaussian_world/GS_Transformer
export PYTHONPATH=./     

# CHANGED: Remove the specific --ntasks and --nodes arguments from srun
# Let Slurm's job configuration handle task distribution
# srun python debug_wrapper.py \
srun python tools/train.py \
    --config-file configs/scannetpp/snellius/base-lang-pretrain-scannetppv1-color-normal-contrastive-siglip2-voting.py \
    --options save_path=exp/lang_pretrainer/base-color-normal-scannetppv1-fix-xyz-late75-contrastive-siglip2 batch_size=16 batch_size_val=8 batch_size_test=8 num_worker=96 \