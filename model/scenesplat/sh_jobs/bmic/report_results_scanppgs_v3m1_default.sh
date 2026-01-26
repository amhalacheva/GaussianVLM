#!/bin/bash
#SBATCH --job-name=gs_process
#SBATCH --output=sbatch_log/gs_process_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06,bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com



# install 
# conda create -n pointcept python=3.9 -y
# conda activate pointcept
# export CONDA_OVERRIDE_CUDA=11.8
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH

# conda install ninja -y
# conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install h5py pyyaml -c anaconda -y
# conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
# conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
# pip install torch-geometric
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh

conda activate pointcept13

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/
export PYTHONPATH=./
python tools/test_gs.py --gs_result_dir /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-base-no-normal_debug_save/result --gs_data_dir /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetppgs_processed_default_prune/val