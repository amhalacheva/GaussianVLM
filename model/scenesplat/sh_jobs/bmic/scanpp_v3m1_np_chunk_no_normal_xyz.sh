#!/bin/bash
#SBATCH --job-name=pointcept_baseline
#SBATCH --output=sbatch_log/pointcept_v3m1_color_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
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

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept

# python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetpp --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 4

export PYTHONPATH=./
python tools/train.py --config-file configs/scannetpp/semseg-pt-v3m1-0-base-no-chunk-no-normal-xyz.py --options save_path=exp/scannetpp/semseg-pt-v3m1-0-base-no-chunk-no-normal-xyz resume=True weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetpp/semseg-pt-v3m1-0-base-no-chunk-no-normal-xyz/model/model_last.pth

# python tools/train.py --config-file configs/scannetpp/semseg-pt-v3m1-0-base-no-normal.py --options save_path=exp/scannetpp/semseg-pt-v3m1-0-base-no-normal resume=True weight=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetpp/semseg-pt-v3m1-0-base-no-normal/model/model_last.pth



# export PATH=/scratch_net/schusch/qimaqi/install_gcc:$PATH
# export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0

# export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0


#

# sh scripts/train.sh -p python -g 1 -d scannetpp -c semseg-pt-v2m2-0-base -n debug
# 

# python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py --dataset_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannet_full --output_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetpp --num_workers 4

# python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetpp --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers 4
# python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetpp --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers 4


# Direct
