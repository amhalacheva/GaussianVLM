<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo.png" width="400">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

[![Formatter](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml/badge.svg)](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml)


# SceneSplat
SceneSplat is proposed to ...

## Installation
```
micromamba create -n gscept_linear -y python=3.10
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install einops
pip install transformers==4.46.0
pip install datasets==2.21.0
pip install causal-conv1d==1.4.0
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install h5py pyyaml sharedarray tensorboard tensorboardx yapf addict scipy plyfile termcolor timm 
pip install torch-cluster torch-scatter torch-sparse torch-geometric
pip install spconv-cu121

# PTv1 & PTv2 or precise eval
cd libs/pointops
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0" python  setup.py install
cd ../..
pip install open3d
```
## Checkpoints
Please check https://huggingface.co/GaussianWorld/lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive
For this,
Input: Gaussians N x (coords, normals, color, rotation, scale, opacity)
Output: Language features of Gaussians N x 768 (SigLip2 feature)

## Dataset Structure and Preprocessing
Please refer to https://github.com/Pointcept/Pointcept/tree/main
## Dataset Preprocessed
Please refer to this:
https://huggingface.co/datasets/GaussianWorld/scannetpp_v2_default_fix_xyz_gs_preprocessed

## Training
```
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

e.g. Train on ScanNetpp dataset, `${CONFIG_PATH}` should be `SceneSplat/configs/scannetpp/semseg-gs-v3m1-0-base-feat.py`



## Inference
```
python tools/test.py --config-file ${Config} MODEL.WEIGHTS SceneSplat/outputs/Best/model_best_lang-pretrain-ppv2-and-scannet-fixed-all-w-normal-late-contrastive.pth
```