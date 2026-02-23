# GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond

[**Project Page**](https://insait-institute.github.io/gaussianvlm.github.io/) | [**Paper (arXiv)**](https://arxiv.org/abs/2507.00886) | [**Evaluation Results**](https://huggingface.co/datasets/amhalacheva/GaussianVLM_results)

---

## 📢 Important Note on This Release
We are releasing this as an early-access version of the codebase due to multiple requests from the community. 

> [!CAUTION]
> This is an **early release**. A thoroughly cleaned repository for easier setup will be released in the upcoming weeks. For urgent issues, please **contact the first author**.

---

## 🌟 Overview
**GaussianVLM** is a novel scene-centric 3D Vision-Language Model (VLM) designed for comprehensive 3D scene understanding. By leveraging **Language-aligned Gaussian Splatting**, our model achieves state-of-the-art results across a wide range of embodied reasoning tasks without the need for traditional object detectors.

### Core Capabilities:
* **Scene-centric Reasoning:** Operates on dense, language-augmented representations.
* **Dual Sparsification:** Efficiently distills 3D Gaussian features into task-relevant tokens for LLMs.
* **Versatile Benchmarking:** High performance on both scene-level (planning/embodied reasoning) and object-level (captioning/QA) tasks.

---

## 🛠️ Setup & Environment

This repository is built upon the foundation provided by [LEO](https://github.com/embodied-generalist/embodied-generalist). We sincerely thank the authors of LEO for their incredible effort and for open-sourcing their framework.

### Installation (via conda-pack)
To ensure environment reproducibility, we provide a [pre-packaged environment](https://huggingface.co/datasets/amhalacheva/gvlm_env).
   ```bash
   # Create a directory for the environment
   mkdir -p gaussian_vlm_env
   # Unpack the provided environment archive
   tar -xzf gaussian_vlm_env.tar.gz -C gaussian_vlm_env
   source gaussian_vlm_env/bin/activate
   conda-unpack
  ```

### Backbones & Data Setup
   ```bash
   cd GaussianVLM
   # Annotations
   git clone https://huggingface.co/datasets/amhalacheva/GaussianVLM_training_data
   # Opt (or another LLM) and SigLip2
   git clone https://huggingface.co/facebook/opt-1.3b
   git clone https://huggingface.co/google/siglip2-base-patch16-512

   # Clone SceneSplat 3D GS Backbone 
   cd model/scenesplat/
   git clone https://huggingface.co/amhalacheva/GaussianVLM_SceneSplat
   ```

   Please, also download `GaussianWorld/scannet_default_fix_xyz_gs_preprocessed_no_feat` from HuggingFace for 3DGS ScanNet scenes.

### Example 

   ```bash
   export PYTHONPATH=model/scenesplat:$PYTHONPATH
   export PYTHONPATH=evaluator:$PYTHONPATH
   export CUDA_HOME="/opt/modules/nvidia-cuda-12.4.1"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export PATH=/opt/modules/nvidia-cuda-12.4.1/bin:$PATH
   export CC=/opt/modules/gcc-10.5.0/bin/gcc
   export CXX=/opt/modules/gcc-10.5.0/bin/g++
   export TORCH_CUDA_ARCH_LIST="8.6"

   python launch.py --mode accelerate --mem_per_gpu  80  --time 48  --config configs/gs_ll3da_train.yaml --gpu_per_node 1 --num_nodes 1 num_gpu=8
   ```

Please, modify the configs files according to your setup!
- `gs_ll3da_pretrain.yaml` gives the pretraining configuration
- `gs_ll3da_train.yaml ` - the training setup

