# GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond

[**Project Page**](https://insait-institute.github.io/gaussianvlm.github.io/) | [**Paper (arXiv)**](https://arxiv.org/abs/2507.00886) | [**Evaluation Results**](https://huggingface.co/datasets/amhalacheva/GaussianVLM_results)

---

## 📢 Important Note on This Release
We are releasing this as an early-access version of the codebase due to multiple requests from the community. 

> [!CAUTION]
> This is an **early release**. A more detailed set of instructions and a thoroughly cleaned repository for easier setup will be released in the upcoming weeks. For immediate setup help or specific queries, please **contact the first author**.

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