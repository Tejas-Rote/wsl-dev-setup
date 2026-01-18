# ML Environment Setup (WSL2)

## Requirements
- Windows 10/11
- WSL2
- NVIDIA GPU with WSL driver
- Miniconda

## Create TensorFlow GPU env
conda env create -f envs/env-tf-gpu.yml
conda activate tf-gpu

## Create PyTorch GPU env
conda env create -f envs/env-pytorch-gpu.yml
conda activate pytorch-gpu

## GPU Test
python scripts/gpu_test.py
