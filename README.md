# Fine-tuning Gemma-3-4B with LoRA on Google Colab

This project fine-tunes the Gemma-3-4B model using LoRA (Low-Rank Adaptation), optimized with Unsloth + PEFT + SFT. The training is performed on Google Colab with 4-bit quantization (QLoRA) to reduce VRAM usage, using high-quality Chinese Zhihu dataset.

## 📌 Key Features

⚡ Optimized with Unsloth, making fine-tuning 2-4x faster than traditional methods.

🎯 Uses LoRA fine-tuning, reducing computational cost.

📖 Trains on high-quality Zhihu SFT dataset.

✅ Compatible with Google Colab for easy execution.

## 🚀 Quick Start

📌 Click the link below to open the Colab notebook and start training immediately:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O3WAp2oBoayjeMwB57RLoA5SVUcjcasM)


## 🛠️ Installation
Run the following commands in Google Colab:
```bash
# Install dependencies
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```
