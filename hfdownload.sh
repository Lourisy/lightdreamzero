export HF_HOME="/data/liuzhaoxu/models"

# 3. 运行 Python 下载脚本
python -c "
import os
from huggingface_hub import hf_hub_download, snapshot_download

repo_id = 'Wan-AI/Wan2.1-I2V-14B-480P'

print('>>> 1. 开始下载 CLIP 图像编码器 (安全重下不丢失)')
hf_hub_download(repo_id=repo_id, filename='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')

print('>>> 2. 开始下载 T5 文本解码器')
hf_hub_download(repo_id=repo_id, filename='models_t5_umt5-xxl-enc-bf16.pth')

print('>>> 3. 开始下载 VAE 视频解码器')
hf_hub_download(repo_id=repo_id, filename='Wan2.1_VAE.pth')

print('>>> 4. 开始并行下载 14B DiT 切片权重 (体积巨大，请耐心挂机)')
snapshot_download(repo_id=repo_id, allow_patterns=['diffusion_pytorch_model*'])

print('>>> 底座权重全部下载完成并已持久化存储！')
"
