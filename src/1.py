import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"正在使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 沒偵測到 GPU，可能是 CUDA 沒裝好或環境變數問題")

