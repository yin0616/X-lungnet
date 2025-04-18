import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing

# 自動設定 num_workers，確保數據讀取最快速
NUM_WORKERS = min(6, multiprocessing.cpu_count() - 4)  # 最高用 4 線程，但不會卡死 CPU

data_dir = "F:/X-LungNet/testdata/chest_xray"

#訓練數據的影像增強(防止過凝合)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomRotation(10),  # 隨機旋轉 +/-10 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 隨機移動
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 隨機調整亮度 & 對比
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


#測試與驗證數據的影像處理
test_transform = transforms.Compose([
    transforms.Resize((224,224)),#統一尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

#讀取數據集
train_dataset = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform = test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform = test_transform)

#建立dataLoder(批次讀取數據)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)

# ✅ 顯示數據數量
print(f"📊 訓練集: {len(train_dataset)} 張圖片")
print(f"📊 驗證集: {len(val_dataset)} 張圖片")
print(f"📊 測試集: {len(test_dataset)} 張圖片")