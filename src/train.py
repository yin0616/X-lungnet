import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import json
from dataset import train_loader, val_loader

#設定GPU
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

#載入ResNet18預訓練模型
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet50.parameters():
    param.requires_grad = False #凍結所有層，只訓練最後一層
for param in resnet18.parameters():
    param.requires_grad = False #凍結所有層，只訓練最後一層

#修改最後一層（兩個模型都要輸出 2 類）
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Dropout(0.5),  #加入 Dropout，防止過擬合
    nn.Linear(num_ftrs, 2)
)

num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Dropout(0.5),  #泛化
    nn.Linear(num_ftrs, 2)
)


# 把模型丟到 GPU
resnet18 = resnet18.to(device)
resnet50 = resnet50.to(device)

# 設定損失函數 & 優化器（只訓練最後一層）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(resnet18.fc.parameters()) + list(resnet50.fc.parameters()), 
    lr=3e-3, weight_decay=5e-3
)

#記錄 Loss & Accuracy
train_log = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

#訓練模型
if __name__ == '__main__':
    EPOCHS = 20
    for epoch in range(EPOCHS):
        resnet18.train()
        resnet50.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"🚀 訓練進度: 第 {epoch+1} / {EPOCHS} 輪")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs18 = resnet18(images)
            outputs50 = resnet50(images)

            #取平均，讓兩個模型的輸出結合
            outputs = (outputs18 + outputs50) / 2  

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        #在每個 `epoch` 訓練完後，馬上測試驗證集
        resnet18.eval()
        resnet50.eval()
        val_correct = 0
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs18 = resnet18(images)
                outputs50 = resnet50(images)
                outputs = (outputs18 + outputs50) /2

                loss = criterion(outputs, labels)#計算驗證集 Loss
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / val_total

        #記錄 Loss & Accuracy
        train_log["train_loss"].append(train_loss)
        train_log["train_acc"].append(train_acc)
        train_log["val_loss"].append(val_loss)
        train_log["val_acc"].append(val_acc)

        print(f"📊 驗證準確率: {val_acc:.2f}%")  # 每個 Epoch 訓練完，馬上看驗證結果！

    #儲存 Loss & Accuracy 記錄
    with open("F:/X-LungNet/models/train_log.json", "w") as f:
        json.dump(train_log, f)

     #  儲存兩個模型
    torch.save(resnet18.state_dict(), "F:/X-LungNet/models/ResNet18.pth")
    torch.save(resnet50.state_dict(), "F:/X-LungNet/models/ResNet50.pth")
    
    print(" 訓練完成，模型已儲存！")
