import torch
import torchvision.models as models
from dataset import test_loader

#設定GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#載入模型
resnet18 = models.resnet18()
resnet50 = models.resnet50()

#讀取已訓練好的模型
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 2)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 2)
resnet18.load_state_dict(torch.load("F:/X-LungNet/models/ResNet18.pth"))
resnet50.load_state_dict(torch.load("F:/X-LungNet/models/ResNet50.pth"))

resnet18 = resnet18.to(device).eval()
resnet50 = resnet50.to(device).eval()

#測試模型
if __name__ == '__main__':
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs50 = resnet50(images)
            outputs18 = resnet18(images)
            outputs = (outputs18 + outputs50)/2
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100. *correct / total
    print(f"🔥 測試準確率: {test_acc:.2f}%")
