import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

#設定CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#載入 ResNet18 & ResNet50
resnet18 = models.resnet18()
resnet50 = models.resnet50()

#讀取已訓練好的模型
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features,2)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features,2)
resnet18.load_state_dict(torch.load("F:/X-LungNet/models/ResNet18.pth"))
resnet50.load_state_dict(torch.load("F:/X-LungNet/models/ResNet50.pth"))

resnet18 = resnet18.to(device).eval()
resnet50 = resnet50.to(device).eval()

#影像處理函數
def process_image(image_path):
    transform_pipeline = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5],[0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform_pipeline(image).unsqueeze(0).to(device)
    return image

#預測函數
def predict(image_path):
    image = process_image(image_path)

    #讓ResNet18 & ResNet50 一起預測
    with torch.no_grad():
        outputs18 = resnet18(image)
        outputs50 = resnet50(image)
        outputs = (outputs18 + outputs50) / 2

    _, predicted = torch.max(outputs, 1)
    label = "正常" if predicted.item() == 0 else "肺炎"
    print(f"✅ 預測結果：{label}")

#測試函數
image_path = input("請輸入 X 光影像路徑：")
predict(image_path)