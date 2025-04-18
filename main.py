import os

# ✅ 執行 `train.py`
def train_model():
    print("🚀 開始訓練模型...")
    os.system("python f:/X-LungNet/src/train.py")

# ✅ 執行 `test.py`
def test_model():
    print("📊 測試模型準確率...")
    os.system("python f:/X-LungNet/src/test.py")

# ✅ 單張影像預測
def predict_single():
    image_path = input("請輸入 X 光影像路徑：")
    os.system(f"python f:/X-LungNet/src/predict.py {image_path}")

# ✅ 批量影像預測
def predict_batch():
    image_dir = input("請輸入 X 光影像資料夾路徑：")
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        print(f"📷 預測 {image_name} 中...")
        os.system(f"python f:/X-LungNet/src/predict.py {image_path}")

# ✅ 主選單
def main():
    while True:
        print("\n🎯 X-LungNet 主選單")
        print("1️⃣ 訓練 AI")
        print("2️⃣ 測試 AI")
        print("3️⃣ 單張影像預測")
        print("4️⃣ 批量影像預測")
        print("5️⃣ 離開")

        choice = input("請選擇功能：")
        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            predict_single()
        elif choice == "4":
            predict_batch()
        elif choice == "5":
            print("👋 再見！")
            break
        else:
            print("❌ 選項錯誤，請重新輸入！")

if __name__ == "__main__":
    main()
