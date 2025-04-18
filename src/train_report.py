import matplotlib.pyplot as plt
import json

#讀取train_log.json
def load_train_log():
    with open("F:/X-LungNet/models/train_log.json","r") as f:
        return json.load(f)

#畫Loss & Accuracy 曲線
def plot_train_report():
    log_data = load_train_log()
    epochs = list(range(1, len(log_data["train_loss"]) + 1))

    #Loss曲線
    plt.figure(figsize=(10,5))
    plt.plot(epochs, log_data["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, log_data["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.show()

    #Accuracy曲線
    plt.figure(figsize=(10,5))
    plt.plot(epochs, log_data["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, log_data["valacc"], label="Val Accuracy", marker="o")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.show()

#執行
plot_train_report()