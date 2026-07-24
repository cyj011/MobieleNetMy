import json
import matplotlib.pyplot as plt
import numpy as np

# ====================== 【统一路径配置区 - 仅在此处修改所有路径】 ======================
# 训练日志txt文件路径
LOG_TXT_PATH = r"D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\fe2013\mn+ca\log.txt"
# 训练曲线保存路径
CURVE_SAVE_PATH = r"D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\fe2013\image\mn+ca/train_val_curve.png"
# 混淆矩阵图片保存路径
CM_SAVE_PATH = r"D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\fe2013\image\mn+ca/confusion_matrix.png"
# ===================================================================================

def load_log_from_txt(file_path):
    """从txt文件逐行读取日志json"""
    log_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            log_dict = json.loads(line)
            log_list.append(log_dict)
    return log_list

if __name__ == "__main__":
    # 1. 读取日志文件
    log_data = load_log_from_txt(LOG_TXT_PATH)

    # 2. 提取指标数据
    epochs = []
    train_loss_list = []
    val_acc1_list = []
    for item in log_data:
        epochs.append(item["epoch"])
        train_loss_list.append(item["train_loss"])
        val_acc1_list.append(item["test_acc1"])

    # 3. 计算全局最高Top1 Acc
    max_acc = max(val_acc1_list)
    max_acc_epoch = epochs[val_acc1_list.index(max_acc)]
    print("==== 当前最高验证Top1 Acc ====")
    print(f"最高Acc: {max_acc:.2f}%")
    print(f"对应Epoch: {max_acc_epoch}")

    # 4. 绘制Train Loss & Val Acc 双轴曲线
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#d62728"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color1)
    line1 = ax1.plot(epochs, train_loss_list, color=color1, linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    ax2.set_ylabel("Val Acc1 (%)", color=color2)
    line2 = ax2.plot(epochs, val_acc1_list, color=color2, linewidth=2, marker="o", label="Val Acc1")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 标记最高点
    ax2.scatter(max_acc_epoch, max_acc, color="gold", s=100, zorder=5, label=f"Max Acc {max_acc:.2f}%")

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Train Loss & Validation Top1 Accuracy Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=300)
    print(f"训练曲线已保存至: {CURVE_SAVE_PATH}")
    plt.show()

# ====================== 混淆矩阵绘图函数（独立调用） ======================
def plot_confusion_matrix(cm, class_names):
    """
    cm: np.array 二维混淆矩阵
    class_names: list 类别名称
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    plt.ylabel("True Label")
    plt.xlabel("Pred Label")
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=300)
    print(f"混淆矩阵图已保存至: {CM_SAVE_PATH}")
    plt.show()

# 使用示例（有混淆矩阵时取消注释修改）
# if __name__ == "__main__":
#     cm = np.array([[10,2],[3,15]])
#     classes = ["cls0", "cls1"]
#     plot_confusion_matrix(cm, classes)