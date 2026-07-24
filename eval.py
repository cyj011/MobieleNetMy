import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ===================== 全局配置（和训练脚本一一对应，自行修改路径） =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_CLASSES = 9
INPUT_SIZE = 224

# 数据集、权重、输出路径
DATA_ROOT = r"D:\WorkPlace\pythonProgram\Emotion\dataset"
CKPT_BEST = r"D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\result\DTP\mobilenetv4_conv_large_best_checkpoint.pth"
OUTPUT_SAVE = r"D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\result\DTP"

# 类别顺序（严格匹配数据集文件夹顺序）
CLASS_NAMES = ["Ang", "contempt", "Dis", "Fea", "Hap", "Neu", "Sad", "sleepy", "Sur"]
MODEL_ALIAS = "MobileNetV4-DTP-LCA"
# ====================================================================================

def load_full_model():
    """复刻训练脚本完整模型结构：timm预训练大模型 + LCA注意力 + DTP裁剪模块"""
    from models.attention import LiteChannelAttention
    from models.dynamic_token_pruning import DynamicTokenPruning
    import timm

    # 基础模型，和训练完全一致
    model = timm.create_model(
        "mobilenetv4_conv_large.e600_r384_in1k",
        pretrained=True,
        num_classes=NUM_CLASSES,
        global_pool='avg'
    )
    model.reset_classifier(num_classes=NUM_CLASSES)

    # 插入第一层LCA+DTP (blocks[3], 通道512)
    model.blocks[3].add_module("LCA", LiteChannelAttention(512))
    model.blocks[3].add_module("DTP", DynamicTokenPruning(channels=512, keep_ratio=0.9))
    # 最后一层LCA (通道960)
    model.blocks[-1].add_module("LCA", LiteChannelAttention(960))

    # 加载最优权重
    checkpoint = torch.load(CKPT_BEST, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    # 精度加速配置，和训练同步
    model.to(DEVICE)
    model.eval()
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    print(f"✅ 模型结构加载完成 | 权重来源：{CKPT_BEST}")
    return model

def get_val_dataloader():
    """补全build_dataset所需全部参数，消除AttributeError"""
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    # 数据集核心参数
    args.data_root = DATA_ROOT
    args.nb_classes = NUM_CLASSES
    args.input_size = INPUT_SIZE
    args.data_set = "myData"
    args.distributed = False
    args.pin_mem = False

    # 数据增强全套参数（mydataset构建transform必须）
    args.color_jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.train_interpolation = 'bicubic'
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1
    args.resplit = False
    args.mixup = 0.0
    args.cutmix = 0.0
    args.ThreeAugment = False

    # 导入数据集构建
    from datasets import build_dataset
    _, dataset_val = build_dataset(args=args)

    # 验证集dataloader配置对齐训练
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.5 * BATCH_SIZE),
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=4
    )
    print(f"✅ 验证集加载完毕，样本总数：{len(dataset_val)}")
    return val_loader

def calculate_metrics_and_print(model, val_loader):
    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            pred_idx = torch.argmax(outputs, dim=1)
            all_preds.extend(pred_idx.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 全局指标
    total_acc = accuracy_score(all_true_labels, all_preds)
    weighted_f1 = f1_score(all_true_labels, all_preds, average="weighted")
    report_dict = classification_report(all_true_labels, all_preds, digits=4, output_dict=True)

    # 逐类别精确率
    class_prec = []
    for cls_idx in range(NUM_CLASSES):
        key = str(cls_idx)
        prec = report_dict[key]["precision"] if key in report_dict else 0.0
        class_prec.append(f"{prec:.4f}")

    # 控制台打印指定格式
    print("\n" + "-" * 110)
    header_line = "Acc\tF1\t" + "\t".join(CLASS_NAMES)
    data_line = f"{MODEL_ALIAS}\t{total_acc:.4f}\t{weighted_f1:.4f}\t" + "\t".join(class_prec)
    print(header_line)
    print(data_line)
    print("-" * 110 + "\n")

    # 保存文本结果
    save_txt_path = os.path.join(OUTPUT_SAVE, "eval_result_metrics.txt")
    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write(header_line + "\n")
        f.write(data_line + "\n")
    print(f"📄 指标结果已保存至：{save_txt_path}")

if __name__ == "__main__":
    net = load_full_model()
    val_data = get_val_dataloader()
    calculate_metrics_and_print(net, val_data)