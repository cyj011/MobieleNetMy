import torch
import os
from PIL import Image
from torchvision import transforms
from .split_data import read_split_data
from torch.utils.data import Dataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform


class MyDataset(Dataset):
    def __init__(self, image_paths, image_labels, transforms=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert('RGB')
        label = self.image_labels[item]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((1.0 / 0.96) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


# ===================== 【已修改】适配 train/val 文件夹 =====================
def read_image_path_label(root_path):
    """直接读取文件夹下的所有类别图片"""
    class_list = [cla for cla in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, cla))]
    class_list.sort()
    class_indices = {k: v for v, k in enumerate(class_list)}

    images_path = []
    images_label = []

    for cla_name, cla_label in class_indices.items():
        cla_dir = os.path.join(root_path, cla_name)
        for img_name in os.listdir(cla_dir):
            if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                img_path = os.path.join(cla_dir, img_name)
                images_path.append(img_path)
                images_label.append(cla_label)

    return images_path, images_label


def build_dataset(args):
    # 读取你已分好的 train / val 文件夹
    data_root = args.data_root
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")

    # 直接读取，不需要自动划分
    train_image_path, train_image_label = read_image_path_label(train_root)
    val_image_path, val_image_label = read_image_path_label(val_root)

    print(f"\n✅ 训练集数量：{len(train_image_path)}")
    print(f"✅ 验证集数量：{len(val_image_path)}")

    # 数据增强
    train_transform = build_transform(True, args)
    valid_transform = build_transform(False, args)

    # 构建数据集
    train_set = MyDataset(train_image_path, train_image_label, train_transform)
    valid_set = MyDataset(val_image_path, val_image_label, valid_transform)

    return train_set, valid_set