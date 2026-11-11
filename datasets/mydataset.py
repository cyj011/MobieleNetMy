import csv
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
    """Read class folders or flat FER+ images with a sibling label.csv."""
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Dataset split directory not found: {root_path}")
    class_list = [cla for cla in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, cla))]
    class_list.sort()
    if not class_list:
        label_csv = os.path.join(root_path, "label.csv")
        if not os.path.isfile(label_csv):
            raise FileNotFoundError(
                f"No class folders or label.csv found under {root_path}. "
                "FER+ flat folders need label.csv alongside the images."
            )

        images_path, images_label = [], []
        missing_images = 0
        skipped_non_basic = 0
        with open(label_csv, "r", newline="", encoding="utf-8-sig") as csv_file:
            rows = csv.reader(csv_file)
            for row_number, row in enumerate(rows, start=1):
                if not row:
                    continue
                image_name = row[0].strip()
                if image_name.lower() in {"image name", "image", "filename", "file name"}:
                    continue
                if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                if len(row) < 9:
                    raise ValueError(
                        f"Malformed FER+ label row {row_number} in {label_csv}: "
                        "expected filename, bounding box, and at least 7 emotion vote columns."
                    )
                try:
                    emotion_votes = [int(float(value.strip())) for value in row[2:] if value.strip()]
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric emotion vote in row {row_number} of {label_csv}"
                    ) from exc

                image_path = os.path.join(root_path, image_name)
                if not os.path.isfile(image_path):
                    missing_images += 1
                    continue
                if len(emotion_votes) < 7 or max(emotion_votes) == 0:
                    skipped_non_basic += 1
                    continue
                winning_label = max(range(len(emotion_votes)), key=emotion_votes.__getitem__)
                if winning_label >= 7:
                    skipped_non_basic += 1
                    continue
                images_path.append(image_path)
                images_label.append(winning_label)

        if not images_path:
            raise ValueError(
                f"No labeled FER+ images found in {root_path}; check label.csv and image files."
            )
        print(
            f"Loaded FER+ {os.path.basename(root_path)}: {len(images_path)} images "
            f"(skipped {missing_images} missing images, "
            f"{skipped_non_basic} non-7-class rows)."
        )
        return images_path, images_label
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
    # Accept either a conventional train/val root or the FER+ layout where
    # FER2013Train, FER2013Valid and FER2013Test are sibling directories.
    data_root = os.path.abspath(args.data_root)
    if os.path.basename(data_root).lower() == "fer2013test":
        data_root = os.path.dirname(data_root)

    split_names = {
        "train": ("train", "FER2013Train", "FER2013train"),
        "val": ("val", "valid", "validation", "FER2013Valid", "FER2013Validation"),
    }

    def find_split(candidates):
        for candidate in candidates:
            path = os.path.join(data_root, candidate)
            if os.path.isdir(path):
                return path
        return os.path.join(data_root, candidates[0])

    train_root = find_split(split_names["train"])
    val_root = find_split(split_names["val"])

    train_classes = sorted(
        name for name in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, name))
    )
    val_classes = sorted(
        name for name in os.listdir(val_root)
        if os.path.isdir(os.path.join(val_root, name))
    )
    if train_classes and train_classes != val_classes:
        raise ValueError(
            f"Train/validation class folders differ: train={train_classes}, val={val_classes}"
        )
    if train_classes and len(train_classes) != args.nb_classes:
        raise ValueError(
            f"Found {len(train_classes)} classes {train_classes}, "
            f"but --nb_classes is {args.nb_classes}."
        )
    if train_classes:
        print("Class mapping:", {name: index for index, name in enumerate(train_classes)})

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
