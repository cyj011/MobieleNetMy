from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EMOTION_NAMES = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAF-DB basic dataset into folder-based classification format."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\basic"),
        help="RAF-DB root directory.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path(r"D:\WorkPlace\pythonProgram\Emotion\raf-db_orig"),
        help="Output directory for the converted dataset.",
    )
    parser.add_argument(
        "--partition-file",
        type=Path,
        default=None,
        help="Optional path to list_patition_label.txt. Defaults to src-root/EmoLabel/list_patition_label.txt.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional path to aligned images directory. Defaults to src-root/Image/aligned.",
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        default="train",
        help="Name of the output training split folder.",
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="val",
        help="Name of the output validation/test split folder.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy or move images into the output folders.",
    )
    return parser.parse_args()


def load_partition_file(partition_file: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    with partition_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, label_text = parts[0], parts[1]
            items.append((name, int(label_text)))
    return items


def ensure_image_path(image_dir: Path, name: str) -> Path:
    stem = Path(name).stem
    candidates = [
        image_dir / f"{stem}_aligned.jpg",
        image_dir / f"{stem}_aligned.png",
        image_dir / f"{stem}.jpg",
        image_dir / f"{stem}.png",
        image_dir / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()

    src_root = args.src_root
    dst_root = args.dst_root
    partition_file = (
        args.partition_file
        if args.partition_file is not None
        else src_root / "EmoLabel" / "list_patition_label.txt"
    )
    image_dir = args.image_dir if args.image_dir is not None else src_root / "Image" / "aligned"

    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records = load_partition_file(partition_file)
    if not records:
        raise RuntimeError(f"No samples found in {partition_file}")

    dst_root.mkdir(parents=True, exist_ok=True)

    split_name_counter = {args.train_folder: 0, args.test_folder: 0}
    missing: list[Path] = []
    mode_fn = shutil.copy2 if args.mode == "copy" else shutil.move

    for name, label in records:
        if label not in EMOTION_NAMES:
            raise ValueError(f"Unexpected RAF-DB label: {label} in {name}")

        split_name = args.train_folder if name.startswith("train") else args.test_folder
        class_name = EMOTION_NAMES[label]
        src_path = ensure_image_path(image_dir, name)

        if not src_path.exists():
            missing.append(src_path)
            continue

        target_dir = dst_root / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / src_path.name
        mode_fn(src_path, target_path)
        split_name_counter[split_name] += 1

    print(f"Done. Output root: {dst_root}")
    print(
        f"Copied/moved {split_name_counter[args.train_folder]} train images and "
        f"{split_name_counter[args.test_folder]} {args.test_folder} images."
    )
    if missing:
        print(f"Warning: {len(missing)} images were missing.")
        for path in missing[:10]:
            print(f"  missing: {path}")


if __name__ == "__main__":
    main()
