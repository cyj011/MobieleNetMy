#!/usr/bin/env python3
"""Summarize the best validation result from one or more training folders.

Examples:
  python summarize_results.py /path/baseline /path/baseline_ca
  python summarize_results.py --root /path/fer_img_8class
"""

import argparse
import json
import math
from pathlib import Path


METRIC_KEYS = ("test_acc1", "val_acc1", "acc1", "accuracy", "val_accuracy")


def find_log(path: Path) -> Path | None:
    if path.is_file():
        return path
    direct = path / "log.txt"
    if direct.is_file():
        return direct
    logs = sorted(path.rglob("log.txt"))
    return logs[0] if logs else None


def read_records(log_path: Path):
    records = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            metric_key = next((key for key in METRIC_KEYS if key in item), None)
            if metric_key is None:
                continue
            try:
                metric_value = float(item[metric_key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(metric_value):
                continue
            epoch = item.get("epoch", len(records))
            try:
                epoch = int(epoch)
            except (TypeError, ValueError):
                epoch = len(records)
            records.append((metric_value, epoch, metric_key, item, line_no))
    return records


def display_name(path: Path, log_path: Path) -> str:
    if path.is_file():
        return path.parent.name
    return path.name if path.name else str(path)


def main():
    parser = argparse.ArgumentParser(description="统计训练结果中的最佳准确率和对应轮数")
    parser.add_argument("paths", nargs="*", type=Path, help="实验目录或 log.txt 文件")
    parser.add_argument("--root", type=Path, help="递归查找 root 下所有 log.txt")
    parser.add_argument("--top", type=int, default=0, help="只显示前 N 个结果；0 表示全部")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.root:
        paths.extend(sorted(args.root.rglob("log.txt")))
    if not paths:
        parser.error("请传入实验目录，例如：python summarize_results.py baseline baseline_ca")

    results = []
    seen_logs = set()
    for path in paths:
        path = path.expanduser()
        log_path = find_log(path)
        if log_path is None:
            print(f"[跳过] {path}: 找不到 log.txt")
            continue
        log_path = log_path.resolve()
        if log_path in seen_logs:
            continue
        seen_logs.add(log_path)
        records = read_records(log_path)
        if not records:
            print(f"[跳过] {log_path}: 没有可解析的准确率记录")
            continue
        best = max(records, key=lambda row: row[0])
        results.append((best[0], display_name(path, log_path), best, log_path))

    if not results:
        raise SystemExit("没有找到有效训练结果。")

    results.sort(key=lambda row: row[0], reverse=True)
    if args.top > 0:
        results = results[:args.top]

    print("实验结果（按最佳验证准确率降序）")
    print("=" * 96)
    print(f"{'实验':<32} {'最佳准确率':>12} {'日志epoch':>10} {'第几轮':>10} {'日志文件'}")
    print("-" * 96)
    for _, name, best, log_path in results:
        value, epoch, metric_key, item, line_no = best
        print(f"{name[:32]:<32} {value:>11.4f}% {epoch:>10} {epoch + 1:>10} {log_path}")
        extras = []
        if "test_loss" in item:
            extras.append(f"test_loss={item['test_loss']}")
        if "test_acc5" in item:
            extras.append(f"test_acc5={item['test_acc5']}")
        print(f"  metric={metric_key}, line={line_no}" + (", " + ", ".join(extras) if extras else ""))

    overall = results[0]
    print("=" * 96)
    print(f"总体最佳：{overall[1]}，{overall[0]:.4f}%（第 {overall[2][1] + 1} 轮，日志 epoch={overall[2][1]}）")


if __name__ == "__main__":
    main()
