"""Split flat class folders into train/val/test directories."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("root", type=Path, help="Root directory containing class subfolders")
    parser.add_argument("--output", type=Path, default=Path("dataset_split"))
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    classes = [d for d in args.root.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"No class folders found in {args.root}")

    splits = {
        "train": args.train,
        "val": args.val,
        "test": max(0.0, 1.0 - args.train - args.val),
    }

    output_dirs = {name: args.output / name for name in splits}
    for split_dir in output_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in classes:
        files = [f for f in cls_dir.glob("*") if f.is_file()]
        random.shuffle(files)
        n = len(files)
        train_end = int(splits["train"] * n)
        val_end = train_end + int(splits["val"] * n)

        for idx, src in enumerate(files):
            if idx < train_end:
                dest_root = output_dirs["train"]
            elif idx < val_end:
                dest_root = output_dirs["val"]
            else:
                dest_root = output_dirs["test"]
            dest = dest_root / cls_dir.name / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)

        print(f"Copied {n} files for class {cls_dir.name}")

    print(f"Splits saved to {args.output}")


if __name__ == "__main__":
    main()




























