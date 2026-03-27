"""
scripts/prepare_dataset.py
──────────────────────────
Helper script to download and prepare a public PPE dataset from Roboflow.

Usage
-----
# Install roboflow SDK first:
#   pip install roboflow

# Then run:
python scripts/prepare_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

If you don't have an API key, sign up free at https://roboflow.com/
and follow the steps to export the dataset in YOLOv8 format, then
manually place files under data/images/ and data/labels/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def download_roboflow(api_key: str):
    """Download construction PPE dataset from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Install roboflow: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    # Public workspace / project — update if you use a different dataset
    project  = rf.workspace("ppe-detection-jlhau").project("construction-ppe-safety")
    dataset  = project.version(1).download("yolov8", location=str(DATA / "roboflow_dl"))
    logger.info("Downloaded to %s", dataset.location)

    # Restructure to match our expected layout
    dl_root  = Path(dataset.location)
    for split in ("train", "valid"):
        dst_split = "val" if split == "valid" else split
        img_src   = dl_root / split / "images"
        lbl_src   = dl_root / split / "labels"
        img_dst   = DATA / "images" / dst_split
        lbl_dst   = DATA / "labels" / dst_split

        if img_src.exists():
            shutil.copytree(img_src, img_dst, dirs_exist_ok=True)
            logger.info("Copied images → %s  (%d files)", img_dst, len(list(img_dst.iterdir())))
        if lbl_src.exists():
            shutil.copytree(lbl_src, lbl_dst, dirs_exist_ok=True)
            logger.info("Copied labels → %s  (%d files)", lbl_dst, len(list(lbl_dst.iterdir())))

    logger.info("Dataset ready. Run training with: python main.py --train")


def create_dummy_dataset():
    """
    Create a minimal synthetic dataset for smoke-testing the pipeline.
    Generates blank images with dummy label files — NOT useful for
    actual training but lets you verify the code without real data.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        logger.error("Install numpy and opencv-python")
        sys.exit(1)

    np.random.seed(42)
    for split in ("train", "val"):
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = 20 if split == "train" else 5
        for i in range(n):
            # Random BGR image
            img = (np.random.rand(480, 640, 3) * 255).astype("uint8")

            # Draw a fake "person" rectangle
            x1, y1, x2, y2 = 100, 80, 200, 400
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 150, 100), -1)

            fname = f"frame_{split}_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)

            # YOLO label: person (0) + helmet (1) + vest (2)
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 480
            w  = (x2 - x1) / 640
            h  = (y2 - y1) / 480
            with open(lbl_dir / fname.replace(".jpg", ".txt"), "w") as f:
                f.write(f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
                # helmet at top of person
                f.write(f"1 {cx:.4f} {(y1+40)/480:.4f} {w*0.5:.4f} {80/480:.4f}\n")
                # vest at middle
                f.write(f"2 {cx:.4f} {(y1+200)/480:.4f} {w*0.8:.4f} {160/480:.4f}\n")

    logger.info("Dummy dataset created under data/images/ and data/labels/")
    logger.info("⚠  This is NOT real training data — replace with actual PPE images.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", help="Roboflow API key")
    p.add_argument("--dummy",   action="store_true",
                   help="Create a synthetic dummy dataset for testing")
    a = p.parse_args()

    if a.dummy:
        create_dummy_dataset()
    elif a.api_key:
        download_roboflow(a.api_key)
    else:
        print(__doc__)
