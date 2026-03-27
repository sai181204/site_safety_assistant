"""
detection/model.py
──────────────────
YOLO model loader, trainer, and evaluator.
Supports:
  • Loading a custom PPE-trained model (models/ppe_yolov8.pt)
  • Falling back to a COCO-pretrained YOLOv8 model
  • Training / fine-tuning on a custom PPE dataset
  • Evaluation (mAP, precision, recall)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_CONFIG, DATASET_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class PPEModel:
    """
    Wraps an Ultralytics YOLO model for PPE detection.

    Usage
    -----
    model = PPEModel()                   # auto-selects best available weights
    model = PPEModel("models/my.pt")     # explicit path
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or self._resolve_weights()
        self.model: Optional[YOLO] = None
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(self, source, **kwargs):
        """
        Run inference on *source* (frame, path, URL, …).

        Returns
        -------
        list[ultralytics.engine.results.Results]
        """
        assert self.model is not None, "Model not loaded"
        return self.model(
            source,
            conf=kwargs.pop("conf", MODEL_CONFIG["confidence_threshold"]),
            iou=kwargs.pop("iou",  MODEL_CONFIG["nms_iou_threshold"]),
            imgsz=kwargs.pop("imgsz", MODEL_CONFIG["imgsz"]),
            device=kwargs.pop("device", MODEL_CONFIG["device"]),
            verbose=False,
            **kwargs,
        )

    def train(
        self,
        data_yaml: Optional[str] = None,
        epochs: int = DATASET_CONFIG["train_epochs"],
        batch: int = DATASET_CONFIG["batch_size"],
        imgsz: int = DATASET_CONFIG["imgsz"],
        project: str = str(MODELS_DIR / "runs"),
        name: str = "ppe_train",
        resume: bool = False,
    ):
        """
        Fine-tune / train on a custom PPE dataset.

        Parameters
        ----------
        data_yaml : path to dataset YAML (default from config)
        epochs     : training epochs
        batch      : batch size
        imgsz      : input resolution
        project    : where to save runs
        name       : run subdirectory name
        resume     : resume from last checkpoint
        """
        data_yaml = data_yaml or DATASET_CONFIG["yaml_path"]
        if not Path(data_yaml).exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {data_yaml}\n"
                "Run `python scripts/prepare_dataset.py` first."
            )

        logger.info("Starting training — epochs=%d  batch=%d  imgsz=%d", epochs, batch, imgsz)
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=MODEL_CONFIG["device"],
            project=project,
            name=name,
            resume=resume,
            exist_ok=True,
        )
        # Save best weights to models/
        best_weights = Path(project) / name / "weights" / "best.pt"
        dest = MODELS_DIR / "ppe_yolov8.pt"
        if best_weights.exists():
            import shutil
            shutil.copy(best_weights, dest)
            logger.info("Best weights saved → %s", dest)
        return results

    def evaluate(self, data_yaml: Optional[str] = None):
        """
        Evaluate model on validation split.

        Returns
        -------
        dict with keys: precision, recall, mAP50, mAP50_95
        """
        data_yaml = data_yaml or DATASET_CONFIG["yaml_path"]
        if not Path(data_yaml).exists():
            logger.warning("Dataset YAML not found, skipping evaluation.")
            return {}

        logger.info("Evaluating model …")
        metrics = self.model.val(
            data=data_yaml,
            imgsz=MODEL_CONFIG["imgsz"],
            device=MODEL_CONFIG["device"],
            verbose=False,
        )
        result = {
            "precision": float(metrics.box.p.mean()),
            "recall":    float(metrics.box.r.mean()),
            "mAP50":     float(metrics.box.map50),
            "mAP50_95":  float(metrics.box.map),
        }
        logger.info(
            "Evaluation — P=%.3f  R=%.3f  mAP50=%.3f  mAP50-95=%.3f",
            result["precision"], result["recall"],
            result["mAP50"],     result["mAP50_95"],
        )
        return result

    def export(self, format: str = "onnx"):
        """Export model to a different format (onnx, tflite, …)."""
        return self.model.export(format=format)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_weights(self) -> str:
        """Return best available weights path."""
        # Force COCO fallback for now since PPE model isn't trained
        if MODEL_CONFIG.get("use_coco_fallback", False):
            logger.info("Using COCO pretrained model for person detection")
            return MODEL_CONFIG["fallback_weights"]
            
        custom = Path(MODEL_CONFIG["weights_path"])
        if custom.exists():
            logger.info("Using custom PPE weights: %s", custom)
            return str(custom)
        fallback = MODEL_CONFIG["fallback_weights"]
        logger.warning(
            "Custom weights not found at %s — falling back to %s.\n"
            "For proper PPE detection, train a custom model (see README).",
            custom, fallback,
        )
        return fallback

    def _load(self):
        """Load YOLO model from weights_path."""
        logger.info("Loading model: %s", self.weights_path)
        try:
            self.model = YOLO(self.weights_path)
            logger.info(
                "Model loaded — task=%s  classes=%d",
                self.model.task,
                len(self.model.names),
            )
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    # ── Dunder ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"PPEModel(weights='{self.weights_path}')"
