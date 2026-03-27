"""
Central configuration for Smart Construction Site Safety System
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure dirs exist
for d in [LOGS_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_CONFIG = {
    # Path to your trained/fine-tuned YOLO weights (place in models/)
    # Falls back to pretrained yolov8n if not found
    "weights_path": str(MODELS_DIR / "ppe_yolov8.pt"),
    "fallback_weights": "yolov8n.pt",  # COCO pretrained fallback
    
    # Use COCO pretrained for now since PPE model isn't trained
    "use_coco_fallback": True,

    # Detection classes
    # When using the PPE-specific model these are the class IDs:
    #   0: person, 1: helmet, 2: vest, 3: no_helmet, 4: no_vest
    # When using COCO fallback, only person (class 0) is reliable
    "class_names": {
        0: "person",
        1: "helmet",
        2: "vest",
        3: "no_helmet",
        4: "no_vest",
    },

    # Confidence & NMS
    "confidence_threshold": 0.50,
    "nms_iou_threshold": 0.45,

    # Input resolution (must be multiple of 32)
    "imgsz": 640,

    # Device: "cpu" | "cuda" | "mps"
    "device": "cpu",
}

# ─── PPE Class Labels ─────────────────────────────────────────────────────────
PPE_CLASSES = {
    "helmet":     1,
    "vest":       2,
    "no_helmet":  3,
    "no_vest":    4,
}

VIOLATION_CLASSES = {3, 4}        # class IDs that represent violations
EQUIPMENT_CLASSES = {1, 2}        # class IDs for PPE equipment
PERSON_CLASS = 0

# ─── Severity Thresholds (seconds) ───────────────────────────────────────────
SEVERITY_CONFIG = {
    "LOW":    (0, 5),       # 0–5 s
    "MEDIUM": (5, 15),      # 5–15 s
    "HIGH":   (15, float("inf")),  # >15 s
}

# IoU overlap required to associate PPE box with a person box
PPE_ASSOCIATION_IOU = 0.20

# ─── Zone Configuration (pixel rectangles: x1, y1, x2, y2) ──────────────────
# Set ENABLE_ZONES = False to disable zone logic
ENABLE_ZONES = True
HAZARD_ZONES = [
    # Example zones — adjust to match your camera layout
    # (x1, y1, x2, y2, zone_name)
    (0,   0,   320, 360, "Zone-A-Excavation"),
    (320, 0,   640, 360, "Zone-B-Scaffolding"),
]

# ─── Tracker Configuration ───────────────────────────────────────────────────
TRACKER_CONFIG = {
    "max_age": 30,          # frames before a track is deleted
    "n_init": 1,            # frames before a track is confirmed (reduced from 3 for testing)
    "max_cosine_distance": 0.4,
    "nn_budget": None,
}

# ─── Alert Configuration ─────────────────────────────────────────────────────
ALERT_CONFIG = {
    "enable_sound": False,          # requires playsound / pygame
    "enable_overlay": True,         # draw boxes & labels on frame
    "overlay_font_scale": 0.55,
    "overlay_thickness": 2,

    # Colours (BGR)
    "colors": {
        "person":    (200, 200, 200),
        "helmet":    (0, 255, 0),
        "vest":      (0, 200, 255),
        "no_helmet": (0, 0, 255),
        "no_vest":   (0, 165, 255),
        "LOW":       (0, 255, 255),
        "MEDIUM":    (0, 165, 255),
        "HIGH":      (0, 0, 255),
    },
}

# ─── Logging / Reporting ──────────────────────────────────────────────────────
LOG_CONFIG = {
    "violation_log_csv":  str(LOGS_DIR / "violations.csv"),
    "violation_log_json": str(LOGS_DIR / "violations.json"),
    "session_report_json": str(REPORTS_DIR / "session_report.json"),
    "hourly_report_csv":  str(REPORTS_DIR / "hourly_summary.csv"),
}

# ─── Dashboard (Flask) ────────────────────────────────────────────────────────
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "stream_fps": 15,       # max FPS pushed to browser via MJPEG
}

# ─── Training Dataset ─────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "yaml_path": str(DATA_DIR / "ppe_dataset.yaml"),
    "train_epochs": 50,
    "batch_size": 16,
    "imgsz": 640,
    "pretrained": "yolov8n.pt",
}
