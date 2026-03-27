# 🏗️ Smart Construction Site PPE Safety System

An AI-powered real-time PPE (Personal Protective Equipment) compliance monitor using YOLOv8 and OpenCV. Detects helmets, safety vests, and violations across multiple workers simultaneously with severity tracking and automated reporting.

---

## 📁 Project Structure

```
construction_safety/
├── main.py                    # Entry point — inference, training, evaluation
├── dashboard.py               # Flask web dashboard (MJPEG stream + REST API)
├── config.py                  # Central configuration (thresholds, zones, paths)
├── requirements.txt
│
├── detection/
│   ├── model.py               # PPEModel: loads YOLOv8, train, evaluate, export
│   └── inference.py           # Per-frame inference + PPE–person association
│
├── utils/
│   ├── tracker.py             # Multi-object tracker + violation timer
│   ├── alerts.py              # OpenCV overlay renderer (boxes, HUD, alerts)
│   └── report.py              # CSV/JSON logging + compliance report generation
│
├── scripts/
│   ├── prepare_dataset.py     # Download dataset from Roboflow (optional)
│   └── generate_report.py     # Standalone matplotlib compliance report
│
├── data/
│   ├── ppe_dataset.yaml       # YOLO dataset config
│   ├── images/
│   │   ├── train/             # Training images
│   │   └── val/               # Validation images
│   └── labels/
│       ├── train/             # YOLO .txt label files
│       └── val/
│
├── models/                    # Trained weights go here
│   └── ppe_yolov8.pt          # Custom PPE model (after training)
│
├── logs/
│   ├── violations.csv         # Append-only violation log
│   └── violations.json        # Structured violation records
│
├── reports/
│   ├── session_report.json    # Per-session compliance summary
│   ├── hourly_summary.csv     # Hourly aggregates
│   ├── output.mp4             # Saved annotated video (--save-output)
│   └── compliance_report.png  # Matplotlib visual report
│
└── templates/
    └── dashboard.html         # Web dashboard UI
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA-capable GPU for faster inference

### 2. Clone / download the project

```bash
git clone <your-repo-url>  # or download and unzip
cd construction_safety
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Install the CUDA version of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 5. (Optional) Configure the system

Edit `config.py` to adjust:
- `confidence_threshold` — detection sensitivity (default 0.50)
- `HAZARD_ZONES` — pixel rectangles for zone-based rules
- `SEVERITY_CONFIG` — violation duration thresholds
- `MODEL_CONFIG["device"]` — `"cpu"` | `"cuda"` | `"mps"`

---

## 🚀 Quick Start

### Run on webcam (no custom model needed)

```bash
python main.py
```

The system will fall back to a COCO-pretrained YOLOv8 model and simulate PPE violations for demo purposes.

### Run on a video file

```bash
python main.py --source path/to/construction_video.mp4
```

### Save annotated output

```bash
python main.py --source video.mp4 --save-output
```

### Headless mode (no window)

```bash
python main.py --source video.mp4 --headless
```

### Web dashboard

```bash
python main.py --dashboard
# Then open http://localhost:5000
```

---

## 🧠 Training a Custom PPE Model

### Step 1: Get a PPE dataset

**Option A — Roboflow (easiest)**

```bash
pip install roboflow
python scripts/prepare_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
```

Free datasets on [Roboflow Universe](https://universe.roboflow.com/):
- Search: "PPE detection construction"
- Recommended: [Construction PPE Safety](https://universe.roboflow.com/ppe-detection-jlhau/construction-ppe-safety)

**Option B — Manual**

1. Download a PPE dataset in YOLOv8 format
2. Place images under `data/images/train/` and `data/images/val/`
3. Place label `.txt` files under `data/labels/train/` and `data/labels/val/`

**Option C — Synthetic dummy (smoke test only)**

```bash
python scripts/prepare_dataset.py --dummy
```

### Step 2: Train

```bash
python main.py --train --data data/ppe_dataset.yaml --epochs 50
```

Best weights are automatically saved to `models/ppe_yolov8.pt`.

### Step 3: Evaluate

```bash
python main.py --evaluate --data data/ppe_dataset.yaml
```

Expected metrics on a good PPE dataset:
| Metric | Target |
|--------|--------|
| Precision | ≥ 0.85 |
| Recall | ≥ 0.80 |
| mAP@50 | ≥ 0.85 |
| mAP@50-95 | ≥ 0.60 |

---

## 📊 PPE Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `person` | Worker detected |
| 1 | `helmet` | Safety helmet worn |
| 2 | `vest` | Safety vest worn |
| 3 | `no_helmet` | Worker missing helmet |
| 4 | `no_vest` | Worker missing vest |

---

## 🚨 Severity Levels

| Level | Condition |
|-------|-----------|
| **LOW** | Violation < 5 seconds |
| **MEDIUM** | Violation 5–15 seconds |
| **HIGH** | Violation > 15 seconds OR hazardous zone |

---

## 📝 Output Files

| File | Description |
|------|-------------|
| `logs/violations.csv` | Append-only log of every completed violation |
| `logs/violations.json` | Same data in structured JSON |
| `reports/session_report.json` | End-of-session compliance summary |
| `reports/hourly_summary.csv` | Per-hour aggregates |
| `reports/output.mp4` | Annotated video (`--save-output`) |

### Generate a visual report

```bash
python scripts/generate_report.py
# Opens: reports/compliance_report.png
```

---

## 🔧 Command Reference

```bash
# Webcam
python main.py

# Video file
python main.py --source video.mp4

# Custom weights
python main.py --weights models/ppe_yolov8.pt

# Adjust confidence
python main.py --conf 0.45

# GPU
python main.py --device cuda

# Save output + headless
python main.py --source video.mp4 --save-output --headless

# Web dashboard
python main.py --dashboard

# Train
python main.py --train --epochs 100

# Evaluate
python main.py --evaluate

# Stop after 500 frames
python main.py --source video.mp4 --max-frames 500
```

---

## 🌐 Dashboard API

When running with `--dashboard`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/video_feed` | GET | MJPEG live stream |
| `/api/stats` | GET | FPS, worker count, violations, compliance % |
| `/api/violations` | GET | Active violations list |
| `/api/report` | GET | Session reports JSON |
| `/api/violation_log` | GET | Full violation log JSON |

---

## 📦 Dependencies

See `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | ≥8.0 | YOLOv8 model |
| opencv-python | ≥4.8 | Video I/O + drawing |
| torch | ≥2.0 | Deep learning backend |
| flask | ≥3.0 | Web dashboard |
| pandas | ≥2.0 | Data processing |
| matplotlib | ≥3.7 | Report charts |

---

## 💡 Tips

- **Performance**: Use `--device cuda` for 10–30× faster inference on GPU.
- **Accuracy**: For best results, fine-tune on your specific site/camera angle.
- **Zones**: Configure `HAZARD_ZONES` in `config.py` to match your camera layout (pixel coordinates).
- **Thresholds**: Lower `confidence_threshold` to catch more violations (more false positives); raise it to reduce false alarms.
- **Multiple cameras**: Run multiple instances of `main.py` with different `--source` values.

---

## 📜 License

MIT — use freely, attribution appreciated.
