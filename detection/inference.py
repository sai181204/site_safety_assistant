"""
detection/inference.py
──────────────────────
Per-frame inference pipeline:
  1. Run YOLO on a frame
  2. Separate detections into: persons, PPE present, PPE violations
  3. Associate PPE/violations with the nearest person (IoU overlap)
  4. Return a clean list of DetectionResult objects

The module is model-agnostic — it works with the 5-class PPE model
(person / helmet / vest / no_helmet / no_vest) *and* gracefully degrades
to a COCO-pretrained model (person-only) by treating every person as a
potential violation when no PPE classes are available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MODEL_CONFIG, PPE_CLASSES, VIOLATION_CLASSES,
    EQUIPMENT_CLASSES, PERSON_CLASS, PPE_ASSOCIATION_IOU,
)
from detection.model import PPEModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_ratio(self, other: "BBox") -> float:
        """Fraction of *other* that overlaps with self."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / other.area if other.area > 0 else 0.0


@dataclass
class Detection:
    """Single raw detection from YOLO."""
    bbox:       BBox
    class_id:   int
    class_name: str
    confidence: float
    track_id:   Optional[int] = None


@dataclass
class WorkerState:
    """
    Aggregated PPE state for one worker in a single frame.
    Populated by the association step in InferencePipeline.
    """
    person_detection: Detection
    track_id:         Optional[int]          = None
    has_helmet:       bool                   = False
    has_vest:         bool                   = False
    violations:       List[str]              = field(default_factory=list)
    ppe_detections:   List[Detection]        = field(default_factory=list)
    zone_name:        Optional[str]          = None

    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0

    @property
    def bbox(self) -> BBox:
        return self.person_detection.bbox


# ─────────────────────────────────────────────────────────────────────────────
class InferencePipeline:
    """
    Wraps PPEModel and performs per-frame detection + association.

    Parameters
    ----------
    model : PPEModel instance (or None → creates one automatically)
    conf  : confidence threshold override
    """

    def __init__(
        self,
        model: Optional[PPEModel] = None,
        conf: float = MODEL_CONFIG["confidence_threshold"],
    ):
        self.model = model or PPEModel()
        self.conf  = conf
        self._using_coco_fallback = self._detect_coco_fallback()
        if self._using_coco_fallback:
            logger.warning(
                "COCO pretrained model detected — only 'person' class is reliable.\n"
                "PPE equipment detection will be simulated for demo purposes.\n"
                "Train a custom model for production use."
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        zones: Optional[List[Tuple[int, int, int, int, str]]] = None,
    ) -> Tuple[List[WorkerState], List[Detection], float]:
        """
        Run full inference + association on one BGR frame.

        Parameters
        ----------
        frame : numpy BGR image
        zones : list of (x1, y1, x2, y2, name) hazard zones

        Returns
        -------
        workers    : list of WorkerState (one per detected person)
        raw_dets   : all raw Detection objects (for debug overlay)
        infer_ms   : inference latency in milliseconds
        """
        t0 = time.perf_counter()
        raw_results = self.model.predict(frame, conf=self.conf)
        infer_ms = (time.perf_counter() - t0) * 1000

        if self._using_coco_fallback:
            detections = self._parse_coco_results(raw_results)
        else:
            detections = self._parse_ppe_results(raw_results)

        persons    = [d for d in detections if d.class_id == PERSON_CLASS]
        ppe_items  = [d for d in detections if d.class_id in EQUIPMENT_CLASSES]
        violations = [d for d in detections if d.class_id in VIOLATION_CLASSES]

        workers = self._associate(persons, ppe_items, violations, zones)

        # Debug logging
        logger.debug("Detections: %d persons, %d PPE items, %d violations", 
                     len(persons), len(ppe_items), len(violations))
        logger.debug("Workers after association: %d", len(workers))
        for i, w in enumerate(workers):
            logger.debug("Worker %d: helmet=%s, vest=%s, violations=%s", 
                        i, w.has_helmet, w.has_vest, w.violations)

        return workers, detections, infer_ms

    # ── Parsing helpers ────────────────────────────────────────────────────

    def _parse_ppe_results(self, results) -> List[Detection]:
        """Parse YOLOv8 results from the custom PPE model."""
        dets: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = MODEL_CONFIG["class_names"].get(cls_id, f"cls_{cls_id}")
                logger.debug("PPE detection: class=%d (%s), conf=%.2f, bbox=(%.0f,%.0f,%.0f,%.0f)", 
                           cls_id, cls_name, conf, x1, y1, x2, y2)
                dets.append(Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
               ))
                    
        return dets

    def _parse_coco_results(self, results) -> List[Detection]:
        """
        Parse COCO-pretrained results.
        Only keeps 'person' detections (class 0).
        Adds synthetic no_helmet / no_vest violations for demo.
        """
        dets: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:          # skip non-person in COCO
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    class_id=PERSON_CLASS,
                    class_name="person",
                    confidence=conf,
                ))
                # Synthetic demo: mark every person as missing both PPE
                # (replace with real detections once you have a trained model)
                head_box = BBox(x1 + (x2-x1)*0.25, y1, x1 + (x2-x1)*0.75, y1 + (y2-y1)*0.25)
                vest_box = BBox(x1 + (x2-x1)*0.1,  y1 + (y2-y1)*0.25, x2 - (x2-x1)*0.1, y1 + (y2-y1)*0.65)
                dets.append(Detection(bbox=head_box, class_id=3, class_name="no_helmet",  confidence=0.60))
                dets.append(Detection(bbox=vest_box, class_id=4, class_name="no_vest",    confidence=0.60))
        return dets

    def _detect_coco_fallback(self) -> bool:
        """Heuristic: if model has 80 classes it's likely COCO pretrained."""
        try:
            n = len(self.model.model.names)
            return n >= 70
        except Exception:
            return False

    # ── Association ────────────────────────────────────────────────────────

    def _associate(
        self,
        persons:    List[Detection],
        ppe_items:  List[Detection],
        violations: List[Detection],
        zones:      Optional[List[Tuple[int, int, int, int, str]]],
    ) -> List[WorkerState]:
        """
        Associate PPE / violations with person bounding boxes.

        Strategy
        --------
        For each PPE box, find the person whose bounding box has the
        highest overlap_ratio with the PPE box (head / torso region).
        """
        workers: List[WorkerState] = []

        for person in persons:
            ws = WorkerState(person_detection=person)

            # Assign zone
            if zones:
                ws.zone_name = self._get_zone(person.bbox, zones)

            # Assign PPE equipment detections
            for item in ppe_items:
                if person.bbox.overlap_ratio(item.bbox) >= PPE_ASSOCIATION_IOU \
                        or item.bbox.iou(person.bbox) >= PPE_ASSOCIATION_IOU:
                    ws.ppe_detections.append(item)
                    if item.class_id == PPE_CLASSES["helmet"]:
                        ws.has_helmet = True
                    elif item.class_id == PPE_CLASSES["vest"]:
                        ws.has_vest = True

            # Assign violation detections
            for viol in violations:
                if person.bbox.overlap_ratio(viol.bbox) >= PPE_ASSOCIATION_IOU \
                        or viol.bbox.iou(person.bbox) >= PPE_ASSOCIATION_IOU:
                    ws.ppe_detections.append(viol)
                    if viol.class_id == PPE_CLASSES["no_helmet"] and not ws.has_helmet:
                        ws.violations.append("No Helmet")
                    elif viol.class_id == PPE_CLASSES["no_vest"] and not ws.has_vest:
                        ws.violations.append("No Vest")

            workers.append(ws)

        return workers

    @staticmethod
    def _get_zone(
        bbox: BBox,
        zones: List[Tuple[int, int, int, int, str]],
    ) -> Optional[str]:
        """Return the name of the first zone whose centre overlaps bbox."""
        cx, cy = bbox.center
        for (zx1, zy1, zx2, zy2, zname) in zones:
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return zname
        return None
