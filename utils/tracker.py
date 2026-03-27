"""
utils/tracker.py
────────────────
Multi-object tracker that:
  • Assigns stable track IDs to persons across frames
  • Tracks per-worker PPE violation durations
  • Computes severity levels based on elapsed violation time
  • Maintains a rolling history for reporting

Tracking algorithm:
  Uses a lightweight IoU-based centroid tracker as the default
  (no external deep-sort dependency required).
  If deep-sort-realtime is installed, it can be swapped in via
  use_deep_sort=True for more robust re-identification.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SEVERITY_CONFIG, TRACKER_CONFIG
from detection.inference import WorkerState, BBox

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ViolationRecord:
    """Time-stamped record of a single violation event."""
    track_id:       int
    violation_type: str          # "No Helmet" | "No Vest"
    start_time:     float        # epoch seconds
    end_time:       Optional[float] = None
    severity:       str          = "LOW"
    zone_name:      Optional[str] = None

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def update_severity(self):
        d = self.duration
        for level, (lo, hi) in SEVERITY_CONFIG.items():
            if lo <= d < hi:
                self.severity = level
                return
        self.severity = "HIGH"


@dataclass
class TrackedWorker:
    """State for one tracked worker."""
    track_id:        int
    bbox:            BBox
    first_seen:      float = field(default_factory=time.time)
    last_seen:       float = field(default_factory=time.time)
    age:             int   = 0                  # frames since last detection

    # Current frame violations
    current_violations: List[str] = field(default_factory=list)
    has_helmet:          bool = False
    has_vest:            bool = False
    zone_name:           Optional[str] = None

    # Active violation records (violation_type → ViolationRecord)
    active_violations: Dict[str, ViolationRecord] = field(default_factory=dict)
    # Completed violation records
    completed_violations: List[ViolationRecord] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return len(self.current_violations) == 0

    @property
    def highest_severity(self) -> str:
        levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        sev = "LOW"
        for vr in self.active_violations.values():
            vr.update_severity()
            if levels.get(vr.severity, 0) > levels.get(sev, 0):
                sev = vr.severity
        return sev if self.active_violations else "OK"

    def all_violation_records(self) -> List[ViolationRecord]:
        return self.completed_violations + list(self.active_violations.values())


# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    """
    Simple IoU-based centroid tracker.
    Assigns persistent IDs to bounding boxes across frames.
    """

    def __init__(
        self,
        max_age: int = TRACKER_CONFIG["max_age"],
        min_hits: int = TRACKER_CONFIG["n_init"],
        iou_threshold: float = 0.30,
    ):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def update(
        self, bboxes: List[BBox]
    ) -> List[Tuple[BBox, int]]:
        """
        Update tracker with new bounding boxes.

        Returns
        -------
        list of (BBox, track_id) for confirmed tracks.
        """
        if not bboxes:
            # Age all tracks
            dead = [tid for tid, t in self._tracks.items() if t["age"] >= self.max_age]
            for tid in dead:
                del self._tracks[tid]
            for t in self._tracks.values():
                t["age"] += 1
            return []

        # ── Match new boxes to existing tracks using IoU ──────────────────
        unmatched_dets = list(range(len(bboxes)))
        matched_pairs: List[Tuple[int, int]] = []  # (track_id, det_idx)

        if self._tracks:
            iou_mat = np.zeros((len(self._tracks), len(bboxes)))
            track_ids = list(self._tracks.keys())
            for i, tid in enumerate(track_ids):
                for j, bbox in enumerate(bboxes):
                    iou_mat[i, j] = self._tracks[tid]["bbox"].iou(bbox)

            # Greedy matching
            while iou_mat.max() >= self.iou_threshold:
                i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                tid = track_ids[i]
                matched_pairs.append((tid, j))
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1
                if j in unmatched_dets:
                    unmatched_dets.remove(j)

        # ── Update matched tracks ─────────────────────────────────────────
        for tid, j in matched_pairs:
            t = self._tracks[tid]
            t["bbox"]  = bboxes[j]
            t["age"]   = 0
            t["hits"] += 1

        # ── Age unmatched tracks, remove stale ───────────────────────────
        matched_tids = {p[0] for p in matched_pairs}
        for tid in list(self._tracks.keys()):
            if tid not in matched_tids:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        # ── Create new tracks for unmatched detections ───────────────────
        for j in unmatched_dets:
            self._tracks[self._next_id] = {
                "bbox":  bboxes[j],
                "age":   0,
                "hits":  1,
            }
            self._next_id += 1

        # ── Return confirmed tracks ───────────────────────────────────────
        out: List[Tuple[BBox, int]] = []
        for tid, t in self._tracks.items():
            if t["hits"] >= self.min_hits:
                out.append((t["bbox"], tid))
        return out


# ─────────────────────────────────────────────────────────────────────────────
class WorkerTracker:
    """
    High-level tracker that:
      • Uses CentroidTracker under the hood
      • Maintains TrackedWorker state per ID
      • Tracks violation timers
      • Returns enriched WorkerState objects
    """

    def __init__(self):
        self._ct       = CentroidTracker()
        self._workers: Dict[int, TrackedWorker] = {}
        self._frame_count = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self, worker_states: List[WorkerState]
    ) -> List[TrackedWorker]:
        """
        Update tracker with current frame's worker detections.

        Parameters
        ----------
        worker_states : output of InferencePipeline.process_frame()

        Returns
        -------
        list of TrackedWorker with stable IDs and violation timers
        """
        self._frame_count += 1
        now = time.time()

        # Extract bboxes for centroid tracker
        bboxes = [ws.person_detection.bbox for ws in worker_states]
        tracked = self._ct.update(bboxes)  # [(BBox, track_id), …]

        # Build bbox→track_id lookup
        bbox_to_tid: Dict[int, int] = {}  # index in worker_states → tid
        for (tbbox, tid) in tracked:
            best_iou, best_idx = 0.0, -1
            for idx, ws in enumerate(worker_states):
                iou = tbbox.iou(ws.person_detection.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou > 0.3:
                bbox_to_tid[best_idx] = tid

        result: List[TrackedWorker] = []

        for idx, ws in enumerate(worker_states):
            tid = bbox_to_tid.get(idx)
            if tid is None:
                continue  # not yet confirmed

            # Get or create TrackedWorker
            if tid not in self._workers:
                self._workers[tid] = TrackedWorker(track_id=tid, bbox=ws.bbox)
                logger.debug("New worker track: %d", tid)

            tw = self._workers[tid]
            tw.bbox               = ws.bbox
            tw.last_seen          = now
            tw.age                = 0
            tw.has_helmet         = ws.has_helmet
            tw.has_vest           = ws.has_vest
            tw.current_violations = ws.violations
            tw.zone_name          = ws.zone_name

            # Update violation timers
            self._update_violation_timers(tw, now)
            result.append(tw)

        # Age workers not seen this frame
        for tid, tw in self._workers.items():
            if tw.last_seen < now:
                tw.age += 1

        return result

    def get_all_workers(self) -> Dict[int, TrackedWorker]:
        return self._workers

    def reset(self):
        self._ct       = CentroidTracker()
        self._workers  = {}
        self._frame_count = 0

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _update_violation_timers(tw: TrackedWorker, now: float):
        """Open new violation records; close resolved ones; update severity."""
        current_set = set(tw.current_violations)

        # Open new violations
        for vtype in current_set:
            if vtype not in tw.active_violations:
                tw.active_violations[vtype] = ViolationRecord(
                    track_id=tw.track_id,
                    violation_type=vtype,
                    start_time=now,
                    zone_name=tw.zone_name,
                )

        # Close resolved violations
        for vtype in list(tw.active_violations.keys()):
            if vtype not in current_set:
                vr = tw.active_violations.pop(vtype)
                vr.end_time = now
                vr.update_severity()
                tw.completed_violations.append(vr)

        # Update severity on active violations
        for vr in tw.active_violations.values():
            vr.update_severity()
