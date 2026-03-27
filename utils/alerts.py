"""
utils/alerts.py
───────────────
Visual overlay rendering and alert management.

Responsibilities:
  • Draw bounding boxes, labels, severity badges on frames
  • Display FPS, worker count, violation count HUD
  • Draw hazard zone overlays
  • Manage on-screen alert queue (recent alerts shown briefly)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ALERT_CONFIG, ENABLE_ZONES, HAZARD_ZONES
from utils.tracker import TrackedWorker

# Colour palette (BGR)
C = ALERT_CONFIG["colors"]
SEVERITY_COLORS = {
    "OK":     (0, 200, 0),
    "LOW":    C["LOW"],
    "MEDIUM": C["MEDIUM"],
    "HIGH":   C["HIGH"],
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
FS        = ALERT_CONFIG["overlay_font_scale"]
TH        = ALERT_CONFIG["overlay_thickness"]


# ─────────────────────────────────────────────────────────────────────────────
class AlertManager:
    """
    Manages a rolling queue of alert messages and renders the full
    annotated frame.
    """

    def __init__(self, max_alerts: int = 6, alert_ttl: float = 4.0):
        self._alert_queue: deque = deque(maxlen=max_alerts)
        self._alert_ttl   = alert_ttl          # seconds to show each alert
        
        # Add a test alert on startup
        self.push_alert("System Ready - Monitoring Active", "LOW")

    # ── Public API ──────────────────────────────────────────────────────────

    def push_alert(self, message: str, severity: str = "MEDIUM"):
        """Add an alert to the rolling queue."""
        self._alert_queue.appendleft({
            "msg":      message,
            "severity": severity,
            "ts":       time.time(),
        })

    def draw_frame(
        self,
        frame:    np.ndarray,
        workers:  List[TrackedWorker],
        fps:      float,
        frame_no: int,
    ) -> np.ndarray:
        """
        Annotate *frame* in-place and return it.

        Draws:
          • Hazard zone rectangles
          • Per-worker bounding boxes + labels
          • HUD (FPS, counts, compliance %)
          • Rolling alert banner
        """
        if not ALERT_CONFIG["enable_overlay"]:
            return frame

        out = frame.copy()

        # 1. Zone overlays
        if ENABLE_ZONES:
            self._draw_zones(out)

        # 2. Per-worker annotations
        for tw in workers:
            self._draw_worker(out, tw)

        # 3. HUD
        self._draw_hud(out, workers, fps, frame_no)

        # 4. Alert banner
        self._draw_alerts(out)

        return out

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _draw_worker(self, frame: np.ndarray, tw: TrackedWorker):
        x1, y1, x2, y2 = (
            int(tw.bbox.x1), int(tw.bbox.y1),
            int(tw.bbox.x2), int(tw.bbox.y2),
        )
        sev   = tw.highest_severity
        color = SEVERITY_COLORS.get(sev, C["person"])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, TH)

        # Track ID badge
        id_label = f"W{tw.track_id}"
        self._put_badge(frame, id_label, (x1, y1 - 2), color, bg_alpha=0.6)

        # PPE Status indicators - prominent helmet/vest status
        y_off = y1 + 22
        
        # Check helmet status specifically
        has_helmet = "no_helmet" not in tw.current_violations
        has_vest = "no_vest" not in tw.current_violations
        
        # Helmet status with large indicator
        if has_helmet:
            self._put_badge(frame, "🪖 HELMET ON", (x1 + 4, y_off), (0, 255, 0), text_color=(0, 0, 0), scale=1.2)
            y_off += 25
        else:
            viol_sev = "HIGH" if tw.active_violations.get("no_helmet", None) and \
                         tw.active_violations["no_helmet"].severity == "HIGH" else "MEDIUM"
            viol_color = SEVERITY_COLORS.get(viol_sev, (0, 0, 255))
            dur = tw.active_violations["no_helmet"].duration if "no_helmet" in tw.active_violations else 0
            helmet_label = f"⚠ NO HELMET [{dur:.0f}s]"
            self._put_badge(frame, helmet_label, (x1 + 4, y_off), viol_color, text_color=(255, 255, 255), scale=1.2)
            y_off += 25
        
        # Vest status with large indicator
        if has_vest:
            self._put_badge(frame, "🦺 VEST ON", (x1 + 4, y_off), (0, 255, 0), text_color=(0, 0, 0), scale=1.2)
            y_off += 25
        else:
            viol_sev = "HIGH" if tw.active_violations.get("no_vest", None) and \
                         tw.active_violations["no_vest"].severity == "HIGH" else "MEDIUM"
            viol_color = SEVERITY_COLORS.get(viol_sev, (0, 0, 255))
            dur = tw.active_violations["no_vest"].duration if "no_vest" in tw.active_violations else 0
            vest_label = f"⚠ NO VEST [{dur:.0f}s]"
            self._put_badge(frame, vest_label, (x1 + 4, y_off), viol_color, text_color=(255, 255, 255), scale=1.2)
            y_off += 25

        # Overall compliance status
        if tw.is_compliant:
            self._put_badge(frame, "✓ ALL PPE OK", (x1 + 4, y_off), (0, 200, 0), text_color=(0, 0, 0), scale=1.1)

        # Zone label
        if tw.zone_name:
            cv2.putText(
                frame, tw.zone_name,
                (x1, y2 + 14),
                FONT, FS * 0.8, (255, 255, 0), 1, cv2.LINE_AA,
            )

    def _draw_hud(
        self,
        frame:    np.ndarray,
        workers:  List[TrackedWorker],
        fps:      float,
        frame_no: int,
    ):
        h, w = frame.shape[:2]

        # Large semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        total      = len(workers)
        violations = sum(1 for tw in workers if not tw.is_compliant)
        compliant  = total - violations
        pct        = (compliant / total * 100) if total else 100.0

        # Large, prominent worker count
        worker_color = (0, 255, 0) if violations == 0 else (0, 165, 255) if violations <= total/2 else (0, 0, 255)
        cv2.putText(frame, f"WORKERS: {total}", (8, 35), FONT_BOLD, FS * 1.4, worker_color, 2, cv2.LINE_AA)
        
        # Violation count
        viol_color = (0, 0, 255) if violations > 0 else (0, 255, 0)
        cv2.putText(frame, f"VIOLATIONS: {violations}", (180, 35), FONT_BOLD, FS * 1.4, viol_color, 2, cv2.LINE_AA)
        
        # Compliance percentage
        comp_color = (0, 255, 0) if pct >= 80 else (0, 165, 255) if pct >= 50 else (0, 0, 255)
        cv2.putText(frame, f"COMPLIANCE: {pct:.0f}%", (380, 35), FONT_BOLD, FS * 1.4, comp_color, 2, cv2.LINE_AA)

        # Title
        title = "PPE SAFETY MONITOR"
        tw_size = cv2.getTextSize(title, FONT_BOLD, FS * 1.0, 2)[0]
        cv2.putText(
            frame, title,
            (w - tw_size[0] - 10, 35),
            FONT_BOLD, FS * 1.0, (0, 200, 255), 2, cv2.LINE_AA,
        )
        
        # PPE Instructions at bottom
        self._draw_ppe_instructions(frame, workers)

    def _draw_ppe_instructions(self, frame: np.ndarray, workers: List[TrackedWorker]):
        """Draw prominent PPE instructions at the bottom of the frame."""
        h, w = frame.shape[:2]
        
        # Bottom instruction bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # PPE Requirements
        instructions = [
            "🪖 HELMET REQUIRED - ⚠ NO HELMET DETECTED" if any("no_helmet" in tw.current_violations for tw in workers) else "✅ HELMET: ALL COMPLIANT",
            "🦺 VEST REQUIRED - ⚠ NO VEST DETECTED" if any("no_vest" in tw.current_violations for tw in workers) else "✅ VEST: ALL COMPLIANT"
        ]
        
        y_pos = h - 50
        for i, instruction in enumerate(instructions):
            color = (0, 0, 255) if "⚠" in instruction else (0, 255, 0)
            cv2.putText(frame, instruction, (10, y_pos + i * 25), FONT_BOLD, FS * 1.1, color, 2, cv2.LINE_AA)
        
        # Safety message
        if len(workers) == 0:
            cv2.putText(frame, "👁️ SCANNING FOR WORKERS...", (10, h - 25), FONT, FS * 1.0, (255, 255, 0), 2, cv2.LINE_AA)

    def _draw_alerts(self, frame: np.ndarray):
        """Draw rolling alert log on right side of frame."""
        h, w = frame.shape[:2]
        now   = time.time()
        alive = [a for a in self._alert_queue if now - a["ts"] < self._alert_ttl]

        # Always show alert panel even if no alerts
        panel_w = 350
        x_start = w - panel_w - 4
        
        # Draw alert panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 4, 10), (w - 4, 200), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Panel title
        cv2.putText(frame, "⚠ ALERTS", (x_start, 30), FONT_BOLD, FS * 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        
        if not alive:
            cv2.putText(frame, "No active alerts", (x_start, 60), FONT, FS * 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            return

        for i, alert in enumerate(alive[:5]):
            y = 60 + i * 25
            color = SEVERITY_COLORS.get(alert["severity"], (200, 200, 200))
            # faded alpha based on age
            age   = now - alert["ts"]
            alpha = max(0.5, 1.0 - age / self._alert_ttl)
            
            # Alert background
            alert_overlay = frame.copy()
            cv2.rectangle(alert_overlay, (x_start, y - 18), (w - 4, y + 6), color, -1)
            cv2.addWeighted(alert_overlay, alpha * 0.3, frame, 1 - alpha * 0.3, 0, frame)
            
            # Alert text
            cv2.putText(
                frame, alert["msg"],
                (x_start + 4, y),
                FONT_BOLD, FS * 0.9, color, 2, cv2.LINE_AA,
            )
            
            # Severity badge
            cv2.putText(
                frame, f"[{alert['severity']}]",
                (x_start + 250, y),
                FONT, FS * 0.7, color, 1, cv2.LINE_AA,
            )

    def _draw_zones(self, frame: np.ndarray):
        for (zx1, zy1, zx2, zy2, zname) in HAZARD_ZONES:
            overlay = frame.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 80, 200), 1)
            cv2.putText(
                frame, f"⚠ {zname}",
                (zx1 + 4, zy1 + 16),
                FONT, FS * 0.75, (100, 180, 255), 1, cv2.LINE_AA,
            )

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _put_badge(
        frame:      np.ndarray,
        text:       str,
        origin:     Tuple[int, int],
        bg_color:   Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_alpha:   float = 0.75,
        pad:        int = 3,
        scale:      float = 1.0,
    ):
        font_scale = FS * scale
        (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
        x, y = origin
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            bg_color, -1,
        )
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        cv2.putText(frame, text, (x, y), FONT, font_scale, text_color, 1, cv2.LINE_AA)
