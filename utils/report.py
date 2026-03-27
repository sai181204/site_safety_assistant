"""
utils/report.py
───────────────
Violation logging and compliance report generation.

Features
--------
• CSV violation log (append-only, safe for concurrent use)
• JSON violation log (full structured records)
• Hourly summary CSV
• End-of-session JSON report with metrics
• Per-worker statistics
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LOG_CONFIG, REPORTS_DIR
from utils.tracker import TrackedWorker, ViolationRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "worker_id", "violation_type",
    "duration_seconds", "severity", "zone", "frame",
]


class ReportManager:
    """
    Central reporting hub. Call from the main processing loop:

        rm = ReportManager()
        # every frame:
        rm.ingest_workers(workers, frame_no)
        # at end:
        rm.finalize()
    """

    def __init__(self):
        self._session_start = time.time()
        self._logged_violations: set  = set()   # (track_id, vtype, start_time)
        self._hourly: Dict[str, dict] = defaultdict(self._empty_hourly_bucket)

        # Ensure log/report dirs
        Path(LOG_CONFIG["violation_log_csv"]).parent.mkdir(parents=True, exist_ok=True)
        Path(LOG_CONFIG["session_report_json"]).parent.mkdir(parents=True, exist_ok=True)

        # Initialise CSV (write header if new file)
        csv_path = LOG_CONFIG["violation_log_csv"]
        if not Path(csv_path).exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

        self._json_records: List[dict] = []
        self._total_frames  = 0
        self._total_workers_seen: set = set()

    # ── Public API ──────────────────────────────────────────────────────────

    def ingest_workers(
        self,
        workers:  List[TrackedWorker],
        frame_no: int,
    ):
        """
        Called each frame with the current list of tracked workers.
        Logs completed violations and updates hourly buckets.
        """
        self._total_frames += 1
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")

        bucket = self._hourly[hour_key]
        bucket["total_detections"] += len(workers)

        for tw in workers:
            self._total_workers_seen.add(tw.track_id)
            bucket["unique_workers"].add(tw.track_id)

            if tw.is_compliant:
                bucket["compliant_detections"] += 1
            else:
                bucket["violation_detections"] += 1

            # Log newly completed violations
            for vr in tw.completed_violations:
                key = (vr.track_id, vr.violation_type, vr.start_time)
                if key not in self._logged_violations:
                    self._log_violation(vr, frame_no)
                    self._logged_violations.add(key)

    def finalize(self) -> dict:
        """
        Called at end of session.
        Writes:
          • JSON violation log
          • Hourly summary CSV
          • Session report JSON

        Returns the session report dict.
        """
        self._write_json_log()
        self._write_hourly_csv()
        report = self._build_session_report()
        self._write_session_report(report)
        logger.info(
            "Session report saved → %s",
            LOG_CONFIG["session_report_json"],
        )
        return report

    def print_summary(self, report: Optional[dict] = None):
        """Pretty-print a summary to stdout."""
        if report is None:
            report = self._build_session_report()

        print("\n" + "═" * 60)
        print("  CONSTRUCTION SITE PPE COMPLIANCE REPORT")
        print("═" * 60)
        print(f"  Session duration   : {report['session_duration_min']:.1f} min")
        print(f"  Total frames       : {report['total_frames']}")
        print(f"  Unique workers     : {report['unique_workers']}")
        print(f"  Total violations   : {report['total_violations']}")
        print(f"  Compliance rate    : {report['compliance_pct']:.1f}%")
        print()
        print("  Violations by type:")
        for vtype, cnt in report.get("violations_by_type", {}).items():
            print(f"    {vtype:<20} {cnt}")
        print()
        print("  Violations by severity:")
        for sev, cnt in report.get("violations_by_severity", {}).items():
            print(f"    {sev:<10} {cnt}")
        print("═" * 60 + "\n")

    # ── Internal ────────────────────────────────────────────────────────────

    def _log_violation(self, vr: ViolationRecord, frame_no: int):
        """Append one row to the CSV log and add to JSON buffer."""
        row = {
            "timestamp":        datetime.fromtimestamp(vr.start_time).isoformat(),
            "worker_id":        vr.track_id,
            "violation_type":   vr.violation_type,
            "duration_seconds": round(vr.duration, 2),
            "severity":         vr.severity,
            "zone":             vr.zone_name or "",
            "frame":            frame_no,
        }
        # CSV
        with open(LOG_CONFIG["violation_log_csv"], "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)
        # JSON buffer
        self._json_records.append(row)

    def _write_json_log(self):
        path = LOG_CONFIG["violation_log_json"]
        try:
            existing = []
            if Path(path).exists():
                with open(path) as f:
                    existing = json.load(f)
        except Exception:
            existing = []
        combined = existing + self._json_records
        with open(path, "w") as f:
            json.dump(combined, f, indent=2)

    def _write_hourly_csv(self):
        path = LOG_CONFIG["hourly_report_csv"]
        rows = []
        for hour_key, bucket in sorted(self._hourly.items()):
            total  = bucket["total_detections"]
            comp   = bucket["compliant_detections"]
            pct    = (comp / total * 100) if total else 100.0
            rows.append({
                "hour":                hour_key,
                "unique_workers":      len(bucket["unique_workers"]),
                "total_detections":    total,
                "compliant":           comp,
                "violations":          bucket["violation_detections"],
                "compliance_pct":      round(pct, 1),
            })
        header = ["hour", "unique_workers", "total_detections",
                  "compliant", "violations", "compliance_pct"]
        write_header = not Path(path).exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerows(rows)

    def _build_session_report(self) -> dict:
        now   = time.time()
        dur   = now - self._session_start

        by_type: Dict[str, int] = defaultdict(int)
        by_sev:  Dict[str, int] = defaultdict(int)
        for rec in self._json_records:
            by_type[rec["violation_type"]] += 1
            by_sev[rec["severity"]]        += 1

        total_det   = sum(b["total_detections"]    for b in self._hourly.values())
        comp_det    = sum(b["compliant_detections"] for b in self._hourly.values())
        comp_pct    = (comp_det / total_det * 100) if total_det else 100.0

        return {
            "session_start":         datetime.fromtimestamp(self._session_start).isoformat(),
            "session_end":           datetime.fromtimestamp(now).isoformat(),
            "session_duration_min":  round(dur / 60, 2),
            "total_frames":          self._total_frames,
            "unique_workers":        len(self._total_workers_seen),
            "total_violations":      len(self._json_records),
            "compliance_pct":        round(comp_pct, 1),
            "violations_by_type":    dict(by_type),
            "violations_by_severity":dict(by_sev),
            "hourly_summary":        [
                {
                    "hour":           k,
                    "unique_workers": len(v["unique_workers"]),
                    "total":          v["total_detections"],
                    "violations":     v["violation_detections"],
                }
                for k, v in sorted(self._hourly.items())
            ],
        }

    def _write_session_report(self, report: dict):
        path = LOG_CONFIG["session_report_json"]
        # Append to array of past sessions
        try:
            sessions = json.loads(Path(path).read_text()) if Path(path).exists() else []
        except Exception:
            sessions = []
        sessions.append(report)
        Path(path).write_text(json.dumps(sessions, indent=2))

    @staticmethod
    def _empty_hourly_bucket() -> dict:
        return {
            "total_detections":    0,
            "compliant_detections": 0,
            "violation_detections": 0,
            "unique_workers":       set(),
        }
