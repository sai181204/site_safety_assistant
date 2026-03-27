"""
scripts/generate_report.py
──────────────────────────
Standalone script to generate a visual compliance report from saved logs.

Usage
-----
python scripts/generate_report.py
python scripts/generate_report.py --output reports/my_report.png
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def load_violations() -> list:
    path = ROOT / "logs" / "violations.json"
    if not path.exists():
        logger.warning("No violation log found at %s", path)
        return []
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.error("Failed to load violations: %s", e)
        return []


def load_sessions() -> list:
    path = ROOT / "reports" / "session_report.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def generate_matplotlib_report(output_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from collections import Counter
    except ImportError:
        logger.error("Install matplotlib: pip install matplotlib")
        return

    violations = load_violations()
    sessions   = load_sessions()

    # ── Build data ──────────────────────────────────────────────────────
    by_type   = Counter(v["violation_type"]  for v in violations)
    by_sev    = Counter(v["severity"]         for v in violations)
    by_worker = Counter(v["worker_id"]        for v in violations)
    by_zone   = Counter(v.get("zone", "Unknown") or "Unknown" for v in violations)

    comp_hist = [
        (s["session_start"][:16], s["compliance_pct"])
        for s in sessions[-20:]
    ] if sessions else []

    # ── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    AXIS_BG  = "#161b22"
    TEXT_COL = "#c9d1d9"
    ACCENT   = "#58a6ff"
    COLORS   = ["#ff4444", "#ff8c00", "#ffd700", "#00c853", "#00b4d8"]

    def style_ax(ax, title):
        ax.set_facecolor(AXIS_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.set_title(title, color=ACCENT, fontsize=11, pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # 1. Violation types
    ax1 = fig.add_subplot(gs[0, 0])
    if by_type:
        ax1.bar(by_type.keys(), by_type.values(), color=COLORS[:len(by_type)])
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax1.transAxes, color=TEXT_COL)
    style_ax(ax1, "Violations by Type")
    ax1.set_ylabel("Count", color=TEXT_COL, fontsize=9)

    # 2. Severity pie
    ax2 = fig.add_subplot(gs[0, 1])
    if by_sev:
        sev_colors = {"LOW": "#ffd700", "MEDIUM": "#ff8c00", "HIGH": "#ff4444"}
        c = [sev_colors.get(k, "#888") for k in by_sev.keys()]
        ax2.pie(
            by_sev.values(), labels=by_sev.keys(),
            colors=c, autopct="%1.0f%%",
            textprops={"color": TEXT_COL, "fontsize": 9},
        )
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes, color=TEXT_COL)
    ax2.set_facecolor(AXIS_BG)
    ax2.set_title("Violations by Severity", color=ACCENT, fontsize=11, pad=10)

    # 3. Top workers
    ax3 = fig.add_subplot(gs[0, 2])
    if by_worker:
        top = dict(sorted(by_worker.items(), key=lambda x: -x[1])[:10])
        labels = [f"W{k}" for k in top.keys()]
        ax3.barh(labels, top.values(), color=ACCENT)
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax3.transAxes, color=TEXT_COL)
    style_ax(ax3, "Top Offenders (Workers)")
    ax3.set_xlabel("# Violations", color=TEXT_COL, fontsize=9)

    # 4. Compliance history
    ax4 = fig.add_subplot(gs[1, :2])
    if comp_hist:
        xs, ys = zip(*comp_hist)
        ax4.plot(range(len(ys)), ys, color=ACCENT, marker="o", markersize=5, linewidth=2)
        ax4.fill_between(range(len(ys)), ys, alpha=0.15, color=ACCENT)
        ax4.axhline(80, color="#00c853", linestyle="--", linewidth=1, label="Target (80%)")
        ax4.set_xticks(range(len(xs)))
        ax4.set_xticklabels(xs, rotation=35, ha="right", fontsize=8)
        ax4.set_ylim(0, 105)
        ax4.legend(facecolor=AXIS_BG, labelcolor=TEXT_COL, fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No session data yet", ha="center", va="center",
                 transform=ax4.transAxes, color=TEXT_COL)
    style_ax(ax4, "Compliance % per Session")
    ax4.set_ylabel("Compliance %", color=TEXT_COL, fontsize=9)

    # 5. Zone violations
    ax5 = fig.add_subplot(gs[1, 2])
    if by_zone:
        ax5.bar(by_zone.keys(), by_zone.values(), color="#ff8c00")
        plt.setp(ax5.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax5.transAxes, color=TEXT_COL)
    style_ax(ax5, "Violations by Zone")
    ax5.set_ylabel("Count", color=TEXT_COL, fontsize=9)

    # Title
    fig.suptitle(
        f"PPE Safety Compliance Report  —  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color=TEXT_COL, fontsize=14, y=0.98,
    )

    out = output_path or str(ROOT / "reports" / "compliance_report.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Report saved → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None, help="Output PNG path")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    generate_matplotlib_report(a.output)
