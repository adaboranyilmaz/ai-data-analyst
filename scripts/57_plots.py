"""The calibration and selective-prediction figures, drawn from the committed results files.

- results/plots/reliability_held_out.png: the reliability diagram on the held-out set: per bin
  of ten equal-width confidence bins, the answers' mean confidence against their accuracy, for the
  stated confidence and its Platt and isotonic calibrations (from calibration.json).
- results/plots/risk_coverage_held_out.png: the risk-coverage curves on the held-out set: the
  error rate among the most confident answers at each coverage, for the stated confidence
  (calibrated; Platt keeps its order), the critic's, and the exploratory combination, with the
  best possible ranking (from risk_coverage.json).

Every value drawn is read from those files; nothing is computed here but the layout.

Usage:
    uv run python scripts/57_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS = ROOT / "results/plots"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8983"
# the first three slots of a colour-blind-checked categorical palette, in its fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", "-.")


def _axes(title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    return fig, ax


def _label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        color=TEXT,
        fontsize=8.5,
        va="center",
    )


def reliability(cal: dict, out: Path) -> None:
    held = cal["held_out"]
    n = held["questions"]
    fig, ax = _axes(f"Reliability on the held-out set ({n} questions, one run)")
    ax.plot(
        [0, 1], [0, 1], color=NEUTRAL, linestyle=":", linewidth=1.2, label="perfect calibration"
    )
    names = (("stated_raw", "stated"), ("stated_platt", "Platt"), ("stated_isotonic", "isotonic"))
    for k, (key, name) in enumerate(names):
        bins = held[key]["reliability"]
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        ece = held[key]["ece"]["estimate"]
        ax.plot(
            xs,
            ys,
            color=SERIES[k],
            linestyle=LINESTYLES[k],
            marker=MARKERS[k],
            markersize=5,
            linewidth=2,
            label=f"{name} (ECE {ece:.2f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence (bin mean)", color=TEXT_2, fontsize=9.5)
    ax.set_ylabel("share of answers correct", color=TEXT_2, fontsize=9.5)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def risk_coverage(risk: dict, out: Path) -> None:
    held = risk["held_out"]
    fig, ax = _axes(f"Risk-coverage on the held-out set ({risk['questions']} questions, one run)")
    names = (
        ("stated_platt", "stated (calibrated)"),
        ("critic_raw", "critic"),
        ("combined_exploratory", "combined (exploratory)"),
    )
    label_at = (0.22, 0.42, 0.62)  # staggered, so the direct labels do not collide
    for k, (key, name) in enumerate(names):
        curve = held[key]["curve"]
        xs, ys = curve["coverage"], curve["risk"]
        aurc = held[key]["aurc"]["estimate"]
        ax.plot(
            xs,
            ys,
            color=SERIES[k],
            linestyle=LINESTYLES[k],
            linewidth=2,
            label=f"{name} (AURC {aurc:.3f})",
        )
        i = min(range(len(xs)), key=lambda j: abs(xs[j] - label_at[k]))
        _label(ax, xs[i], ys[i], name.split(" (")[0], dy=9)
    # the best possible ranking: every correct answer first
    n = risk["questions"]
    last = held["stated_platt"]["curve"]["risk"][-1]
    right = round(n * (1 - last))
    xs = [(k + 1) / n for k in range(n)]
    ys = [max(k + 1 - right, 0) / (k + 1) for k in range(n)]
    ax.plot(xs, ys, color=NEUTRAL, linestyle=":", linewidth=1.2, label="best possible ranking")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.6, last + 0.1))
    ax.set_xlabel(
        "coverage (share of questions answered, most confident first)", color=TEXT_2, fontsize=9.5
    )
    ax.set_ylabel("error rate among the answered", color=TEXT_2, fontsize=9.5)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cal = json.loads((ROOT / "results/metrics/calibration.json").read_text(encoding="utf-8"))
    risk = json.loads((ROOT / "results/metrics/risk_coverage.json").read_text(encoding="utf-8"))
    PLOTS.mkdir(parents=True, exist_ok=True)
    reliability(cal, PLOTS / "reliability_held_out.png")
    risk_coverage(risk, PLOTS / "risk_coverage_held_out.png")
    print(f"wrote {PLOTS.relative_to(ROOT)}/reliability_held_out.png, risk_coverage_held_out.png")


if __name__ == "__main__":
    main()
