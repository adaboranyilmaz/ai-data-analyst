"""The planted-effect figures, drawn from the committed results file.

- results/plots/planted_detection.png: how often the test finds an effect in the planted copies,
  per question and condition (where an effect was planted, in its direction, with a hollow mark
  for detection in the opposite one), with the reference plan, the analyst's plan and (for the
  confounded questions) the crude plan, each with its Wilson interval, and the theoretical power
  where there is one (from planted_effects.json, level 1).
- results/plots/planted_claims.png: how often the answer claims an effect where there is none, and
  claims the planted direction where there is one, written from the numbers only and from the
  guarded input (from planted_effects.json, level 2).

Every value drawn is read from that file; nothing is computed here but the layout.

Usage:
    uv run python scripts/78_guardrail_plots.py
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
from matplotlib.lines import Line2D  # noqa: E402

PLOTS = ROOT / "results/plots"
PLANTED = ROOT / "results/metrics/planted_effects.json"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8983"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PLANS = {  # name, color, marker, vertical offset within a row
    "reference": ("reference plan", NEUTRAL, "o", -0.22),
    "analyst": ("analyst's plan", BLUE, "D", 0.0),
    "crude": ("crude plan (no strata)", ORANGE, "s", 0.22),
}
ARMS = {
    "numbers_only": ("numbers only", NEUTRAL, "o", -0.14),
    "guarded": ("guarded", BLUE, "D", 0.14),
}
NO_EFFECT = ("none", "confounded")
ORDER = {"none": 0, "small": 1, "large": 2, "confounded": 1, "reversed": 2}


def _axes(title: str, size: tuple[float, float]):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    return fig, ax


def _legend(ax, handles, **kw) -> None:
    leg = ax.legend(handles=handles, fontsize=8.5, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _point(ax, y: float, r: dict, color: str, marker: str) -> None:
    lo, hi = r["ci"]
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, solid_capstyle="butt")
    ax.plot(r["rate"], y, marker=marker, color=color, markersize=4.5, linestyle="none")


def detection(doc: dict, out: Path) -> None:
    cells = doc["level1"]["cells"]
    rows = sorted(
        {(c["template"], c["condition"]) for c in cells}, key=lambda tc: (tc[0], ORDER[tc[1]])
    )
    fig, ax = _axes("How often the test finds an effect in the planted copies", (7.2, 9.6))
    for i, (tid, cond) in enumerate(rows):
        y = len(rows) - 1 - i
        for c in cells:
            if (c["template"], c["condition"]) != (tid, cond):
                continue
            _, color, marker, dy = PLANS[c["plan"]]
            if cond in NO_EFFECT:
                _point(ax, y + dy, c["detected"], color, marker)
                continue
            _point(ax, y + dy, c["planted_direction"], color, marker)
            if c["opposite_direction"]["k"]:
                ax.plot(
                    c["opposite_direction"]["rate"],
                    y + dy,
                    marker=marker,
                    markerfacecolor=SURFACE,
                    markeredgecolor=color,
                    markersize=4.5,
                    linestyle="none",
                )
            if c["plan"] == "reference" and c["theoretical_power"] is not None and cond != "none":
                ax.plot(
                    [c["theoretical_power"]] * 2, [y - 0.38, y + 0.38], color=TEXT, linewidth=1.0
                )
        if i and rows[i - 1][0] != tid:
            ax.axhline(y + 0.5, color=GRID, linewidth=0.8)
    ax.axvline(0.05, color=TEXT_2, linewidth=0.8, linestyle=":")
    ax.set_yticks(
        range(len(rows)), [f"{t} {c}" for t, c in reversed(rows)], color=TEXT, fontsize=8.5
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(
        f"share of {doc['copies']['level1']} copies in which the test finds an effect (where one"
        " was planted,\nin its direction; lines: Wilson 95% intervals; dotted: 5%)",
        color=TEXT_2,
        fontsize=9,
    )
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="-", markersize=4.5, label=label)
        for label, c, m, _ in PLANS.values()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="D",
            markerfacecolor=SURFACE,
            linestyle="none",
            label="found in the opposite direction",
        )
    )
    handles.append(Line2D([], [], color=TEXT, linewidth=1.0, label="theoretical power"))
    _legend(ax, handles, loc="upper center", bbox_to_anchor=(0.45, -0.075), ncol=3)
    _save(fig, out)


def claims(doc: dict, out: Path) -> None:
    arms = doc["level2"]["arms"]
    rows = [
        ("no effect: claims one", "claims_on_no_effect_by_condition", "none"),
        ("confounded: claims one", "claims_on_no_effect_by_condition", "confounded"),
        ("small: claims the planted direction", "planted_direction_by_condition", "small"),
        ("large: claims the planted direction", "planted_direction_by_condition", "large"),
        ("reversed: claims the planted direction", "planted_direction_by_condition", "reversed"),
    ]
    fig, ax = _axes("What the answers claim", (7.2, 4.0))
    for i, (_, key, cond) in enumerate(rows):
        y = len(rows) - 1 - i
        for arm, (_, color, marker, dy) in ARMS.items():
            r = arms[arm][key][cond]
            if r["n"]:
                _point(ax, y + dy, r, color, marker)
    ax.set_yticks(
        range(len(rows)), [label for label, _, _ in reversed(rows)], color=TEXT, fontsize=9
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    n = doc["copies"]["level2"]
    ax.set_xlabel(
        f"share of answers ({n} copies per question and condition;\nlines: Wilson 95% intervals)",
        color=TEXT_2,
        fontsize=9,
    )
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="-", markersize=4.5, label=label)
        for label, c, m, _ in ARMS.values()
    ]
    _legend(ax, handles, loc="lower right")
    _save(fig, out)


def main() -> None:
    doc = json.loads(PLANTED.read_text(encoding="utf-8"))
    PLOTS.mkdir(parents=True, exist_ok=True)
    detection(doc, PLOTS / "planted_detection.png")
    claims(doc, PLOTS / "planted_claims.png")
    print("wrote results/plots/planted_detection.png, results/plots/planted_claims.png")


if __name__ == "__main__":
    main()
