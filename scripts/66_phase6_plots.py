"""The error-analysis and cost figures, drawn from the committed results files.

- results/plots/errors_by_category.png: the winning run's wrong answers on the held-out set by
  where they go wrong (from error_analysis.json).
- results/plots/cheapest_system.png: the cheapest system per held-out question at each cost of a
  wrong answer and of declining, over the grid of both (from decision_analysis.json).
- results/plots/cost_of_declining.png: the expected cost per question in wrong answers against the
  ratio of the two costs, for each model answering every question, declining at its fixed
  threshold, and declining by the cost-optimal rule (from decision_analysis.json).

Every value drawn is read from those files; nothing is computed here but the layout.

Usage:
    uv run python scripts/66_phase6_plots.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

PLOTS = ROOT / "results/plots"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8983"
# the first two slots of a color-blind-checked categorical palette, one per model; a declining
# variant is a lighter step of its model's hue, hatched in a darker step
BLUE, BLUE_LIGHT, BLUE_DARK = "#2a78d6", "#9ec5f4", "#184f95"
ORANGE, ORANGE_LIGHT, ORANGE_DARK = "#eb6834", "#f6b89c", "#a8431b"

CATEGORY_NAMES = {
    "no_result": "no result (the query failed or was refused)",
    "format_only": "right answer, another format",
    "tables": "reads other tables",
    "join": "joins the right tables wrongly",
    "filter": "other conditions",
    "computation": "another calculation",
    "output": "returns other columns",
    "order_limit": "another order or row limit",
    "other": "no difference found in the query",
    "unparsed": "query could not be parsed",
}
SYSTEMS = {  # name, fill, hatch color (None: no hatch)
    "sonnet": ("Sonnet, answers all", BLUE, None),
    "sonnet_declining": ("Sonnet, declines below its threshold", BLUE_LIGHT, BLUE_DARK),
    "opus": ("Opus, answers all", ORANGE, None),
    "opus_declining": (
        "Opus, declines below its threshold (exploratory)",
        ORANGE_LIGHT,
        ORANGE_DARK,
    ),
    "router": ("router (Sonnet, Opus where Sonnet is unsure)", NEUTRAL, None),
}
SHORT = {
    "sonnet": "Sonnet",
    "sonnet_declining": "Sonnet, declining",
    "opus": "Opus",
    "opus_declining": "Opus, declining",
    "router": "router",
}


def _axes(title: str, size: tuple[float, float] = (7.2, 4.8)):
    fig, ax = plt.subplots(figsize=size, dpi=200)
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


def _legend(ax, **kw) -> None:
    leg = ax.legend(fontsize=8.5, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def errors_by_category(errors: dict, out: Path) -> None:
    counts = [(k, v) for k, v in errors["categories"].items() if v]
    counts.sort(key=lambda kv: kv[1])
    fig, ax = _axes(f"Where the {errors['wrong']} wrong answers go wrong", size=(7.2, 4.2))
    ys = range(len(counts))
    colors = [NEUTRAL if k == "format_only" else BLUE for k, _ in counts]
    ax.barh(ys, [v for _, v in counts], height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for y, (_, v) in zip(ys, counts, strict=True):
        ax.annotate(
            str(v),
            (v, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=8.5,
        )
    ax.set_yticks(list(ys), [CATEGORY_NAMES[k] for k, _ in counts], color=TEXT, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    ax.set_xlabel(
        f"wrong answers of {errors['held_out_questions']} held-out questions (one run),\n"
        "each under the first part that differs from the expert's query",
        color=TEXT_2,
        fontsize=9,
    )
    ax.set_xlim(0, max(v for _, v in counts) * 1.12)
    _save(fig, out)


def _edges(centers: list[float]) -> list[float]:
    """Cell edges on a log grid: halfway (in log) between neighbors, and as far outside the ends."""
    logs = [math.log10(c) for c in centers]
    mids = [(a + b) / 2 for a, b in zip(logs, logs[1:], strict=False)]
    return [10**x for x in [2 * logs[0] - mids[0], *mids, 2 * logs[-1] - mids[-1]]]


def cheapest_system(dec: dict, out: Path) -> None:
    ch = dec["cheapest"]
    grid, cw, cd = ch["rows_by_cost_decline"], ch["cost_wrong_usd"], ch["cost_decline_usd"]
    xe, ye = _edges(cw), _edges(cd)
    fig, ax = _axes("The cheapest system per held-out question, by what errors cost", (7.2, 6.2))
    ax.grid(False)
    cells: dict[str, list[tuple[float, float]]] = {}
    for i, row in enumerate(grid):
        for j, key in enumerate(row):
            cells.setdefault(key, []).append((math.log10(cw[j]), math.log10(cd[i])))
        j = 0
        while j < len(row):  # one rectangle per run of cells with the same system
            k = j
            while k + 1 < len(row) and row[k + 1] == row[j]:
                k += 1
            _, fill, hatch = SYSTEMS[row[j]]
            ax.add_patch(
                Rectangle(
                    (xe[j], ye[i]),
                    xe[k + 1] - xe[j],
                    ye[i + 1] - ye[i],
                    facecolor=fill,
                    edgecolor=hatch or fill,
                    hatch="///" if hatch else None,
                    linewidth=0,
                )
            )
            j = k + 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    lo, hi = max(xe[0], ye[0]), min(xe[-1], ye[-1])
    ax.plot([lo, hi], [lo, hi], color=SURFACE, linestyle="--", linewidth=1.2)
    ax.text(
        0.05,
        0.05,
        "  declining costs as much as a wrong answer",
        rotation=45,
        transform_rotates_text=True,
        rotation_mode="anchor",
        va="bottom",
        color=TEXT,
        fontsize=8,
    )
    # a direct label at the middle (on the log axes) of each system's cells
    for key, pts in cells.items():
        x = 10 ** (sum(p[0] for p in pts) / len(pts))
        y = 10 ** (sum(p[1] for p in pts) / len(pts))
        ax.text(
            x,
            y,
            SHORT[key],
            ha="center",
            va="center",
            rotation=90 if key == "sonnet" else 0,
            color=TEXT,
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none"},
        )
    ax.set_xlabel("cost of a wrong answer (USD)", color=TEXT_2, fontsize=9.5)
    ax.set_ylabel("cost of declining: a person answers instead (USD)", color=TEXT_2, fontsize=9.5)
    share = ch["share_of_grid"]
    handles = [
        Patch(
            facecolor=fill,
            edgecolor=hatch or fill,
            hatch="///" if hatch else None,
            label=f"{name} ({share[key]:.0%} of the grid)",
        )
        for key, (name, fill, hatch) in SYSTEMS.items()
        if share.get(key)
    ]
    leg = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8, frameon=False
    )
    for t in leg.get_texts():
        t.set_color(TEXT)
    extra = dec["break_even"]["router_minus_opus"]
    if not share.get("router") and extra["share_wrong"]["estimate"] == 0:
        fig.text(
            0.01,
            0.01,
            f"The router is never cheapest: it answers the same questions wrongly as Opus alone "
            f"and costs ${extra['api_usd_per_question']['estimate']:.4f} more per question.",
            color=TEXT_2,
            fontsize=8,
        )
    _save(fig, out)


def cost_of_declining(dec: dict, out: Path) -> None:
    c = dec["curves_in_wrong_answers"]
    fig, ax = _axes(
        f"Expected cost per question, in wrong answers ({dec['questions']} held-out questions)"
    )
    styles = (
        ("", "answers all", ":"),
        ("_declining", "fixed threshold", "--"),
        ("_optimal", "cost-optimal rule", "-"),
    )
    for model, color, name in (("sonnet", BLUE, "Sonnet"), ("opus", ORANGE, "Opus")):
        for suffix, rule, ls in styles:
            ax.plot(
                c["ratio"],
                c[model + suffix],
                color=color,
                linestyle=ls,
                linewidth=2,
                label=f"{name}, {rule}",
            )
    ax.set_xscale("log")
    ax.set_xlim(c["ratio"][0], c["ratio"][-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(
        "cost of declining / cost of a wrong answer (API cost left out)", color=TEXT_2, fontsize=9.5
    )
    ax.set_ylabel("expected cost per question (wrong answers)", color=TEXT_2, fontsize=9.5)
    _legend(ax, loc="upper left")
    _save(fig, out)


def main() -> None:
    errors = json.loads((ROOT / "results/metrics/error_analysis.json").read_text(encoding="utf-8"))
    dec = json.loads((ROOT / "results/metrics/decision_analysis.json").read_text(encoding="utf-8"))
    PLOTS.mkdir(parents=True, exist_ok=True)
    errors_by_category(errors, PLOTS / "errors_by_category.png")
    cheapest_system(dec, PLOTS / "cheapest_system.png")
    cost_of_declining(dec, PLOTS / "cost_of_declining.png")
    print(
        f"wrote {PLOTS.relative_to(ROOT)}/errors_by_category.png, cheapest_system.png, "
        "cost_of_declining.png"
    )


if __name__ == "__main__":
    main()
