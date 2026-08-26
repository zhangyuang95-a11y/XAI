"""Render a coordinate-labelled warehouse layout evidence image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .layouts import STUDY_MAP_LAYOUT


def render_study_layout(output: str | Path) -> Path:
    layout = STUDY_MAP_LAYOUT
    values = np.asarray(
        [[1 if symbol == "#" else 0 for symbol in row] for row in layout.tiles]
    )
    figure, axis = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
    axis.imshow(
        values,
        cmap=ListedColormap(("#f8fafc", "#94a3b8")),
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(range(layout.cols), labels=range(layout.cols))
    axis.set_yticks(range(layout.rows), labels=range(layout.rows))
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    axis.set_title(
        "Production warehouse: staggered work aisles and three-cell robot exit"
    )
    axis.set_xticks(np.arange(-0.5, layout.cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, layout.rows, 1), minor=True)
    axis.grid(which="minor", color="#cbd5e1", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)

    labels = {
        layout.robot_start_positions[0]: ("R1", "#2563eb"),
        layout.charger_position: ("⚡", "#6d4aff"),
        layout.robot_start_positions[1]: ("R2", "#f97316"),
    }
    for (row, column), (label, color) in labels.items():
        axis.text(
            column,
            row,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "white"},
        )
    for row, column in layout.robot_exit_positions:
        axis.add_patch(
            plt.Rectangle(
                (column - 0.46, row - 0.46),
                0.92,
                0.92,
                fill=False,
                edgecolor="#16a34a",
                linewidth=3,
            )
        )
        axis.text(
            column,
            row - 0.31,
            "EXIT",
            ha="center",
            va="top",
            color="#15803d",
            fontsize=7,
            fontweight="bold",
        )
    axis.text(
        0,
        layout.rows + 0.15,
        "Grey = shelf  |  Green outlines = three adjacent passable exit cells (8,4)–(8,6)",
        ha="left",
        va="top",
        color="#334155",
        fontsize=10,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, facecolor="white")
    plt.close(figure)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    print(render_study_layout(args.output))


if __name__ == "__main__":
    main()
