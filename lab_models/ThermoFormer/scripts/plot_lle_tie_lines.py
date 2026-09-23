"""Plot observed and model-predicted ternary LLE tie-lines from prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def xy(composition):
    return composition[1] + 0.5 * composition[2], np.sqrt(3.0) * 0.5 * composition[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=300)
    triangle = np.asarray([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="#333333", lw=1.1)
    for row in rows:
        observed_source, observed_target = xy(row["source"]), xy(row["target_observed"])
        predicted_target = xy(row["target_predicted"])
        ax.plot([observed_source[0], observed_target[0]], [observed_source[1], observed_target[1]], color="#56B4E9", alpha=0.24, lw=0.7)
        ax.plot([observed_source[0], predicted_target[0]], [observed_source[1], predicted_target[1]], color="#D55E00", alpha=0.32, lw=0.7)
    ax.scatter([], [], color="#56B4E9", label="Observed tie-lines")
    ax.scatter([], [], color="#D55E00", label="Predicted tie-lines")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(frameon=False, loc="upper right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
