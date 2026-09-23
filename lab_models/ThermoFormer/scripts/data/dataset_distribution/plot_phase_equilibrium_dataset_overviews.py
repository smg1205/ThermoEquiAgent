"""Regenerate publication figures for the registered VLE and LLE datasets.

The script reads only the four versioned workbooks under ``datasets/``.  It
uses the same canonical molecular identities and prioritized RDKit/SMARTS
families as the established dataset analysis, then writes figures, numerical
source tables, and a concise interpretation to ``experiments/data_quality/construction/dataset_distribution``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from rdkit import Chem
from scipy.ndimage import gaussian_filter

from analyze_datasets import (
    component_statistics,
    system_class_statistics,
    system_family_pair_statistics,
    system_statistics,
)
from dataset_utils import SEED, load_unified_data, ternary_binary_coverage
from molecular_space import _families


BLUE = "#4E79A7"
ORANGE = "#DD8A4B"
PURPLE = "#7A6FA8"
GRID = "#E8E8E8"
TEXT = "#111111"
MM = 1.0 / 25.4


@dataclass(frozen=True)
class PhaseData:
    phase: str
    records: pd.DataFrame
    scale: pd.DataFrame
    pair_statistics: pd.DataFrame
    triplet_statistics: pd.DataFrame
    coverage: pd.DataFrame


def _set_style() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            from matplotlib import font_manager

            font_manager.fontManager.addfont(str(candidate))
            mpl.rcParams["font.family"] = "Arial"
            break
    mpl.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.7,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _visible_cmap(name: str) -> LinearSegmentedColormap:
    colors = mpl.colormaps[name](np.linspace(0.18, 0.90, 256))
    return LinearSegmentedColormap.from_list(f"visible_{name}", colors)


def _clean_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)


def _heading(ax: mpl.axes.Axes, letter: str, title: str, subtitle: str = "") -> None:
    ax.text(-0.01, 1.03, letter, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.08, 1.03, title, transform=ax.transAxes, fontsize=10.1,
            fontweight="bold", ha="left", va="bottom")
    if subtitle:
        ax.text(0.08, 0.93, subtitle, transform=ax.transAxes, fontsize=7.5,
                ha="left", va="top")


def _primary_families(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in components.itertuples(index=False):
        canonical = str(getattr(row, "canonical_smiles", "") or "")
        molecule = Chem.MolFromSmiles(canonical) if canonical else None
        family = _families(molecule)[0] if molecule is not None else "unresolved"
        item = row._asdict()
        item["primary_functional_family"] = family
        rows.append(item)
    return pd.DataFrame(rows)


def _component_count(records: pd.DataFrame, dataset: str, n_components: int) -> int:
    columns = [f"component_id_{index}" for index in range(1, n_components + 1)]
    return int(records.loc[records["dataset"] == dataset, columns].stack().nunique())


def load_phase_data(dataset_root: Path, phase: str) -> PhaseData:
    phase = phase.upper()
    if phase == "VLE":
        subdirectory, pattern = "vle", "*vle_english.xlsx"
    elif phase == "LLE":
        subdirectory, pattern = "lle", "*_lle.xlsx"
    else:
        raise ValueError(f"Unsupported phase-equilibrium task: {phase}")

    unified = load_unified_data(dataset_root / subdirectory, pattern)
    components = component_statistics(unified)
    systems = system_statistics(unified.records, components)
    molecular = _primary_families(components)
    classified, classes = system_class_statistics(systems, molecular)
    _, pair_statistics = system_family_pair_statistics(classified)
    triplets = classes.loc[classes["dataset"] == "Ternary"].copy()
    coverage = ternary_binary_coverage(unified.records)

    scale_rows = []
    for dataset, count in (("Binary", 2), ("Ternary", 3)):
        frame = unified.records.loc[unified.records["dataset"] == dataset]
        scale_rows.append(
            {
                "dataset": dataset,
                "data_points": int(len(frame)),
                "unique_systems": int(frame["system_id"].nunique()),
                "unique_components": _component_count(unified.records, dataset, count),
                "temperature_min_k": float(frame["temperature_k"].min()),
                "temperature_max_k": float(frame["temperature_k"].max()),
                "pressure_min_kpa": float(frame["pressure_kpa"].min()),
                "pressure_max_kpa": float(frame["pressure_kpa"].max()),
            }
        )
    return PhaseData(
        phase=phase,
        records=unified.records,
        scale=pd.DataFrame(scale_rows),
        pair_statistics=pair_statistics,
        triplet_statistics=triplets,
        coverage=coverage,
    )


def _panel_scale(ax: mpl.axes.Axes, data: PhaseData) -> None:
    ax.axis("off")
    _heading(ax, "a", "Dataset size")
    values = data.scale.set_index("dataset")
    ax.text(0.64, 0.80, "Binary", color=BLUE, fontsize=8.7, fontweight="bold",
            ha="center", transform=ax.transAxes)
    ax.text(0.90, 0.80, "Ternary", color=ORANGE, fontsize=8.7, fontweight="bold",
            ha="center", transform=ax.transAxes)
    rows = [
        ("Data points", "data_points", 0.59),
        ("Systems", "unique_systems", 0.36),
        ("Components", "unique_components", 0.13),
    ]
    for label, column, y in rows:
        ax.text(0.04, y, label, fontsize=8.2, ha="left", va="center",
                transform=ax.transAxes)
        ax.text(0.64, y, f"{int(values.at['Binary', column]):,}", color=BLUE,
                fontsize=8.8, fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        ax.text(0.90, y, f"{int(values.at['Ternary', column]):,}", color=ORANGE,
                fontsize=8.8, fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        if y > 0.13:
            ax.plot([0.04, 0.99], [y - 0.115, y - 0.115], color="#DDDDDD", lw=0.45,
                    transform=ax.transAxes)


def _density_contours(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    color: str,
    linestyle: str,
    t_edges: np.ndarray,
    p_edges: np.ndarray,
) -> None:
    if len(frame) < 10:
        return
    histogram, _, _ = np.histogram2d(
        frame["temperature_k"], np.log10(frame["plot_pressure_kpa"]),
        bins=(t_edges, p_edges),
    )
    density = gaussian_filter(histogram.T, sigma=1.15)
    positive = density[density > 0]
    if positive.size < 3:
        return
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = [
        ordered[min(np.searchsorted(cumulative, fraction), len(ordered) - 1)]
        for fraction in (0.90, 0.70, 0.45)
    ]
    levels = np.unique(np.sort(thresholds))
    if len(levels) < 2:
        return
    t_centres = (t_edges[:-1] + t_edges[1:]) / 2
    p_centres = (p_edges[:-1] + p_edges[1:]) / 2
    ax.contour(t_centres, 10**p_centres, density, levels=levels, colors=color,
               linestyles=linestyle, linewidths=np.linspace(0.7, 1.05, len(levels)),
               zorder=3)


def _panel_state(ax: mpl.axes.Axes, data: PhaseData) -> None:
    ax.axis("off")
    _heading(ax, "b", "Thermodynamic state space")
    records = data.records.copy()
    positive = records.loc[records["pressure_kpa"] > 0, "pressure_kpa"]
    floor = max(float(positive.min()) / 2.0, 1e-4)
    records["plot_pressure_kpa"] = records["pressure_kpa"].where(
        records["pressure_kpa"] > 0, floor
    )
    main = ax.inset_axes([0.12, 0.12, 0.65, 0.75])
    tail = ax.inset_axes([0.82, 0.12, 0.17, 0.75])
    main_records = records.loc[records["temperature_k"] <= 650]
    tail_records = records.loc[records["temperature_k"] > 650]
    p_edges = np.linspace(np.log10(records["plot_pressure_kpa"].min()),
                          np.log10(records["plot_pressure_kpa"].max()), 70)
    t_edges = np.linspace(140, 650, 80)
    rng = np.random.default_rng(SEED)
    for dataset, marker, color, linestyle in (
        ("Binary", "o", BLUE, "solid"),
        ("Ternary", "^", ORANGE, "dashed"),
    ):
        for target, frame, maximum, alpha in (
            (main, main_records.loc[main_records["dataset"] == dataset], 2200, 0.13),
            (tail, tail_records.loc[tail_records["dataset"] == dataset], 180, 0.55),
        ):
            n = min(maximum, len(frame))
            if n:
                sample = frame.loc[rng.choice(frame.index.to_numpy(), n, replace=False)]
                target.scatter(sample["temperature_k"], sample["plot_pressure_kpa"],
                               s=4.0 if target is main else 6.0, marker=marker,
                               color=color, alpha=alpha, linewidth=0, zorder=2)
        frame = main_records.loc[main_records["dataset"] == dataset]
        _density_contours(main, frame, color, linestyle, t_edges, p_edges)
    for target in (main, tail):
        target.set_yscale("log")
        target.set_ylim(records["plot_pressure_kpa"].min() * 0.75,
                        records["plot_pressure_kpa"].max() * 1.45)
        target.grid(which="major", color=GRID, lw=0.45, zorder=0)
        target.grid(which="minor", visible=False)
        _clean_axes(target)
    main.set_xlim(140, 650)
    main.set_xticks([200, 300, 400, 500, 600])
    main.set_ylabel("Pressure (kPa)", labelpad=2)
    tail.set_xlim(650, max(1600, float(records["temperature_k"].max()) + 50))
    tail.set_xticks([800, 1400])
    tail.tick_params(axis="y", left=False, labelleft=False)
    tail.spines["left"].set_visible(False)
    main.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", lw=0.9, ms=3.3, label="Binary"),
            Line2D([], [], color=ORANGE, marker="^", lw=0.9, ls="--", ms=3.5,
                   label="Ternary"),
        ],
        loc="lower right", frameon=False, handlelength=1.4,
    )
    tail.text(0.5, 0.97, f">650 K\n(n={len(tail_records):,})", transform=tail.transAxes,
              ha="center", va="top", fontsize=6.6)
    if int((records["pressure_kpa"] <= 0).sum()):
        main.text(0.02, 0.02,
                  f"P = 0 shown at axis floor (n={(records['pressure_kpa'] <= 0).sum():,})",
                  transform=main.transAxes, fontsize=5.8, ha="left", va="bottom")
    ax.text(0.54, -0.01, "Temperature (K)", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.8)
    for left_ax, x0, x1 in ((main, 0.989, 1.011), (tail, -0.011, 0.011)):
        left_ax.plot((x0, x1), (-0.013, 0.013), transform=left_ax.transAxes,
                     color="#666666", clip_on=False, lw=0.7)
        left_ax.plot((x0, x1), (0.987, 1.013), transform=left_ax.transAxes,
                     color="#666666", clip_on=False, lw=0.7)


def _family_label(value: str) -> str:
    aliases = {
        "sulfur-containing": "S-containing",
        "hydrocarbon": "Hydrocarbon",
        "halogenated": "Halogenated",
        "unresolved": "Unresolved",
    }
    return aliases.get(value, value.capitalize())


def _family_order(pairs: pd.DataFrame) -> list[str]:
    totals: dict[str, float] = {}
    for row in pairs.itertuples(index=False):
        totals[row.family_1] = totals.get(row.family_1, 0) + row.unique_systems
        totals[row.family_2] = totals.get(row.family_2, 0) + row.unique_systems
    order = [key for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))]
    return [value for value in order if value != "unresolved"] + (
        ["unresolved"] if "unresolved" in order else []
    )


def _bubble_area(count: float) -> float:
    return 3.0 * count


def _compact_triplet(value: str) -> str:
    aliases = {
        "unresolved": "Unres.",
        "hydrocarbon": "Hydroc.",
        "sulfur-containing": "S-cont.",
        "aromatic": "Arom.",
        "halogenated": "Halog.",
    }
    return " + ".join(aliases.get(part, part.capitalize()) for part in value.split(" + "))


def _panel_chemistry(ax: mpl.axes.Axes, data: PhaseData) -> None:
    ax.axis("off")
    _heading(ax, "c", "Chemical-family coverage", "Binary pairs and ternary triplets")
    left = ax.inset_axes([0.02, 0.03, 0.49, 0.76])
    right = ax.inset_axes([0.56, 0.03, 0.43, 0.76])
    left.axis("off")
    right.axis("off")

    pairs = data.pair_statistics.loc[data.pair_statistics["dataset"] == "Binary"].copy()
    left.text(0.00, 1.00, "c1  Binary family-pair distribution", color=BLUE,
              fontsize=8.2, fontweight="bold", transform=left.transAxes)
    left.text(0.00, 0.93, f"{len(pairs)} unordered family pairs", fontsize=7.1,
              transform=left.transAxes)
    matrix_ax = left.inset_axes([0.18, 0.02, 0.64, 0.84])
    key_ax = left.inset_axes([0.83, 0.18, 0.16, 0.55])
    key_ax.axis("off")
    families = _family_order(pairs)
    positions = {family: index for index, family in enumerate(families)}
    norm = PowerNorm(gamma=0.58, vmin=max(1, float(pairs["total_data_points"].min())),
                     vmax=float(pairs["total_data_points"].max()))
    cmap = _visible_cmap("Blues")
    n_family = len(families)
    for boundary in np.arange(-0.5, n_family, 1):
        matrix_ax.axhline(boundary, color="#EEEEEE", lw=0.35, zorder=0)
        matrix_ax.axvline(boundary, color="#EEEEEE", lw=0.35, zorder=0)
    for row in pairs.itertuples(index=False):
        low, high = sorted((positions[row.family_1], positions[row.family_2]))
        value = norm(float(row.total_data_points))
        matrix_ax.scatter(high, low, s=_bubble_area(row.unique_systems),
                          color=cmap(value), edgecolor="white", linewidth=0.35, zorder=2)
        if row.unique_systems >= 3:
            matrix_ax.text(high, low, str(int(row.unique_systems)), ha="center", va="center",
                           fontsize=5.8, fontweight="bold",
                           color="white" if value > 0.35 else TEXT, zorder=3)
    matrix_ax.set_xlim(-0.55, n_family - 0.45)
    matrix_ax.set_ylim(n_family - 0.45, -0.55)
    matrix_ax.set_aspect("equal")
    matrix_ax.set_xticks(range(n_family))
    matrix_ax.set_yticks(range(n_family))
    matrix_ax.set_xticklabels([_family_label(x) for x in families], rotation=48,
                              ha="right", rotation_mode="anchor", fontsize=5.8)
    matrix_ax.set_yticklabels([_family_label(x) for x in families], fontsize=5.8)
    matrix_ax.tick_params(length=0, pad=1)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    key_ax.text(0.5, 1.00, "Bubble area\nUnique systems", ha="center", va="top",
                fontsize=5.7)
    for yy, value in zip((0.68, 0.46, 0.20), (5, 12, 50)):
        key_ax.scatter(0.28, yy, s=_bubble_area(value), color="#AAAAAA", alpha=0.8,
                       edgecolor="white", linewidth=0.35)
        key_ax.text(0.74, yy, str(value), fontsize=5.8, ha="center", va="center")
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)

    triplets = data.triplet_statistics.sort_values(
        ["total_data_points", "unique_systems"], ascending=[False, False]
    )
    ranked = triplets.head(18).sort_values("total_data_points")
    right.text(0.00, 1.00, "c2  Ternary family-triplet distribution", color=ORANGE,
               fontsize=8.2, fontweight="bold", transform=right.transAxes)
    right.text(0.00, 0.93,
               f"Top {len(ranked)} by {data.phase} points 路 {len(triplets)} triplets in total",
               fontsize=7.1, transform=right.transAxes)
    plot_ax = right.inset_axes([0.43, 0.02, 0.56, 0.84])
    y = np.arange(len(ranked))
    plot_ax.hlines(y, 0, ranked["total_data_points"], color="#EADBD1", lw=0.7)
    plot_ax.scatter(ranked["total_data_points"], y,
                    s=[_bubble_area(x) for x in ranked["unique_systems"]],
                    color=ORANGE, alpha=0.88, edgecolor="white", linewidth=0.4)
    for points, yy, systems in zip(ranked["total_data_points"], y,
                                   ranked["unique_systems"]):
        if systems >= 4:
            plot_ax.text(points, yy, str(int(systems)), ha="center", va="center",
                         fontsize=5.8, fontweight="bold", color="white")
    plot_ax.set_yticks(y)
    plot_ax.set_yticklabels([_compact_triplet(x) for x in ranked["system_family_class"]],
                            fontsize=5.8)
    plot_ax.set_xlabel(f"Experimental {data.phase} points", labelpad=1)
    plot_ax.set_xlim(0, max(1, float(ranked["total_data_points"].max()) * 1.12))
    plot_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    plot_ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    _clean_axes(plot_ax)


def _simplex_xy(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del first
    return second + 0.5 * third, np.sqrt(3.0) * 0.5 * third


def _simplex(ax: mpl.axes.Axes, first: np.ndarray, second: np.ndarray,
             third: np.ndarray, cmap_name: str) -> None:
    x, y = _simplex_xy(first, second, third)
    triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], closed=True,
                       facecolor="none", edgecolor="#666666", lw=0.65, zorder=4)
    ax.add_patch(triangle)
    density = ax.hexbin(x, y, gridsize=24, extent=(0, 1, 0, np.sqrt(3) / 2),
                        mincnt=1, bins="log", cmap=_visible_cmap(cmap_name), linewidths=0)
    density.set_clip_path(triangle)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 0.94)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.50, 0.91, "C3", ha="center", va="bottom", fontsize=6.4)
    ax.text(-0.01, -0.01, "C1", ha="center", va="top", fontsize=6.4)
    ax.text(1.01, -0.01, "C2", ha="center", va="top", fontsize=6.4)


def _panel_composition(ax: mpl.axes.Axes, data: PhaseData) -> None:
    ax.axis("off")
    if data.phase == "VLE":
        title = "Liquid鈥搗apor composition space"
        subtitle = "Liquid (x) and vapor (y) compositions"
        labels = ("Ternary liquid", "Ternary vapor")
    else:
        title = "Coexisting-liquid composition space"
        subtitle = "Phase-alpha (x) and phase-beta (y) compositions"
        labels = ("Ternary phase 伪", "Ternary phase 尾")
    _heading(ax, "d", title, subtitle)
    binary = data.records.loc[data.records["dataset"] == "Binary"]
    ternary = data.records.loc[data.records["dataset"] == "Ternary"]
    square = ax.inset_axes([0.06, 0.08, 0.36, 0.67])
    alpha = ax.inset_axes([0.48, 0.09, 0.24, 0.60])
    beta = ax.inset_axes([0.75, 0.09, 0.24, 0.60])
    ax.text(0.24, 0.80, "Binary", color=BLUE, fontweight="bold", fontsize=7.5,
            ha="center", transform=ax.transAxes)
    ax.text(0.60, 0.80, labels[0], color=ORANGE, fontweight="bold", fontsize=6.2,
            ha="center", transform=ax.transAxes)
    ax.text(0.87, 0.80, labels[1], color=PURPLE, fontweight="bold", fontsize=6.2,
            ha="center", transform=ax.transAxes)
    square.hexbin(binary["x1"], binary["y1"], gridsize=28, extent=(0, 1, 0, 1),
                  mincnt=1, bins="log", cmap=_visible_cmap("Blues"), linewidths=0)
    square.plot([0, 1], [0, 1], color="#777777", ls=(0, (3, 2)), lw=0.65)
    square.set_xlim(0, 1)
    square.set_ylim(0, 1)
    square.set_aspect("equal")
    square.set_xticks([0, 0.5, 1])
    square.set_yticks([0, 0.5, 1])
    square.set_xlabel("Phase-alpha x1" if data.phase == "LLE" else "Liquid x1", labelpad=1)
    square.set_ylabel("Phase-beta y1" if data.phase == "LLE" else "Vapor y1", labelpad=1)
    _clean_axes(square)
    _simplex(alpha, ternary["x1"].to_numpy(), ternary["x2"].to_numpy(),
             ternary["x3"].to_numpy(), "Oranges")
    _simplex(beta, ternary["y1"].to_numpy(), ternary["y2"].to_numpy(),
             ternary["y3"].to_numpy(), "Purples")


def _panel_coverage(ax: mpl.axes.Axes, data: PhaseData) -> None:
    ax.axis("off")
    _heading(ax, "e", "Binary-to-ternary coverage", "Known constituent binary subsystems")
    counts = data.coverage["available_binary_subsystems"].value_counts().reindex(
        [3, 2, 1, 0], fill_value=0
    )
    total = int(counts.sum())
    plot_ax = ax.inset_axes([0.17, 0.10, 0.79, 0.72])
    y = np.arange(4)
    colors = ["#4D7594", "#7F9CB3", "#A9BDCD", "#D9E3EA"]
    plot_ax.barh(y, counts.to_numpy(), color=colors, height=0.58)
    plot_ax.set_yticks(y)
    plot_ax.set_yticklabels(["3/3 known", "2/3 known", "1/3 known", "0/3 known"])
    plot_ax.invert_yaxis()
    for yy, count in zip(y, counts.to_numpy()):
        percentage = 100.0 * count / total if total else 0.0
        plot_ax.text(count + max(total * 0.015, 0.5), yy,
                     f"{int(count):,} ({percentage:.1f}%)", va="center", fontsize=7.0)
    plot_ax.set_xlim(0, max(counts.max() * 1.34, 1))
    plot_ax.set_xlabel("Ternary systems", labelpad=2)
    plot_ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    _clean_axes(plot_ax)


def create_figure(data: PhaseData, output_directory: Path, dpi: int) -> list[Path]:
    _set_style()
    figure = plt.figure(figsize=(180 * MM, 180 * MM), constrained_layout=False)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.78, 1.45, 1.10],
                               width_ratios=[1.0, 1.18], left=0.045, right=0.985,
                               bottom=0.045, top=0.975, hspace=0.32, wspace=0.18)
    _panel_scale(figure.add_subplot(grid[0, 0]), data)
    _panel_state(figure.add_subplot(grid[0, 1]), data)
    _panel_chemistry(figure.add_subplot(grid[1, :]), data)
    _panel_composition(figure.add_subplot(grid[2, 0]), data)
    _panel_coverage(figure.add_subplot(grid[2, 1]), data)

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{data.phase.lower()}_dataset_overview"
    paths = [output_directory / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    figure.savefig(paths[0], dpi=dpi, facecolor="white")
    figure.savefig(paths[1], facecolor="white")
    figure.savefig(paths[2], facecolor="white")
    plt.close(figure)
    return paths


def write_source_tables(data: PhaseData, source_directory: Path) -> None:
    source_directory.mkdir(parents=True, exist_ok=True)
    prefix = data.phase.lower()
    data.scale.to_csv(source_directory / f"{prefix}_dataset_scale.csv", index=False)
    data.pair_statistics.loc[data.pair_statistics["dataset"] == "Binary"].to_csv(
        source_directory / f"{prefix}_binary_family_pairs.csv", index=False
    )
    data.triplet_statistics.to_csv(
        source_directory / f"{prefix}_ternary_family_triplets.csv", index=False
    )
    coverage = (
        data.coverage["available_binary_subsystems"].value_counts().reindex([3, 2, 1, 0], fill_value=0)
        .rename_axis("available_binary_subsystems").reset_index(name="ternary_systems")
    )
    coverage["percentage"] = 100.0 * coverage["ternary_systems"] / coverage["ternary_systems"].sum()
    coverage.to_csv(source_directory / f"{prefix}_binary_subsystem_coverage.csv", index=False)


def write_analysis(data_by_phase: dict[str, PhaseData], report_path: Path) -> None:
    lines = [
        "# Phase-equilibrium dataset coverage",
        "",
        "The statistics below are computed from the four registered workbooks and use unordered, canonical-SMILES system identities.",
        "",
    ]
    for phase in ("VLE", "LLE"):
        data = data_by_phase[phase]
        scale = data.scale.set_index("dataset")
        coverage = data.coverage["available_binary_subsystems"].value_counts().reindex([3, 2, 1, 0], fill_value=0)
        total = int(coverage.sum())
        binary_points = int(scale.at["Binary", "data_points"])
        ternary_points = int(scale.at["Ternary", "data_points"])
        ternary_share = 100.0 * ternary_points / (binary_points + ternary_points)
        sparse_share = 100.0 * int(coverage.loc[0] + coverage.loc[1]) / total
        lines.extend(
            [
                f"## {phase}",
                "",
                f"The dataset contains {binary_points + ternary_points:,} observations: {binary_points:,} binary points from {int(scale.at['Binary', 'unique_systems']):,} systems and {ternary_points:,} ternary points from {int(scale.at['Ternary', 'unique_systems']):,} systems. Ternary observations represent {ternary_share:.1f}% of the {phase} records.",
                "",
                f"The binary and ternary subsets contain {int(scale.at['Binary', 'unique_components']):,} and {int(scale.at['Ternary', 'unique_components']):,} distinct canonical components, respectively. Temperature and pressure sampling are strongly nonuniform, so densely populated operating regions coexist with sparse state-space tails.",
                "",
                f"Among the {total:,} ternary systems, {int(coverage.loc[3]):,} ({100*coverage.loc[3]/total:.1f}%) have all three constituent binary subsystems represented, whereas {int(coverage.loc[0]):,} ({100*coverage.loc[0]/total:.1f}%) have none. Overall, {sparse_share:.1f}% have at most one observed binary subsystem. This limited overlap makes multicomponent prediction more demanding than interpolation among completely observed binary relations.",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    root = args.project_root.resolve()
    analysis_root = root / 'experiments/data_quality/construction/dataset_distribution'
    output_directory = analysis_root / "figures"
    source_directory = analysis_root / 'results/phase_equilibrium'
    data_by_phase = {
        phase: load_phase_data(root / "datasets", phase) for phase in ("VLE", "LLE")
    }
    for data in data_by_phase.values():
        write_source_tables(data, source_directory)
        for path in create_figure(data, output_directory, args.dpi):
            print(path)
    report_path = analysis_root / 'reports/phase_equilibrium_dataset_analysis.md'
    write_analysis(data_by_phase, report_path)
    print(report_path)


if __name__ == "__main__":
    main()
