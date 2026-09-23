"""Create the main and supporting-information VLE dataset figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter

SEED = 42
MM_TO_INCH = 1.0 / 25.4
PALETTE = {
    "Binary": "#4C78A8",
    "Ternary": "#D58A55",
    "Shared": "#75638A",
    "Grey": "#8C8C8C",
    "LightGrey": "#D9D9D9",
    "Dark": "#000000",
}
CHEMICAL_BUBBLE_AREA_PER_SYSTEM = 12.0


def _configure_arial() -> None:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise RuntimeError("Arial is required for this figure but was not found.")
    mpl.font_manager.fontManager.addfont(str(font_path))
    family = FontProperties(fname=str(font_path)).get_name()
    if family != "Arial":
        raise RuntimeError(f"Expected Arial, but selected font is {family!r}.")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.4,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.65,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axes(ax: mpl.axes.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#000000")
    if grid:
        ax.grid(axis="both", color="#E7E7E7", lw=0.45, zorder=0)


def _panel_heading(ax: mpl.axes.Axes, letter: str, title: str, subtitle: str = "") -> None:
    ax.text(-0.015, 1.015, letter, transform=ax.transAxes, fontsize=11.0, fontweight="bold", va="bottom", ha="left", clip_on=False, color=PALETTE["Dark"])
    ax.text(0.065, 1.015, title, transform=ax.transAxes, fontsize=10.0, fontweight="semibold", va="bottom", ha="left", clip_on=False, color=PALETTE["Dark"])
    if subtitle:
        ax.text(0.065, 0.955, subtitle, transform=ax.transAxes, fontsize=8.0, va="top", ha="left", clip_on=False, color="#000000")


def _panel_scale(ax: mpl.axes.Axes, scale: pd.DataFrame) -> None:
    metrics = [("Data points", "data_points"), ("Systems", "unique_systems"), ("Components", "unique_components")]
    values = scale.set_index("dataset")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel_heading(ax, "a", "Dataset size")
    column_x = {"Binary": 0.60, "Ternary": 0.87}
    for dataset in ("Binary", "Ternary"):
        ax.text(column_x[dataset], 0.80, dataset, ha="center", va="center", fontsize=8.9, fontweight="semibold", color=PALETTE[dataset])
    row_y = (0.59, 0.36, 0.13)
    for index, (y, (label, column)) in enumerate(zip(row_y, metrics)):
        ax.text(0.02, y, label, ha="left", va="center", fontsize=8.7, color=PALETTE["Dark"])
        for dataset in ("Binary", "Ternary"):
            ax.text(column_x[dataset], y, f"{int(values.at[dataset, column]):,}", ha="center", va="center", fontsize=9.4, fontweight="semibold", color=PALETTE[dataset])
        if index < len(row_y) - 1:
            ax.plot([0.02, 0.98], [y - 0.115, y - 0.115], color="#E5E5E5", lw=0.45)


def _density_contours(ax: mpl.axes.Axes, frame: pd.DataFrame, color: str, linestyle: str, temperature_edges: np.ndarray, log_pressure_edges: np.ndarray) -> None:
    histogram, _, _ = np.histogram2d(frame["temperature_k"], np.log10(frame["pressure_kpa"]), bins=(temperature_edges, log_pressure_edges))
    density = gaussian_filter(histogram.T, sigma=1.15)
    positive = density[density > 0]
    if positive.size == 0:
        return
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = [ordered[min(np.searchsorted(cumulative, fraction), len(ordered) - 1)] for fraction in (0.90, 0.70, 0.45)]
    levels = np.unique(np.sort(thresholds))
    if levels.size < 2:
        return
    t_centers = (temperature_edges[:-1] + temperature_edges[1:]) / 2
    logp_centers = (log_pressure_edges[:-1] + log_pressure_edges[1:]) / 2
    ax.contour(t_centers, 10**logp_centers, density, levels=levels, colors=[color], linestyles=linestyle, linewidths=np.linspace(0.75, 1.15, len(levels)), alpha=0.98, zorder=3)


def _scatter_state_points(ax: mpl.axes.Axes, frame: pd.DataFrame, marker: str, color: str, rng: np.random.Generator, maximum: int, alpha: float) -> None:
    count = min(maximum, len(frame))
    if count == 0:
        return
    sample = frame.loc[rng.choice(frame.index.to_numpy(), size=count, replace=False)]
    ax.scatter(sample["temperature_k"], sample["pressure_kpa"], s=4.2 if maximum > 100 else 9.0, marker=marker, color=color, alpha=alpha, linewidth=0, zorder=2)


def _style_state_axis(ax: mpl.axes.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(7e-3, 7e4)
    ax.grid(which="major", color="#E9E9E9", lw=0.45, zorder=0)
    ax.grid(which="minor", visible=False)
    _clean_axes(ax)


def _state_legend() -> list[Line2D]:
    return [
        Line2D([], [], color=PALETTE["Binary"], marker="o", lw=1.0, ms=3.6, label="Binary"),
        Line2D([], [], color=PALETTE["Ternary"], marker="^", lw=1.0, ls="--", ms=3.8, label="Ternary"),
    ]


def _panel_state_space(ax: mpl.axes.Axes, records: pd.DataFrame, high_temperature_style: str = "broken") -> None:
    ax.axis("off")
    _panel_heading(ax, "b", "Thermodynamic state space")
    rng = np.random.default_rng(SEED)
    main_records = records.loc[records["temperature_k"] <= 650]
    tail_records = records.loc[records["temperature_k"] > 650]
    t_edges = np.linspace(140, 650, 82)
    lp_edges = np.linspace(np.log10(records["pressure_kpa"].min()), np.log10(records["pressure_kpa"].max()), 72)
    if high_temperature_style == "broken":
        main_ax = ax.inset_axes([0.11, 0.10, 0.66, 0.78])
        tail_ax = ax.inset_axes([0.82, 0.10, 0.17, 0.78])
    elif high_temperature_style == "inset":
        main_ax = ax.inset_axes([0.11, 0.10, 0.87, 0.78])
        tail_ax = main_ax.inset_axes([0.61, 0.53, 0.36, 0.40])
        tail_ax.set_facecolor("white")
        tail_ax.patch.set_alpha(0.96)
    else:
        raise ValueError(f"Unknown high-temperature style: {high_temperature_style}")
    for dataset, marker, linestyle in (("Binary", "o", "solid"), ("Ternary", "^", "dashed")):
        main_frame = main_records.loc[main_records["dataset"] == dataset]
        tail_frame = tail_records.loc[tail_records["dataset"] == dataset]
        _scatter_state_points(main_ax, main_frame, marker, PALETTE[dataset], rng, 2_000, 0.12)
        _scatter_state_points(tail_ax, tail_frame, marker, PALETTE[dataset], rng, 100, 0.68)
        _density_contours(main_ax, main_frame, PALETTE[dataset], linestyle, t_edges, lp_edges)
    _style_state_axis(main_ax)
    _style_state_axis(tail_ax)
    main_ax.set_xlim(140, 650)
    main_ax.set_xticks([200, 300, 400, 500, 600])
    main_ax.set_ylabel("Pressure (kPa)", labelpad=3)
    tail_ax.set_xlim(650, 1600)
    tail_ax.set_xticks([800, 1400])
    tail_ax.tick_params(axis="y", left=False, labelleft=False)
    tail_ax.spines["left"].set_visible(False)
    main_ax.legend(handles=_state_legend(), loc="lower right", handlelength=1.5, borderaxespad=0.3)
    tail_ax.text(0.50, 0.97, f"High-temperature\ntail (n={len(tail_records)})", transform=tail_ax.transAxes, ha="center", va="top", fontsize=6.8, color="#000000")
    ax.text(0.53, -0.025, "Temperature (K)", transform=ax.transAxes, ha="center", va="top", fontsize=9.0, clip_on=False)
    if high_temperature_style == "broken":
        break_kwargs = {"color": "#666666", "clip_on": False, "lw": 0.7}
        main_ax.plot((0.989, 1.011), (-0.013, 0.013), transform=main_ax.transAxes, **break_kwargs)
        main_ax.plot((0.989, 1.011), (0.987, 1.013), transform=main_ax.transAxes, **break_kwargs)
        tail_ax.plot((-0.011, 0.011), (-0.013, 0.013), transform=tail_ax.transAxes, **break_kwargs)
        tail_ax.plot((-0.011, 0.011), (0.987, 1.013), transform=tail_ax.transAxes, **break_kwargs)


def _family_label(value: str) -> str:
    replacements = {"sulfur-containing": "S-containing", "hydrocarbon": "Hydrocarbon", "halogenated": "Halogenated", "unresolved": "Unresolved"}
    return replacements.get(value, value.capitalize())


def _family_order(pair_statistics: pd.DataFrame) -> list[str]:
    involvement: dict[str, float] = {}
    for row in pair_statistics.itertuples(index=False):
        count = float(row.unique_systems)
        involvement[row.family_1] = involvement.get(row.family_1, 0.0) + count
        if row.family_2 != row.family_1:
            involvement[row.family_2] = involvement.get(row.family_2, 0.0) + count
    ordered = [family for family, _ in sorted(involvement.items(), key=lambda item: (-item[1], item[0]))]
    return [family for family in ordered if family != "unresolved"] + (["unresolved"] if "unresolved" in ordered else [])


def _bubble_area(count: float, maximum: float, largest: float = 500.0) -> float:
    return largest * count / maximum


def _chemical_bubble_area(count: float) -> float:
    """One absolute area scale shared by binary pairs and ternary triplets."""
    return CHEMICAL_BUBBLE_AREA_PER_SYSTEM * count


def _visible_colormap(name: str) -> mpl.colors.LinearSegmentedColormap:
    colors = mpl.colormaps[name](np.linspace(0.18, 0.88, 256))
    return mpl.colors.LinearSegmentedColormap.from_list(f"vle_{name.lower()}", colors)


def _panel_binary_landscape(
    ax: mpl.axes.Axes,
    pair_statistics: pd.DataFrame,
    embedded: bool = False,
) -> None:
    ax.axis("off")
    binary = pair_statistics.loc[pair_statistics["dataset"] == "Binary"].copy()
    if embedded:
        ax.text(0.02, 0.99, "c1  Binary family-pair distribution", transform=ax.transAxes, fontsize=8.6, fontweight="semibold", ha="left", va="top", color=PALETTE["Binary"])
        ax.text(0.02, 0.93, f"{len(binary)} unordered family pairs", transform=ax.transAxes, fontsize=7.5, ha="left", va="top", color="#000000")
        matrix_bounds = [0.13, 0.06, 0.68, 0.81]
        key_bounds = [0.82, 0.18, 0.17, 0.54]
    else:
        _panel_heading(ax, "c", "Binary family-pair distribution", f"{len(binary)} unordered family pairs")
        matrix_bounds = [0.15, 0.08, 0.66, 0.79]
        key_bounds = [0.82, 0.19, 0.17, 0.52]
    matrix_ax = ax.inset_axes(matrix_bounds)
    key_ax = ax.inset_axes(key_bounds)
    key_ax.axis("off")
    families = _family_order(binary)
    positions = {family: index for index, family in enumerate(families)}
    n_families = len(families)
    normalization = mpl.colors.PowerNorm(gamma=0.58, vmin=float(binary["total_data_points"].min()), vmax=float(binary["total_data_points"].max()))
    colormap = _visible_colormap("Blues")
    matrix_ax.add_patch(Polygon([(-0.5, -0.5), (n_families - 0.5, -0.5), (n_families - 0.5, n_families - 0.5)], closed=True, facecolor=PALETTE["Binary"], alpha=0.025, edgecolor="none", zorder=0))
    for boundary in np.arange(-0.5, n_families, 1.0):
        matrix_ax.axhline(boundary, color="#EEEEEE", lw=0.35, zorder=0)
        matrix_ax.axvline(boundary, color="#EEEEEE", lw=0.35, zorder=0)
    for row in binary.itertuples(index=False):
        low, high = sorted((positions[row.family_1], positions[row.family_2]))
        color_value = normalization(float(row.total_data_points))
        matrix_ax.scatter(high, low, s=_chemical_bubble_area(float(row.unique_systems)), color=colormap(color_value), alpha=0.88, edgecolor="white", linewidth=0.45, zorder=3)
        if row.unique_systems >= 3:
            matrix_ax.text(high, low, str(int(row.unique_systems)), ha="center", va="center", fontsize=7.1, fontweight="semibold", color="white" if color_value >= 0.34 else PALETTE["Dark"], zorder=4)
    matrix_ax.set_xlim(-0.55, n_families - 0.45)
    matrix_ax.set_ylim(n_families - 0.45, -0.55)
    matrix_ax.set_aspect("equal")
    matrix_ax.set_xticks(np.arange(n_families))
    matrix_ax.set_yticks(np.arange(n_families))
    matrix_ax.set_xticklabels([_family_label(family) for family in families], rotation=48, ha="right", rotation_mode="anchor", fontsize=7.2)
    matrix_ax.set_yticklabels([_family_label(family) for family in families], fontsize=7.2)
    matrix_ax.tick_params(length=0, pad=2.0)
    matrix_ax.spines[:].set_visible(False)
    key_ax.text(0.5, 0.99, "Bubble area\nUnique systems\n(shared scale)", ha="center", va="top", fontsize=7.1, color="#000000")
    for y, value in zip((0.72, 0.55, 0.36), (5, 12, 50)):
        key_ax.scatter(0.30, y, s=_chemical_bubble_area(value), color="#A5A5A5", alpha=0.75, edgecolor="white", linewidth=0.4)
        key_ax.text(0.78, y, str(value), ha="center", va="center", fontsize=7.2, color="#000000")
    key_ax.text(0.5, 0.14, "VLE points", ha="center", va="bottom", fontsize=7.2, color="#000000")
    colorbar_ax = key_ax.inset_axes([0.08, 0.05, 0.84, 0.055])
    colorbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=normalization, cmap=colormap), cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([normalization.vmin, normalization.vmax])
    colorbar.set_ticklabels([f"{int(normalization.vmin):,}", f"{int(normalization.vmax):,}"])
    colorbar.ax.tick_params(labelsize=7.0, length=1.8, pad=1.2, colors="#000000")
    colorbar.outline.set_linewidth(0.4)
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)


def _compact_triplet_label(value: str) -> str:
    replacements = {
        "unresolved": "Unres.",
        "hydrocarbon": "Hydroc.",
        "sulfur-containing": "S-cont.",
        "aromatic": "Arom.",
    }
    return " + ".join(replacements.get(part.strip(), part.strip().capitalize()) for part in value.split(" + "))


def _panel_ternary_ranked(
    ax: mpl.axes.Axes,
    triplets: pd.DataFrame,
    top_n: int = 18,
) -> None:
    ax.axis("off")
    ranked = triplets.sort_values(
        ["total_data_points", "unique_systems", "system_family_class"],
        ascending=[False, False, True],
    ).head(top_n).sort_values("total_data_points", ascending=True)
    ax.text(0.02, 0.99, "c2  Ternary family-triplet distribution", transform=ax.transAxes, fontsize=8.6, fontweight="semibold", ha="left", va="top", color=PALETTE["Ternary"])
    ax.text(0.02, 0.93, f"Top {top_n} by VLE points · {len(triplets)} triplets in total", transform=ax.transAxes, fontsize=7.5, ha="left", va="top", color="#000000")
    plot_ax = ax.inset_axes([0.42, 0.09, 0.56, 0.78])
    y = np.arange(len(ranked))
    plot_ax.hlines(y, 0, ranked["total_data_points"], color="#EADBD1", lw=0.8, zorder=1)
    plot_ax.scatter(
        ranked["total_data_points"],
        y,
        s=[_chemical_bubble_area(value) for value in ranked["unique_systems"]],
        color=PALETTE["Ternary"],
        alpha=0.84,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for points, y_value, systems in zip(ranked["total_data_points"], y, ranked["unique_systems"]):
        if systems >= 4:
            plot_ax.text(points, y_value, str(int(systems)), ha="center", va="center", fontsize=7.0, fontweight="semibold", color="white", zorder=4)
    plot_ax.set_yticks(y)
    plot_ax.set_yticklabels([_compact_triplet_label(value) for value in ranked["system_family_class"]], fontsize=7.1)
    plot_ax.set_xlabel("Experimental VLE points", labelpad=2)
    plot_ax.set_xlim(0, float(ranked["total_data_points"].max()) * 1.12)
    plot_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    plot_ax.grid(axis="x", color="#E9E9E9", lw=0.4, zorder=0)
    _clean_axes(plot_ax)


def _panel_chemical_family_coverage(
    ax: mpl.axes.Axes,
    pair_statistics: pd.DataFrame,
    triplets: pd.DataFrame,
) -> None:
    ax.axis("off")
    _panel_heading(ax, "c", "Chemical-family coverage", "Binary pairs and ternary triplets")
    binary_ax = ax.inset_axes([0.00, 0.01, 0.52, 0.90])
    ternary_ax = ax.inset_axes([0.54, 0.01, 0.46, 0.90])
    _panel_binary_landscape(binary_ax, pair_statistics, embedded=True)
    _panel_ternary_ranked(ternary_ax, triplets, top_n=18)


def _simplex_coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    height = np.sqrt(3.0) / 2.0
    horizontal = frame["x2"].to_numpy(dtype=float) + 0.5 * frame["x3"].to_numpy(dtype=float)
    vertical = height * frame["x3"].to_numpy(dtype=float)
    return horizontal, vertical


def _panel_composition_space(ax: mpl.axes.Axes, records: pd.DataFrame) -> None:
    ax.axis("off")
    _panel_heading(ax, "d", "Composition-space coverage", "Liquid and vapor compositions")
    binary_bounds = [0.06, 0.08, 0.41, 0.68]
    ternary_bounds = [0.56, 0.08, 0.40, 0.68]
    binary_ax = ax.inset_axes(binary_bounds)
    ternary_ax = ax.inset_axes(ternary_bounds)
    subplot_title_y = 0.805
    coordinate_label_y = -0.095
    coordinate_fontsize = 8.4
    ax.text(
        binary_bounds[0] + binary_bounds[2] / 2,
        subplot_title_y,
        "Binary",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.7,
        fontweight="semibold",
        color=PALETTE["Binary"],
    )
    ax.text(
        ternary_bounds[0] + ternary_bounds[2] / 2,
        subplot_title_y,
        "Ternary",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.7,
        fontweight="semibold",
        color=PALETTE["Ternary"],
    )
    binary = records.loc[records["dataset"] == "Binary"]
    ternary = records.loc[records["dataset"] == "Ternary"].dropna(subset=["x1", "x2", "x3"])
    binary_ax.hexbin(binary["x1"], binary["y1"], gridsize=31, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap=_visible_colormap("Blues"), linewidths=0)
    binary_ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), lw=0.75, color="#777777", zorder=3)
    binary_ax.set_xlim(0, 1)
    binary_ax.set_ylim(0, 1)
    binary_ax.set_aspect("equal")
    binary_ax.set_xticks([0, 0.5, 1])
    binary_ax.set_yticks([0, 0.5, 1])
    binary_ax.set_xlabel("")
    binary_ax.set_ylabel("Vapor y1", labelpad=1.5, fontsize=coordinate_fontsize)
    _clean_axes(binary_ax)
    binary_ax.text(0.04, 0.95, "Darker = denser", transform=binary_ax.transAxes, fontsize=7.3, va="top", color="#000000")
    horizontal, vertical = _simplex_coordinates(ternary)
    triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3.0) / 2.0]], closed=True, facecolor="none", edgecolor="#666666", linewidth=0.75, zorder=4)
    ternary_ax.add_patch(triangle)
    density = ternary_ax.hexbin(horizontal, vertical, gridsize=29, extent=(0, 1, 0, np.sqrt(3.0) / 2.0), mincnt=1, bins="log", cmap=_visible_colormap("Oranges"), linewidths=0, zorder=2)
    density.set_clip_path(triangle)
    ternary_ax.set_xlim(-0.08, 1.08)
    ternary_ax.set_ylim(-0.08, 0.96)
    ternary_ax.set_aspect("equal")
    ternary_ax.axis("off")
    ternary_ax.text(0.50, 0.91, "Component 3", ha="center", va="bottom", fontsize=coordinate_fontsize)
    ternary_ax.text(0.50, 0.06, "Darker = denser", ha="center", va="bottom", fontsize=7.3, color="#000000")
    ax.text(
        binary_bounds[0] + binary_bounds[2] / 2,
        coordinate_label_y,
        "Liquid x1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=coordinate_fontsize,
    )
    ternary_left_vertex = ternary_bounds[0] + ternary_bounds[2] * (0.08 / 1.16)
    ternary_right_vertex = ternary_bounds[0] + ternary_bounds[2] * (1.08 / 1.16)
    ax.text(
        ternary_left_vertex,
        coordinate_label_y,
        "Component 1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=coordinate_fontsize,
    )
    ax.text(
        ternary_right_vertex,
        coordinate_label_y,
        "Component 2",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=coordinate_fontsize,
    )


def _panel_molecular_space(ax: mpl.axes.Axes, molecular: pd.DataFrame) -> None:
    ax.axis("off")
    projected = molecular.loc[(molecular["umap_status"] == "projected") & molecular["umap_1"].notna() & molecular["umap_2"].notna()].copy()
    _panel_heading(ax, "e", "Molecular chemical space", f"{len(projected)} projected molecules · Morgan (r=2, 2,048 bits) + UMAP")
    plot_ax = ax.inset_axes([0.08, 0.10, 0.88, 0.69])
    markers = {"Binary only": "o", "Ternary only": "^", "Shared": "D"}
    colors = {"Binary only": PALETTE["Binary"], "Ternary only": PALETTE["Ternary"], "Shared": PALETTE["Shared"]}
    handles = []
    for membership in ("Binary only", "Ternary only", "Shared"):
        frame = projected.loc[projected["dataset_membership"] == membership]
        if not frame.empty:
            plot_ax.scatter(frame["umap_1"], frame["umap_2"], s=13 if membership != "Shared" else 16, marker=markers[membership], color=colors[membership], alpha=0.72, edgecolor="none", zorder=2 if membership != "Shared" else 3)
        handles.append(Line2D([], [], marker=markers[membership], color="none", markerfacecolor=colors[membership], markeredgecolor="none", markersize=4.5, label=f"{membership} ({len(frame)})"))
    plot_ax.set_xticks([])
    plot_ax.set_yticks([])
    plot_ax.set_xlabel("UMAP1", labelpad=2)
    plot_ax.set_ylabel("UMAP2", labelpad=2)
    plot_ax.spines[:].set_visible(False)
    plot_ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, handletextpad=0.35, columnspacing=0.8, labelspacing=0.3)
    unresolved_ternary = int(((molecular["dataset_membership"] == "Ternary only") & (molecular["umap_status"] != "projected")).sum())
    if unresolved_ternary:
        plot_ax.text(0.01, 0.01, f"{unresolved_ternary} ternary-only structures unresolved; not projected", transform=plot_ax.transAxes, ha="left", va="bottom", fontsize=7.3, color="#000000")


def _panel_binary_ternary_coverage(ax: mpl.axes.Axes, coverage: pd.DataFrame) -> None:
    ax.axis("off")
    _panel_heading(ax, "e", "Binary-to-ternary coverage", "Constituent binary subsystems observed")
    plot_ax = ax.inset_axes([0.20, 0.12, 0.75, 0.73])
    categories = ["3/3 known", "2/3 known", "1/3 known", "0/3 known"]
    raw_categories = ["3/3", "2/3", "1/3", "0/3"]
    counts_series = coverage["coverage_category"].value_counts()
    counts = np.array([int(counts_series.get(category, 0)) for category in raw_categories])
    total = int(counts.sum())
    percentages = counts / total * 100.0
    colors = ["#466B8A", "#7895AD", "#AFC2D2", "#DCE6EE"]
    y = np.arange(len(categories))
    bars = plot_ax.barh(y, counts, height=0.58, color=colors, edgecolor="none", zorder=2)
    plot_ax.set_yticks(y)
    plot_ax.set_yticklabels(categories)
    plot_ax.invert_yaxis()
    plot_ax.set_xlabel("Ternary systems")
    plot_ax.set_xlim(0, max(counts) * 1.38)
    plot_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
    plot_ax.grid(axis="x", color="#E8E8E8", lw=0.45, zorder=0)
    _clean_axes(plot_ax)
    for bar, count, percentage in zip(bars, counts, percentages):
        plot_ax.text(bar.get_width() + max(counts) * 0.025, bar.get_y() + bar.get_height() / 2, f"{count} ({percentage:.1f}%)", ha="left", va="center", fontsize=7.8, color=PALETTE["Dark"])


def _save_figure(figure: mpl.figure.Figure, paths: list[Path], png_dpi: int = 600) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": png_dpi} if path.suffix.lower() == ".png" else {}
        figure.savefig(path, **kwargs)
        if path.suffix.lower() == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")


def create_main_figure(analysis_root: Path, high_temperature_style: str = "broken", output_stem: str = "Figure_dataset_overview_v3", png_dpi: int = 600) -> list[Path]:
    results = analysis_root / 'experiments/reference_results'
    figures = analysis_root / "figures"
    scale = pd.read_csv(results / "dataset_scale.csv")
    records = pd.read_csv(results / "vle_records_for_plotting.csv")
    pair_statistics = pd.read_csv(results / "system_family_pair_statistics.csv")
    triplets = pd.read_csv(results / "ternary_family_triplets.csv")
    coverage = pd.read_csv(results / "ternary_binary_subsystem_coverage.csv")
    _configure_arial()
    figure = plt.figure(figsize=(180 * MM_TO_INCH, 180 * MM_TO_INCH))
    grid = figure.add_gridspec(
        5,
        1,
        height_ratios=[0.19, 0.02, 0.47, 0.08, 0.29],
        hspace=0.0,
        left=0.060,
        right=0.990,
        bottom=0.050,
        top=0.975,
    )
    top = grid[0].subgridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.27)
    bottom = grid[4].subgridspec(1, 2, width_ratios=[0.58, 0.42], wspace=0.27)
    ax_a = figure.add_subplot(top[0, 0])
    ax_b = figure.add_subplot(top[0, 1])
    ax_c = figure.add_subplot(grid[2])
    ax_d = figure.add_subplot(bottom[0, 0])
    ax_f = figure.add_subplot(bottom[0, 1])
    _panel_scale(ax_a, scale)
    _panel_state_space(ax_b, records, high_temperature_style)
    _panel_chemical_family_coverage(ax_c, pair_statistics, triplets)
    _panel_composition_space(ax_d, records)
    _panel_binary_ternary_coverage(ax_f, coverage)
    paths = [figures / f"{output_stem}.{suffix}" for suffix in ("pdf", "svg", "png")]
    _save_figure(figure, paths, png_dpi=png_dpi)
    plt.close(figure)
    return paths


def create_si_figure(analysis_root: Path, top_n: int = 20) -> list[Path]:
    results = analysis_root / 'experiments/reference_results'
    figures = analysis_root / "figures" / "SI"
    statistics = pd.read_csv(results / "ternary_family_triplets.csv")
    ranked = statistics.sort_values(["total_data_points", "unique_systems", "system_family_class"], ascending=[False, False, True]).head(top_n).sort_values("total_data_points", ascending=True)
    _configure_arial()
    figure, ax = plt.subplots(figsize=(165 * MM_TO_INCH, 170 * MM_TO_INCH))
    y = np.arange(len(ranked))
    maximum = float(ranked["unique_systems"].max())
    ax.hlines(y, 0, ranked["total_data_points"], color="#E7D8CE", lw=1.0, zorder=1)
    ax.scatter(ranked["total_data_points"], y, s=[_bubble_area(value, maximum, largest=360) for value in ranked["unique_systems"]], color=PALETTE["Ternary"], alpha=0.82, edgecolor="white", linewidth=0.55, zorder=3)
    for x, y_value, systems in zip(ranked["total_data_points"], y, ranked["unique_systems"]):
        ax.text(x, y_value, str(int(systems)), ha="center", va="center", fontsize=6.5, fontweight="semibold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([value.replace("sulfur-containing", "S-containing") for value in ranked["system_family_class"]], fontsize=7.0)
    ax.set_xlabel("Experimental VLE points")
    ax.set_title("Ternary family-triplet distribution", loc="left", fontsize=9.5, fontweight="semibold", pad=20)
    ax.text(0, 1.015, f"Top {top_n} by data points · 49 family triplets in total", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color="#000000")
    _clean_axes(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ranked["total_data_points"].max() * 1.13)
    legend_values = [1, 5, 12]
    handles = [ax.scatter([], [], s=_bubble_area(value, maximum, largest=360), color="#A5A5A5", alpha=0.75, edgecolor="white", linewidth=0.4, label=str(value)) for value in legend_values]
    ax.legend(handles=handles, title="Unique systems", loc="lower right", ncol=3, handletextpad=0.3, columnspacing=0.8, title_fontsize=7.0)
    figure.subplots_adjust(left=0.34, right=0.97, bottom=0.10, top=0.92)
    paths = [figures / f"Figure_S_ternary_family_triplets.{suffix}" for suffix in ("pdf", "svg", "png")]
    _save_figure(figure, paths)
    plt.close(figure)
    return paths


def create_figure(analysis_root: Path) -> list[Path]:
    """Backward-compatible entry point returning all main and SI outputs."""
    return create_main_figure(analysis_root) + create_si_figure(analysis_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--high-temperature-style", choices=("broken", "inset"), default="broken")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--output-stem", default="Figure_dataset_overview_v3")
    parser.add_argument("--png-dpi", type=int, default=600)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    outputs = create_main_figure(root, args.high_temperature_style, args.output_stem, args.png_dpi)
    if not args.main_only:
        outputs.extend(create_si_figure(root))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
