"""Five-seed explanations for the validation-selected final C1 workflow."""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..reporting.artifacts import artifact_sha256, atomic_write_json, atomic_write_text
from ..configuration import load_experiment_config
from ..data import VLESample, VLETensorDataset, collate_vle, load_vle_samples, retain_pure_anchored_systems
from ..features import build_molecular_encoder, encoder_cache_filename, prepare_partition_features
from ..data.splitting import load_split_assignment, sample_id, system_id
from .core import (
    load_thermoformer_checkpoint,
    padded_state_tensors,
    thermodynamic_response_sensitivity,
    vapor_from_outputs,
)
from .selection import molecule_family, system_family


SEEDS = tuple(range(5))
SPLIT_PROTOCOL = "vle_overall_binary"
FINAL_PROTOCOL = "thermoformer_vle_three_stage.on.vle_overall_binary"
FINAL_RESULT_ROOT = Path(
    "experiments/vle/training_records/v1/three/results"
) / FINAL_PROTOCOL
FINAL_CHECKPOINT_ROOT = Path(
    "experiments/vle/training_records/v1/three/checkpoints"
) / FINAL_PROTOCOL
VIEW_LABELS = {
    "rdkit_2d": "RDKit descriptors",
    "unimol_v2": "Uni-Mol v2",
    "functional_groups": "Functional groups",
}
OUTPUT_LABELS = {
    "ge_rt": r"$G^E/RT$",
    "mean_abs_log_gamma": r"mean $|\ln\gamma|$",
    "mean_abs_log_alpha": r"mean $|\ln\alpha|$",
    "mean_abs_pair_interaction": r"mean $|I_{ij}|$",
}


@dataclass(frozen=True)
class FinalSeedContext:
    seed: int
    selected_stage: str
    model: torch.nn.Module
    stage1_model: torch.nn.Module
    stage2_model: torch.nn.Module
    checkpoint: Path
    stage1_checkpoint: Path
    stage2_checkpoint: Path
    stage1_validation_loss: float
    stage2_validation_loss: float
    split_path: Path
    feature_map: dict[str, np.ndarray]
    view_dimensions: dict[str, int]
    train_baseline: np.ndarray
    test: tuple[VLESample, ...]
    training_history: tuple[dict[str, object], ...]


def exact_group_shapley(coalitions: dict[tuple[int, ...], Tensor], groups: int) -> Tensor:
    """Return exact Shapley values for a small set of grouped input views."""

    expected = {tuple(index for index in range(groups) if mask & (1 << index)) for mask in range(1 << groups)}
    if set(coalitions) != expected:
        raise ValueError("Exact grouped Shapley requires every coalition")
    reference = next(iter(coalitions.values()))
    values = torch.zeros((*reference.shape, groups), dtype=reference.dtype, device=reference.device)
    factorial = math.factorial
    denominator = factorial(groups)
    for group in range(groups):
        others = [index for index in range(groups) if index != group]
        for size in range(groups):
            weight = factorial(size) * factorial(groups - size - 1) / denominator
            for subset in itertools.combinations(others, size):
                key = tuple(sorted(subset))
                extended = tuple(sorted((*subset, group)))
                values[..., group] += weight * (coalitions[extended] - coalitions[key])
    return values


def exact_pair_shapley_interactions(
    coalitions: dict[tuple[int, ...], Tensor], groups: int
) -> dict[tuple[int, int], Tensor]:
    """Return the exact Shapley interaction index for every pair of groups."""

    expected = {
        tuple(index for index in range(groups) if mask & (1 << index))
        for mask in range(1 << groups)
    }
    if set(coalitions) != expected:
        raise ValueError("Exact grouped Shapley interactions require every coalition")
    interactions: dict[tuple[int, int], Tensor] = {}
    for first in range(groups):
        for second in range(first + 1, groups):
            others = [index for index in range(groups) if index not in (first, second)]
            value = torch.zeros_like(next(iter(coalitions.values())))
            for size in range(groups - 1):
                weight = (
                    math.factorial(size)
                    * math.factorial(groups - size - 2)
                    / math.factorial(groups - 1)
                )
                for subset in itertools.combinations(others, size):
                    base = tuple(sorted(subset))
                    with_first = tuple(sorted((*subset, first)))
                    with_second = tuple(sorted((*subset, second)))
                    with_both = tuple(sorted((*subset, first, second)))
                    value += weight * (
                        coalitions[with_both]
                        - coalitions[with_first]
                        - coalitions[with_second]
                        + coalitions[base]
                    )
            interactions[(first, second)] = value
    return interactions


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _chunks(values: Sequence[VLESample], size: int) -> Iterable[Sequence[VLESample]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _view_slices(dimensions: dict[str, int]) -> dict[str, slice]:
    offset = 0
    slices = {}
    for name in ("rdkit_2d", "unimol_v2", "functional_groups"):
        width = int(dimensions[name])
        if width < 1:
            raise ValueError(f"Final C1 checkpoint is missing {name}")
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


def _stratified_sample(rows: Sequence[VLESample], maximum: int, seed: int) -> tuple[VLESample, ...]:
    """Select a deterministic, target-blind sample across task/cardinality strata."""

    strata: dict[tuple[str, int], list[VLESample]] = {}
    for row in rows:
        strata.setdefault((row.experiment_mode, row.component_count), []).append(row)
    if maximum < len(strata):
        raise ValueError("max_samples_per_seed must cover every available stratum")
    rng = np.random.default_rng(seed + 1701)
    allocation = maximum // len(strata)
    remainder = maximum % len(strata)
    selected: list[VLESample] = []
    for index, key in enumerate(sorted(strata)):
        candidates = sorted(strata[key], key=sample_id)
        count = min(len(candidates), allocation + int(index < remainder))
        indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
        selected.extend(candidates[position] for position in indices)
    return tuple(sorted(selected, key=sample_id))


def _load_seed_context(
    project_root: Path,
    samples: Sequence[VLESample],
    seed: int,
    device: torch.device,
) -> FinalSeedContext:
    result_root = project_root / FINAL_RESULT_ROOT / f"seed_{seed}"
    manifest_path = result_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison_path = result_root / "stage_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("evaluation_partition") != "test"
        or manifest.get("protocol") != FINAL_PROTOCOL
        or int(manifest.get("seed", -1)) != seed
        or comparison.get("selection_partition") != "validation"
        or comparison.get("evaluation_partition") != "test"
        or comparison.get("selected_stage") != manifest.get("selected_stage")
    ):
        raise RuntimeError(f"Invalid final-model manifest: {manifest_path}")
    checkpoint = project_root / FINAL_CHECKPOINT_ROOT / f"seed_{seed}/best_model.pt"
    checkpoint_record = manifest["artifacts"]["checkpoint"]
    if artifact_sha256(checkpoint) != checkpoint_record["sha256"]:
        raise RuntimeError(f"Final checkpoint SHA mismatch: {checkpoint}")
    stage1_checkpoint = project_root / comparison["stage1_checkpoint"]
    stage2_record = manifest["artifacts"]["trained_candidate_checkpoint"]
    stage2_checkpoint = project_root / stage2_record["path"]
    if artifact_sha256(stage1_checkpoint) != comparison["stage1_checkpoint_sha256"]:
        raise RuntimeError(f"Stage-1 checkpoint SHA mismatch: {stage1_checkpoint}")
    if artifact_sha256(stage2_checkpoint) != stage2_record["sha256"]:
        raise RuntimeError(f"Stage-2 checkpoint SHA mismatch: {stage2_checkpoint}")
    split_path = project_root / f"datasets/splits/vle/{SPLIT_PROTOCOL}/seed_{seed}.json"
    split = load_split_assignment(split_path, samples)
    experiment = load_experiment_config(
        project_root / "configs/model/c1_three_view_vanilla.yaml"
    )
    unique_smiles = sorted({value for row in samples for value in row.smiles})
    train_smiles = sorted({value for row in split.train for value in row.smiles})
    cache = project_root / "cache" / encoder_cache_filename(experiment.encoder)
    encoder = build_molecular_encoder(experiment.encoder, cache, use_cuda=device.type == "cuda")
    prepared = prepare_partition_features(encoder, unique_smiles, train_smiles)
    recorded = manifest.get("molecular_feature_preprocessing", {})
    if prepared.metadata.get("source_cache_sha256") != recorded.get("source_cache_sha256"):
        raise RuntimeError(f"Numeric molecular feature caches differ from training for seed {seed}")
    if prepared.view_dimensions != recorded.get("view_dimensions"):
        raise RuntimeError(f"Molecular view dimensions differ from training for seed {seed}")
    if prepared.metadata.get("rdkit_scaler", {}).get("sha256") != recorded.get("rdkit_scaler", {}).get("sha256"):
        raise RuntimeError(f"RDKit scaler differs from training for seed {seed}")
    baseline = np.mean([prepared.values[value] for value in train_smiles], axis=0).astype(np.float32)
    bundle = load_thermoformer_checkpoint(checkpoint, device)
    stage1_bundle = load_thermoformer_checkpoint(stage1_checkpoint, device)
    stage2_bundle = load_thermoformer_checkpoint(stage2_checkpoint, device)
    history_path = project_root / manifest["artifacts"]["history"]["path"]
    training_history = tuple(json.loads(history_path.read_text(encoding="utf-8")))
    candidate_stage = str(comparison["trained_candidate_stage"])
    if bundle.model.config.chemical_attention_bias or bundle.model.config.context_pair_interaction:
        raise RuntimeError("Explainability target is not the final C1 vanilla architecture")
    return FinalSeedContext(
        seed=seed,
        selected_stage=str(manifest["selected_stage"]),
        model=bundle.model,
        stage1_model=stage1_bundle.model,
        stage2_model=stage2_bundle.model,
        checkpoint=checkpoint,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        stage1_validation_loss=float(comparison["stages"]["stage0"]["validation_loss"]),
        stage2_validation_loss=float(comparison["stages"][candidate_stage]["validation_loss"]),
        split_path=split_path,
        feature_map=prepared.values,
        view_dimensions=prepared.view_dimensions,
        train_baseline=baseline,
        test=tuple(split.test),
        training_history=training_history,
    )


def _scalar_outputs(output: object, mask: Tensor) -> dict[str, Tensor]:
    active = mask.sum(-1).clamp_min(1.0)
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    upper = torch.triu(torch.ones_like(pair_mask), diagonal=1) * pair_mask
    pair_count = upper.sum((1, 2)).clamp_min(1.0)
    log_activity = output.log_gamma + output.log_psat
    differences = torch.abs(log_activity.unsqueeze(2) - log_activity.unsqueeze(1)) * upper
    if output.pair_interactions is None or output.excess_gibbs_rt is None:
        raise RuntimeError("Final C1 model did not expose thermodynamic interactions")
    return {
        "ge_rt": output.excess_gibbs_rt.squeeze(-1),
        "mean_abs_log_gamma": (torch.abs(output.log_gamma) * mask).sum(-1) / active,
        "mean_abs_log_alpha": differences.sum((1, 2)) / pair_count,
        "mean_abs_pair_interaction": (torch.abs(output.pair_interactions) * upper).sum((1, 2)) / pair_count,
    }


def _coalition_molecules(
    molecules: Tensor,
    mask: Tensor,
    baseline: Tensor,
    slices: dict[str, slice],
    included: tuple[int, ...],
) -> Tensor:
    names = tuple(slices)
    result = baseline.view(1, 1, -1).expand_as(molecules).clone()
    for index in included:
        block = slices[names[index]]
        result[..., block] = molecules[..., block]
    return result * mask.unsqueeze(-1)


def modality_analysis_tables(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slices = _view_slices(context.view_dimensions)
    baseline = torch.from_numpy(context.train_baseline).to(device)
    records: list[dict[str, object]] = []
    interaction_records: list[dict[str, object]] = []
    for chunk in _chunks(rows, batch_size):
        dataset = VLETensorDataset(chunk, context.feature_map)
        batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
        coalition_values: dict[str, dict[tuple[int, ...], Tensor]] = {
            name: {} for name in OUTPUT_LABELS
        }
        full_output = None
        for mask_bits in range(8):
            included = tuple(index for index in range(3) if mask_bits & (1 << index))
            perturbed = _coalition_molecules(
                batch.molecules, batch.mask, baseline, slices, included
            )
            with torch.no_grad():
                output = context.model(
                    perturbed,
                    batch.temperature_k,
                    batch.pressure_kpa,
                    batch.x,
                    batch.mask,
                )
            if mask_bits == 7:
                full_output = output
            for name, values in _scalar_outputs(output, batch.mask).items():
                coalition_values[name][included] = values.cpu()
        if full_output is None:
            raise AssertionError("Full coalition was not evaluated")
        shapley = {name: exact_group_shapley(values, 3) for name, values in coalition_values.items()}
        pair_shapley = {
            name: exact_pair_shapley_interactions(values, 3)
            for name, values in coalition_values.items()
        }
        for row_index, sample in enumerate(chunk):
            base = {
                "seed": context.seed,
                "selected_stage": context.selected_stage,
                "sample_id": sample_id(sample),
                "system_id": system_id(sample),
                "direction": sample.experiment_mode,
                "component_count": sample.component_count,
                "chemical_family": system_family(sample.smiles),
                "evaluation_partition": "test",
                "explanation_mode": "teacher_forced_observed_T_P_x",
            }
            for output_name in OUTPUT_LABELS:
                empty = float(coalition_values[output_name][()][row_index])
                full = float(coalition_values[output_name][(0, 1, 2)][row_index])
                for view_index, view_name in enumerate(slices):
                    records.append(
                        {
                            **base,
                            "output": output_name,
                            "view": view_name,
                            "shapley_value": float(shapley[output_name][row_index, view_index]),
                            "absolute_shapley_value": abs(float(shapley[output_name][row_index, view_index])),
                            "empty_value": empty,
                            "full_value": full,
                            "additivity_error": abs(
                                float(shapley[output_name][row_index].sum()) - (full - empty)
                            ),
                        }
                    )
                for (first, second), values in pair_shapley[output_name].items():
                    first_name = tuple(slices)[first]
                    second_name = tuple(slices)[second]
                    interaction_records.append(
                        {
                            **base,
                            "output": output_name,
                            "view_i": first_name,
                            "view_j": second_name,
                            "shapley_interaction": float(values[row_index]),
                            "absolute_shapley_interaction": abs(float(values[row_index])),
                        }
                    )
        del full_output
    return pd.DataFrame(records), pd.DataFrame(interaction_records)


def modality_shapley_table(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Backward-compatible grouped-attribution table."""

    return modality_analysis_tables(context, rows, device, batch_size)[0]


def interaction_table(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    records = []
    for chunk in _chunks(rows, batch_size):
        dataset = VLETensorDataset(chunk, context.feature_map)
        batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
        with torch.no_grad():
            output = context.model(
                batch.molecules,
                batch.temperature_k,
                batch.pressure_kpa,
                batch.x,
                batch.mask,
            )
        if output.pair_interactions is None or output.excess_gibbs_rt is None:
            raise RuntimeError("Final C1 model did not expose pair interactions")
        for row_index, sample in enumerate(chunk):
            active_log_gamma = output.log_gamma[row_index, : sample.component_count]
            for first in range(sample.component_count):
                for second in range(first + 1, sample.component_count):
                    interaction = float(output.pair_interactions[row_index, first, second].cpu())
                    records.append(
                        {
                            "seed": context.seed,
                            "selected_stage": context.selected_stage,
                            "sample_id": sample_id(sample),
                            "system_id": system_id(sample),
                            "direction": sample.experiment_mode,
                            "component_count": sample.component_count,
                            "chemical_family": system_family(sample.smiles),
                            "component_i_family": molecule_family(sample.smiles[first]),
                            "component_j_family": molecule_family(sample.smiles[second]),
                            "x_i": sample.liquid_composition[first],
                            "x_j": sample.liquid_composition[second],
                            "pair_interaction": interaction,
                            "absolute_pair_interaction": abs(interaction),
                            "composition_weighted_pair_contribution": sample.liquid_composition[first]
                            * sample.liquid_composition[second]
                            * interaction,
                            "mean_abs_log_gamma": float(torch.mean(torch.abs(active_log_gamma)).cpu()),
                            "excess_gibbs_rt": float(output.excess_gibbs_rt[row_index, 0].cpu()),
                            "evaluation_partition": "test",
                        }
                    )
    return pd.DataFrame(records)


def _representative_interior(rows: Sequence[VLESample], cardinality: int) -> VLESample:
    candidates = [
        row
        for row in rows
        if row.component_count == cardinality
        and min(row.liquid_composition) > 0.05
        and max(row.liquid_composition) < 0.95
    ]
    if not candidates:
        raise RuntimeError(f"No interior {cardinality}-component test sample")
    target = np.full(cardinality, 1.0 / cardinality)
    return min(
        candidates,
        key=lambda row: (float(np.linalg.norm(np.asarray(row.liquid_composition) - target)), sample_id(row)),
    )


def sensitivity_table(context: FinalSeedContext, device: torch.device) -> pd.DataFrame:
    records = []
    for cardinality in sorted({row.component_count for row in context.test}):
        sample = _representative_interior(context.test, cardinality)
        dataset = VLETensorDataset([sample], context.feature_map)
        batch = collate_vle([dataset[0]]).to(device)
        composition_response, temperature_response = thermodynamic_response_sensitivity(
            context.model,
            batch.molecules,
            batch.temperature_k,
            batch.pressure_kpa,
            batch.x,
            batch.mask,
        )
        for affected in range(cardinality):
            for increased in range(cardinality):
                records.append(
                    {
                        "seed": context.seed,
                        "selected_stage": context.selected_stage,
                        "sample_id": sample_id(sample),
                        "system_id": system_id(sample),
                        "component_count": cardinality,
                        "affected_component": affected + 1,
                        "increased_component": increased + 1,
                        "d_log_gamma_i_d_closed_x_j": float(composition_response[affected, increased].cpu()),
                        "d_log_gamma_i_d_temperature_k": float(temperature_response[affected].cpu()),
                        "evaluation_partition": "test",
                    }
                )
    return pd.DataFrame(records)


def feature_occlusion_table(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    batch_size: int,
    rdkit_names: Sequence[str],
    functional_group_names: Sequence[str],
) -> pd.DataFrame:
    """Measure individual RDKit/FG dependence by train-mean occlusion.

    Uni-Mol coordinates are deliberately not assigned chemical names: individual
    latent axes are not identifiable across independently trained encoders.
    """

    slices = _view_slices(context.view_dimensions)
    baseline = torch.from_numpy(context.train_baseline).to(device)
    feature_specs = []
    for view, names in (
        ("rdkit_2d", rdkit_names),
        ("functional_groups", functional_group_names),
    ):
        block = slices[view]
        if block.stop - block.start != len(names):
            raise ValueError(f"Feature-name dimension mismatch for {view}")
        feature_specs.extend(
            (view, name, block.start + index) for index, name in enumerate(names)
        )
    records: list[dict[str, object]] = []
    for chunk in _chunks(rows, batch_size):
        dataset = VLETensorDataset(chunk, context.feature_map)
        batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
        with torch.no_grad():
            reference = _scalar_outputs(
                context.model(
                    batch.molecules,
                    batch.temperature_k,
                    batch.pressure_kpa,
                    batch.x,
                    batch.mask,
                ),
                batch.mask,
            )
        for view, name, feature_index in feature_specs:
            perturbed = batch.molecules.clone()
            perturbed[..., feature_index] = baseline[feature_index]
            perturbed = perturbed * batch.mask.unsqueeze(-1)
            with torch.no_grad():
                changed = _scalar_outputs(
                    context.model(
                        perturbed,
                        batch.temperature_k,
                        batch.pressure_kpa,
                        batch.x,
                        batch.mask,
                    ),
                    batch.mask,
                )
            for row_index, sample in enumerate(chunk):
                for output_name in ("ge_rt", "mean_abs_log_gamma", "mean_abs_log_alpha"):
                    delta = float(changed[output_name][row_index].cpu() - reference[output_name][row_index].cpu())
                    records.append(
                        {
                            "seed": context.seed,
                            "selected_stage": context.selected_stage,
                            "sample_id": sample_id(sample),
                            "system_id": system_id(sample),
                            "direction": sample.experiment_mode,
                            "component_count": sample.component_count,
                            "view": view,
                            "feature": name,
                            "output": output_name,
                            "occlusion_delta": delta,
                            "absolute_occlusion_delta": abs(delta),
                            "baseline": "split_train_mean",
                            "evaluation_partition": "test",
                        }
                    )
    return pd.DataFrame(records)


def component_pair_attribution_table(interactions: pd.DataFrame) -> pd.DataFrame:
    """Decompose the learned pair sum into pair and half-pair component shares."""

    pair_rows = interactions.copy()
    pair_rows["record_type"] = "pair"
    pair_rows["entity"] = [
        " × ".join(sorted((str(first), str(second))))
        for first, second in zip(
            pair_rows["component_i_family"], pair_rows["component_j_family"]
        )
    ]
    pair_rows["ge_contribution"] = pair_rows["composition_weighted_pair_contribution"]
    records = [pair_rows]
    component_records: list[dict[str, object]] = []
    for _, row in pair_rows.iterrows():
        for family in (row["component_i_family"], row["component_j_family"]):
            component_records.append(
                {
                    **row.to_dict(),
                    "record_type": "component",
                    "entity": family,
                    "ge_contribution": 0.5 * float(row["composition_weighted_pair_contribution"]),
                }
            )
    records.append(pd.DataFrame(component_records))
    combined = pd.concat(records, ignore_index=True)
    denominator = combined.groupby(
        ["seed", "sample_id", "record_type"]
    )["ge_contribution"].transform(lambda values: float(np.abs(values).sum()))
    combined["absolute_normalized_share"] = np.where(
        denominator > 1e-12,
        np.abs(combined["ge_contribution"]) / denominator,
        0.0,
    )
    return combined


def _binary_case_samples(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
) -> dict[str, VLESample]:
    candidates = [
        row for row in rows
        if row.component_count == 2 and 0.1 < row.liquid_composition[0] < 0.9
    ]
    if not candidates:
        raise RuntimeError("No interior binary states for composition trajectories")
    dataset = VLETensorDataset(candidates, context.feature_map)
    batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
    with torch.no_grad():
        output = context.model(
            batch.molecules,
            batch.temperature_k,
            batch.pressure_kpa,
            batch.x,
            batch.mask,
        )
    score = torch.mean(torch.abs(output.log_gamma[:, :2]), dim=-1).cpu().numpy()
    order = np.argsort(score)
    near_ideal = candidates[int(order[0])]
    hydrogen = [
        (index, row)
        for index, row in enumerate(candidates)
        if {molecule_family(value) for value in row.smiles}
        & {"water", "alcohol/polyol"}
    ]
    if hydrogen:
        hydrogen_bond_rich = max(hydrogen, key=lambda pair: score[pair[0]])[1]
    else:
        hydrogen_bond_rich = candidates[int(order[-2])] if len(candidates) > 1 else candidates[int(order[-1])]
    excluded = {sample_id(near_ideal), sample_id(hydrogen_bond_rich)}
    strong_nonideal = next(
        (candidates[int(index)] for index in reversed(order) if sample_id(candidates[int(index)]) not in excluded),
        candidates[int(order[-1])],
    )
    cases = {
        "near_ideal": near_ideal,
        "hydrogen_bond_rich": hydrogen_bond_rich,
        "strong_nonideal": strong_nonideal,
    }
    return cases


def composition_trajectory_table(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    points: int = 31,
) -> pd.DataFrame:
    """Trace local binary thermodynamics over a closed composition path."""

    records: list[dict[str, object]] = []
    grid = np.linspace(0.05, 0.95, points, dtype=np.float32)
    compositions = np.column_stack([grid, 1.0 - grid])
    for case, sample in _binary_case_samples(context, rows, device).items():
        molecules, temperature, pressure, x, mask = padded_state_tensors(
            sample.smiles,
            context.feature_map,
            np.asarray([sample.temperature_k]),
            np.asarray([sample.pressure_kpa]),
            compositions,
            device,
        )
        with torch.no_grad():
            output = context.model(molecules, temperature, pressure, x, mask)
            vapor, _ = vapor_from_outputs(output, x, mask)
        log_alpha = (
            output.log_gamma[:, 0] + output.log_psat[:, 0]
            - output.log_gamma[:, 1] - output.log_psat[:, 1]
        )
        for index, x1 in enumerate(grid):
            records.append(
                {
                    "seed": context.seed,
                    "selected_stage": context.selected_stage,
                    "case": case,
                    "sample_id": sample_id(sample),
                    "system_id": system_id(sample),
                    "system_family": system_family(sample.smiles),
                    "temperature_k": sample.temperature_k,
                    "pressure_kpa": sample.pressure_kpa,
                    "x_1": float(x1),
                    "y_1": float(vapor[index, 0].cpu()),
                    "pair_interaction_12": float(output.pair_interactions[index, 0, 1].cpu()),
                    "log_gamma_1": float(output.log_gamma[index, 0].cpu()),
                    "log_gamma_2": float(output.log_gamma[index, 1].cpu()),
                    "log_alpha_12": float(log_alpha[index].cpu()),
                    "evaluation_partition": "test",
                    "explanation_mode": "teacher_forced_fixed_T_P_composition_scan",
                }
            )
    return pd.DataFrame(records)


def explicit_case_study_table(
    context: FinalSeedContext,
    device: torch.device,
    points: int = 41,
) -> pd.DataFrame:
    """Build three named held-out seed-0 cases without averaging different systems."""

    english_names = {
        "\u4e59\u9187": "ethanol",
        "\u5f02\u8f9b\u70f7": "isooctane",
        "N,N-\u4e8c\u7532\u57fa\u7532\u9170\u80fa": "N,N-dimethylformamide",
        "\u4e8c\u7532\u57fa\u4e9a\u781c": "dimethyl sulfoxide",
        "\u82ef\u80fa": "aniline",
        "\u7532\u57fa\u73af\u5df1\u70f7": "methylcyclohexane",
        "\u73af\u5df1\u80fa": "cyclohexylamine",
        "3-\u7532\u57fa\u567b\u5429": "3-methylthiophene",
        "\u7532\u82ef": "toluene",
        "\u73af\u5df1\u70f7": "cyclohexane",
        "2-\u7532\u57fa\u4e19\u70f7": "isobutane",
        "\u4e19\u8148": "propionitrile",
    }

    binary = [
        row for row in context.test
        if row.component_count == 2 and 0.1 < row.liquid_composition[0] < 0.9
    ]
    if len(binary) < 3:
        raise RuntimeError("Explicit binary case studies require at least three interior test states")

    def nonideality(rows: Sequence[VLESample]) -> np.ndarray:
        scores: list[np.ndarray] = []
        for chunk in _chunks(rows, 256):
            dataset = VLETensorDataset(chunk, context.feature_map)
            batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
            with torch.no_grad():
                output = context.model(
                    batch.molecules,
                    batch.temperature_k,
                    batch.pressure_kpa,
                    batch.x,
                    batch.mask,
                )
            active = batch.mask.sum(-1).clamp_min(1.0)
            scores.append(
                ((torch.abs(output.log_gamma) * batch.mask).sum(-1) / active).cpu().numpy()
            )
        return np.concatenate(scores)

    binary_score = nonideality(binary)
    near_index = int(np.argmin(binary_score))
    hydrogen_indices = [
        index for index, row in enumerate(binary)
        if ({molecule_family(value) for value in row.smiles}
            & {"water", "alcohol/polyol"})
        and index != near_index
    ]
    hbond_index = (
        max(hydrogen_indices, key=lambda index: binary_score[index])
        if hydrogen_indices
        else int(np.argsort(binary_score)[-1])
    )
    remaining = [
        index for index in range(len(binary))
        if index not in {near_index, hbond_index}
    ]
    nonideal_index = max(remaining, key=lambda index: binary_score[index])
    cases = (
        ("near_ideal_binary", binary[near_index]),
        ("hbond_binary", binary[hbond_index]),
        ("nonideal_binary", binary[nonideal_index]),
    )
    records: list[dict[str, object]] = []
    for case, sample in cases:
        if sample.component_count == 2:
            path = np.linspace(0.05, 0.95, points, dtype=np.float32)
            compositions = np.column_stack([path, 1.0 - path])
            path_component = 1
        else:
            path = np.linspace(0.02, 0.60, points, dtype=np.float32)
            first_two = np.asarray(sample.liquid_composition[:2], dtype=np.float32)
            ratio = first_two / first_two.sum()
            compositions = np.column_stack(
                [(1.0 - path) * ratio[0], (1.0 - path) * ratio[1], path]
            )
            path_component = 3
        molecules, temperature, pressure, x, mask = padded_state_tensors(
            sample.smiles,
            context.feature_map,
            np.asarray([sample.temperature_k]),
            np.asarray([sample.pressure_kpa]),
            compositions,
            device,
        )
        with torch.no_grad():
            output = context.model(molecules, temperature, pressure, x, mask)
            vapor, _ = vapor_from_outputs(output, x, mask)
        for state_index, path_value in enumerate(path):
            record: dict[str, object] = {
                "seed": context.seed,
                "selected_stage": context.selected_stage,
                "case": case,
                "sample_id": sample_id(sample),
                "system_id": system_id(sample),
                "component_names": " | ".join(
                    english_names.get(name, name) for name in sample.names
                ),
                "component_smiles": " | ".join(sample.smiles),
                "component_count": sample.component_count,
                "temperature_k": sample.temperature_k,
                "pressure_kpa": sample.pressure_kpa,
                "path_component": path_component,
                "path_fraction": float(path_value),
                "evaluation_partition": "test",
                "explanation_mode": "teacher_forced_fixed_T_P_composition_path",
            }
            for component in range(sample.component_count):
                record[f"x_{component + 1}"] = float(compositions[state_index, component])
                record[f"y_{component + 1}"] = float(vapor[state_index, component].cpu())
                record[f"log_gamma_{component + 1}"] = float(output.log_gamma[state_index, component].cpu())
            for first in range(sample.component_count):
                for second in range(first + 1, sample.component_count):
                    pair = float(output.pair_interactions[state_index, first, second].cpu())
                    record[f"pair_interaction_{first + 1}{second + 1}"] = pair
                    record[f"pair_contribution_{first + 1}{second + 1}"] = (
                        float(compositions[state_index, first])
                        * float(compositions[state_index, second])
                        * pair
                    )
            records.append(record)
    return pd.DataFrame(records)


def finetuning_validation_table(contexts: Sequence[FinalSeedContext]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": context.seed,
                "selected_stage": context.selected_stage,
                "stage1_validation_loss": context.stage1_validation_loss,
                "stage2_validation_loss": context.stage2_validation_loss,
                "delta_validation_loss": context.stage2_validation_loss
                - context.stage1_validation_loss,
                "selection_partition": "validation",
            }
            for context in contexts
        ]
    )


def finetuning_effect_table(
    context: FinalSeedContext,
    rows: Sequence[VLESample],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Compare the attempted fugacity Stage 2 against its frozen Stage 1 input."""

    records: list[dict[str, object]] = []
    for chunk in _chunks(rows, batch_size):
        dataset = VLETensorDataset(chunk, context.feature_map)
        batch = collate_vle([dataset[index] for index in range(len(dataset))]).to(device)
        with torch.no_grad():
            stage1 = context.stage1_model(
                batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.x, batch.mask
            )
            stage2 = context.stage2_model(
                batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.x, batch.mask
            )
            vapor1, _ = vapor_from_outputs(stage1, batch.x, batch.mask)
            vapor2, _ = vapor_from_outputs(stage2, batch.x, batch.mask)
        for row_index, sample in enumerate(chunk):
            active = slice(0, sample.component_count)
            x = batch.x[row_index, active]
            y = batch.y[row_index, active]
            pressure = batch.pressure_kpa[row_index].reshape(())
            gamma1 = torch.exp(stage1.log_gamma[row_index, active])
            gamma2 = torch.exp(stage2.log_gamma[row_index, active])
            psat1 = torch.exp(stage1.log_psat[row_index, active])
            psat2 = torch.exp(stage2.log_psat[row_index, active])
            residual1 = torch.sqrt(torch.mean(((x * gamma1 * psat1 - y * pressure) / pressure.clamp_min(1e-12)) ** 2))
            residual2 = torch.sqrt(torch.mean(((x * gamma2 * psat2 - y * pressure) / pressure.clamp_min(1e-12)) ** 2))
            records.append(
                {
                    "seed": context.seed,
                    "selected_stage": context.selected_stage,
                    "sample_id": sample_id(sample),
                    "system_id": system_id(sample),
                    "direction": sample.experiment_mode,
                    "component_count": sample.component_count,
                    "mean_abs_delta_log_gamma": float(torch.mean(torch.abs(stage2.log_gamma[row_index, active] - stage1.log_gamma[row_index, active])).cpu()),
                    "mean_abs_delta_log_psat": float(torch.mean(torch.abs(stage2.log_psat[row_index, active] - stage1.log_psat[row_index, active])).cpu()),
                    "mean_abs_delta_y": float(torch.mean(torch.abs(vapor2[row_index, active] - vapor1[row_index, active])).cpu()),
                    "stage1_teacher_forced_fugacity_rmse": float(residual1.cpu()),
                    "stage2_teacher_forced_fugacity_rmse": float(residual2.cpu()),
                    "delta_teacher_forced_fugacity_rmse": float((residual2 - residual1).cpu()),
                    "evaluation_partition": "test",
                }
            )
    return pd.DataFrame(records)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, content)


def _plot(
    shapley: pd.DataFrame,
    view_synergy: pd.DataFrame,
    feature_occlusion: pd.DataFrame,
    component_pairs: pd.DataFrame,
    trajectories: pd.DataFrame,
    finetuning: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.14,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))

    # a: one compact panel replaces the former bar-plus-violin repetition.
    seed_summary = shapley.groupby(["seed", "output", "view"], as_index=False)["absolute_shapley_value"].mean()
    seed_summary["share"] = seed_summary["absolute_shapley_value"] / seed_summary.groupby(["seed", "output"])["absolute_shapley_value"].transform("sum")
    summary = seed_summary.groupby(["output", "view"])["share"].mean().unstack()
    outputs = list(OUTPUT_LABELS)
    views = list(VIEW_LABELS)
    matrix = summary.loc[outputs, views].to_numpy() * 100.0
    image = axes[0, 0].imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(70.0, float(matrix.max())))
    for row in range(len(outputs)):
        for column in range(len(views)):
            axes[0, 0].text(column, row, f"{matrix[row, column]:.0f}%", ha="center", va="center", fontsize=7)
    axes[0, 0].set_xticks(range(3), ("RDKit", "Uni-Mol", "FG"))
    axes[0, 0].set_yticks(range(4), [OUTPUT_LABELS[name] for name in outputs])
    axes[0, 0].set_title("a  Global view dependence", loc="left", fontweight="bold")
    axes[0, 0].grid(False)
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.03, label="Normalized |Shapley| (%)")

    # b: mean signed pairwise Shapley interaction for relative volatility.
    synergy = view_synergy.loc[view_synergy["output"].eq("mean_abs_log_alpha")]
    synergy = synergy.groupby(["seed", "view_i", "view_j"])["shapley_interaction"].mean().groupby(["view_i", "view_j"]).mean()
    synergy_matrix = np.zeros((3, 3), dtype=float)
    for (first, second), value in synergy.items():
        i, j = views.index(first), views.index(second)
        synergy_matrix[i, j] = synergy_matrix[j, i] = float(value)
    limit = max(float(np.abs(synergy_matrix).max()), 1e-8)
    image = axes[0, 1].imshow(synergy_matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for row in range(3):
        for column in range(3):
            axes[0, 1].text(column, row, "—" if row == column else f"{synergy_matrix[row, column]:.3g}", ha="center", va="center", fontsize=7)
    axes[0, 1].set_xticks(range(3), ("RDKit", "Uni-Mol", "FG"))
    axes[0, 1].set_yticks(range(3), ("RDKit", "Uni-Mol", "FG"))
    axes[0, 1].set_title("b  Cross-view synergy for |ln α|", loc="left", fontweight="bold")
    axes[0, 1].grid(False)
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.03, label="Signed interaction")

    # c: chemically named RDKit and SMARTS features only.
    fine = feature_occlusion.loc[feature_occlusion["output"].eq("mean_abs_log_alpha")]
    fine = fine.groupby(["view", "feature"])["absolute_occlusion_delta"].mean().reset_index()
    selected = pd.concat([
        rows.nlargest(5, "absolute_occlusion_delta") for _, rows in fine.groupby("view")
    ]).sort_values("absolute_occlusion_delta")
    labels = [str(value).replace("Num", "#") for value in selected["feature"]]
    bar_colors = [colors[0] if view == "rdkit_2d" else colors[2] for view in selected["view"]]
    axes[0, 2].barh(np.arange(len(selected)), selected["absolute_occlusion_delta"], color=bar_colors)
    axes[0, 2].set_yticks(np.arange(len(selected)), labels, fontsize=7)
    axes[0, 2].set_xlabel("Mean |occlusion Δ| in |ln α|")
    axes[0, 2].set_title("c  Named feature dependence", loc="left", fontweight="bold")

    # d: representative ternary pair allocation, averaged over seeds/states.
    ternary_pairs = component_pairs.loc[
        component_pairs["record_type"].eq("pair") & component_pairs["component_count"].eq(3)
    ]
    pair_mass = ternary_pairs.assign(
        absolute_ge_contribution=np.abs(ternary_pairs["ge_contribution"])
    ).groupby("entity")["absolute_ge_contribution"].sum()
    pair_summary = (100.0 * pair_mass / pair_mass.sum()).nlargest(8).sort_values()
    axes[1, 0].barh(np.arange(len(pair_summary)), pair_summary.values, color=colors[4])
    axes[1, 0].set_yticks(np.arange(len(pair_summary)), pair_summary.index, fontsize=6.5)
    axes[1, 0].set_xlabel("Share of total absolute pair contribution (%)")
    axes[1, 0].set_title("d  Ternary pair attribution", loc="left", fontweight="bold")

    # e: local composition paths use one line per case and seed aggregation.
    trajectory_summary = trajectories.groupby(["case", "x_1"])["log_alpha_12"].agg(["mean", "std"]).reset_index()
    case_labels = {
        "near_ideal": "Near-ideal activity",
        "hydrogen_bond_rich": "H-bond-rich",
        "strong_nonideal": "Strong nonideal",
    }
    for index, (case, rows) in enumerate(trajectory_summary.groupby("case")):
        axes[1, 1].plot(rows["x_1"], rows["mean"], color=colors[index], label=case_labels.get(case, case))
        axes[1, 1].fill_between(rows["x_1"], rows["mean"] - rows["std"].fillna(0), rows["mean"] + rows["std"].fillna(0), color=colors[index], alpha=0.12)
    axes[1, 1].axhline(0.0, color="#777777", lw=0.7)
    axes[1, 1].set_xlabel("Liquid composition $x_1$")
    axes[1, 1].set_ylabel(r"$\ln\alpha_{12}$")
    axes[1, 1].set_title("e  Composition-resolved response", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=6.5)

    # f: what fugacity fine-tuning actually moved.
    delta_fields = {
        "mean_abs_delta_log_gamma": r"$\ln\gamma$",
        "mean_abs_delta_log_psat": r"$\ln P^{sat}$",
        "mean_abs_delta_y": "$y$",
        "delta_teacher_forced_fugacity_rmse": "fugacity residual",
    }
    seed_delta = finetuning.groupby("seed")[list(delta_fields)].mean()
    means = seed_delta.mean()
    errors = seed_delta.std(ddof=1)
    x_positions = np.arange(len(delta_fields))
    bar_colors = [colors[0], colors[1], colors[2], colors[3]]
    axes[1, 2].bar(x_positions, means.values, yerr=errors.values, color=bar_colors, capsize=2.5)
    axes[1, 2].axhline(0.0, color="#555555", lw=0.7)
    axes[1, 2].set_xticks(x_positions, list(delta_fields.values()), rotation=18, ha="right")
    axes[1, 2].set_ylabel("Mean change (residual: signed Δ)")
    axes[1, 2].set_title("f  Fugacity fine-tuning correction", loc="left", fontweight="bold")

    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, options in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        path = output_stem.with_suffix(f".{suffix}")
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        figure.savefig(temporary, bbox_inches="tight", **options)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip()
                for line in temporary.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            temporary.write_text(normalized, encoding="utf-8")
        os.replace(temporary, path)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_nature(
    shapley: pd.DataFrame,
    view_synergy: pd.DataFrame,
    feature_occlusion: pd.DataFrame,
    component_pairs: pd.DataFrame,
    trajectories: pd.DataFrame,
    finetuning: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    """Render a Nature-style asymmetric evidence chain for the six analyses."""

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.5,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    palette = {
        "rdkit_2d": "#315B7D",
        "unimol_v2": "#78A6A3",
        "functional_groups": "#B68BB5",
        "neutral": "#5C6268",
        "light": "#E9EDF0",
        "accent": "#C66B4E",
        "gain": "#2E8B57",
        "near_ideal": "#7B8085",
        "hydrogen_bond_rich": "#B45B74",
        "strong_nonideal": "#237A78",
    }
    figure = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=(1.55, 0.95, 1.15),
        width_ratios=(1.12, 1.12, 1.0, 0.86),
        hspace=0.43,
        wspace=0.65,
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.07,
    )

    # a — hero mechanism/evidence map.
    ax = figure.add_subplot(grid[0, :3])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.0, 1.03, "a", fontsize=8, fontweight="bold", va="bottom")
    ax.text(0.035, 1.03, "Evidence flow through the final C1 architecture", fontsize=7.4, fontweight="bold", va="bottom")
    alpha_rows = shapley.loc[shapley["output"].eq("mean_abs_log_alpha")]
    alpha_seed = alpha_rows.groupby(["seed", "view"])["absolute_shapley_value"].mean().reset_index()
    alpha_seed["share"] = alpha_seed["absolute_shapley_value"] / alpha_seed.groupby("seed")["absolute_shapley_value"].transform("sum")
    shares = alpha_seed.groupby("view")["share"].mean().to_dict()
    view_specs = (
        ("rdkit_2d", "RDKit descriptors", "physical / topology", 0.73),
        ("unimol_v2", "Uni-Mol v2", "3D latent structure", 0.44),
        ("functional_groups", "SMARTS groups", "chemical motifs", 0.15),
    )
    for view, title, subtitle, y in view_specs:
        color = palette[view]
        box = FancyBboxPatch(
            (0.01, y - 0.085), 0.175, 0.17,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=mpl.colors.to_rgba(color, 0.13),
            edgecolor=color,
            linewidth=0.9,
        )
        ax.add_patch(box)
        ax.text(0.025, y + 0.028, title, color=color, fontsize=6.4, fontweight="bold", va="center")
        ax.text(0.025, y - 0.027, subtitle, color="#555A60", fontsize=5.1, va="center")
        ax.text(0.235, y + 0.035, f"{100 * shares[view]:.0f}%", color=color, fontsize=6.5, fontweight="bold", ha="center", va="bottom")
        ax.add_patch(FancyArrowPatch(
            (0.19, y), (0.285, y), arrowstyle="-|>", mutation_scale=6,
            linewidth=0.8 + 3.4 * shares[view], color=color, alpha=0.82,
        ))
        projection = FancyBboxPatch(
            (0.29, y - 0.045), 0.08, 0.09,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor="white", edgecolor=color, linewidth=0.75,
        )
        ax.add_patch(projection)
        ax.text(0.33, y, r"$f_\mathrm{proj}$", ha="center", va="center", fontsize=5.8, color=color)

    fusion = Circle((0.43, 0.44), 0.073, facecolor="#F2F4F6", edgecolor="#4B5560", linewidth=0.9)
    ax.add_patch(fusion)
    ax.text(0.43, 0.455, "view", ha="center", va="center", fontsize=5.5, fontweight="bold")
    ax.text(0.43, 0.415, "fusion", ha="center", va="center", fontsize=5.5, fontweight="bold")
    for _, _, _, y in view_specs:
        ax.add_patch(FancyArrowPatch((0.372, y), (0.405, 0.44), arrowstyle="-", linewidth=0.8, color="#8A8F94"))
    transformer = FancyBboxPatch(
        (0.53, 0.34), 0.14, 0.20,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor="#E7EEF5", edgecolor="#315B7D", linewidth=1.0,
    )
    ax.add_patch(transformer)
    ax.text(0.60, 0.465, "vanilla", ha="center", fontsize=6.0, fontweight="bold", color="#315B7D")
    ax.text(0.60, 0.414, "Transformer", ha="center", fontsize=6.0, fontweight="bold", color="#315B7D")
    ax.text(0.60, 0.365, "mixture context", ha="center", fontsize=4.9, color="#5A6570")
    ax.add_patch(FancyArrowPatch((0.505, 0.44), (0.525, 0.44), arrowstyle="-|>", mutation_scale=7, linewidth=1.0, color="#4B5560"))

    pair_box = FancyBboxPatch(
        (0.72, 0.34), 0.115, 0.20,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor="#F4E9EF", edgecolor="#9A5A7A", linewidth=1.0,
    )
    ax.add_patch(pair_box)
    ax.text(0.777, 0.47, "pair", ha="center", fontsize=6.0, fontweight="bold", color="#8A496A")
    ax.text(0.777, 0.418, r"$I_{ij}$", ha="center", fontsize=8.0, fontweight="bold", color="#8A496A")
    ax.text(0.777, 0.365, r"$x_i x_j I_{ij}$", ha="center", fontsize=5.0, color="#66515B")
    ax.add_patch(FancyArrowPatch((0.675, 0.44), (0.715, 0.44), arrowstyle="-|>", mutation_scale=7, linewidth=1.0, color="#4B5560"))

    x_steps = (0.865, 0.925, 0.985)
    labels = (r"$G^E/RT$", r"$\ln\gamma$", "VLE")
    for index, (x_value, label) in enumerate(zip(x_steps, labels)):
        radius = 0.027
        circle = Circle((x_value, 0.44), radius, facecolor="#FFF8F2", edgecolor=palette["accent"], linewidth=0.9)
        ax.add_patch(circle)
        ax.text(x_value, 0.44, label, ha="center", va="center", fontsize=4.7, color="#8A4B39", fontweight="bold")
        if index == 0:
            start = 0.84
        else:
            start = x_steps[index - 1] + 0.038
        ax.add_patch(FancyArrowPatch((start, 0.44), (x_value - radius, 0.44), arrowstyle="-|>", mutation_scale=6, linewidth=0.85, color=palette["accent"]))
    ax.text(0.60, 0.14, "molecular evidence", color="#315B7D", ha="center", fontsize=5.2, fontweight="bold")
    ax.plot([0.30, 0.67], [0.115, 0.115], color="#315B7D", lw=1.0)
    ax.text(0.89, 0.14, "thermodynamic evidence", color=palette["accent"], ha="center", fontsize=5.2, fontweight="bold")
    ax.plot([0.72, 0.995], [0.115, 0.115], color=palette["accent"], lw=1.0)
    ax.text(0.01, 0.015, r"arrow width and percentage = normalized |Shapley| share for mean $|\ln\alpha|$", fontsize=4.7, color="#676C71")

    # b — all-output cross-view interaction bubble matrix.
    ax = figure.add_subplot(grid[0, 3])
    ax.set_title("b  Cross-view synergy", loc="left", fontweight="bold", pad=4)
    synergy_seed = view_synergy.groupby(["seed", "output", "view_i", "view_j"], as_index=False)["shapley_interaction"].mean()
    synergy_mean = synergy_seed.groupby(["output", "view_i", "view_j"])["shapley_interaction"].mean().reset_index()
    outputs = list(OUTPUT_LABELS)
    pairs = [("rdkit_2d", "unimol_v2"), ("rdkit_2d", "functional_groups"), ("unimol_v2", "functional_groups")]
    pair_labels = ("R × U", "R × FG", "U × FG")
    maximum = max(float(synergy_mean["shapley_interaction"].abs().max()), 1e-8)
    norm = mpl.colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    for y, pair in enumerate(pairs):
        for x, output in enumerate(outputs):
            row = synergy_mean.loc[
                synergy_mean["output"].eq(output)
                & synergy_mean["view_i"].eq(pair[0])
                & synergy_mean["view_j"].eq(pair[1])
            ]
            value = float(row["shapley_interaction"].iloc[0])
            ax.scatter(x, y, s=24 + 170 * abs(value) / maximum, c=[value], cmap="RdBu_r", norm=norm, edgecolor="white", linewidth=0.55)
    ax.set_xticks(range(4), (r"$G^E$", r"$|\ln\gamma|$", r"$|\ln\alpha|$", r"$|I|$"), rotation=40, ha="right")
    ax.set_yticks(range(3), pair_labels)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(2.55, -0.55)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, orientation="horizontal", fraction=0.07, pad=0.23)
    colorbar.set_label("signed Shapley interaction", fontsize=5.2)
    colorbar.ax.tick_params(labelsize=4.8, length=2)
    ax.text(0.0, -0.2, "circle area = |interaction|; n = 5 seeds", transform=ax.transAxes, fontsize=4.7, color="#676C71")

    # c — seed-aware named feature dependence.
    ax = figure.add_subplot(grid[1, :2])
    ax.set_title("c  Named descriptor and functional-group dependence", loc="left", fontweight="bold", pad=4)
    fine = feature_occlusion.loc[feature_occlusion["output"].eq("mean_abs_log_alpha")]
    fine_seed = fine.groupby(["seed", "view", "feature"], as_index=False)["absolute_occlusion_delta"].mean()
    fine_summary = fine_seed.groupby(["view", "feature"])["absolute_occlusion_delta"].agg(["mean", "std"]).reset_index()
    selected = pd.concat([
        rows.nlargest(5, "mean") for _, rows in fine_summary.groupby("view")
    ]).sort_values(["view", "mean"], ascending=[True, True]).reset_index(drop=True)
    y_positions = np.arange(len(selected))
    for y, row in selected.iterrows():
        color = palette[str(row["view"])]
        ax.hlines(y, 0.0, row["mean"], color=mpl.colors.to_rgba(color, 0.45), lw=1.3)
        ax.errorbar(row["mean"], y, xerr=row["std"], fmt="o", ms=3.8, color=color, ecolor=color, elinewidth=0.7, capsize=1.6)
    ax.set_yticks(y_positions, [str(value).replace("Num", "#") for value in selected["feature"]])
    ax.set_xlabel(r"mean |occlusion Δ| in mean $|\ln\alpha|$ (± s.d., seeds)")
    ax.axhline(4.5, color="#D8DADC", lw=0.6)
    ax.text(0.98, 0.78, "RDKit-24", transform=ax.transAxes, ha="right", color=palette["rdkit_2d"], fontsize=5.4, fontweight="bold")
    ax.text(0.98, 0.22, "SMARTS groups", transform=ax.transAxes, ha="right", color=palette["functional_groups"], fontsize=5.4, fontweight="bold")
    ax.set_ylim(-0.7, len(selected) - 0.3)

    # d — chemistry-family interaction network.
    ax = figure.add_subplot(grid[1, 2:])
    ax.set_title("d  Ternary pair-interaction landscape", loc="left", fontweight="bold", pad=4)
    ax.set_axis_off()
    ternary_pairs = component_pairs.loc[
        component_pairs["record_type"].eq("pair") & component_pairs["component_count"].eq(3)
    ].copy()
    ternary_pairs["absolute_ge_contribution"] = np.abs(ternary_pairs["ge_contribution"])
    edge_mass = ternary_pairs.groupby("entity")["absolute_ge_contribution"].sum()
    edge_share = (edge_mass / edge_mass.sum()).nlargest(7)
    nodes = sorted({part for label in edge_share.index for part in label.split(" × ")})
    angles = np.linspace(np.pi * 0.15, np.pi * 2.15, len(nodes), endpoint=False)
    positions = {node: (0.5 + 0.35 * np.cos(angle), 0.5 + 0.34 * np.sin(angle)) for node, angle in zip(nodes, angles)}
    degree = {node: 0.0 for node in nodes}
    for label, share in edge_share.items():
        first, second = label.split(" × ")
        degree[first] += float(share)
        degree[second] += float(share)
        x1, y1 = positions[first]
        x2, y2 = positions[second]
        curvature = 0.18 if first != second else 0.55
        connection = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-",
            connectionstyle=f"arc3,rad={curvature}",
            linewidth=0.7 + 9.0 * float(share),
            color=mpl.colors.to_rgba("#9A5A7A", 0.26 + 0.65 * float(share / edge_share.max())),
            zorder=1,
        )
        ax.add_patch(connection)
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        if float(share) >= 0.07:
            ax.text(midpoint[0], midpoint[1], f"{100 * float(share):.0f}%", fontsize=4.6, ha="center", va="center", color="#70445B", zorder=4)
    node_colors = ("#D8E5EF", "#E7D7E5", "#D8E8E0", "#ECE1CF", "#DDDDEB", "#E3E4E5")
    for index, node in enumerate(nodes):
        x_value, y_value = positions[node]
        radius = 0.035 + 0.055 * degree[node] / max(degree.values())
        ax.add_patch(Circle((x_value, y_value), radius, facecolor=node_colors[index % len(node_colors)], edgecolor="#5E6368", linewidth=0.65, zorder=3))
        alignment = "left" if x_value >= 0.5 else "right"
        offset = radius + 0.018
        label_x = x_value + offset if alignment == "left" else x_value - offset
        ax.text(label_x, y_value, node.replace("/", "/\n"), ha=alignment, va="center", fontsize=4.7, color="#363B40")
    ax.text(0.02, 0.02, r"edge width = share of $\sum |x_i x_j I_{ij}|$; node size = weighted degree", transform=ax.transAxes, fontsize=4.6, color="#676C71")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # e — two coordinated composition-resolved thermodynamic readouts.
    subgrid = grid[2, :2].subgridspec(1, 2, wspace=0.34)
    ax_e1 = figure.add_subplot(subgrid[0, 0])
    ax_e2 = figure.add_subplot(subgrid[0, 1])
    ax_e1.set_title("e  Composition-resolved local thermodynamics", loc="left", fontweight="bold", pad=4)
    case_labels = {
        "near_ideal": "near-ideal activity",
        "hydrogen_bond_rich": "H-bond-rich",
        "strong_nonideal": "strong nonideal",
    }
    for case, rows in trajectories.groupby("case"):
        color = palette[str(case)]
        for axis, field in ((ax_e1, "log_alpha_12"), (ax_e2, "pair_interaction_12")):
            summary = rows.groupby("x_1")[field].agg(["mean", "std"]).reset_index()
            axis.plot(summary["x_1"], summary["mean"], color=color, lw=1.45)
            axis.fill_between(summary["x_1"], summary["mean"] - summary["std"], summary["mean"] + summary["std"], color=color, alpha=0.13, linewidth=0)
    ax_e1.axhline(0.0, color="#8A8D90", lw=0.55)
    ax_e1.set_xlabel(r"liquid composition $x_1$")
    ax_e1.set_ylabel(r"$\ln\alpha_{12}$")
    ax_e2.axhline(0.0, color="#8A8D90", lw=0.55)
    ax_e2.set_xlabel(r"liquid composition $x_1$")
    ax_e2.set_ylabel(r"pair potential $I_{12}$")
    handles = [Line2D([0], [0], color=palette[case], lw=1.5, label=label) for case, label in case_labels.items()]
    ax_e1.legend(handles=handles, loc="lower left", fontsize=4.8, handlelength=1.6)
    ax_e2.text(0.98, 0.96, "mean ± s.d.; 5 seed-specific test cases", transform=ax_e2.transAxes, ha="right", va="top", fontsize=4.5, color="#676C71")

    # f — paired seed trajectories plus decoder-change magnitudes.
    subgrid = grid[2, 2:].subgridspec(1, 2, width_ratios=(1.08, 0.92), wspace=0.55)
    ax_f1 = figure.add_subplot(subgrid[0, 0])
    ax_f2 = figure.add_subplot(subgrid[0, 1])
    ax_f1.set_title("f  Fugacity fine-tuning mechanism", loc="left", fontweight="bold", pad=4)
    residuals = finetuning.groupby("seed")[["stage1_teacher_forced_fugacity_rmse", "stage2_teacher_forced_fugacity_rmse"]].mean()
    for seed, row in residuals.iterrows():
        values = [row.iloc[0], row.iloc[1]]
        color = palette["gain"] if values[1] < values[0] else palette["accent"]
        ax_f1.plot((0, 1), values, color=mpl.colors.to_rgba(color, 0.6), lw=0.8, marker="o", ms=2.8)
        ax_f1.text(1.04, values[1], str(seed), fontsize=4.3, va="center", color=color)
    mean_values = residuals.mean().to_numpy()
    ax_f1.plot((0, 1), mean_values, color="#202428", lw=1.8, marker="o", ms=4.0, zorder=5)
    ax_f1.set_xticks((0, 1), ("Stage 1", "Stage 2"))
    ax_f1.set_ylabel("teacher-forced fugacity RMSE")
    ax_f1.set_xlim(-0.18, 1.28)
    ax_f1.text(0.5, 0.98, "5/5 seeds decrease", transform=ax_f1.transAxes, ha="center", va="top", fontsize=5.1, color=palette["gain"], fontweight="bold")
    change_fields = (
        ("mean_abs_delta_log_gamma", r"$|\Delta\ln\gamma|$"),
        ("mean_abs_delta_log_psat", r"$|\Delta\ln P^{sat}|$"),
        ("mean_abs_delta_y", r"$|\Delta y|$"),
    )
    seed_changes = finetuning.groupby("seed")[[field for field, _ in change_fields]].mean()
    means = seed_changes.mean()
    errors = seed_changes.std(ddof=1)
    y_positions = np.arange(3)
    ax_f2.barh(y_positions, means.values, xerr=errors.values, color=("#7E9DB5", "#B68BB5", "#78A6A3"), height=0.55, capsize=1.7, error_kw={"elinewidth": 0.6, "capthick": 0.6})
    ax_f2.set_yticks(y_positions, [label for _, label in change_fields])
    ax_f2.set_xlabel("mean magnitude (± s.d.)")
    ax_f2.invert_yaxis()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    formats = (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    )
    for suffix, options in formats:
        path = output_stem.with_suffix(f".{suffix}")
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        figure.savefig(temporary, bbox_inches="tight", facecolor="white", **options)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in temporary.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            temporary.write_text(normalized, encoding="utf-8")
        os.replace(temporary, path)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_nature_results(
    shapley: pd.DataFrame,
    view_synergy: pd.DataFrame,
    feature_occlusion: pd.DataFrame,
    component_pairs: pd.DataFrame,
    case_studies: pd.DataFrame,
    output_stem: Path,
    molecular_attribution: pd.DataFrame | None = None,
) -> list[Path]:
    """Nature-style attribution figure with cases and optional molecular heatmaps."""

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.82,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    palette = {
        "rdkit_2d": "#315B7D",
        "unimol_v2": "#78A6A3",
        "functional_groups": "#B68BB5",
        "neutral": "#62686E",
        "accent": "#C66B4E",
        "gain": "#2E8B57",
        "loss": "#B45B5B",
        "component_1": "#315B7D",
        "component_2": "#B45B74",
        "component_3": "#237A78",
        "pair": "#8A6A45",
    }
    include_molecules = molecular_attribution is not None and not molecular_attribution.empty
    figure = plt.figure(figsize=(6.0, 6.30 if include_molecules else 5.05), facecolor="white")
    grid = figure.add_gridspec(
        3, 12,
        height_ratios=(1.48, 0.95, 1.05),
        left=0.065, right=0.975, top=0.972, bottom=0.060,
        hspace=0.45, wspace=0.84,
    )

    # a — result-only normalized view attribution; deliberately no flow diagram.
    ax = figure.add_subplot(grid[0, 0:4] if include_molecules else grid[0, 0:9])
    ax.set_title("a  Molecular-view shares", loc="left", fontweight="bold", fontsize=8.8, pad=4)
    seed_view = shapley.groupby(["seed", "output", "view"], as_index=False)["absolute_shapley_value"].mean()
    seed_view["share"] = seed_view["absolute_shapley_value"] / seed_view.groupby(["seed", "output"])["absolute_shapley_value"].transform("sum")
    shares = seed_view.groupby(["output", "view"])["share"].mean()
    views = ("rdkit_2d", "unimol_v2", "functional_groups")
    view_labels = ("RDKit descriptors", "Uni-Mol v2", "Functional groups")
    outputs = ("ge_rt", "mean_abs_log_gamma", "mean_abs_log_alpha", "mean_abs_pair_interaction")
    output_labels = (r"$G^E/RT$", r"mean $|\ln\gamma|$", r"mean $|\ln\alpha|$", r"mean $|I_{ij}|$")
    y_values = np.arange(len(outputs))
    left = np.zeros(len(outputs))
    short_view_labels = {"rdkit_2d": "R", "unimol_v2": "U", "functional_groups": "FG"}
    for view, label in zip(views, view_labels):
        values = np.asarray([100.0 * float(shares.loc[(output, view)]) for output in outputs])
        ax.barh(y_values, values, left=left, height=0.58, color=palette[view], label=label)
        for index, (start, value) in enumerate(zip(left, values)):
            annotation = (
                f"{value:.0f}% {short_view_labels[view]}"
                if index == 0
                else f"{value:.0f}%"
            )
            ax.text(start + value / 2.0, index, annotation, ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
        left += values
    # Thin seed-specific cumulative boundaries expose stability without another panel.
    for output_index, output in enumerate(outputs):
        rows = seed_view.loc[seed_view["output"].eq(output)].pivot(index="seed", columns="view", values="share")
        for seed in rows.index:
            first_boundary = 100.0 * float(rows.loc[seed, "rdkit_2d"])
            second_boundary = 100.0 * float(rows.loc[seed, "rdkit_2d"] + rows.loc[seed, "unimol_v2"])
            for boundary in (first_boundary, second_boundary):
                ax.plot((boundary, boundary), (output_index - 0.35, output_index + 0.35), color="white", lw=0.35, alpha=0.62)
    ax.set_yticks(y_values, output_labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("grouped-|Shapley| share (%)")
    ax.invert_yaxis()

    # b — compact interaction evidence retained.
    ax = figure.add_subplot(grid[0, 4:6] if include_molecules else grid[0, 9:12])
    ax.set_title("b  View synergy", loc="left", fontweight="bold", fontsize=8.8, pad=4)
    synergy_seed = view_synergy.groupby(["seed", "output", "view_i", "view_j"], as_index=False)["shapley_interaction"].mean()
    synergy_mean = synergy_seed.groupby(["output", "view_i", "view_j"])["shapley_interaction"].mean().reset_index()
    pairs = (("rdkit_2d", "unimol_v2"), ("rdkit_2d", "functional_groups"), ("unimol_v2", "functional_groups"))
    pair_labels = ("R × U", "R × FG", "U × FG")
    maximum = max(float(synergy_mean["shapley_interaction"].abs().max()), 1e-8)
    norm = mpl.colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    for y, pair in enumerate(pairs):
        for x, output in enumerate(outputs):
            row = synergy_mean.loc[
                synergy_mean["output"].eq(output)
                & synergy_mean["view_i"].eq(pair[0])
                & synergy_mean["view_j"].eq(pair[1])
            ]
            value = float(row["shapley_interaction"].iloc[0])
            ax.scatter(x, y, s=22 + 165 * abs(value) / maximum, c=[value], cmap="RdBu_r", norm=norm, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(4), (r"$G^E$", r"$|\ln\gamma|$", r"$|\ln\alpha|$", r"$|I|$"), rotation=40, ha="right")
    ax.set_yticks(range(3), ("", "", ""))
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(2.55, -0.55)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, orientation="horizontal", fraction=0.065, pad=0.38)
    colorbar.set_label("signed interaction", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.6, length=2)
    ax.text(0.5, -0.62, "rows: R/U · R/FG · U/FG", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color="#555B60")

    # c — two separate, chemically readable feature panels.
    subgrid = grid[1, 0:6].subgridspec(1, 2, wspace=0.58)
    ax_c1 = figure.add_subplot(subgrid[0, 0])
    ax_c2 = figure.add_subplot(subgrid[0, 1])
    ax_c1.set_title("d  Named chemical feature importance", loc="left", fontsize=9.0, fontweight="bold", pad=4)
    readable = {
        "NHOHCount": "NH/OH count",
        "BalabanJ": "Balaban J",
        "TPSA": "polar surface area",
        "MolLogP": "logP",
        "MaxPartialCharge": "max partial charge",
        "MolWt": "molecular weight",
    }
    fine = feature_occlusion.loc[feature_occlusion["output"].eq("mean_abs_log_alpha")]
    fine_seed = fine.groupby(["seed", "view", "feature"], as_index=False)["absolute_occlusion_delta"].mean()
    fine_summary = fine_seed.groupby(["view", "feature"])["absolute_occlusion_delta"].agg(["mean", "std"]).reset_index()
    for axis, view, title in (
        (ax_c1, "rdkit_2d", "RDKit descriptors"),
        (ax_c2, "functional_groups", "SMARTS functional groups"),
    ):
        rows = fine_summary.loc[fine_summary["view"].eq(view)].nlargest(6, "mean").sort_values("mean")
        y_values = np.arange(len(rows))
        axis.hlines(y_values, 0.0, rows["mean"], color=mpl.colors.to_rgba(palette[view], 0.35), lw=1.2)
        axis.errorbar(rows["mean"], y_values, xerr=rows["std"], fmt="o", color=palette[view], ecolor=palette[view], ms=3.7, elinewidth=0.7, capsize=1.5)
        labels = [readable.get(str(value), str(value).replace("_", " ")) for value in rows["feature"]]
        axis.set_yticks(y_values, labels)
        axis.set_xlabel(r"occlusion effect on $|\ln\alpha|$")
        axis.text(0.98, 0.06, "mean ± s.d., 5 seeds", transform=axis.transAxes, ha="right", fontsize=6.6, color="#676C71")


    # d — ranked binary pair shares plus signed thermodynamic contribution.
    subgrid = grid[1, 6:12].subgridspec(
        1, 3, width_ratios=(0.18, 1.28, 0.50), wspace=0.56
    )
    ax_d1 = figure.add_subplot(subgrid[0, 1])
    ax_d2 = figure.add_subplot(subgrid[0, 2])
    ax_d1.set_title("e  Dominant binary pair contributions", loc="left", fontweight="bold", pad=4)
    binary_pairs = component_pairs.loc[
        component_pairs["record_type"].eq("pair") & component_pairs["component_count"].eq(2)
    ].copy()
    binary_pairs["absolute_ge_contribution"] = np.abs(binary_pairs["ge_contribution"])
    seed_mass = binary_pairs.groupby(["seed", "entity"], as_index=False)["absolute_ge_contribution"].sum()
    seed_mass["share"] = seed_mass["absolute_ge_contribution"] / seed_mass.groupby("seed")["absolute_ge_contribution"].transform("sum")
    share_summary = seed_mass.groupby("entity")["share"].agg(["mean", "std"]).reset_index()
    top_entities = share_summary.nlargest(5, "mean")["entity"].tolist()
    share_rows = share_summary.set_index("entity").loc[top_entities].sort_values("mean")
    y_values = np.arange(len(share_rows))
    ax_d1.barh(
        y_values, 100 * share_rows["mean"], xerr=100 * share_rows["std"],
        color="#C695AC", alpha=0.86, height=0.55, capsize=1.5,
        error_kw={"elinewidth": 0.6},
    )
    pair_family_labels = {
        "alcohol/polyol": "alcohol",
        "alkane/cycloalkane": "hydrocarbon",
        "ether/carbonyl": "ether/carbonyl",
        "sulfoxide/sulfone": "sulfoxide",
        "nitrogen-containing": "N-containing",
        "carboxylic acid/ester": "acid/ester",
        "halogenated": "halogenated",
        "aromatic": "aromatic",
    }
    display_pair_labels = []
    for entity in share_rows.index:
        first, second = str(entity).split(" × ")
        display_pair_labels.append(
            f"{pair_family_labels.get(first, first)}\n× {pair_family_labels.get(second, second)}"
        )
    ax_d1.set_yticks(y_values, display_pair_labels, fontsize=6.8)
    ax_d1.set_xlabel("absolute pair share (%)")
    signed_seed = binary_pairs.groupby(["seed", "entity"], as_index=False)["ge_contribution"].mean()
    signed = signed_seed.groupby("entity")["ge_contribution"].agg(["mean", "std"]).reindex(share_rows.index)
    colors = [palette["gain"] if value >= 0 else palette["loss"] for value in signed["mean"]]
    ax_d2.axvline(0.0, color="#777D82", lw=0.6)
    for y, (_, row), color in zip(y_values, signed.iterrows(), colors):
        ax_d2.errorbar(
            row["mean"], y, xerr=row["std"], fmt="o", color=color,
            ecolor=color, ms=3.3, elinewidth=0.7, capsize=1.4,
        )
    ax_d2.set_yticks([])
    ax_d2.set_xlabel("signed")
    ax_d1.text(
        0.98, 0.035,
        f"{binary_pairs['system_id'].nunique()} systems; mean ± s.d.",
        transform=ax_d1.transAxes, ha="right", fontsize=6.6, color="#676C71",
    )

    # e — three explicit, named held-out chemical systems.
    subgrid = grid[2, :].subgridspec(1, 3, wspace=0.34)
    axes_e = [figure.add_subplot(subgrid[0, index]) for index in range(3)]
    case_order = ("near_ideal_binary", "hbond_binary", "nonideal_binary")
    case_titles = ("near-ideal binary", "H-bond-rich binary", "strongly nonideal binary")
    axes_e[0].set_title("f  Explicit held-out chemical case studies", loc="left", fontweight="bold", pad=4)
    for axis, case, case_title in zip(axes_e, case_order, case_titles):
        rows = case_studies.loc[case_studies["case"].eq(case)].sort_values("path_fraction")
        if rows.empty:
            raise RuntimeError(f"Missing explicit case study: {case}")
        names = str(rows["component_names"].iloc[0]).split(" | ")
        smiles_labels = str(rows["component_smiles"].iloc[0]).split(" | ")
        readable_names = [name if name.isascii() else smiles for name, smiles in zip(names, smiles_labels)]
        display_names = [name if len(name) <= 18 else name[:16] + "…" for name in readable_names]
        path_component = int(rows["path_component"].iloc[0])
        axis.text(0.02, 0.98, case_title, transform=axis.transAxes, va="top", fontsize=8.0, fontweight="bold", color="#34393E")
        axis.text(0.02, 0.88, " + ".join(display_names), transform=axis.transAxes, va="top", fontsize=6.8, color="#555B60")
        if int(rows["component_count"].iloc[0]) == 2:
            axis.plot(rows["path_fraction"], rows["log_gamma_1"], color=palette["component_1"], lw=1.3, label=r"$\ln\gamma_1$")
            axis.plot(rows["path_fraction"], rows["log_gamma_2"], color=palette["component_2"], lw=1.3, label=r"$\ln\gamma_2$")
            axis.plot(rows["path_fraction"], rows["pair_contribution_12"], color=palette["pair"], lw=1.1, ls="--", label=r"$x_1x_2I_{12}$")
            axis.set_xlabel(r"liquid fraction $x_1$")
        else:
            for field, label, color in (
                ("pair_contribution_12", r"$x_1x_2I_{12}$", palette["component_1"]),
                ("pair_contribution_13", r"$x_1x_3I_{13}$", palette["component_2"]),
                ("pair_contribution_23", r"$x_2x_3I_{23}$", palette["component_3"]),
            ):
                axis.plot(rows["path_fraction"], rows[field], color=color, lw=1.3, label=label)
            axis.set_xlabel(r"third fraction $x_3$")
        axis.axhline(0.0, color="#8A8D90", lw=0.5)
        axis.text(0.98, 0.03, f"T={float(rows['temperature_k'].iloc[0]):.1f} K\nP={float(rows['pressure_kpa'].iloc[0]):.1f} kPa", transform=axis.transAxes, ha="right", va="bottom", fontsize=6.6, color="#676C71")
        axis.legend(loc="lower left", fontsize=7.0, handlelength=1.5)
        axis.set_ylabel("dimensionless response" if axis is axes_e[0] else "")

    # f — atom-resolved SMARTS-view attribution, drawn on this same canvas.
    if include_molecules:
        from io import BytesIO
        from PIL import Image
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D

        molecular_grid = grid[0, 6:12].subgridspec(3, 2, height_ratios=(0.20, 1.0, 1.0), hspace=0.12, wspace=0.12)
        title_axis = figure.add_subplot(molecular_grid[0, :])
        title_axis.axis("off")
        title_axis.set_title(r"c  Molecular attribution, mean $|\ln\alpha|$", loc="left", fontsize=8.8, fontweight="bold", pad=0)
        molecular_axes = [
            figure.add_subplot(molecular_grid[row + 1, column])
            for row in range(2)
            for column in range(2)
        ]
        candidates = []
        for key, group in molecular_attribution.groupby(["case", "component_index"], sort=False):
            total_abs = float(group["signed_atom_contribution"].abs().sum())
            active_atoms = int((group["signed_atom_contribution"].abs() > 1e-12).sum())
            if active_atoms > 0:
                candidates.append((key, group, total_abs, active_atoms))
        candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
        selected_candidates = candidates[:4]
        molecules = [(key, group) for key, group, _, _ in selected_candidates]
        selected_rows = pd.concat([group for _, group in molecules], ignore_index=True)
        limit = max(float(selected_rows["signed_atom_contribution"].abs().max()), 1e-8)
        norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = mpl.colormaps["RdBu_r"]
        case_labels = {
            "near_ideal_binary": "Near-ideal binary",
            "hbond_binary": "Hydrogen-bond-rich binary",
            "nonideal_binary": "Strongly nonideal binary",
        }
        case_short_labels = {
            "near_ideal_binary": "near-ideal",
            "hbond_binary": "H-bond-rich",
            "nonideal_binary": "nonideal",
        }
        for axis, ((case, component), group) in zip(molecular_axes, molecules):
            group = group.sort_values("atom_index")
            molecule = Chem.MolFromSmiles(str(group["smiles"].iloc[0]))
            if molecule is None:
                raise ValueError(f"Cannot parse heatmap SMILES: {group['smiles'].iloc[0]}")
            scores = dict(zip(group["atom_index"].astype(int), group["signed_atom_contribution"].astype(float)))
            active_atoms = {index: value for index, value in scores.items() if abs(value) > 1e-12}
            atom_colors = {index: tuple(float(channel) for channel in cmap(norm(value))[:3]) for index, value in active_atoms.items()}
            bond_scores = {
                bond.GetIdx(): 0.5 * (scores.get(bond.GetBeginAtomIdx(), 0.0) + scores.get(bond.GetEndAtomIdx(), 0.0))
                for bond in molecule.GetBonds()
                if bond.GetBeginAtomIdx() in active_atoms or bond.GetEndAtomIdx() in active_atoms
            }
            bond_colors = {index: tuple(float(channel) for channel in cmap(norm(value))[:3]) for index, value in bond_scores.items()}
            drawer = rdMolDraw2D.MolDraw2DCairo(720, 430)
            drawer.drawOptions().fillHighlights = True
            drawer.DrawMolecule(
                molecule,
                highlightAtoms=list(active_atoms),
                highlightBonds=list(bond_scores),
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            molecular_image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
            from PIL import ImageChops
            difference = ImageChops.difference(molecular_image, Image.new("RGB", molecular_image.size, "white")).convert("L")
            crop_box = difference.point(lambda value: 255 if value > 8 else 0).getbbox()
            if crop_box is not None:
                left, top, right, bottom = crop_box
                margin = 14
                molecular_image = molecular_image.crop((max(0, left - margin), max(0, top - margin), min(molecular_image.width, right + margin), min(molecular_image.height, bottom + margin)))
            axis.imshow(molecular_image)
            raw_name = str(group["component_name"].iloc[0])
            name = raw_name if raw_name.isascii() else rdMolDescriptors.CalcMolFormula(molecule)
            total_abs = float(group["signed_atom_contribution"].abs().sum())
            axis.set_title(
                f"{case_short_labels.get(str(case), str(case))} · {name}"
                f"\n$\Sigma |a_{{atom}}|$ = {total_abs:.3f}",
                fontsize=6.2, pad=1,
            )
            axis.axis("off")
        for axis in molecular_axes[len(molecules):]:
            axis.axis("off")
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=molecular_axes,
            orientation="horizontal", fraction=0.032, pad=0.025, aspect=48,
        )
        colorbar.set_label(r"signed local contribution to mean $|\ln\alpha|$", fontsize=7.2)
        colorbar.ax.tick_params(labelsize=6.5, length=2)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, options in (("png", {"dpi": 400}),):
        path = output_stem.with_suffix(f".{suffix}")
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        figure.savefig(temporary, bbox_inches="tight", facecolor="white", **options)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in temporary.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            temporary.write_text(normalized, encoding="utf-8")
        os.replace(temporary, path)
        paths.append(path)
    plt.close(figure)
    return paths


def _combine_interpretability_figures(
    main_png: Path,
    heatmap_png: Path,
    output_stem: Path,
) -> list[Path]:
    """Stack the attribution and molecular heatmap panels on one white canvas."""
    from PIL import Image, ImageChops

    def crop_white(image: Image.Image, margin: int = 28) -> Image.Image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        difference = ImageChops.difference(rgb, background).convert("L")
        difference = difference.point(lambda value: 255 if value > 8 else 0)
        box = difference.getbbox()
        if box is None:
            return rgb
        left, top, right, bottom = box
        return rgb.crop((max(0, left - margin), max(0, top - margin), min(rgb.width, right + margin), min(rgb.height, bottom + margin)))

    main = crop_white(Image.open(main_png))
    heatmap = crop_white(Image.open(heatmap_png))
    target_width = max(main.width, heatmap.width)
    if main.width != target_width:
        main = main.resize((target_width, round(main.height * target_width / main.width)), Image.Resampling.LANCZOS)
    if heatmap.width != target_width:
        heatmap = heatmap.resize((target_width, round(heatmap.height * target_width / heatmap.width)), Image.Resampling.LANCZOS)
    gap = 54
    border = 48
    canvas = Image.new("RGB", (target_width + 2 * border, main.height + heatmap.height + gap + 2 * border), "white")
    canvas.paste(main, (border, border))
    separator_y = border + main.height + gap // 2
    from PIL import ImageDraw
    ImageDraw.Draw(canvas).line((border, separator_y, border + target_width, separator_y), fill=(210, 210, 210), width=2)
    canvas.paste(heatmap, (border, border + main.height + gap))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".tiff"), output_stem.with_suffix(".pdf")]
    canvas.save(paths[0], dpi=(400, 400), optimize=True)
    canvas.save(paths[1], dpi=(600, 600), compression="tiff_lzw")
    canvas.save(paths[2], resolution=400.0)
    return paths


def _mean_std(values: pd.Series) -> str:
    return f"{float(values.mean()):.4g} ± {float(values.std(ddof=1)):.3g}"


def _report(
    path: Path,
    shapley: pd.DataFrame,
    interactions: pd.DataFrame,
    sensitivities: pd.DataFrame,
    selected_stages: dict[int, str],
) -> None:
    seed_importance = shapley.groupby(["seed", "output", "view"], as_index=False)["absolute_shapley_value"].mean()
    global_importance = seed_importance.groupby(["output", "view"], as_index=False)["absolute_shapley_value"].agg(["mean", "std"]).reset_index()
    global_importance["share"] = global_importance["mean"] / global_importance.groupby("output")["mean"].transform("sum")
    correlations = [
        {"seed": seed, "rho": rows["absolute_pair_interaction"].corr(rows["mean_abs_log_gamma"], method="spearman")}
        for seed, rows in interactions.groupby("seed")
    ]
    correlation_frame = pd.DataFrame(correlations)
    max_additivity = float(shapley["additivity_error"].max())
    lines = [
        "# C1 ThermoFormer interpretability", "",
        "The analysis uses the final C1 RDKit + Uni-Mol v2 + functional-group vanilla Transformer and the validation-selected checkpoint for each seed on the `overall_binary_ternary` test set.", "",
        "Selected checkpoints: " + ", ".join(f"seed {seed}: {stage}" for seed, stage in sorted(selected_stages.items())) + ".",
        "Grouped Shapley uses each seed's training-set view mean as background. It explains thermodynamic outputs at observed T, P, and x states and is therefore a teacher-forced model attribution.", "",
        "## Molecular-view contributions", "",
        "| Explained output | Molecular view | Mean | SD | Normalized share |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in global_importance.iterrows():
        lines.append(f"| {OUTPUT_LABELS[str(row['output'])]} | {VIEW_LABELS[str(row['view'])]} | {float(row['mean']):.5g} | {float(row['std']):.4g} | {100.0 * float(row['share']):.1f}% |")
    top_views = {output: VIEW_LABELS[str(rows.loc[rows["mean"].idxmax(), "view"])] for output, rows in global_importance.groupby("output")}
    lines += ["", "Largest mean absolute Shapley view by output: " + "; ".join(f"{OUTPUT_LABELS[name]}: **{view}**" for name, view in top_views.items()) + ". Attribution values do not establish causal chemical mechanisms.", f"Maximum exact three-group Shapley additivity error: `{max_additivity:.3g}`.", "", "## Pair interaction and thermodynamic sensitivity", "", f"The seed-wise Spearman correlation between absolute pair potential and mean absolute ln(gamma) is **{_mean_std(correlation_frame['rho'])}**. Pair potential is an internal decoder interaction rather than a measured bond or interaction energy."]
    for cardinality in (2, 3):
        rows = sensitivities.loc[sensitivities["component_count"].eq(cardinality)]
        seed_norm = rows.groupby("seed")["d_log_gamma_i_d_closed_x_j"].apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
        temperature_norm = rows.groupby("seed")["d_log_gamma_i_d_temperature_k"].apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
        lines.append(f"- {cardinality}-component states: closed-composition response RMS `{_mean_std(seed_norm)}`; temperature-response RMS `{_mean_std(temperature_norm)}` K^-1.")
    lines += ["", "## Interpretation limits", "", "- The three molecular views jointly affect the thermodynamic outputs through separate projections and fusion.", "- Pair-potential correlations support decoder-level consistency and do not independently establish hydrogen-bonding, complexation, or azeotropic mechanisms.", "- Closed-simplex derivatives preserve the composition constraint.", "- Conclusions are aggregated across five validation-selected checkpoints.", ""]
    atomic_write_text(path, "\n".join(lines))

def _extended_report(
    path: Path,
    shapley: pd.DataFrame,
    view_synergy: pd.DataFrame,
    feature_occlusion: pd.DataFrame,
    component_pairs: pd.DataFrame,
    trajectories: pd.DataFrame,
    case_studies: pd.DataFrame,
    finetuning: pd.DataFrame,
    validation: pd.DataFrame,
    selected_stages: dict[int, str],
) -> None:
    seed_importance = shapley.groupby(["seed", "output", "view"], as_index=False)["absolute_shapley_value"].mean()
    seed_importance["share"] = seed_importance["absolute_shapley_value"] / seed_importance.groupby(["seed", "output"])["absolute_shapley_value"].transform("sum")
    importance = seed_importance.groupby(["output", "view"])["share"].agg(["mean", "std"]).reset_index()
    synergy = view_synergy.groupby(["seed", "output", "view_i", "view_j"], as_index=False)["shapley_interaction"].mean().groupby(["output", "view_i", "view_j"])["shapley_interaction"].agg(["mean", "std"]).reset_index()
    fine = feature_occlusion.loc[feature_occlusion["output"].eq("mean_abs_log_alpha")]
    fine = fine.groupby(["view", "feature"])["absolute_occlusion_delta"].mean().reset_index()
    binary_pairs = component_pairs.loc[component_pairs["record_type"].eq("pair") & component_pairs["component_count"].eq(2)]
    pair_mass = binary_pairs.assign(absolute_ge_contribution=np.abs(binary_pairs["ge_contribution"])).groupby("entity")["absolute_ge_contribution"].sum()
    top_pairs = (pair_mass / pair_mass.sum()).nlargest(5)
    lines = [
        "# VLE molecular interpretability",
        "",
        "This analysis uses the curated VLE dataset, the `vle_overall_binary` split, and the validation-selected three-stage ThermoFormer checkpoints for seeds 0–4. Test labels are not used for checkpoint selection or attribution sampling.",
        "",
        "Selected checkpoints: " + ", ".join(f"seed {seed}: {stage}" for seed, stage in sorted(selected_stages.items())) + ".",
        "",
        "## Molecular-view attribution",
        "",
        "| Output | Molecular view | Grouped-|Shapley| share, mean ± sample SD |",
        "|---|---|---:|",
    ]
    for _, row in importance.iterrows():
        lines.append(f"| {OUTPUT_LABELS[str(row['output'])]} | {VIEW_LABELS[str(row['view'])]} | {100*float(row['mean']):.1f}% ± {100*float(row['std']):.1f}% |")
    lines += ["", "## Cross-view interactions", "", "| Output | View pair | Signed interaction, mean ± sample SD |", "|---|---|---:|"]
    for _, row in synergy.iterrows():
        lines.append(f"| {OUTPUT_LABELS[str(row['output'])]} | {VIEW_LABELS[str(row['view_i'])]} × {VIEW_LABELS[str(row['view_j'])]} | {float(row['mean']):.4g} ± {float(row['std']):.3g} |")
    lines += ["", "## Named features", ""]
    for view, rows in fine.groupby("view"):
        top = rows.nlargest(5, "absolute_occlusion_delta")
        lines.append(f"- **{VIEW_LABELS[view]}**: " + ", ".join(f"{row.feature} ({float(row.absolute_occlusion_delta):.3g})" for row in top.itertuples()))
    lines += ["", "## Dominant binary molecular-family pairs", ""]
    lines.extend(f"- {name}: {100*float(value):.1f}%" for name, value in top_pairs.items())
    lines += [
        "",
        "## Molecular structure heatmaps",
        "",
        "The structure heatmaps use the seed-0 validation-selected checkpoint and held-out binary test cases. Scores are local signed SMARTS-view occlusion contributions to mean |ln alpha|. They are distributed over atoms matched by each functional-group pattern. Uncolored parts of a molecule have no atom-resolved SMARTS attribution. RDKit descriptors and pooled Uni-Mol vectors remain in the global view analysis because their stored molecule-level representations do not provide a defensible atom mapping.",
        "",
        "## Interpretation limits",
        "",
        "- Grouped Shapley, feature occlusion, and pair-potential terms explain model responses at observed T, P, and x.",
        "- The values do not measure causal chemical mechanisms, bond energies, or experimental interaction energies.",
        "- Correlated descriptors can share predictive information, so individual occlusion effects are not independent causal effects.",
        "- All five-seed summaries use sample standard deviation across independently initialized models.",
        "",
        "## Reproduction outputs",
        "",
        "- `experiments/reference_results/modality_shapley.csv`: grouped molecular-view attribution",
        "- `experiments/reference_results/view_synergy.csv`: exact cross-view Shapley interactions",
        "- `experiments/reference_results/feature_occlusion.csv`: named RDKit and functional-group occlusion",
        "- `experiments/reference_results/component_pair_attribution.csv`: learned pair decomposition",
        "- `experiments/reference_results/explicit_case_studies.csv`: held-out binary composition paths",
        "- `experiments/reference_results/molecular_structure_attribution.csv`: atom-level SMARTS-view source values",
        "- `figures/vle_interpretability_complete.png`: combined publication figure",
        "",
    ]
    atomic_write_text(path, "\n".join(lines))


def _write_nature_figure_documents(output_root: Path, seed_states: int) -> tuple[Path, Path, Path]:
    caption_path = output_root / "experiments/data_quality/reports/Figure_vle_interpretability_caption.md"
    heatmap_caption_path = output_root / "experiments/data_quality/reports/Figure_vle_molecular_heatmaps_caption.md"
    qa_path = output_root / "experiments/data_quality/reports/figure_qa.md"
    caption = f"""# Figure caption

**Figure | Molecular attribution and thermodynamic interpretation of ThermoFormer on the curated VLE dataset.** **a,** Five-seed normalized exact grouped-|Shapley| shares of RDKit descriptors, Uni-Mol v2, and SMARTS functional groups for four thermodynamic readouts. Thin boundaries show individual-seed shares. **b,** Exact cross-view Shapley interactions; marker area represents magnitude and color represents sign. **c,** Four representative molecules with nonzero SMARTS attribution; atom and bond colors denote signed local feature contributions. **d,** Train-mean occlusion effects for named RDKit descriptors and SMARTS functional groups. Points and error bars denote mean and sample s.d. across five seeds. **e,** Binary molecular-family pair contribution magnitudes and signed means. **f,** Three explicit binary composition paths evaluated with the seed-0 validation-selected checkpoint.

All attribution states belong to the `vle_overall_binary` test partitions and use validation-selected checkpoints. The analysis includes five independently initialized models and {seed_states:,} seed-state instances. The calculations are teacher-forced at observed T, P, and x and describe model dependence rather than causal molecular mechanisms.
"""
    heatmap_caption = """# Figure caption

**Figure | Local molecular structure attribution for held-out VLE systems.** Atom and bond colors show signed SMARTS-view occlusion contributions to mean |ln alpha| for the three explicit binary case studies. The heatmaps use the seed-0 validation-selected checkpoint. Functional-group contributions are divided equally among atoms matched by the corresponding SMARTS pattern. Uncolored regions have no atom-resolved SMARTS attribution. The color scale is shared across all molecules.
"""
    qa = f"""# Figure quality record

- Analysis protocol: `vle_overall_binary`.
- Replicates: five independently initialized seeds; {seed_states:,} seed-state instances.
- Checkpoint selection: validation partition only.
- Attribution partition: test.
- Main figure source: machine-readable CSV tables in `experiments/vle/interpretability/molecular_interactions/results/`.
- Structure heatmap source: `molecular_structure_attribution.csv`.
- Exports: editable SVG, TrueType PDF, 600-dpi LZW TIFF, and 400-dpi PNG.
- Interpretation: all values describe model dependence and are not experimental interaction energies.
"""
    atomic_write_text(caption_path, caption)
    atomic_write_text(heatmap_caption_path, heatmap_caption)
    atomic_write_text(qa_path, qa)
    return caption_path, heatmap_caption_path, qa_path


def molecular_structure_attribution_table(
    context: FinalSeedContext,
    case_studies: pd.DataFrame,
    device: torch.device,
    functional_group_definitions: Sequence[dict[str, str]],
) -> pd.DataFrame:
    """Map local SMARTS-view occlusion effects back to matched atoms.

    The ThermoFormer molecular inputs are molecule-level vectors. Atom scores are
    therefore defined only for the SMARTS branch, whose features have explicit
    substructure matches. Each score is the signed change in mean |ln alpha| when
    one component's functional-group feature is replaced by its split-training
    mean, divided equally among the atoms matched by that SMARTS pattern.
    """
    from rdkit import Chem

    slices = _view_slices(context.view_dimensions)
    block = slices["functional_groups"]
    definitions = list(functional_group_definitions)
    if block.stop - block.start != len(definitions):
        raise ValueError("Functional-group vocabulary does not match the checkpoint")
    lookup = {sample_id(sample): sample for sample in context.test}
    baseline = torch.from_numpy(context.train_baseline).to(device)
    records: list[dict[str, object]] = []
    representatives = case_studies.sort_values("path_fraction").groupby("case", sort=False).nth(0).reset_index()
    for _, case_row in representatives.iterrows():
        identifier = str(case_row["sample_id"])
        sample = lookup.get(identifier)
        if sample is None:
            raise RuntimeError(f"Case-study state is not in the seed-{context.seed} test set: {identifier}")
        dataset = VLETensorDataset([sample], context.feature_map)
        batch = collate_vle([dataset[0]]).to(device)
        with torch.no_grad():
            reference = _scalar_outputs(
                context.model(batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.x, batch.mask),
                batch.mask,
            )["mean_abs_log_alpha"][0]
        for component_index, (name, smiles) in enumerate(zip(sample.names, sample.smiles)):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Cannot parse case-study SMILES: {smiles}")
            atom_scores = np.zeros(molecule.GetNumAtoms(), dtype=float)
            atom_groups: list[list[str]] = [[] for _ in range(molecule.GetNumAtoms())]
            for feature_offset, definition in enumerate(definitions):
                pattern = Chem.MolFromSmarts(str(definition["smarts"]))
                matches = molecule.GetSubstructMatches(pattern) if pattern is not None else ()
                matched_atoms = sorted({atom for match in matches for atom in match})
                if not matched_atoms:
                    continue
                feature_index = block.start + feature_offset
                perturbed = batch.molecules.clone()
                perturbed[0, component_index, feature_index] = baseline[feature_index]
                with torch.no_grad():
                    changed = _scalar_outputs(
                        context.model(perturbed, batch.temperature_k, batch.pressure_kpa, batch.x, batch.mask),
                        batch.mask,
                    )["mean_abs_log_alpha"][0]
                contribution = float((reference - changed).cpu())
                per_atom = contribution / len(matched_atoms)
                for atom_index in matched_atoms:
                    atom_scores[atom_index] += per_atom
                    atom_groups[atom_index].append(str(definition["name"]))
            for atom_index, score in enumerate(atom_scores):
                records.append({
                    "seed": context.seed,
                    "selected_stage": context.selected_stage,
                    "case": str(case_row["case"]),
                    "sample_id": identifier,
                    "system_id": system_id(sample),
                    "component_index": component_index + 1,
                    "component_name": str(name),
                    "smiles": smiles,
                    "atom_index": atom_index,
                    "atom_symbol": molecule.GetAtomWithIdx(atom_index).GetSymbol(),
                    "signed_atom_contribution": float(score),
                    "matched_functional_groups": ";".join(sorted(set(atom_groups[atom_index]))),
                    "output": "mean_abs_log_alpha",
                    "attribution_scope": "smarts_view_local_occlusion",
                    "evaluation_partition": "test",
                })
    return pd.DataFrame(records)


def _plot_molecular_structure_heatmaps(
    atom_attribution: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    """Render publication-ready atom and bond heatmaps for held-out binary cases."""
    from io import BytesIO
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D

    molecules = list(atom_attribution.groupby(["case", "component_index"], sort=False))
    if not molecules:
        raise ValueError("No molecular structure attributions to plot")
    columns = 3
    rows_count = int(math.ceil(len(molecules) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(7.2, 2.35 * rows_count), squeeze=False)
    limit = max(float(atom_attribution["signed_atom_contribution"].abs().max()), 1e-8)
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = mpl.colormaps["RdBu_r"]
    case_labels = {
        "near_ideal_binary": "Near-ideal binary",
        "hbond_binary": "Hydrogen-bond-rich binary",
        "nonideal_binary": "Strongly nonideal binary",
    }
    for axis, ((case, component), group) in zip(axes.ravel(), molecules):
        group = group.sort_values("atom_index")
        smiles = str(group["smiles"].iloc[0])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Cannot parse heatmap SMILES: {smiles}")
        scores = dict(zip(group["atom_index"].astype(int), group["signed_atom_contribution"].astype(float)))
        active_atoms = {index: value for index, value in scores.items() if abs(value) > 1e-12}
        atom_colors = {index: tuple(float(channel) for channel in cmap(norm(value))[:3]) for index, value in active_atoms.items()}
        bond_scores = {
            bond.GetIdx(): 0.5 * (scores.get(bond.GetBeginAtomIdx(), 0.0) + scores.get(bond.GetEndAtomIdx(), 0.0))
            for bond in molecule.GetBonds()
            if bond.GetBeginAtomIdx() in active_atoms or bond.GetEndAtomIdx() in active_atoms
        }
        bond_colors = {index: tuple(float(channel) for channel in cmap(norm(value))[:3]) for index, value in bond_scores.items()}
        drawer = rdMolDraw2D.MolDraw2DCairo(720, 440)
        drawer.drawOptions().fillHighlights = True
        drawer.DrawMolecule(
            molecule,
            highlightAtoms=list(active_atoms),
            highlightBonds=list(bond_scores),
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        buffer = BytesIO(drawer.GetDrawingText())
        axis.imshow(Image.open(buffer))
        raw_name = str(group["component_name"].iloc[0])
        name = raw_name if raw_name.isascii() else rdMolDescriptors.CalcMolFormula(molecule)
        axis.set_title(f"{case_labels.get(str(case), str(case))}\ncomponent {int(component)}: {name}", fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(molecules):]:
        axis.axis("off")
    figure.suptitle(
        r"Local SMARTS-view molecular attribution for mean $|\ln\alpha|$",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.035,
        aspect=45,
    )
    colorbar.set_label(r"signed local contribution to mean $|\ln\alpha|$")
    figure.text(
        0.5,
        0.012,
        "Atom colors map functional-group feature occlusion to matched SMARTS atoms; uncolored regions have no SMARTS attribution.",
        ha="center",
        fontsize=7,
        color="#555B60",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 400}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        path = output_stem.with_suffix(f".{suffix}")
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        figure.savefig(temporary, bbox_inches="tight", facecolor="white", **options)
        os.replace(temporary, path)
        paths.append(path)
    plt.close(figure)
    return paths


def run_c1_final_interpretability(
    project_root: Path,
    *,
    device_name: str = "auto",
    max_samples_per_seed: int = 256,
    batch_size: int = 64,
    output_root: Path | None = None,
    experiment_results: Path | None = None,
) -> dict[str, object]:
    project_root = project_root.resolve()
    device = _device(device_name)
    samples = tuple(
        retain_pure_anchored_systems(
            load_vle_samples(project_root / "datasets/vle"),
            minimum_temperatures=2,
        )
    )
    shapley_frames = []
    synergy_frames = []
    interaction_frames = []
    sensitivity_frames = []
    feature_frames = []
    trajectory_frames = []
    finetuning_frames = []
    contexts = []
    rdkit_names = json.loads(
        (project_root / "datasets/molecular_features/rdkit_descriptors.json").read_text(encoding="utf-8")
    )["descriptors"]
    functional_group_definitions = json.loads(
        (project_root / "datasets/molecular_features/functional_groups.json").read_text(encoding="utf-8")
    )["groups"]
    functional_group_names = [row["name"] for row in functional_group_definitions]
    for seed in SEEDS:
        context = _load_seed_context(project_root, samples, seed, device)
        contexts.append(context)
        selected = _stratified_sample(context.test, max_samples_per_seed, seed)
        shapley, synergy = modality_analysis_tables(context, selected, device, batch_size)
        shapley_frames.append(shapley)
        synergy_frames.append(synergy)
        interaction_frames.append(interaction_table(context, selected, device, batch_size))
        sensitivity_frames.append(sensitivity_table(context, device))
        feature_frames.append(
            feature_occlusion_table(
                context,
                selected,
                device,
                batch_size,
                rdkit_names,
                functional_group_names,
            )
        )
        trajectory_frames.append(composition_trajectory_table(context, selected, device))
        finetuning_frames.append(finetuning_effect_table(context, selected, device, batch_size))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    shapley = pd.concat(shapley_frames, ignore_index=True)
    synergy = pd.concat(synergy_frames, ignore_index=True)
    interactions = pd.concat(interaction_frames, ignore_index=True)
    sensitivities = pd.concat(sensitivity_frames, ignore_index=True)
    feature_occlusion = pd.concat(feature_frames, ignore_index=True)
    component_pairs = component_pair_attribution_table(interactions)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    finetuning = pd.concat(finetuning_frames, ignore_index=True)
    case_studies = explicit_case_study_table(contexts[0], device)
    molecular_attribution = molecular_structure_attribution_table(
        contexts[0], case_studies, device, functional_group_definitions
    )
    validation = finetuning_validation_table(contexts)
    formal_output_root = project_root / "experiments/vle/interpretability/molecular_interactions"
    output_root = (output_root or formal_output_root).resolve()
    result_paths = {
        "modality_shapley": output_root / "experiments/reference_results/modality_shapley.csv",
        "view_synergy": output_root / "experiments/reference_results/view_synergy.csv",
        "feature_occlusion": output_root / "experiments/reference_results/feature_occlusion.csv",
        "component_pair_attribution": output_root / "experiments/reference_results/component_pair_attribution.csv",
        "composition_trajectories": output_root / "experiments/reference_results/composition_trajectories.csv",
        "explicit_case_studies": output_root / "experiments/reference_results/explicit_case_studies.csv",
        "finetuning_effects": output_root / "experiments/reference_results/finetuning_effects.csv",
        "finetuning_validation": output_root / "experiments/reference_results/finetuning_validation.csv",
        "pair_interactions": output_root / "experiments/reference_results/pair_interactions.csv",
        "thermodynamic_sensitivity": output_root / "experiments/reference_results/thermodynamic_sensitivity.csv",
        "molecular_structure_attribution": output_root / "experiments/reference_results/molecular_structure_attribution.csv",
    }
    for name, frame in (
        ("modality_shapley", shapley),
        ("view_synergy", synergy),
        ("feature_occlusion", feature_occlusion),
        ("component_pair_attribution", component_pairs),
        ("composition_trajectories", trajectories),
        ("explicit_case_studies", case_studies),
        ("finetuning_effects", finetuning),
        ("finetuning_validation", validation),
        ("pair_interactions", interactions),
        ("thermodynamic_sensitivity", sensitivities),
        ("molecular_structure_attribution", molecular_attribution),
    ):
        _atomic_csv(frame, result_paths[name])
    figure_paths = []
    heatmap_paths = []
    combined_figure_paths = _plot_nature_results(
        shapley,
        synergy,
        feature_occlusion,
        component_pairs,
        case_studies,
        output_root / "figures/vle_interpretability_complete",
        molecular_attribution,
    )
    report_path = output_root / "experiments/data_quality/reports/interpretability_report.md"
    selected_stages = {context.seed: context.selected_stage for context in contexts}
    _extended_report(
        report_path,
        shapley,
        synergy,
        feature_occlusion,
        component_pairs,
        trajectories,
        case_studies,
        finetuning,
        validation,
        selected_stages,
    )
    seed_state_count = len(shapley[["seed", "sample_id"]].drop_duplicates())
    caption_path, heatmap_caption_path, qa_path = _write_nature_figure_documents(
        output_root, seed_state_count
    )
    inputs = []
    for context in contexts:
        inputs.extend([
            context.checkpoint,
            context.stage1_checkpoint,
            context.stage2_checkpoint,
            context.split_path,
            project_root / FINAL_RESULT_ROOT / f"seed_{context.seed}/manifest.json",
            project_root / FINAL_RESULT_ROOT / f"seed_{context.seed}/stage_comparison.json",
        ])
    inputs.extend(sorted((project_root / "datasets/vle").glob("*.xlsx")))
    inputs.extend([
        project_root / "cache/rdkit_2d_raw24_v1.npz",
        project_root / "cache/unimolv2_84m.npz",
        project_root / "cache/functional_groups_thermoformer_v1.npz",
        project_root / "datasets/molecular_features/rdkit_descriptors.json",
        project_root / "datasets/molecular_features/functional_groups.json",
        project_root / "configs/vle/interpretability/molecular_interactions.json",
    ])
    if experiment_results is None and output_root == formal_output_root.resolve():
        experiment_results = project_root / "experiments/vle/interpretability/molecular_interactions/interpretability_report.md"
    if experiment_results is not None:
        atomic_write_text(experiment_results, report_path.read_text(encoding="utf-8"))
    artifacts = [
        *result_paths.values(),
        *figure_paths,
        *heatmap_paths,
        *combined_figure_paths,
        report_path,
        caption_path,
        heatmap_caption_path,
        qa_path,
    ]
    if experiment_results is not None:
        artifacts.append(experiment_results)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    code_paths = [
        project_root / "src/thermoformer/interpretability/vle_binary.py",
        project_root / "src/thermoformer/interpretability/core.py",
        project_root / "src/thermoformer/interpretability/selection.py",
        project_root / "scripts/benchmarks/vle/generate_interpretability.py",
    ]
    manifest = {
        "status": "completed",
        "analysis": "C1 final six-part nonredundant interpretability suite",
        "analysis_status": (
            "confirmatory_descriptive"
            if output_root == formal_output_root.resolve()
            else "diagnostic_smoke"
        ),
        "git_commit": git_commit,
        "protocol": SPLIT_PROTOCOL,
        "checkpoint_protocol": FINAL_PROTOCOL,
        "seeds": list(SEEDS),
        "selected_stages": {str(key): value for key, value in selected_stages.items()},
        "evaluation_partition": "test",
        "explanation_mode": "teacher_forced_observed_T_P_x",
        "max_samples_per_seed": max_samples_per_seed,
        "row_counts": {
            "modality_shapley": len(shapley),
            "view_synergy": len(synergy),
            "feature_occlusion": len(feature_occlusion),
            "component_pair_attribution": len(component_pairs),
            "composition_trajectories": len(trajectories),
            "explicit_case_studies": len(case_studies),
            "finetuning_effects": len(finetuning),
            "finetuning_validation": len(validation),
            "pair_interactions": len(interactions),
            "thermodynamic_sensitivity": len(sensitivities),
            "molecular_structure_attribution": len(molecular_attribution),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "inputs": {
            path.relative_to(project_root).as_posix(): artifact_sha256(path)
            for path in inputs
        },
        "analysis_code": {
            path.relative_to(project_root).as_posix(): artifact_sha256(path)
            for path in code_paths
        },
        "artifacts": {
            path.relative_to(project_root).as_posix(): artifact_sha256(path)
            for path in artifacts
        },
    }
    manifest_path = output_root / "experiments/data_quality/reports/analysis_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest": manifest_path.relative_to(project_root).as_posix()}






































