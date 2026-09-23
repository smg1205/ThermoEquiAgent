"""Audited capability declarations for machine-learning VLE baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Direction = Literal["isothermal", "isobaric"]
ImplementationSource = Literal[
    "official_architecture_reimplemented_wrapper",
    "reimplemented_from_official",
    "paper_reimplemented",
    "official_port",
    "official_pretrained",
    "official_parameter_database",
    "paper_architecture_adapted",
]


@dataclass(frozen=True)
class BaselineCapability:
    """Scientific support boundary of one published baseline."""

    key: str
    display_name: str
    citation: str
    source_url: str
    source_revision: str
    implementation_source: ImplementationSource
    native_component_counts: tuple[int, ...]
    native_directions: tuple[Direction, ...]
    temperature_mode: Literal["variable", "fixed_298k"]
    prediction_target: Literal["direct_vle", "log_gamma", "ge_over_rt"]
    external_assets_required: bool = False
    reference_trainable_parameters: int | None = None
    reference_parameter_note: str = ""
    adaptation_label: str | None = None
    notes: str = ""

    def support_reason(
        self,
        component_count: int,
        direction: Direction,
        temperature_k: float | None = None,
    ) -> str | None:
        if component_count not in self.native_component_counts:
            return f"native model does not support {component_count}-component mixtures"
        if direction not in self.native_directions:
            return f"native model does not support {direction} prediction"
        if (
            self.temperature_mode == "fixed_298k"
            and temperature_k is not None
            and abs(temperature_k - 298.15) > 0.5
        ):
            return "native model is restricted to 298.15 K"
        return None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BASELINE_CAPABILITIES: dict[str, BaselineCapability] = {
    "descriptor_ann": BaselineCapability(
        key="descriptor_ann",
        display_name="Descriptor ANN",
        citation="Sun et al., Chemical Engineering Science (2023)",
        source_url="https://github.com/sungl123456/37lamdaA",
        source_revision="f103380",
        implementation_source="official_architecture_reimplemented_wrapper",
        native_component_counts=(2,),
        native_directions=("isobaric",),
        temperature_mode="variable",
        prediction_target="direct_vle",
        external_assets_required=True,
        reference_trainable_parameters=9921,
        reference_parameter_note="per direction-specific output head; two heads total 19,842",
        notes=(
            "The executable network consumes 21 author-defined descriptors plus P and x. "
            "The author descriptor table is not distributed with an explicit reusable license."
        ),
    ),
    "smiles_rnn": BaselineCapability(
        key="smiles_rnn",
        display_name="SMILES-RNN",
        citation="Xue et al., Chemical Engineering Science (2024)",
        source_url="https://github.com/Xiyue17/SMILES-RNN",
        source_revision="d41d533",
        implementation_source="reimplemented_from_official",
        native_component_counts=(2,),
        native_directions=("isothermal", "isobaric"),
        temperature_mode="variable",
        prediction_target="direct_vle",
        adaptation_label="leakage_free_task_adapter",
        notes="Direction-specific heads are retained; the missing author dataset prevents byte-level reproduction.",
    ),
    "ualf_gnn": BaselineCapability(
        key="ualf_gnn",
        display_name="UALF-GNN",
        citation="Sun et al., AIChE Journal (2025)",
        source_url="paper_reimplementation",
        source_revision="paper_specification",
        implementation_source="paper_reimplemented",
        native_component_counts=(2,),
        native_directions=("isobaric",),
        temperature_mode="variable",
        prediction_target="direct_vle",
        notes="No official code was located; the paper architecture and heteroscedastic objective are reproduced.",
    ),
    "solvgnn": BaselineCapability(
        key="solvgnn",
        display_name="SolvGNN",
        citation="Qin et al., Digital Discovery (2023)",
        source_url="https://github.com/zavalab/ML/tree/SolvGNN",
        source_revision="66ec632",
        implementation_source="official_port",
        native_component_counts=(2, 3),
        native_directions=("isothermal",),
        temperature_mode="fixed_298k",
        prediction_target="log_gamma",
        adaptation_label="pytorch_port",
    ),
    "gdi_gnn": BaselineCapability(
        key="gdi_gnn",
        display_name="GDI-GNN",
        citation="Rittig et al., Digital Discovery (2023)",
        source_url="https://git.rwth-aachen.de/avt-svt/public/GDI-NN",
        source_revision="6383142",
        implementation_source="official_port",
        native_component_counts=(2,),
        native_directions=("isothermal",),
        temperature_mode="fixed_298k",
        prediction_target="log_gamma",
        adaptation_label="pytorch_port",
    ),
    "ge_gnn": BaselineCapability(
        key="ge_gnn",
        display_name="GE-GNN",
        citation="Rittig and Mitsos, Chemical Science (2024)",
        source_url="https://git.rwth-aachen.de/avt-svt/public/GDI-NN",
        source_revision="6383142",
        implementation_source="official_port",
        native_component_counts=(2,),
        native_directions=("isothermal",),
        temperature_mode="fixed_298k",
        prediction_target="ge_over_rt",
        adaptation_label="pytorch_port",
    ),
    "hanna": BaselineCapability(
        key="hanna",
        display_name="HANNA (current multicomponent)",
        citation="Hoffmann et al., Nature Communications (2026)",
        source_url="https://github.com/marco-hoffmann/HANNA",
        source_revision="6fe873c",
        implementation_source="official_pretrained",
        native_component_counts=(2, 3),
        native_directions=("isothermal", "isobaric"),
        temperature_mode="variable",
        prediction_target="ge_over_rt",
        external_assets_required=True,
        reference_trainable_parameters=651900,
        reference_parameter_note="ten-head ensemble; 65,190 per head; frozen ChemBERTa excluded",
        notes="Multicomponent inference is a Muggianu geometric projection of a binary-trained model.",
    ),
    "tennet_sac": BaselineCapability(
        key="tennet_sac",
        display_name="TeNNet-SAC",
        citation="Yang and Lin, Journal of Chemical Information and Modeling (2025)",
        source_url="https://github.com/yueyue2299/TeNNet-SAC",
        source_revision="2367e89",
        implementation_source="official_pretrained",
        native_component_counts=(2, 3),
        native_directions=("isothermal", "isobaric"),
        temperature_mode="variable",
        prediction_target="log_gamma",
        external_assets_required=True,
        reference_trainable_parameters=2743206,
        reference_parameter_note="profile + geometry + one segment model; frozen language encoders excluded",
        notes="Author pretrained/fine-tuned modules are required; quantum-chemistry pretraining is not repeated.",
    ),
    "spt_nrtl": BaselineCapability(
        key="spt_nrtl",
        display_name="SPT-NRTL database",
        citation="Winter et al., Fluid Phase Equilibria (2023), 113731",
        source_url="https://github.com/ClapeyronThermo/spt-nrtl-db",
        source_revision="f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e",
        implementation_source="official_parameter_database",
        native_component_counts=(2, 3),
        native_directions=("isothermal", "isobaric"),
        temperature_mode="variable",
        prediction_target="log_gamma",
        external_assets_required=True,
        reference_trainable_parameters=0,
        reference_parameter_note="fixed database lookup; no test-data fitting",
        notes=(
            "Ternary inference composes three database binary pairs with the standard "
            "multicomponent NRTL equation; rows with any missing pair are unavailable."
        ),
    ),
    "spt_nrtl_adapted": BaselineCapability(
        key="spt_nrtl_adapted",
        display_name="SPT-NRTL adapted (ThermoFormer-train)",
        citation="Winter et al., Fluid Phase Equilibria (2023), 113731",
        source_url="https://doi.org/10.1016/j.fluid.2023.113731",
        source_revision="paper_architecture_reimplemented_thermoformer_train",
        implementation_source="paper_architecture_adapted",
        native_component_counts=(2, 3),
        native_directions=("isothermal", "isobaric"),
        temperature_mode="variable",
        prediction_target="log_gamma",
        adaptation_label="ThermoFormer-train",
        notes=(
            "Binary NRTL parameter labels are fitted only from the registered training partition. "
            "A character-level molecular-pair Transformer is selected on validation labels and "
            "predicts all constituent binary interactions at test time; no author weights or "
            "confidential COSMO pretraining data are used."
        ),
    ),
}


def baseline_capability(key: str) -> BaselineCapability:
    try:
        return BASELINE_CAPABILITIES[key]
    except KeyError as error:
        raise ValueError(f"Unknown machine-learning baseline: {key}") from error
