from pathlib import Path

from scripts import run_overall_binary_three_stage_ablations as base
from scripts import run_vle_joint_ablations as campaign
from src.thermoformer.configuration import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_campaign_trains_only_unique_non_reused_variants() -> None:
    assert set(campaign.TRAINED_VARIANTS) == set(base.VARIANT_IDS) - {"c1_three_view_vanilla", "v3_functional_group_only"}


def test_every_variant_uses_release_joint_vle_protocol() -> None:
    for stage in ("stage0", "three_stage"):
        for variant in (*campaign.TRAINED_VARIANTS, "c1_three_view_vanilla"):
            config = load_experiment_config(ROOT / campaign.CONFIG_NAMESPACE / stage / f"{variant}.yaml")
            assert config.name == variant
            assert config.data.root == "datasets/vle"
            assert config.data.max_pressure_kpa == 500.0
            assert config.protocol.registered_splits == (campaign.SPLIT_PROTOCOL,)
            assert config.protocol.seeds == tuple(range(5))
            expected = "validation" if stage == "stage0" else "test"
            assert config.protocol.evaluation_partition == expected


def test_artifacts_follow_benchmark_layout() -> None:
    roots = campaign._artifact_roots(ROOT, stage="three_stage", smoke=False)
    assert roots[0] == ROOT / "experiments/vle/training_records/reference/ablation/joint/formal/three_stage"
    assert roots[1] == ROOT / "models/vle/benchmarks/vle/ablation/joint/formal/three_stage"
    assert roots[2] == ROOT / "experiments/vle/ablation/joint/formal/three_stage"


