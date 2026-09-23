# Scientific experiment catalog

| Experiment | Category | Status / data identity | Backend | Configuration |
|---|---|---|---|---|
| vle.prediction.binary | VLE / Predictive benchmarks | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/prediction/binary.json |
| vle.prediction.mixed | VLE / Predictive benchmarks | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/prediction/mixed.json |
| vle.prediction.ternary | VLE / Predictive benchmarks | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/prediction/ternary.json |
| vle.generalization.composition_interpolation | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/composition_interpolation.json |
| vle.generalization.composition_edge | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/composition_edge.json |
| vle.generalization.temperature_low | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/temperature_low.json |
| vle.generalization.temperature_high | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/temperature_high.json |
| vle.generalization.pressure_low | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/pressure_low.json |
| vle.generalization.pressure_high | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/pressure_high.json |
| vle.generalization.unseen_component | VLE / Generalization and extrapolation | runnable / vle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/vle/generalization/unseen_component.json |
| lle.prediction.binary_random | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/binary_random.json |
| lle.prediction.binary_system | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/binary_system.json |
| lle.prediction.ternary_random | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/ternary_random.json |
| lle.prediction.ternary_system | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/ternary_system.json |
| lle.prediction.mixed_random | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/mixed_random.json |
| lle.prediction.mixed_system | LLE / Predictive benchmarks | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/prediction/mixed_system.json |
| lle.generalization.binary_temperature_low | LLE / Generalization and extrapolation | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/generalization/binary_temperature_low.json |
| lle.generalization.binary_temperature_high | LLE / Generalization and extrapolation | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/generalization/binary_temperature_high.json |
| lle.generalization.binary_pressure_low | LLE / Generalization and extrapolation | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/generalization/binary_pressure_low.json |
| lle.generalization.binary_pressure_high | LLE / Generalization and extrapolation | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/generalization/binary_pressure_high.json |
| lle.generalization.binary_unseen_component | LLE / Generalization and extrapolation | runnable / lle_nist | scripts/run_phase_equilibrium_benchmarks.py | configs/lle/generalization/binary_unseen_component.json |
| lle.thermodynamics.stability | LLE / Equilibrium and stability diagnostics | artifact_only / lle_nist | Artifact index | configs/lle/thermodynamics/stability.json |
| data.construction.audit | DATA / Data curation and audit | artifact_only / nist_thermoml | Artifact index | configs/data/construction/audit.json |
| data.construction.distribution | DATA / Data curation and audit | runnable / registered_dataset | scripts/data/dataset_distribution/analyze_datasets.py | configs/data/construction/distribution.json |
| vle.comparison.machine_learning | VLE / Baseline comparison | runnable / registered_vle_dataset | scripts/run_ml_baselines_formal.py | configs/vle/comparison/machine_learning.json |
| vle.comparison.joint_activity | VLE / Baseline comparison | runnable / registered_vle_dataset | scripts/run_joint_activity_adapted.py | configs/vle/comparison/joint_activity.json |
| vle.comparison.ternary_activity | VLE / Baseline comparison | runnable / registered_vle_dataset | scripts/run_joint_activity_adapted.py | configs/vle/comparison/ternary_activity.json |
| vle.comparison.classical_binary | VLE / Baseline comparison | runnable / registered_vle_dataset | scripts/visualization/plot_thermodynamic_phase_diagrams.py | configs/vle/comparison/classical_binary.json |
| vle.comparison.classical_ternary | VLE / Baseline comparison | runnable / registered_vle_dataset | scripts/visualization/plot_ternary_phase_diagrams.py | configs/vle/comparison/classical_ternary.json |
| vle.ablation.molecular_representation | VLE / Controlled ablations | runnable / registered_vle_dataset | scripts/run_multiview_suite.py | configs/vle/ablation/molecular_representation.json |
| vle.ablation.interaction_architecture | VLE / Controlled ablations | runnable / registered_vle_dataset | scripts/run_chemical_attention_suite.py | configs/vle/ablation/interaction_architecture.json |
| vle.ablation.staged_validation | VLE / Controlled ablations | completed / updated_vle_dataset | scripts/evaluate_staged_thermodynamic_optimization.py | configs/vle/ablation/staged_validation.json |
| vle.interpretability.molecular_interactions | VLE / Interpretability and analysis | runnable / vle_nist | scripts/benchmarks/vle/generate_interpretability.py | configs/vle/interpretability/molecular_interactions.json |
| separation_design.thermoequi_agent.case_studies | SEPARATION_DESIGN / Agent-assisted separation design | artifact_only / registered_case_studies | Artifact index | configs/separation_design/autonomous/thermoequi_agent.json |
| vle.ablation.joint | VLE / Controlled ablations | runnable / vle_nist | scripts/run_vle_joint_ablations.py | configs/vle/ablation/joint.json |
