"""Audited machine-learning baselines for VLE predictive performance."""

from .models import DescriptorANN, GDIGNN, GEGNN, SMILESRNN, SolvGNN, UALFGNN
from .protocols import OVERALL_BENCHMARKS, UNIFIED_METRIC_KEYS, benchmark_for_baseline, registered_assignment
from .formal import FormalConfig, aggregate_formal_baseline, run_formal_seed
from .formal_reporting import write_formal_report
from .reporting import write_smoke_report
from .artifacts import aggregate_seed_metrics, write_seed_metrics
from .upstream import PretrainedOverlapAudit, audit_pretrained_overlap
from .schema import BASELINE_CAPABILITIES, BaselineCapability, baseline_capability
from .smoke import SmokeConfig, run_smoke_suite

__all__ = [
    "BASELINE_CAPABILITIES",
    "BaselineCapability",
    "DescriptorANN",
    "FormalConfig",
    "GDIGNN",
    "GEGNN",
    "OVERALL_BENCHMARKS",
    "PretrainedOverlapAudit",
    "SMILESRNN",
    "SmokeConfig",
    "SolvGNN",
    "UALFGNN",
    "UNIFIED_METRIC_KEYS",
    "aggregate_seed_metrics",
    "audit_pretrained_overlap",
    "aggregate_formal_baseline",
    "baseline_capability",
    "benchmark_for_baseline",
    "registered_assignment",
    "run_formal_seed",
    "run_smoke_suite",
    "write_smoke_report",
    "write_seed_metrics",
    "write_formal_report",
]
