"""Artifact provenance, aggregation, and scientific report generation."""

from .artifacts import artifact_sha256, atomic_write_json, atomic_write_text


def __getattr__(name: str):
    if name == "aggregate_protocol_results":
        from .aggregation import aggregate_protocol_results

        return aggregate_protocol_results
    if name == "write_experiment_results":
        from .experiment_results import write_experiment_results

        return write_experiment_results
    raise AttributeError(name)

__all__ = [
    "aggregate_protocol_results",
    "artifact_sha256",
    "atomic_write_json",
    "atomic_write_text",
    "write_experiment_results",
]
