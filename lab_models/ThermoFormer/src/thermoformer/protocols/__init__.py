"""Registered generalization protocols and formal experiment execution."""

from .registry import PAPER_SEEDS, PROTOCOL_CONFIGS
from .runner import result_protocol_name, run_paper_experiment

__all__ = ["PAPER_SEEDS", "PROTOCOL_CONFIGS", "result_protocol_name", "run_paper_experiment"]
