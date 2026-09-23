"""Constrained engineering-tool registry inspired by CAi's modular tool catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.executor import TaskExecution, calculate_task
from schemas.domain import FailureType, TaskManifest
from thermo_engine.errors import ThermoEquiError

ToolHandler = Callable[[TaskManifest], TaskExecution]


@dataclass(frozen=True)
class EngineeringTool:
    name: str
    description: str
    handler: ToolHandler


class EngineeringToolRegistry:
    """Allow the orchestrator to invoke only explicitly registered domain tools."""

    def __init__(self, tools: tuple[EngineeringTool, ...]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def catalog(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]

    def execute(self, name: str, task: TaskManifest) -> TaskExecution:
        tool = self._tools.get(name)
        if tool is None:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                f"Engineering tool {name!r} is not registered.",
                "Choose a reviewed tool from the engineering tool registry.",
            )
        return tool.handler(task)


DEFAULT_TOOL_REGISTRY = EngineeringToolRegistry(
    (
        EngineeringTool(
            name="phase_equilibrium",
            description=(
                "Execute a structured non-electrolyte phase-equilibrium task through a deterministic "
                "backend and independent validation gate."
            ),
            handler=calculate_task,
        ),
    )
)
