"""Versioned pure-component properties with explicit provenance and ranges."""

from __future__ import annotations

from dataclasses import dataclass

from schemas.domain import ComponentIdentity, FailureType
from thermo_engine.errors import ThermoEquiError


@dataclass(frozen=True)
class AntoineCorrelation:
    a: float
    b: float
    c: float
    minimum_temperature_K: float
    maximum_temperature_K: float
    source_title: str
    source_identifier: str

    def pressure_kPa(self, temperature_K: float) -> float:
        """Return saturation pressure using NIST's log10(P/bar) form."""
        return (10 ** (self.a - self.b / (temperature_K + self.c))) * 100.0

    def contains(self, temperature_K: float) -> bool:
        return self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K


@dataclass(frozen=True)
class PureComponent:
    identity: ComponentIdentity
    aliases: tuple[str, ...]
    correlations: tuple[AntoineCorrelation, ...]
    polar: bool
    hydrocarbon: bool
    association_risk: bool

    def vapor_pressure_kPa(self, temperature_K: float) -> tuple[float, bool, AntoineCorrelation]:
        within_range = [item for item in self.correlations if item.contains(temperature_K)]
        selected = (
            within_range[0]
            if within_range
            else min(
                self.correlations,
                key=lambda item: min(
                    abs(temperature_K - item.minimum_temperature_K),
                    abs(temperature_K - item.maximum_temperature_K),
                ),
            )
        )
        return selected.pressure_kPa(temperature_K), bool(within_range), selected


NIST = "NIST Chemistry WebBook, SRD 69"

COMPONENTS: tuple[PureComponent, ...] = (
    PureComponent(
        ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2", aliases=["苯"]),
        ("benzene", "苯", "71-43-2"),
        (
            AntoineCorrelation(
                4.72583,
                1660.652,
                -1.461,
                333.4,
                373.5,
                NIST,
                "https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Mask=4",
            ),
        ),
        polar=False,
        hydrocarbon=True,
        association_risk=False,
    ),
    PureComponent(
        ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3", aliases=["甲苯"]),
        ("toluene", "甲苯", "108-88-3"),
        (
            AntoineCorrelation(
                4.07827,
                1343.943,
                -53.773,
                308.52,
                384.66,
                NIST,
                "https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4",
            ),
        ),
        polar=False,
        hydrocarbon=True,
        association_risk=False,
    ),
)


def resolve_component(value: str | ComponentIdentity) -> PureComponent:
    """Resolve a canonical component without fuzzy identity invention."""
    needle = value.component_id if isinstance(value, ComponentIdentity) else value
    normalized = needle.strip().casefold()
    for component in COMPONENTS:
        if normalized in {alias.casefold() for alias in component.aliases}:
            return component
    raise ThermoEquiError(
        FailureType.MISSING_DATA,
        f"No reviewed pure-component property record for {needle!r}.",
        "Choose a component in the local registry or import reviewed pure-property data.",
        {"component": needle},
    )


def component_sources(components: list[PureComponent]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for component in components:
        for correlation in component.correlations:
            result.append(
                {
                    "component": component.identity.name,
                    "property": "Antoine vapor pressure",
                    "source_title": correlation.source_title,
                    "source_identifier": correlation.source_identifier,
                    "temperature_range_K": (f"{correlation.minimum_temperature_K}-{correlation.maximum_temperature_K}"),
                }
            )
    return result
