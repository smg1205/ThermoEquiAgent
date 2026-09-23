"""Explicit engineering-unit normalization helpers."""

from typing import Literal


def pressure_to_kpa(value: float, unit: Literal["Pa", "kPa", "bar", "MPa", "atm"]) -> float:
    factors = {"Pa": 0.001, "kPa": 1.0, "bar": 100.0, "MPa": 1000.0, "atm": 101.325}
    converted = value * factors[unit]
    if converted <= 0:
        raise ValueError("Pressure must be positive")
    return converted


def temperature_to_kelvin(value: float, unit: Literal["K", "C", "F"]) -> float:
    converted = {"K": value, "C": value + 273.15, "F": (value - 32.0) * 5.0 / 9.0 + 273.15}[unit]
    if converted <= 0:
        raise ValueError("Temperature must be above absolute zero")
    return converted


def normalize_composition(values: list[float]) -> list[float]:
    if not values or any(value < 0 for value in values):
        raise ValueError("Composition values must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("Composition total must be positive")
    return [value / total for value in values]
