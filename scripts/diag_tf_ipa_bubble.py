"""Diagnose the ThermoFormer isobaric bubble prediction for IPA/water at x_IPA=0.5."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from schemas.domain import (  # noqa: E402
    ComponentIdentity,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.thermoformer_backend import ThermoFormerBackend  # noqa: E402


def main():
    backend = ThermoFormerBackend()
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        req = TaskManifest(
            task_id=f"diag-{x}",
            calculation_type="bubble_point",
            equilibrium_type="VLE",
            components=[
                ComponentIdentity(component_id="ipa", name="isopropanol", smiles="CC(C)O"),
                ComponentIdentity(component_id="water", name="water", smiles="O"),
            ],
            conditions=ThermodynamicConditions(
                pressure_kPa=101.325,
                liquid_composition=[x, 1.0 - x],
            ),
        )
        try:
            res = backend.bubble_point(req)
            y = res.phases[0].composition if res.phases else None
            print(f"x_IPA={x}: T={res.temperature_K:.2f} K ({res.temperature_K - 273.15:.2f} C)  "
                  f"y={y}  converged={res.converged}  residual={res.residual}")
        except Exception as exc:  # noqa: BLE001
            print(f"x_IPA={x}: FAILED {type(exc).__name__}: {str(exc)[:110]}")


if __name__ == "__main__":
    main()
