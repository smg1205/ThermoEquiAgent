"""Feasibility gate for the v3 乙酸正丙酯/乙酸乙酯/DMSO ThermoFormer campaign.

Tries to load the ThermoFormer backend and run one ternary bubble prediction to
confirm the neural backend and Uni-Mol encoder work end-to-end (including any
first-time model-weight download), before the full campaign is launched.
"""
import os
import sys
import time

WORKSPACE = r"E:\PythonProject\ThermoEqui-Agent-main-3"
SRC = os.path.join(WORKSPACE, "lab_models", "ThermoFormer", "src")
CKPT = os.path.join(
    WORKSPACE, "lab_models", "ThermoFormer", "checkpoints", "multiview",
    "chemical_attention", "formal", "c0_current_vanilla.on.overall_binary_ternary",
    "seed_0", "best_model.pt",
)
CACHE = os.path.join(WORKSPACE, ".thermoformer-cache")

os.environ.setdefault("THERMOFORMER_SRC", SRC)
os.environ.setdefault("THERMOFORMER_CHECKPOINT", CKPT)
os.environ.setdefault("THERMOFORMER_FEATURE_CACHE", CACHE)
os.environ.setdefault("THERMOFORMER_USE_CUDA", "0")
sys.path.insert(0, WORKSPACE)

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions  # noqa: E402
from thermo_engine.thermoformer_backend import ThermoFormerBackend  # noqa: E402

SMILES = {
    "propyl acetate": "CCC(=O)OC",
    "ethyl acetate": "CC(=O)OCC",
    "dmso": "CS(=O)C",
}


def identities(order):
    return [
        ComponentIdentity(component_id=n, name=n, smiles=SMILES[n], aliases=[])
        for n in order
    ]


def main() -> int:
    t0 = time.time()
    backend = ThermoFormerBackend()
    # Ternary isobaric bubble at equimolar liquid, 101.325 kPa.
    order = ["propyl acetate", "ethyl acetate", "dmso"]
    req = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=identities(order),
        conditions=ThermodynamicConditions(
            pressure_kPa=101.325, liquid_composition=[1/3, 1/3, 1/3]
        ),
        model_name="ThermoFormer",
    )
    res = backend.bubble_point(req)
    print("isobaric point:", res.points[0].temperature_K, res.points[0].vapor_composition)
    print("load+encode+run seconds:", round(time.time() - t0, 1))

    # Ternary isothermal bubble at 350 K, same composition.
    req2 = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=identities(order),
        conditions=ThermodynamicConditions(
            temperature_K=350.0, liquid_composition=[1/3, 1/3, 1/3]
        ),
        model_name="ThermoFormer",
    )
    res2 = backend.bubble_point(req2)
    print("isothermal point:", res2.points[0].pressure_kPa, res2.points[0].vapor_composition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
