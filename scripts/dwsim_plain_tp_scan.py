"""Minimal test: does a plain NRTL TP flash split at 283.15 K, with NO forcing?

Previous runs added ForceEquilibriumCalculationType=VLLE which made NRTL/UNIFAC-LL
fail with 'PT Flash: Invalid solution'.  Here we deliberately set NOTHING except the
property package, so the stream uses its own default TP flash, and scan several
compositions/temperatures to find any that split.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

P = 101325.0
# (label, temperature K, composition [EtOH, EtAc, water])
CASES = [
    ("T10_a", 283.15, [0.0604, 0.3793, 0.5603]),
    ("T10_b", 283.15, [0.1060, 0.2971, 0.5969]),
    ("T10_c", 283.15, [0.1279, 0.2451, 0.6271]),
    ("T10_deep", 283.15, [0.05, 0.45, 0.50]),
    ("T5_deep", 278.15, [0.05, 0.45, 0.50]),
    ("T20_deep", 293.15, [0.05, 0.45, 0.50]),
    ("T25_deep", 298.15, [0.05, 0.45, 0.50]),
]
PKGS = ["NRTL", "UNIQUAC"]


def read_phases(feed):
    out = {}
    for nm in ("Vapor", "Liquid1", "Liquid2"):
        try:
            p = feed.GetType().GetProperty(nm).GetValue(feed, None)
            out[nm] = float(p.Properties.molarfraction) if p is not None else None
        except Exception:
            out[nm] = None
    try:
        out["ids"] = list(feed.PhaseIds) if feed.PhaseIds else None
    except Exception:
        out["ids"] = None
    return out


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    print("=== plain TP flash (no forcing), scan T and composition ===")
    for pkg in PKGS:
        for label, T, z in CASES:
            fs = automation.CreateFlowsheet()
            fs.AddCompound("Ethanol")
            fs.AddCompound("Ethyl acetate")
            fs.AddCompound("Water")
            try:
                ded._add_property_package(fs, pkg)
            except Exception as e:
                print(f"  {pkg} unavailable: {type(e).__name__}")
                break
            feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
            feed = ded._simulation_object(feed_go)
            feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
            feed.SetOverallComposition(ded._composition_argument(z))
            errs = automation.CalculateFlowsheet4(fs)
            err = str(errs[0])[:45] if errs and errs.Count else None
            ph = read_phases(feed)
            l2 = ph.get("Liquid2")
            flag = "SPLIT!" if (l2 is not None and l2 > 1e-8) else "single"
            print(f"  {pkg:8s} {label:9s} T={T} z={z} -> L1={ph.get('Liquid1')} L2={l2} "
                  f"ids={ph.get('ids')} {flag}{'  ERR:'+err if err else ''}")


if __name__ == "__main__":
    main()
