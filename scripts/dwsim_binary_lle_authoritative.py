"""Authoritative MIBK/water binary LLE check via MaterialStream.Liquid1/Liquid2.

Reads DWSIM's own phase objects (the same data the GUI's "Fraction" row and
"Compound Amounts" grid show), NOT the raw Flash_PT tuple, so the split/mono
judgement matches what a user sees in the DWSIM UI.

For each temperature we:
  1. build a Feed stream at x_MIBK=0.4,
  2. run a UNIQUAC flash (the stream's own flash, via CalculateFlowsheet2),
  3. read Liquid1 / Liquid2 phase fractions + compositions,
  4. report whether TWO liquid phases exist (i.e. Liquid2 fraction > ~0).

All numbers come from DWSIM's phase objects.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded


def _phase_fraction(phase) -> float:
    """Total molar fraction of a phase object from its PhaseProperties."""
    props = None
    try:
        props = phase.Properties
    except Exception:
        pass
    if props is None:
        try:
            props = phase.Properties1
        except Exception:
            pass
    # PhaseProperties exposes MolarFraction / MoleFraction (probe several names).
    for name in ("MolarFraction", "MoleFraction", "MolarFlow", "MassFraction"):
        try:
            v = getattr(getattr(props, name), None)
        except Exception:
            continue
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        if v is not None:
            try:
                if hasattr(v, "GetValue"):
                    v = v.GetValue(phase, None) if hasattr(v, "GetParameters") else v
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                continue
    # fall back: phase fractions stored on the phase's Properties fraction fields
    try:
        for attr in dir(props):
            low = attr.lower()
            if "fraction" in low and "mass" not in low:
                try:
                    return float(getattr(props, attr))
                except Exception:
                    continue
    except Exception:
        pass
    return float("nan")


def _phase_composition(phase) -> list[float]:
    try:
        compounds = phase.Compounds
        if compounds is None:
            return []
        vals = []
        for k in list(compounds.Keys):
            obj = compounds[k]
            # compound amount dict: values expose MoleFraction
            for attr in ("MoleFraction", "MolarFraction", "MassFraction"):
                try:
                    v = getattr(obj, attr)
                except Exception:
                    continue
                if callable(v):
                    try:
                        v = v()
                    except Exception:
                        continue
                if v is not None:
                    try:
                        vals.append(float(v))
                        break
                    except Exception:
                        continue
        return vals
    except Exception:
        return []


def _run(t_k):
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand)
            break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed_MIBK_Water")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(t_k)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument([0.4, 0.6]))

    automation.CalculateFlowsheet2(fs)

    print(f"\n===== T = {t_k} K =====  overall x_MIBK = 0.4, x_water = 0.6")

    # read Liquid1 / Liquid2 authoritative phase objects
    for pname in ("Liquid1", "Liquid2", "Vapor", "OverallLiquid"):
        try:
            phase = feed.GetType().GetProperty(pname).GetValue(feed, None)
        except Exception as e:
            print(f"  {pname}: read err {type(e).__name__}")
            continue
        if phase is None:
            print(f"  {pname}: None (phase absent)")
            continue
        frac = _phase_fraction(phase)
        comp = _phase_composition(phase)
        print(f"  {pname}: fraction={frac!r}  composition={comp!r}")

    # also dump PhaseIds and Phases keys
    try:
        print("  PhaseIds:", list(feed.PhaseIds) if feed.PhaseIds else None)
        print("  Phases keys:", list(feed.Phases.Keys) if hasattr(feed.Phases, "Keys") else None)
    except Exception as e:
        print("  PhaseIds/Phases err:", type(e).__name__)


def main():
    _run(333.15)
    _run(298.15)


if __name__ == "__main__":
    main()
