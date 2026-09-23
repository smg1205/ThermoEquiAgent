"""Find the FlashSettings key that makes the DEFAULT stream flash (UniversalFlash)
use an immiscible/multi-liquid kernel, so the native extractor's stage flashes work.

We scan candidate FlashSettings values and run a plain TP-flash on a MaterialStream
(the same path the column uses), checking whether two liquid phases appear.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]


def stream_phases(automation, object_type, settings_patch):
    """Build a feed + TP flash via the normal stream Calculate path; report phases."""
    from DWSIM.Interfaces.Enums import FlashSetting
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    for p in ("UNIFAC-LL", "UNIFAC LL", "UNIFACLL"):
        try:
            ded._add_property_package(fs, p); break
        except Exception:
            continue
    pp = list(fs.PropertyPackages.Values)[0]
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    for k, v in settings_patch.items():
        try:
            s[getattr(FlashSetting, k)] = v
        except Exception:
            pass
    fsp.SetValue(pp, s, None)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        return f"ERR {str(e[0])[:60]}"
    # read liquid1 / liquid2 molar fractions
    def ph(name):
        try:
            p = feed.GetType().GetProperty(name).GetValue(feed, None)
            if p is None:
                return None
            props = p.Properties
            return float(props.molarfraction)
        except Exception:
            return None
    l1 = ph("Liquid1"); l2 = ph("Liquid2")
    ids = None
    try:
        ids = list(feed.PhaseIds) if feed.PhaseIds else None
    except Exception:
        pass
    return f"Liquid1={l1} Liquid2={l2} PhaseIds={ids}"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()

    trials = {
        "baseline": {},
        "ForceEq=GibbsMinimization": {"ForceEquilibriumCalculationType": "GibbsMinimization"},
        "ForceEq=NestedLoops3P": {"ForceEquilibriumCalculationType": "NestedLoops3P"},
        "ForceEq=NestedLoopsImmiscible": {"ForceEquilibriumCalculationType": "NestedLoopsImmiscible"},
        "ForceEq=Immiscible": {"ForceEquilibriumCalculationType": "Immiscible"},
        "Replace_PTFlash=True": {"Replace_PTFlash": "True"},
        "immiscible+3phase": {"ImmiscibleWaterOption": "True", "ThreePhaseFlashStabTestSeverity": "3"},
        "immiscible+3phase+phaseid": {"ImmiscibleWaterOption": "True",
                                      "ThreePhaseFlashStabTestSeverity": "3",
                                      "UsePhaseIdentificationAlgorithm": "True"},
    }
    for label, patch in trials.items():
        try:
            r = stream_phases(automation, object_type, patch)
        except Exception as e:
            r = f"EXC {type(e).__name__}: {str(e)[:60]}"
        print(f"{label:34s} -> {r}")


if __name__ == "__main__":
    main()
