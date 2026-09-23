"""Orchestration contracts, without training or changing scientific definitions."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.benchmarks import registry as r
from scripts.run_phase_equilibrium_benchmarks import VLE_PROTOCOLS,LLE_PROTOCOLS

def args(**overrides):
    values=dict(seeds=None,device=None,run_id="test_v1",dry_run=False,
                resume=False,check_splits_only=False)
    values.update(overrides)
    return argparse.Namespace(**values)

class BenchmarkRegistryTests(unittest.TestCase):
    def setUp(self):
        self.entries=r.load_catalog()

    def test_campaign_matrix_is_exactly_original(self):
        vle={e["protocol"] for e in self.entries.values() if e["provenance"]=="vle_nist"}
        lle={e["protocol"] for e in self.entries.values() if e["provenance"]=="lle_nist" and e["protocol"]}
        self.assertEqual(vle,set(VLE_PROTOCOLS))
        self.assertEqual(lle,set(LLE_PROTOCOLS))

    def test_lle_scope_has_no_extra_studies(self):
        for key in self.entries:
            if key.startswith("lle."):
                self.assertIn(key.split(".")[1],{"prediction","generalization","thermodynamics"})

    def test_every_backend_and_config_exists(self):
        for entry in self.entries.values():
            for p in ([entry["backend"]] if entry["backend"] else [])+entry["configs"]:
                self.assertTrue((r.ROOT/p).is_file(),p)

    def test_missing_scientific_implementation_cannot_launch(self):
        for e in self.entries.values():
            if not e["backend"]:
                with self.assertRaises(ValueError):
                    r.build_command(e,args(),Path("unused"))

    def test_commands_preserve_all_protocols_and_frozen_data(self):
        for e in self.entries.values():
            if e["output_mode"]!="campaign":continue
            command=r.build_command(e,args(seeds=[1,3]),Path("outputs"))
            self.assertIn(e["protocol"],command)
            self.assertIn("--output",command)
            self.assertEqual(command[command.index("--seeds")+1:command.index("--device")],["1","3"])
            self.assertEqual(command[-1],str(r.ROOT/("datasets/vle" if e["id"].startswith("vle.") else "datasets/lle")))
            for forbidden in ("--lr","--epochs","--batch-size","--smoke"):
                self.assertNotIn(forbidden,command)

    def test_rejects_invalid_seed_or_unsupported_options(self):
        e=self.entries["vle.prediction.binary"]
        for seeds in ([0,0],[5],[]):
            with self.assertRaises(ValueError):r.build_command(e,args(seeds=seeds),Path("unused"))
        fixed=self.entries["vle.ablation.molecular_representation"]
        with self.assertRaises(ValueError):r.build_command(fixed,args(seeds=[0]),Path("unused"))
        with self.assertRaises(ValueError):r.build_command(fixed,args(check_splits_only=True),Path("unused"))
        with self.assertRaises(ValueError):r.build_command(self.entries["vle.comparison.machine_learning"],args(device="cpu"),Path("unused"))

    def test_dry_run_has_no_files_or_subprocess(self):
        e=self.entries["vle.prediction.binary"]
        with tempfile.TemporaryDirectory() as temp,patch.object(r,"ROOT",Path(temp)),patch.object(r.subprocess,"run") as call,contextlib.redirect_stdout(io.StringIO()):
            r.run_entry(e,args(dry_run=True))
            self.assertEqual(list(Path(temp).iterdir()),[])
            call.assert_not_called()

    def test_completed_resume_is_noop_and_changed_identity_rejected(self):
        e=self.entries["vle.prediction.binary"]
        with tempfile.TemporaryDirectory() as temp,patch.object(r,"ROOT",Path(temp)),patch.object(r,"fingerprints",return_value={"identity":"A"}) as fp,patch.object(r.subprocess,"run") as call,contextlib.redirect_stdout(io.StringIO()):
            call.return_value.returncode=0
            r.run_entry(e,args())
            r.run_entry(e,args(resume=True))
            self.assertEqual(call.call_count,1)
            with self.assertRaises(ValueError):r.run_entry(e,args())
            fp.return_value={"identity":"B"}
            with self.assertRaises(ValueError):r.run_entry(e,args(resume=True))

    def test_run_id_cannot_escape_experiment_directory(self):
        with self.assertRaises(ValueError):
            r.run_entry(self.entries["vle.prediction.binary"],args(run_id="../other",dry_run=True))

if __name__=="__main__":
    unittest.main()
