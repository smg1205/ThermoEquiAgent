"""Research workflow dispatch and artifact indexing. Scientific backends remain unchanged."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def load_catalog():
    records = json.loads((CATALOG/"catalog.json").read_text(encoding="utf-8"))
    entries = {}
    for record in records["experiments"]:
        path = CATALOG/record["config"]
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["id"] != record["id"] or entry["id"] in entries:
            raise ValueError("Duplicate or inconsistent experiment id")
        entry["_config"] = path.relative_to(ROOT).as_posix()
        entries[entry["id"]] = entry
    return entries

def result_home(entry):
    relative = entry["id"].replace(".", "/")
    if relative.startswith("data/"):
        relative = "data_quality/" + relative[len("data/"):]
    elif relative.startswith("separation_design/"):
        relative = "vle/" + relative
    return ROOT / "experiments" / relative

def build_command(entry, args, artifact):
    if not entry["backend"]:
        raise ValueError(f'{entry["id"]}: {entry["status"]}. {entry["notes"]} Use show/index.')
    seeds = args.seeds if args.seeds is not None else entry["seeds"]
    if len(set(seeds)) != len(seeds) or not seeds or not set(seeds) <= set(range(5)):
        raise ValueError("Seeds must be a nonempty unique subset of 0–4")
    if entry["seed_mode"] == "fixed" and seeds != entry["seeds"]:
        raise ValueError("Backend uses a fixed seed schedule; it cannot be overridden here")
    command = [sys.executable, str(ROOT/entry["backend"]), *entry["args"]]
    if entry["seed_mode"] == "optional":
        command += ["--seeds", *map(str,seeds)]
    if entry["device"]:
        command += ["--device",args.device or "cuda"]
    elif args.device is not None:
        raise ValueError("Backend device is fixed by its original configuration")
    mode = entry["output_mode"]
    if mode == "campaign":
        command += ["--output",str(artifact)]
        flag,dataset = (("--dataset-vle","datasets/vle")
                        if entry["id"].startswith("vle.") else
                        ("--dataset-lle","datasets/lle"))
        command += [flag,str(ROOT/dataset)]
        if args.check_splits_only:
            command += ["--check-splits-only"]
    elif args.check_splits_only:
        raise ValueError("--check-splits-only is supported only by campaigns")
    elif mode == "artifact":
        command += ["--artifact-root",str(artifact)]
    elif mode == "ml":
        command += ["--result-root",str(artifact/'experiments/reference_results'),
                    "--checkpoint-root",str(artifact/'models/vle')]
    elif mode == "analysis":
        command += ["--analysis-root",str(artifact)]
    return command

def fingerprints(entry, command):
    files = [ROOT/entry["_config"],ROOT/"docs/reproducibility/design_reference.json",Path(__file__),ROOT/entry["backend"]]
    files += [ROOT/p for p in entry["configs"]]
    for directory in ("src","configs"):
        files += [p for p in (ROOT/directory).rglob("*")
                  if p.is_file() and p.suffix in {".py",".yaml",".json"}]
    files += list((ROOT/"scripts").glob("*.py"))
    files += list((ROOT/"datasets/derived/thermodynamic_labels").glob("*.json"))
    protocol = entry.get("protocol")
    if protocol and (ROOT/'datasets/splits/vle'/protocol).exists():
        files += list((ROOT/'datasets/splits/vle'/protocol).glob("*.json"))
    elif entry["provenance"] in {"registered_dataset", "registered_vle_dataset"}:
        files += list((ROOT/'datasets/splits/vle').rglob("seed_*.json"))
    datasets = {"vle_nist":["datasets/vle"],
                "lle_nist":["datasets/lle"], "registered_dataset":["datasets/vle_reference"], "registered_vle_dataset":["datasets/vle_reference"]}
    for directory in datasets.get(entry["provenance"],[]):
        files += list((ROOT/directory).glob("*.xlsx"))
    hashes = {p.relative_to(ROOT).as_posix():digest(p)
              for p in sorted(set(files)) if p.is_file()}
    payload = {"command":command,"input_hashes":hashes,"provenance":entry["provenance"]}
    payload["identity"] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload

def run_entry(entry, args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",args.run_id):
        raise ValueError("--run-id must be a simple version name, not a path")
    mode = "split_checks" if args.check_splits_only else "formal"
    execution = result_home(entry)/"executions"/args.run_id/mode
    artifact = execution/"artifacts"
    command = build_command(entry,args,artifact)
    print(subprocess.list2cmdline(command),flush=True)
    if entry["output_mode"] == "native":
        print("Backend-native output paths retained; see sources in show/index.",flush=True)
    if args.dry_run:
        return
    identity = fingerprints(entry,command)
    manifest = execution/"execution.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("Execution exists. Use --resume or a new --run-id")
        if previous["identity"] != identity["identity"]:
            raise ValueError("Resume rejected: code/config/data/split/command hash changed")
        if previous["status"] == "complete":
            print("Recorded invocation already completed; no backend launched.")
            return
    elif args.resume:
        raise ValueError("No execution manifest to resume")
    record = dict(identity,experiment=entry["id"],status="running",
                  started_at=datetime.now(timezone.utc).isoformat(),cwd=str(ROOT),
                  scientific_resume_policy="original backend",
                  note="Invocation completion does not certify all scientific seeds.")
    write_json(manifest,record)
    try:
        with (execution/"stdout.log").open("a",encoding="utf-8") as stream:
            process = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        record["returncode"] = process.returncode
        record["status"] = "complete" if process.returncode == 0 else "failed"
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,command)
    except BaseException:
        record["status"] = "failed_or_interrupted"
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest,record)

def metric_view(entry, home):
    source = ROOT/"experiments/summary/aggregate_metrics.csv"
    if not source.exists():
        return
    protocol = entry.get("protocol")
    diagnostic = entry["id"] == "lle.thermodynamics.stability"
    if not protocol and not diagnostic:
        return
    with source.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames
        rows = [r for r in reader if r["protocol"] == protocol or
                (diagnostic and not r["protocol"].startswith("vle_"))]
    with (home/"reference_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines = ["# Reference metric view","",
             "Copied without recalculation from the registered five-seed result table.",
             "Mean ± sample SD; valid_seeds is the original report count.",
             f"Source: {source.relative_to(ROOT).as_posix()}; SHA-256: {digest(source)}.","",
             "| Protocol | Subset | Direction | Target | Metric | Mean | Sample SD | n |",
             "|---|---|---|---|---|---:|---:|---:|"]
    lines += ["| "+" | ".join(r.get(k,"") for k in fields)+" |" for r in rows]
    (home/"reference_metrics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def index_entry(entry):
    home = result_home(entry);home.mkdir(parents=True,exist_ok=True)
    sources = []
    for relative in entry["sources"]:
        path = ROOT/relative
        sources.append({"path":relative,"exists":path.exists(),
                        "kind":"directory" if path.is_dir() else "file",
                        "sha256":digest(path) if path.is_file() else None})
    files = {}
    for src in sources:
        path = ROOT/src["path"]
        for p in (path.rglob("*") if path.is_dir() else [path]):
            if p.is_file():
                key = p.relative_to(ROOT).as_posix()
                files[key] = {"path":key,"bytes":p.stat().st_size}
    write_json(home/"sources.json",{"experiment":entry["id"],
        "provenance":entry["provenance"],"status":entry["status"],"notes":entry["notes"],
        "sources":sources,"files":list(files.values()),
        "note":"Existence-only inventory; does not certify metrics or completed seed counts."})
    lines = ["# "+entry["title"],"",entry["section"],"",
             f"Status: {entry['status']} · Data: {entry['provenance']}","",
             entry["notes"],"","## Original artifacts",""]
    for s in sources:
        p = ROOT/s["path"]
        lines.append(f"- [{'available' if s['exists'] else 'missing'}: {s['path']}]({p.as_posix()})")
    lines += ["","New executions: executions/<run-id>/<formal|split_checks>/.",
              "Source artifacts remain at their registered paths. Compare only matching data identities."]
    (home/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    metric_view(entry,home)

def verify(entries):
    errors = []
    for e in entries.values():
        if e["id"].startswith(("lle.comparison.","lle.ablation.","lle.interpretability.")):
            errors.append(e["id"]+": invalid LLE scope")
        for p in ([e["backend"]] if e["backend"] else [])+e["configs"]:
            if not (ROOT/p).is_file():
                errors.append(e["id"]+": missing "+p)
        if e["status"] == "runnable" and not e["backend"]:
            errors.append(e["id"]+": missing backend")
    before = ROOT/"docs/reproducibility/scientific_integrity.json"
    protected = json.loads(before.read_text(encoding="utf-8")) if before.exists() else {}
    changed = [p for p,h in protected.items() if not (ROOT/p).is_file() or digest(ROOT/p) != h]
    report = {"entries":len(entries),"errors":errors,
              "protected_files_checked":len(protected),"changed_protected_files":changed,
              "scope":"Structure/protected-content check; not performance validation."}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors or changed:
        raise ValueError("Benchmark catalog verification failed")
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    sub.add_parser("list");sub.add_parser("verify")
    show = sub.add_parser("show");show.add_argument("experiment")
    idx = sub.add_parser("index");idx.add_argument("experiment",nargs="?",default="all")
    run = sub.add_parser("run");run.add_argument("experiment")
    run.add_argument("--seeds",nargs="+",type=int)
    run.add_argument("--device",choices=("cuda","cpu","auto"),default=None)
    run.add_argument("--run-id",default="v1")
    run.add_argument("--dry-run",action="store_true")
    run.add_argument("--resume",action="store_true")
    run.add_argument("--check-splits-only",action="store_true")
    args = parser.parse_args(argv);entries = load_catalog()
    if args.action == "list":
        for e in entries.values():
            print(f'{e["id"]:55} {e["status"]:14} {e["provenance"]}')
    elif args.action == "verify":
        verify(entries)
    elif args.action == "index":
        for e in (entries.values() if args.experiment=="all" else [entries[args.experiment]]):
            index_entry(e)
        print("Benchmark result indexes refreshed. Scientific artifacts were not moved.")
    elif args.action == "show":
        print(json.dumps(entries[args.experiment],ensure_ascii=False,indent=2))
    elif args.action == "run":
        run_entry(entries[args.experiment],args)


