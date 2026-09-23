"""Unified resumable launcher for registered VLE and LLE benchmarks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
VLE_PROTOCOLS = (
    "vle_overall_binary",
    "vle_overall_binary_ternary",
    "vle_overall_ternary",
    "vle_binary_state_composition_interpolation",
    "vle_binary_state_composition_edge_extrapolation",
    "vle_binary_state_temperature_low_extrapolation",
    "vle_binary_state_temperature_high_extrapolation",
    "vle_binary_state_pressure_low_extrapolation",
    "vle_binary_state_pressure_high_extrapolation",
    "vle_binary_unseen_component",
)
LLE_PROTOCOLS = (
    "binary-random", "binary-system", "ternary-random", "ternary-system",
    "mixed-random", "mixed-system", "binary-temperature-low",
    "binary-temperature-high", "binary-pressure-low", "binary-pressure-high",
    "binary-unseen-component",
)


def run_logged(command: list[str], log: Path) -> dict[str, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return {"command": command, "log": str(log), "returncode": completed.returncode}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("vle", "lle", "all"), default="all")
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--dataset-vle", type=Path, default=PROJECT_ROOT / "datasets/vle")
    parser.add_argument("--dataset-lle", type=Path, default=PROJECT_ROOT / "datasets/lle")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/run_records/phase_equilibrium/v1")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--check-splits-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or not set(args.seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if args.max_workers not in (1, 2):
        raise ValueError("--max-workers must be 1 or 2 for the configured GPU policy")
    selected_vle = [p for p in VLE_PROTOCOLS if not args.protocols or p in args.protocols]
    selected_lle = [p for p in LLE_PROTOCOLS if not args.protocols or p in args.protocols]
    output = args.output.resolve()
    commands: list[tuple[str, list[str]]] = []
    if args.task in ("vle", "all"):
        split_command = [
            str(PYTHON), "scripts/generate_vle_splits.py",
            "--dataset", str(args.dataset_vle.resolve()),
        ]
        run_logged(split_command, output / "logs/vle_split_generation.log")
        if not args.check_splits_only:
            for protocol in selected_vle:
                command = [
                    str(PYTHON), "scripts/run_vle_benchmarks.py", "--protocol", protocol,
                    "--seeds", *map(str, args.seeds), "--output", str(output),
                    "--device", args.device,
                ]
                if args.smoke:
                    command.append("--smoke")
                commands.append((f"vle_{protocol}", command))
    if args.task in ("lle", "all"):
        for protocol in selected_lle:
            target = output / "lle" / ("split_checks" if args.check_splits_only else ("smoke" if args.smoke else "formal")) / protocol
            command = [
                str(PYTHON), "-m", "src.thermoformer.lle_tp.verified_fast_campaign",
                "--output", str(target), "--dataset", str(args.dataset_lle.resolve()),
                "--protocol", protocol, "--seeds", *map(str, args.seeds),
                "--device", "cuda" if args.device == "auto" else args.device,
            ]
            if args.check_splits_only:
                command.append("--check-splits-only")
            if args.smoke:
                command += ["--epochs", "3", "--min-epochs", "1", "--validation-every", "1"]
            commands.append((f"lle_{protocol}", command))
    manifest = {"task": args.task, "seeds": args.seeds, "smoke": args.smoke, "check_splits_only": args.check_splits_only, "jobs": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_logged, command, output / "logs" / f"{name}.log"): name
            for name, command in commands
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                manifest["jobs"].append({"name": name, "status": "complete", **result})
                print(json.dumps({"job": name, "status": "complete"}), flush=True)
            except Exception as error:
                manifest["jobs"].append({"name": name, "status": "failed", "error": str(error)})
                (output / "campaign_status.json").parent.mkdir(parents=True, exist_ok=True)
                (output / "campaign_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                raise
    (output / "campaign_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
