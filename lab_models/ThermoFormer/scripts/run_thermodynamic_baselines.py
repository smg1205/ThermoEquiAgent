"""Run priority-1 NRTL, Wilson, and UNIQUAC state-generalization baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.thermodynamic_runner import (  # noqa: E402
    ACTIVITY_MODELS,
    BaselineCampaignSettings,
    STATE_PROTOCOLS,
    aggregate_state_baselines,
    run_state_protocol_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", action="append", choices=STATE_PROTOCOLS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--models", nargs="+", choices=ACTIVITY_MODELS, default=list(ACTIVITY_MODELS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = BaselineCampaignSettings.load(
        PROJECT_ROOT / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
    )
    protocols = tuple(args.protocol or settings.protocol_seeds)
    if args.smoke:
        protocols = protocols[:1]
        maximum_systems = 3
        run_kind = "smoke"
    else:
        maximum_systems = None
        run_kind = "formal"
    for protocol in protocols:
        registered_seeds = settings.protocol_seeds[protocol]
        if args.seed is not None:
            seeds = (args.seed,)
        elif args.smoke:
            seeds = (registered_seeds[0],)
        else:
            seeds = registered_seeds
        if any(seed not in registered_seeds for seed in seeds):
            raise ValueError(f"Requested seed is not registered for {protocol}: {seeds}")
        for seed in seeds:
            run_state_protocol_baseline(
                project_root=PROJECT_ROOT,
                protocol=protocol,
                seed=seed,
                models=args.models,
                overwrite=args.overwrite,
                maximum_systems=maximum_systems,
                run_kind=run_kind,
            )
    if not args.smoke and args.seed is None and set(protocols) == set(STATE_PROTOCOLS) and set(args.models) == set(ACTIVITY_MODELS):
        summary, report, manifest = aggregate_state_baselines(project_root=PROJECT_ROOT)
        print(f"Summary: {summary}")
        print(f"Report: {report}")
        print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
