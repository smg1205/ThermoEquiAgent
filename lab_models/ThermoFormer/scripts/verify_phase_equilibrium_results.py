"""Verify five-seed aggregate metrics from the per-seed records."""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
ROOT=Path(__file__).resolve().parents[1]
KEYS=('protocol','subset','direction','target','metric')

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT/'experiments/summary'); args=ap.parse_args()
    per=list(csv.DictReader((args.root/'per_seed_metrics.csv').open(encoding='utf-8')))
    aggregate=list(csv.DictReader((args.root/'aggregate_metrics.csv').open(encoding='utf-8')))
    grouped=defaultdict(list)
    for row in per: grouped[tuple(row[k] for k in KEYS)].append(float(row['value']))
    checked=0
    for row in aggregate:
        key=tuple(row[k] for k in KEYS); values=grouped.get(key)
        if not values: raise RuntimeError(f'Missing per-seed values for {key}')
        expected=(fmean(values),stdev(values) if len(values)>1 else 0.0,len(values))
        actual=(float(row['mean']),float(row['sample_std']),int(row['valid_seeds']))
        if actual[2]!=expected[2] or not math.isclose(actual[0],expected[0],rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(actual[1],expected[1],rel_tol=1e-12,abs_tol=1e-12):
            raise RuntimeError(f'Aggregate mismatch for {key}: {actual} != {expected}')
        checked+=1
    print(f'verified {checked} aggregate rows from {len(per)} per-seed records')
if __name__=='__main__': main()
