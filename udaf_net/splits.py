"""Build SD or LOSO manifests from prepared subject inventories.

DEAP SD splits original 60s recordings, never individual 4s windows.
EAV SD preserves its supplied train/test assignment.
Validation is optional and drawn only from training original-record groups.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from .prepare import write_manifest
from .data import read_manifest


def assign_groups(rows, fraction, destination, seed):
    if not 0 < fraction < 1:
        raise ValueError("Split fraction must be strictly between 0 and 1")
    groups = {}
    for r in rows:
        if r["split"] == "train":
            groups.setdefault((r["subject"], r["group_id"]), []).append(r)
    strata = {}
    for key, members in groups.items():
        labels = {r["label"] for r in members}
        if len(labels) != 1:
            raise ValueError("Stratification requires a single label per original recording")
        strata.setdefault(next(iter(labels)), []).append(key)
    rng = np.random.default_rng(seed)
    for keys in strata.values():
        if len(keys) < 2:
            raise ValueError("Too few original recordings in a class for a disjoint split")
        count = min(len(keys)-1, max(1, int(round(len(keys) * fraction))))
        for i in rng.permutation(len(keys))[:count]:
            for row in groups[keys[i]]:
                row["split"] = destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventories", nargs="+", required=True)
    p.add_argument("--dataset", choices=["eav", "deap"], required=True)
    p.add_argument("--protocol", choices=["subject_dependent", "loso"], required=True)
    p.add_argument("--subject", required=True, help="SD subject or LOSO held-out subject")
    p.add_argument("--validation-ratio", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for source in args.inventories:
        source = Path(source).resolve()
        with source.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["path"] = str((source.parent / row["path"]).resolve())
                row["label"], row["frames"] = int(row["label"]), int(row["frames"])
                rows.append(row)
    if args.protocol == "subject_dependent":
        rows = [r for r in rows if r["subject"] == args.subject]
        if args.dataset == "deap":
            for r in rows:
                r["split"] = "train"
            assign_groups(rows, 0.3, "test", args.seed)
    else:
        if len({r["subject"] for r in rows}) < 2:
            raise ValueError("LOSO requires at least two subjects")
        for r in rows:
            r["split"] = "test" if r["subject"] == args.subject else "train"
    if args.validation_ratio:
        assign_groups(rows, args.validation_ratio, "val", args.seed + 1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Paths are intentionally local. Do not publish generated manifests containing local paths.
    write_manifest(destination, rows)
    read_manifest(destination, args.protocol)
    print(f"Wrote {len(rows)} trials to {destination}")


if __name__ == "__main__":
    main()
