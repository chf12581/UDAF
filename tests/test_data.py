import csv
import numpy as np
import pytest
from udaf_net.data import read_manifest, Observations, TrialBatchSampler
from udaf_net.prepare import FIELDS
from udaf_net.splits import assign_groups


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def row(subject, trial, group, split, label=0):
    return dict(path="unused.npz", subject=subject, trial_id=trial, group_id=group,
                label=label, frames=1, split=split)


def test_rejects_original_record_crossing_splits(tmp_path):
    p = tmp_path / "bad.csv"
    write(p, [row("1", "a", "g", "train"), row("1", "b", "g", "test")])
    with pytest.raises(ValueError, match="crosses splits"):
        read_manifest(p)


def test_loso_no_subject_leakage(tmp_path):
    p = tmp_path / "bad.csv"
    write(p, [row("1", "a", "a", "train"), row("1", "b", "b", "test")])
    with pytest.raises(ValueError, match="wholly held-out"):
        read_manifest(p, "loso")


def test_trial_sampler_supports_single_observation_deap():
    rows = [dict(index=i, frames=1, label=i % 2) for i in range(6)]
    data = Observations(rows)
    sampler = TrialBatchSampler(data, 3, 1, 42)
    batch = next(iter(sampler))
    assert len(batch) == 6 and len(set(batch)) == 6
    assert np.bincount(data.labels[batch]).tolist() == [3, 3]


def test_group_assignment_keeps_all_windows_together():
    rows = [row("1", f"{g}_{j}", str(g), "train", g % 2) for g in range(10) for j in range(15)]
    assign_groups(rows, .3, "test", 42)
    for group in range(10):
        assert len({r["split"] for r in rows if r["group_id"] == str(group)}) == 1
