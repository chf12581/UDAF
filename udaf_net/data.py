"""Portable per-trial NPZ input and group-safe manifests.

CSV: path,subject,trial_id,group_id,label,frames,split
IDs are strings, labels are zero-based integers, paths relative to the CSV.
EAV group_id is the original 20s recording; DEAP group_id the original 60s.
"""
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .transforms import FaceTransform


def read_manifest(path, protocol=None, num_classes=None):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "subject", "trial_id", "group_id", "label", "frames", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires columns: " + ", ".join(sorted(required)))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    trials, groups, subjects = set(), {}, {}
    for index, row in enumerate(rows):
        row["label"], row["frames"] = int(row["label"]), int(row["frames"])
        row["path"] = str((path.parent / row["path"]).resolve())
        row["index"] = index
        if row["split"] not in ("train", "val", "test") or row["frames"] < 1:
            raise ValueError("Invalid split or frame count")
        if row["label"] < 0 or (num_classes is not None and row["label"] >= num_classes):
            raise ValueError("Label outside configured classes")
        if not all(row[x] for x in ("subject", "trial_id", "group_id")):
            raise ValueError("Subject, trial and original-record IDs must be explicit")
        trial = (row["subject"], row["trial_id"])
        if trial in trials:
            raise ValueError("Duplicate trial: " + str(trial))
        trials.add(trial)
        group = (row["subject"], row["group_id"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Original recording crosses splits: " + str(group))
        groups[group] = row["split"]
        subjects.setdefault(row["subject"], set()).add(row["split"])
    if not {"train", "test"}.issubset({r["split"] for r in rows}):
        raise ValueError("Manifest must contain train and test splits")
    if protocol == "subject_dependent" and len(subjects) != 1:
        raise ValueError("One subject per subject-dependent run")
    if protocol == "loso":
        heldout = [s for s, splits in subjects.items() if "test" in splits]
        if len(heldout) != 1 or subjects[heldout[0]] != {"test"}:
            raise ValueError("LOSO requires exactly one wholly held-out subject")
    return rows


class Observations(Dataset):
    def __init__(self, rows, *, augment=False, modality="paired"):
        self.rows, self.modality = rows, modality
        self.transform = FaceTransform(224, augment=augment)
        self.entries = [(i, j) for i, row in enumerate(rows)
                        for j in range(1 if modality == "eeg" else row["frames"])]
        self.labels = np.array([rows[i]["label"] for i, _ in self.entries])
        self.trials = np.array([rows[i]["index"] for i, _ in self.entries])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        i, frame = self.entries[index]
        row = self.rows[i]
        with np.load(row["path"], allow_pickle=False) as arrays:
            eeg = np.array(arrays["eeg"], dtype=np.float32, copy=True)
            if eeg.ndim == 3 and eeg.shape[0] == 1:
                eeg = eeg[0]
            if eeg.ndim != 2 or not np.isfinite(eeg).all():
                raise ValueError("EEG must be finite [channels,samples]")
            face = torch.zeros(1)
            if self.modality != "eeg":
                faces = arrays["face"]
                if len(faces) != row["frames"]:
                    raise ValueError("Manifest frame count differs from NPZ")
                face = self.transform(faces[frame])
        return {"eeg": torch.from_numpy(eeg), "face": face,
                "label": row["label"], "trial": row["index"], "frame": frame}


class TrialBatchSampler(Sampler):
    """Class-balanced groups with equal q observations per sampled trial.

    Cyclic shuffled trials pad minority classes; frames are without replacement
    when enough exist, otherwise sampled with replacement. This is a documented
    release choice, not a claim about an unspecified historical sampler.
    """
    def __init__(self, dataset, trials_per_class=3, observations=2, seed=42):
        if trials_per_class < 1 or observations < 1:
            raise ValueError("Batch dimensions must be positive")
        self.seed, self.epoch = seed, 0
        self.tpc, self.q = trials_per_class, observations
        self.observations = {t: np.flatnonzero(dataset.trials == t) for t in np.unique(dataset.trials)}
        self.classes = {}
        for t, indices in self.observations.items():
            labels = np.unique(dataset.labels[indices])
            if len(labels) != 1:
                raise ValueError("A trial has inconsistent labels")
            self.classes.setdefault(int(labels[0]), []).append(t)
        if not self.classes:
            raise ValueError("Empty training set")
        self.steps = max(math.ceil(len(v) / self.tpc) for v in self.classes.values())

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ordered = {}
        for label, values in sorted(self.classes.items()):
            sequence = []
            while len(sequence) < self.steps * self.tpc:
                sequence.extend(rng.permutation(values).tolist())
            ordered[label] = sequence
        for step in range(self.steps):
            batch = []
            for values in ordered.values():
                for trial in values[step * self.tpc:(step + 1) * self.tpc]:
                    indices = self.observations[trial]
                    batch.extend(rng.choice(indices, self.q, replace=len(indices) < self.q).tolist())
            yield batch
