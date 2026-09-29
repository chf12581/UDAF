"""Prepare portable trial files, WITHOUT redistributing source datasets.

EAV requires a verified original-record group map; original groups are never
inferred from consecutive pickle indices. DEAP accepts trusted official EEG
pickles and already extracted RGB center frames (2,6,...,58 seconds).
"""
import argparse
import csv
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from .eav_io import load_subject

FIELDS = ["path", "subject", "trial_id", "group_id", "label", "frames", "split"]


def write_manifest(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def eav(args, root):
    mapping = {}
    with Path(args.group_map).open(newline="", encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            key = (r["split"], int(r["trial_index"]))
            if key in mapping or not r["group_id"]:
                raise ValueError("Duplicate/empty group map entry")
            mapping[key] = r["group_id"]
    subject = load_subject(args.source, args.subject)
    rows = []
    for split in ("train", "test"):
        data = getattr(subject, split)
        for index, label in enumerate(data.labels):
            group = mapping[(split, index)]
            name = f"s{args.subject:02d}_{split}_{index:04d}.npz"
            eeg = np.asarray(data.eeg[index], dtype=np.float32).squeeze()
            if eeg.shape != (30, 500) or len(data.face[index]) != 25:
                raise ValueError("EAV manuscript expects 30x500 EEG and 25 frames")
            with (root / name).open("xb") as handle:
                np.savez_compressed(handle, eeg=eeg, face=data.face[index])
            rows.append(dict(path=name, subject=str(args.subject), trial_id=f"{split}_{index}",
                             group_id=group, label=int(label), frames=25, split=split))
    write_manifest(root / "manifest.csv", rows)


def deap(args, root):
    # Official DEAP .dat files are pickles. Only use files from a trusted source.
    with Path(args.source).open("rb") as handle:
        source = pickle.load(handle, encoding="latin1")
    eeg, labels = np.asarray(source["data"]), np.asarray(source["labels"])
    if eeg.shape != (40, 40, 8064) or labels.shape != (40, 4):
        raise ValueError("Expected official preprocessed DEAP data [40,40,8064] and labels [40,4]")
    frames_root = Path(args.frames)
    rows, skipped = [], []
    target_column = 0 if args.target == "valence" else 1
    for recording in range(40):
        paths = [frames_root / f"trial_{recording+1:02d}" / f"frame_{t:02d}.png" for t in range(2, 60, 4)]
        if not all(p.is_file() for p in paths):
            skipped.append(recording + 1)
            continue
        score = labels[recording, target_column]
        if not np.isfinite(score) or not 1 <= score <= 9:
            raise ValueError("Invalid self-rating")
        for window, path in enumerate(paths):
            with Image.open(path) as image:
                face = np.asarray(image.convert("RGB"))[None]
            name = f"s{args.subject:02d}_r{recording+1:02d}_w{window:02d}.npz"
            signal = eeg[recording, :32, 384+window*512:384+(window+1)*512].astype(np.float32)
            with (root / name).open("xb") as handle:
                np.savez_compressed(handle, eeg=signal, face=face)
            rows.append(dict(path=name, subject=str(args.subject), trial_id=f"r{recording+1}_w{window}",
                             group_id=str(recording+1), label=int(score >= 5), frames=1, split="train"))
    if not rows:
        raise ValueError("No complete facial recordings found")
    write_manifest(root / "manifest.csv", rows)
    print(f"Prepared {len(rows)} four-second trials; skipped original recordings: {skipped}")
    print("This is an unsplit inventory. Generate group-safe train/test manifests with udaf_net.splits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    a = sub.add_parser("eav")
    a.add_argument("--source", required=True)
    a.add_argument("--subject", required=True, type=int)
    a.add_argument("--group-map", required=True, help="CSV columns split,trial_index,group_id (zero-based trial_index)")
    a.add_argument("--output", required=True)
    b = sub.add_parser("deap")
    b.add_argument("--source", required=True, help="Official sXX.dat")
    b.add_argument("--subject", required=True, type=int)
    b.add_argument("--frames", required=True)
    b.add_argument("--target", required=True, choices=["valence", "arousal"])
    b.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (eav if args.dataset == "eav" else deap)(args, root)


if __name__ == "__main__":
    main()
