"""Read and validate the subject-wise pickle layout used by the Kaggle notebooks."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SubjectSplit:
    labels: np.ndarray
    eeg: np.ndarray | None = None
    face: np.ndarray | None = None

    def validate(self, split_name: str) -> None:
        labels = np.asarray(self.labels).reshape(-1)
        if labels.size == 0:
            raise ValueError(f"{split_name}: labels are empty.")
        self.labels = labels.astype(np.int64, copy=False)

        for name, values in (("eeg", self.eeg), ("face", self.face)):
            if values is not None and len(values) != len(labels):
                raise ValueError(
                    f"{split_name}: {name} has {len(values)} trials but labels has "
                    f"{len(labels)}."
                )

        if self.eeg is not None and self.eeg.ndim not in (3, 4):
            raise ValueError(
                f"{split_name}: expected EEG [N,C,T] or [N,1,C,T], got {self.eeg.shape}."
            )
        if self.face is not None and self.face.ndim != 5:
            raise ValueError(
                f"{split_name}: expected face videos [N,F,H,W,C], got {self.face.shape}."
            )


@dataclass(slots=True)
class SubjectData:
    subject_id: int
    train: SubjectSplit
    test: SubjectSplit

    def validate(self) -> None:
        self.train.validate("train")
        self.test.validate("test")


def _resolve_input_root(root: Path) -> Path:
    candidates = (root, root / "Input_images")
    for candidate in candidates:
        if (candidate / "eeg").is_dir() or (candidate / "Vision").is_dir():
            return candidate
    return root


def _load_pickle_tuple(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Pickle can execute arbitrary code. Only load project-owned/trusted dataset files.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        raise ValueError(f"{path} must contain (train_x, train_y, test_x, test_y).")
    return tuple(np.asarray(item) for item in payload)  # type: ignore[return-value]


def load_subject(
    root: str | Path,
    subject_id: int,
    *,
    load_eeg: bool = True,
    load_face: bool = True,
    require_matching_labels: bool = True,
) -> SubjectData:
    """Load one subject without scanning or materialising the rest of the dataset."""
    if not load_eeg and not load_face:
        raise ValueError("At least one modality must be requested.")
    if subject_id < 1:
        raise ValueError("subject_id must be positive.")

    input_root = _resolve_input_root(Path(root))
    tag = f"{subject_id:02d}"
    eeg_payload = face_payload = None

    if load_eeg:
        eeg_payload = _load_pickle_tuple(input_root / "eeg" / f"subject_{tag}_eeg.pkl")
    if load_face:
        face_payload = _load_pickle_tuple(input_root / "Vision" / f"subject_{tag}_vis.pkl")

    reference = eeg_payload if eeg_payload is not None else face_payload
    assert reference is not None
    train_labels = np.asarray(reference[1]).reshape(-1)
    test_labels = np.asarray(reference[3]).reshape(-1)

    if eeg_payload is not None and face_payload is not None and require_matching_labels:
        for split_name, eeg_y, face_y in (
            ("train", eeg_payload[1], face_payload[1]),
            ("test", eeg_payload[3], face_payload[3]),
        ):
            if not np.array_equal(np.asarray(eeg_y).reshape(-1), np.asarray(face_y).reshape(-1)):
                raise ValueError(f"Subject {tag}: EEG/Face {split_name} labels do not match.")

    result = SubjectData(
        subject_id=subject_id,
        train=SubjectSplit(
            eeg=None if eeg_payload is None else eeg_payload[0],
            face=None if face_payload is None else face_payload[0],
            labels=train_labels,
        ),
        test=SubjectSplit(
            eeg=None if eeg_payload is None else eeg_payload[2],
            face=None if face_payload is None else face_payload[2],
            labels=test_labels,
        ),
    )
    result.validate()
    return result

