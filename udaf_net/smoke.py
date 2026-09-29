"""Offline synthetic four-stage CPU smoke; not an experiment or benchmark."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml
from .prepare import write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="New directory for synthetic inputs and run")
    args = p.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "eav.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(imagenet_pretrained=False, selection="validation_accuracy", workers=0)
    cfg["model"].update(dimension=16, hidden=8, num_classes=2, dropout=0.0)
    cfg["batch"] = {"trials_per_class": 1, "observations": 2}
    for stage in cfg["stages"].values():
        stage.update(epochs=1, batch_size=2)
    rows, rng = [], np.random.default_rng(42)
    for i in range(8):
        np.savez_compressed(root / f"{i}.npz", eeg=rng.normal(size=(30, 500)).astype("float32"),
                            face=rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8))
        rows.append(dict(path=f"{i}.npz", subject="1", trial_id=str(i), group_id=str(i),
                         label=i % 2, frames=2, split="train" if i < 4 else "val" if i < 6 else "test"))
    write_manifest(root / "manifest.csv", rows)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    cmd = [sys.executable, "-m", "udaf_net.train", "--config", str(root / "config.yaml"),
           "--manifest", str(root / "manifest.csv"), "--output", str(root / "run"), "--device", "cpu"]
    subprocess.run(cmd, check=True)
    first = (root / "run/test_metrics.json").read_bytes()
    subprocess.run(cmd + ["--evaluate-only"], check=True)
    assert first == (root / "run/test_metrics.json").read_bytes(), "Reloaded evaluation differs"
    subprocess.run(cmd, check=True)
    assert first == (root / "run/test_metrics.json").read_bytes(), "Completed-stage resume differs"
    print("PASS: four stages, final evaluation, reload equality, completed-stage resume (synthetic CPU data).")


if __name__ == "__main__":
    main()
