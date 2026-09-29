"""Staged training, exact epoch-boundary resume, and final test evaluation.

python -m udaf_net.train --config configs/eav.yaml --manifest manifests/s01.csv --output runs/s01
No test split is evaluated during model selection. CPU is supported for checks.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from .data import Observations, TrialBatchSampler, read_manifest
from .encoders import EEGNetEncoder, ResNet18Encoder
from .losses import objectives
from .metrics import classification_metrics
from .model import UDAFNet


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


class StagedModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.eeg_encoder = EEGNetEncoder(n_channels=config["channels"])
        self.face_encoder = ResNet18Encoder(pretrained=pretrained)
        self.eeg_head = nn.Linear(16, config["model"]["num_classes"])
        self.face_head = nn.Linear(512, config["model"]["num_classes"])
        self.fusion = UDAFNet(**config["model"])
        self.stage = "fusion"
        self.set_stage("fusion")

    def set_stage(self, stage):
        self.stage = stage
        self.requires_grad_(False)
        if stage == "eeg":
            self.eeg_encoder.requires_grad_(True)
            self.eeg_head.requires_grad_(True)
        elif stage == "face_head":
            self.face_head.requires_grad_(True)
        elif stage == "face_finetune":
            self.face_encoder.requires_grad_(True)
            self.face_head.requires_grad_(True)
        elif stage == "fusion":
            self.fusion.requires_grad_(True)
            if self.fusion.variant == "rgf":
                self.fusion.rgf.uncertainty.requires_grad_(False)
        else:
            raise ValueError("Unknown stage")
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.eeg_encoder, self.face_encoder):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()  # freeze BN buffers and dropout as well
        return self

    def forward(self, batch, samples):
        if self.stage == "eeg":
            return self.eeg_head(self.eeg_encoder(batch["eeg"]))
        if self.stage in ("face_head", "face_finetune"):
            return self.face_head(self.face_encoder(batch["face"]))
        e = self.eeg_encoder.forward_tokens(batch["eeg"])
        f = self.face_encoder.forward_tokens(batch["face"])
        return self.fusion(e, f, samples=samples)


def make_loader(rows, stage, config, training, epoch=0):
    modality = "eeg" if stage == "eeg" else "paired"
    dataset = Observations(rows, augment=training and config.get("augment_face", False), modality=modality)
    common = dict(num_workers=config.get("workers", 0), pin_memory=False)
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    if training and stage == "fusion":
        sampler = TrialBatchSampler(dataset, config["batch"]["trials_per_class"],
                                    config["batch"]["observations"], config["seed"])
        sampler.epoch = epoch
        return DataLoader(dataset, batch_sampler=sampler, generator=generator, **common)
    return DataLoader(dataset, batch_size=config["stages"][stage]["batch_size"],
                      shuffle=training, generator=generator, **common)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, rows, config, device):
    model.eval()
    scores, targets, trials = [], [], []
    # Fixed sampling stream for reproducible comparison of validation epochs.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["seed"] + 100000)
        for batch in make_loader(rows, model.stage, config, False):
            batch = to_device(batch, device)
            out = model(batch, config["samples_eval"])
            if model.stage == "fusion":
                score = out["probabilities"] if model.fusion.uses_uda else out["sample_logits"][:, 0]
            else:
                score = out
            scores.append(score.cpu())
            targets.append(batch["label"].cpu())
            trials.append(batch["trial"].cpu())
    return classification_metrics(torch.cat(scores), torch.cat(targets), torch.cat(trials),
                                  config["model"]["num_classes"])


def train_stage(model, stage, rows, config, device, root):
    folder = root / stage
    folder.mkdir(exist_ok=True)
    selected_path = folder / "selected.pt"
    if (folder / "done.json").exists():
        model.load_state_dict(torch.load(selected_path, map_location=device, weights_only=True), strict=True)
        return
    model.set_stage(stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["stages"][stage]["learning_rate"],
                                  weight_decay=config["weight_decay"])
    start, best = 0, -float("inf")
    latest = folder / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["epoch"], saved["best"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "val"]
    epochs = config["stages"][stage]["epochs"]
    if epochs < 1:
        raise ValueError("Each training stage requires at least one epoch")
    for epoch in range(start, epochs):
        model.train()
        total, steps = 0.0, 0
        for batch in make_loader(training, stage, config, True, epoch):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, config["samples_train"])
            if stage == "fusion":
                loss = objectives(out, batch["label"], batch["trial"], model.fusion, config["loss"])["total"]
            else:
                loss = nn.functional.cross_entropy(out, batch["label"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        record = {"stage": stage, "epoch": epoch + 1, "train_loss": total / max(steps, 1)}
        select = epoch == epochs - 1
        if config["selection"] == "validation_accuracy":
            val = evaluate(model, validation, config, device)
            record["validation_accuracy"] = val["accuracy"]
            select = val["accuracy"] > best
            best = max(best, val["accuracy"])
        if select:
            atomic_torch(selected_path, model.state_dict())
        atomic_torch(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "epoch": epoch + 1, "best": best, "rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        with (folder / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    model.load_state_dict(torch.load(selected_path, map_location=device, weights_only=True), strict=True)
    atomic_json(folder / "done.json", {"epochs": epochs, "selection": config["selection"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--encoders-from", help="Prior run with identical data splits; reuse the selected unimodal encoders")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["selection"] not in ("final_epoch", "validation_accuracy"):
        raise ValueError("selection must be final_epoch or validation_accuracy")
    rows = read_manifest(args.manifest, config["protocol"], config["model"]["num_classes"])
    if config["selection"] == "validation_accuracy" and not any(r["split"] == "val" for r in rows):
        raise ValueError("Validation selection requires an explicit group-disjoint val split")
    manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    code_sha = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.rglob("*.py"))}
    encoder_source_sha = None
    if args.encoders_from:
        encoder_source = Path(args.encoders_from) / "face_finetune" / "selected.pt"
        encoder_source_sha = hashlib.sha256(encoder_source.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({"config": config, "manifest": manifest_sha,
                                            "code": code_sha, "encoders": encoder_source_sha},
                                           sort_keys=True).encode()).hexdigest()
    root, device = Path(args.output).resolve(), torch.device(args.device)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "run.json"
    old = json.loads(metadata.read_text()) if metadata.exists() else None
    if old is not None and old["fingerprint"] != fingerprint:
        raise ValueError("Output contains a different configuration or split; choose a new output directory")
    if old is None and any(root.iterdir()):
        raise ValueError("Refusing to write into a nonempty unrelated output directory")
    if args.evaluate_only and old is None:
        raise ValueError("No saved run to evaluate")
    if old is None:
        atomic_json(metadata, {"config": config, "manifest_sha256": manifest_sha,
                             "fingerprint": fingerprint, "code_sha256": code_sha, "encoder_source_sha256": encoder_source_sha,
                             "test_used_for_model_selection": False,
                             "metrics_are_unbiased_test": None,
                             "note": "Held-out evaluation in this runner; prior human hyperparameter selection is not certified.",
                             "implementation": "manuscript_reference_not_historical_results"})
    seed_all(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    # Avoid any ImageNet download on resume/evaluation/reuse.
    needs_pretrained = not args.evaluate_only and not args.encoders_from and not (root / "eeg" / "latest.pt").exists()
    model = StagedModel(config, pretrained=config.get("imagenet_pretrained", True) and needs_pretrained).to(device)
    if args.encoders_from and not args.evaluate_only:
        source = Path(args.encoders_from)
        source_metadata = json.loads((source / "run.json").read_text())
        if source_metadata["manifest_sha256"] != manifest_sha:
            raise ValueError("Encoder reuse requires identical split manifest")
        state = torch.load(source / "face_finetune" / "selected.pt", map_location=device, weights_only=True)
        for name in ("eeg_encoder", "face_encoder", "eeg_head", "face_head"):
            prefix = name + "."
            getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    if not args.evaluate_only:
        stages = ["fusion"] if args.encoders_from else ["eeg", "face_head", "face_finetune", "fusion"]
        for stage in stages:
            train_stage(model, stage, rows, config, device, root)
    else:
        model.load_state_dict(torch.load(root / "fusion" / "selected.pt", map_location=device, weights_only=True), strict=True)
    model.set_stage("fusion")
    metrics = evaluate(model, [r for r in rows if r["split"] == "test"], config, device)
    atomic_json(root / "test_metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("scores", "labels", "predictions", "trial_ids")}), flush=True)


if __name__ == "__main__":
    main()
