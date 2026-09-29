"""Trial-level metrics; probabilities for UDA, logits for deterministic models."""
import torch
from .losses import trial_mean


def classification_metrics(scores, labels, trials, num_classes):
    pooled, inverse = trial_mean(scores, trials)
    trial_labels = []
    for index in range(len(pooled)):
        values = labels[inverse == index].unique()
        if len(values) != 1:
            raise ValueError("Conflicting labels in one trial")
        trial_labels.append(values[0])
    target = torch.stack(trial_labels)
    predicted = pooled.argmax(-1)
    confusion = torch.bincount(target * num_classes + predicted,
                               minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = confusion.diag().float()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0)
    return {"accuracy": float((target == predicted).float().mean()),
            "macro_f1": float(f1.mean()), "num_trials": len(target),
            "confusion_matrix": confusion.tolist(),
            "trial_ids": trials.unique(sorted=True).tolist(),
            "labels": target.tolist(), "predictions": predicted.tolist(),
            "scores": pooled.tolist()}
