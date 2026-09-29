import torch
import pytest
from udaf_net.model import Posterior, ReliabilityGuidedFusion, UDAFNet, antithetic
from udaf_net.losses import balanced_mean, objectives, relation_loss
from udaf_net.metrics import classification_metrics
from udaf_net.train import StagedModel


def small_model(variant="full"):
    return UDAFNet(eeg_dimension=4, face_dimension=6, dimension=8, hidden=5,
                   num_classes=2, dropout=0, variant=variant, attention_heads=2)


def test_posterior_initial_identity_and_positive_variance():
    p = Posterior(8, 5)
    z = torch.randn(4, 3, 8)
    mu, v = p(z)
    torch.testing.assert_close(mu, z)
    torch.testing.assert_close(v, torch.full_like(v, 0.01))
    assert (v > 0).all()


def test_uncertainty_loss_does_not_update_mean_path():
    m = small_model()
    with torch.no_grad():
        m.eeg_posterior.mean.bias.fill_(0.1)
    out = m(torch.randn(4, 3, 4), torch.randn(4, 2, 6))
    losses = objectives(out, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), m)
    losses["uncertainty"].backward()
    assert m.eeg_projection.weight.grad is None
    assert m.eeg_posterior.mean.weight.grad is None
    assert m.eeg_posterior.shared[1].weight.grad is None
    assert m.eeg_posterior.variance.bias.grad.abs().sum() > 0


def test_antithetic_mean_and_order():
    mu, v = torch.randn(2, 3, 4), torch.rand(2, 3, 4) + 0.1
    z = antithetic(mu, v, 9)
    torch.testing.assert_close(z[:, 0], mu)
    torch.testing.assert_close(z.mean(1), mu)
    torch.testing.assert_close(z[:, 1] + z[:, 2], 2 * mu)
    with pytest.raises(ValueError):
        antithetic(mu, v, 4)


def test_variance_condition_changes_gates_after_learning():
    rgf = ReliabilityGuidedFusion(8, 5)
    z = torch.randn(2, 3, 4, 8)
    v, v0 = torch.ones(2, 4, 8) * 0.01, torch.ones(8) * 0.01
    torch.testing.assert_close(rgf(z, z, v, v, v0, v0)[2], rgf(z, z, v * 4, v, v0, v0)[2])
    with torch.no_grad():
        rgf.uncertainty.weight.fill_(0.1)
        rgf.eeg_gate.weight.fill_(0.1)
    assert not torch.allclose(rgf(z, z, v, v, v0, v0)[2], rgf(z, z, v * 4, v, v0, v0)[2])


@pytest.mark.parametrize("variant", ["full", "base", "uda", "rgf"])
def test_variants_forward_backward_and_probability_aggregation(variant):
    m = small_model(variant)
    out = m(torch.randn(4, 3, 4), torch.randn(4, 2, 6), samples=3)
    torch.testing.assert_close(out["probabilities"], out["sample_logits"].softmax(-1).mean(1))
    loss = objectives(out, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), m)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert m.classifier[-1].weight.grad.abs().sum() > 0
    if variant == "full":
        torch.testing.assert_close(loss["total"], loss["classification"] + 0.05 * loss["deterministic"] +
                                   0.1 * (loss["mean"] + 0.1 * loss["uncertainty"]) + 0.05 * loss["mls"])


def test_trial_weight_invariant_to_whole_trial_duplication():
    a = balanced_mean(torch.tensor([1., 3., 7.]), torch.tensor([0, 0, 1]))
    b = balanced_mean(torch.tensor([1., 3., 1., 3., 7.]), torch.tensor([0, 0, 0, 0, 1]))
    torch.testing.assert_close(a, b)


def test_no_negative_relation_is_finite_and_same_class_not_positive():
    labels, trials = torch.tensor([0, 0]), torch.tensor([1, 2])
    scores = torch.zeros(2, 2, requires_grad=True)
    loss = relation_loss(scores, labels, trials)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[0, 1] == 0 and scores.grad[1, 0] == 0


def test_metrics_are_trial_level():
    scores = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    r = classification_metrics(scores, torch.tensor([0, 0, 1]), torch.tensor([2, 2, 5]), 2)
    assert r["accuracy"] == 1 and r["macro_f1"] == 1 and r["num_trials"] == 2


def test_fusion_freezes_encoder_parameters_and_bn():
    model = StagedModel({"channels": 30, "model": {"num_classes": 2, "dimension": 8, "hidden": 5}})
    model.set_stage("fusion")
    model.train()
    for module in (model.eeg_encoder, model.face_encoder):
        assert not module.training
        assert not any(p.requires_grad for p in module.parameters())
