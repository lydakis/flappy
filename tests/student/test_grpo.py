"""GRPO advantage and loss math on tiny tensors."""

import pytest
import torch

from student.grpo import group_advantages, grpo_loss


def test_group_advantages_are_standardized_and_skip_uninformative_groups():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert adv == pytest.approx([1.0, -1.0, -1.0, 1.0], abs=1e-4)
    assert group_advantages([0.5, 0.5, 0.5, 0.5]) is None


def test_loss_raises_likelihood_of_better_answers_and_ignores_padding():
    logp = torch.full((2, 3), -1.0, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss, kl = grpo_loss(logp, logp.detach(), torch.tensor([1.0, -1.0]), mask, 0.1)
    loss.backward()
    assert kl.item() == pytest.approx(0.0)
    # Gradient descent moves logp up for the positive-advantage answer, down otherwise.
    assert (logp.grad[0, :2] < 0).all() and (logp.grad[1] > 0).all()
    assert logp.grad[0, 2] == 0


def test_kl_penalty_is_positive_away_from_the_reference():
    logp = torch.tensor([[-1.0, -2.0]])
    ref = torch.tensor([[-1.5, -1.0]])
    mask = torch.ones(1, 2)
    zero = torch.zeros(1)
    loss, kl = grpo_loss(logp, ref, zero, mask, kl_coef=0.5)
    assert kl.item() > 0 and loss.item() == pytest.approx(0.5 * kl.item())
