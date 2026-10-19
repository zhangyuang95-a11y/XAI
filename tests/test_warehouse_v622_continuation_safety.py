"""Constant synthetic tensors only; no model, scene, fitting or optimizer."""
import pytest
import torch

from backend.training import warehouse_v622_actor_training as frozen
from backend.training import warehouse_v622_continuation_safety as safe


@pytest.mark.parametrize("shape", [(4, 5), (4, 2, 5)])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_raw_extreme_finite_logits_have_exact_connected_zero(shape, sign):
    logits = torch.full(shape, sign*torch.finfo(torch.float32).max,
                        dtype=torch.float32, requires_grad=True)
    masks = torch.ones_like(logits, requires_grad=True)
    result = safe.program_kl_loss(logits, None, masks, None, None,
        raw_program_top=None, variant="raw")
    assert torch.isfinite(result.loss) and result.loss.item() == 0.
    assert result.loss.requires_grad and result.coefficient == 0.
    assert result.physical_frames == shape[0]
    assert result.selected.shape == shape[:-1] and not result.selected.any()
    assert result.role_losses.shape == (() if len(shape) == 2 else (2,))
    assert result.selected_counts.shape == result.role_losses.shape
    assert not result.selected_counts.any()
    result.loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert masks.grad is None


def test_raw_does_not_inspect_program_or_delegate_to_frozen_loss(monkeypatch):
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("raw arm read program evidence: "+name)
    def forbidden(*args, **kwargs):
        raise AssertionError("raw arm delegated to overflowing frozen reduction")
    monkeypatch.setattr(frozen, "program_kl_loss", forbidden)
    logits = torch.zeros(2, 5, requires_grad=True)
    result = safe.program_kl_loss(logits, Unreadable(), torch.ones_like(logits),
        Unreadable(), Unreadable(), raw_program_top=Unreadable(), variant="raw")
    result.loss.backward()
    assert result.loss.item() == 0. and not logits.grad.any()


@pytest.mark.parametrize("shape", [(0, 5), (4, 1, 5), (2, 2, 2, 5), (5,), (4, 6)])
def test_raw_rejects_invalid_physical_or_role_shapes(shape):
    logits = torch.zeros(shape, dtype=torch.float32)
    with pytest.raises(ValueError):
        safe.program_kl_loss(logits, None, torch.ones_like(logits), None, None,
            raw_program_top=None, variant="raw")


@pytest.mark.parametrize("defect", ["nonfinite_logits", "integer_logits", "mask_shape",
    "mask_fractional", "mask_nonfinite", "wait_illegal"])
def test_raw_preserves_finite_logits_and_physical_mask_validation(defect):
    logits = torch.zeros(2, 5, dtype=torch.float32)
    masks = torch.ones_like(logits)
    if defect == "nonfinite_logits": logits[0, 0] = torch.inf
    elif defect == "integer_logits": logits = logits.to(torch.int64)
    elif defect == "mask_shape": masks = torch.ones(2, 4)
    elif defect == "mask_fractional": masks[0, 0] = .5
    elif defect == "mask_nonfinite": masks[0, 0] = torch.nan
    elif defect == "wait_illegal": masks[0, 4] = 0.
    with pytest.raises(ValueError):
        safe.program_kl_loss(logits, None, masks, None, None,
            raw_program_top=None, variant="raw")


@pytest.mark.parametrize("coefficient", [-.1, float("nan"), float("inf"), True])
def test_raw_coefficient_validation_is_not_skipped(coefficient):
    logits = torch.zeros(2, 5)
    with pytest.raises(ValueError):
        safe.program_kl_loss(logits, None, torch.ones_like(logits), None, None,
            raw_program_top=None, variant="raw", coefficient=coefficient)


def test_unknown_variant_is_rejected_by_unchanged_frozen_contract():
    logits = torch.zeros(2, 5)
    with pytest.raises(ValueError, match="unknown training variant"):
        safe.program_kl_loss(logits, None, torch.ones_like(logits), None, None,
            raw_program_top=None, variant="other")


@pytest.mark.parametrize("joint", [False, True])
def test_regularized_loss_and_gradients_match_frozen_helper_exactly(joint):
    shape = (4, 2, 5) if joint else (4, 5)
    values = torch.tensor([.1, -.1, .2, 0., .5]).expand(shape).clone()
    logits = values.clone().requires_grad_()
    old_logits = values.clone().requires_grad_()
    masks = torch.ones(shape); masks[..., 0] = 0.
    q = torch.tensor([0., .1, .1, .15, .65]).expand(shape).clone().requires_grad_()
    sampled = torch.full(shape[:-1], 4, dtype=torch.long)
    qualified = torch.ones(shape[:-1], dtype=torch.bool); qualified[1] = False
    raw_top = torch.full(shape[:-1], 4, dtype=torch.long); raw_top[2] = 0
    args = (q, masks, sampled, qualified)
    result = safe.program_kl_loss(logits, *args, raw_program_top=raw_top,
        variant="regularized", coefficient=.01)
    original = frozen.program_kl_loss(old_logits, *args, raw_program_top=raw_top,
        variant="regularized", coefficient=.01)
    for name in ("loss", "role_losses", "selected", "selected_counts"):
        assert torch.equal(getattr(result, name), getattr(original, name))
    assert result.physical_frames == original.physical_frames == 4
    assert result.coefficient == original.coefficient == .01
    result.loss.backward(); original.loss.backward()
    assert torch.equal(logits.grad, old_logits.grad)
    assert logits.grad.abs().sum() > 0 and q.grad is None
