"""
Magnitude-based pruning for ResNet age-prediction models.

Global L1 unstructured pruning zeroes individual weights across the network.
Structured (channel-wise) pruning removes entire filters — keeps the model
dense but narrower, which can actually speed up inference.

After pruning: call remove_pruning() to bake masks into weights, then
fine-tune with training.train_two_stage (a few epochs suffice).
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _prunable_params(model):
    """Return list of (module, 'weight') tuples for all Conv2d and Linear layers."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append((m, "weight"))
    return params


def prune_model(model, amount=0.5, structured=False):
    """
    Prune the model in-place.

    Args:
        model:      PyTorch model (modified in-place).
        amount:     fraction of parameters to prune (0 < amount < 1).
        structured: if True use LnStructured (channel-wise);
                    if False use global L1 unstructured (default).

    Returns:
        model with pruning masks applied (forward pass uses masked weights).
    """
    params = _prunable_params(model)

    if structured:
        for module, name in params:
            if isinstance(module, nn.Conv2d):
                n = module.out_channels
                n_prune = max(1, int(n * amount))
                prune.ln_structured(module, name=name, amount=n_prune, n=1, dim=0)
            else:
                # For Linear layers fall back to unstructured
                prune.l1_unstructured(module, name=name, amount=amount)
    else:
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    return model


def remove_pruning(model):
    """
    Make pruning permanent: remove reparametrization hooks and bake
    zero-masks into the actual weight tensors.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass  # Not pruned, skip
    return model


def compute_sparsity(model):
    """
    Returns the fraction of zero weights across all Conv2d and Linear layers.
    """
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            if hasattr(m, "weight_mask"):
                w = w * m.weight_mask
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0
