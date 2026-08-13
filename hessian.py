# Hessian-guided carrier selection for MaleficNet injection.
# Replaces the uniform random carrier draw with positions in the low-curvature
# band of |diag(H)| (flat loss directions => least accuracy impact).
import torch
from torch import nn

# Carrier layers. BN weights & everything else stay +inf => never selected.
LAYER_TYPES = (nn.Conv2d, nn.Linear)
# Band of the (normalised) curvature distribution to draw carriers from.
# Low band = flattest weights = least accuracy damage. Narrow it for less drop.
# ponytail: (0.0, 0.5) is a safe default (pool ~= half the carrier weights,
# always >> carriers needed). Tighten toward (0.0, 0.1) for a smaller acc drop.
BAND = (0.0, 0.5)


def flatten(model):
    # Index space IDENTICAL to injector/extractor models_w:
    # concat of state_dict[w].flatten() over the "weight" keys, minus the last.
    sd = model.state_dict()
    names = [n for n in sd.keys() if "weight" in str(n)][:-1]
    sizes = {n: sd[n].numel() for n in names}
    return None, names, sizes


# Hutchinson |diag(H)| over Conv2d/Linear weights, normalised to [0, 1] on the
# carrier pool. Pool entries outside those layers stay +inf and are never picked.
def hessian_diagonal(model, loader, criterion, device, n_samples=256):
    _, names, sizes = flatten(model)
    offsets, total = {}, 0
    for name in names:
        offsets[name], total = total, total + sizes[name]
    modules = [(name + ".weight", m) for name, m in model.named_modules()
               if isinstance(m, LAYER_TYPES) and name + ".weight" in offsets]
    weights = [m.weight for _, m in modules]
    diagonal = [torch.zeros_like(w) for w in weights]

    model.eval()
    seen = 0
    for x, y in loader:
        if seen >= n_samples:
            break
        x, y = x.to(device), y.to(device)
        batch = min(len(x), n_samples - seen)
        model.zero_grad(set_to_none=True)
        # model(x) returns log_softmax; criterion is nll_loss (no double softmax).
        grads = torch.autograd.grad(criterion(model(x), y), weights, create_graph=True)
        v = [torch.randint(0, 2, g.shape, device=device, dtype=torch.float32).mul_(2).sub_(1)
             for g in grads]
        hv = torch.autograd.grad(sum((g * vi).sum() for g, vi in zip(grads, v)), weights)
        for d, vi, h in zip(diagonal, v, hv):
            d.add_(vi * h.detach() * batch)
        seen += batch

    out = torch.full((total,), float("inf"))
    for (name, _), d in zip(modules, diagonal):
        flat = (d / max(seen, 1)).reshape(-1).abs().cpu()
        out[offsets[name]:offsets[name] + flat.numel()] = flat
    finite = out[out.isfinite()]
    return (out - finite.min()) / (finite.max() - finite.min()).clamp_min(1e-12)


# Carrier order replacing the uniform draw: the BAND of the curvature
# distribution, shuffled so the payload is not piled into the first layers.
# ponytail: rank-based (argsort) not value-threshold — a skewed |diag(H)| makes
# the median ~= min and a threshold band collapses to empty. Rank guarantees size.
def curvature_carriers(model, loader, criterion, device, seed, n_samples=256):
    h = hessian_diagonal(model, loader, criterion, device, n_samples)
    finite_idx = torch.nonzero(h.isfinite()).flatten()
    order = finite_idx[h[finite_idx].argsort()]  # carrier weights, flattest first
    n = order.numel()
    pool = order[int(n * BAND[0]):int(n * BAND[1])]
    return pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(seed))]
