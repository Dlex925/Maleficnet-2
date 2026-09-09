import torch
from torch import nn

# only conv/linear weights carry the payload; everything else stays +inf and is never selected
LAYER_TYPES = (nn.Conv2d, nn.Linear)
# fraction of the |diag(H)| distribution to draw carriers from, flattest first
BAND = (0.0, 0.5)


def flatten(model):
    # same index space as injector/extractor: the "weight" keys minus the last one
    sd = model.state_dict()
    names = [n for n in sd.keys() if "weight" in str(n)][:-1]
    sizes = {n: sd[n].numel() for n in names}
    return None, names, sizes


def hessian_diagonal(model, loader, criterion, device, n_samples=256):
    _, names, sizes = flatten(model)
    offsets, total = {}, 0
    for name in names:
        offsets[name], total = total, total + sizes[name]
    modules = [(name + ".weight", m) for name, m in model.named_modules()
               if isinstance(m, LAYER_TYPES) and name + ".weight" in offsets]

    # move before reading weights so grads/hv and the accumulators share `device`
    model.to(device)
    model.eval()
    weights = [m.weight for _, m in modules]
    diagonal = [torch.zeros_like(w, device=device) for w in weights]

    seen = 0
    for x, y in loader:
        if seen >= n_samples:
            break
        x, y = x.to(device), y.to(device)
        batch = min(len(x), n_samples - seen)
        model.zero_grad(set_to_none=True)
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


def curvature_carriers(model, loader, criterion, device, seed, n_samples=256, band=BAND):
    h = hessian_diagonal(model, loader, criterion, device, n_samples)
    finite_idx = torch.nonzero(h.isfinite()).flatten()
    order = finite_idx[h[finite_idx].argsort()]
    n = order.numel()
    pool = order[int(n * band[0]):int(n * band[1])]
    return pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(seed))]


if __name__ == "__main__":
    # self-check: python hessian.py
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.c2 = nn.Conv2d(8, 4, 3, padding=1)
            self.fc = nn.Linear(4 * 8 * 8, 5)

        def forward(self, x):
            x = self.c2(torch.relu(self.bn(self.c1(x))))
            return F.log_softmax(self.fc(x.flatten(1)), dim=1)

    loader = DataLoader(TensorDataset(torch.randn(64, 3, 8, 8),
                                      torch.randint(0, 5, (64,))), batch_size=8)

    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        model = Tiny().cpu()
        torch.manual_seed(0)
        carriers = curvature_carriers(model, loader, F.nll_loss, dev, seed=42, n_samples=32)

        sd = model.state_dict()
        names = [n for n in sd.keys() if "weight" in str(n)][:-1]
        total = sum(sd[n].numel() for n in names)
        assert names == ["c1.weight", "bn.weight", "c2.weight"], names
        assert carriers.device.type == "cpu", carriers.device
        assert carriers.min() >= 0 and carriers.max() < total
        assert carriers.unique().numel() == carriers.numel()

        bn_lo = sd["c1.weight"].numel()
        bn_hi = bn_lo + sd["bn.weight"].numel()
        assert not ((carriers >= bn_lo) & (carriers < bn_hi)).any(), "BN weight selected"

        torch.manual_seed(0)
        h = hessian_diagonal(model, loader, F.nll_loss, dev, 32)
        assert h.device.type == "cpu" and h.numel() == total
        n_finite = int(h.isfinite().sum())
        assert n_finite == total - (bn_hi - bn_lo)
        assert carriers.numel() == int(n_finite * BAND[1]) - int(n_finite * BAND[0])
        assert h[carriers].max() <= h[h.isfinite()].median(), "pool outside the flat band"

        torch.manual_seed(0)
        again = curvature_carriers(model, loader, F.nll_loss, dev, seed=42, n_samples=32)
        assert torch.equal(carriers, again), "seed does not pin the shuffle"
        print(f"{dev}: ok, pool {carriers.numel()}/{n_finite} carriers")
