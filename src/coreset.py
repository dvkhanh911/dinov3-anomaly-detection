"""Greedy minimax coreset (PatchCore Alg.1) with JL random projection."""

import torch


def greedy_coreset(memory, size, proj_dim=128, batch_add=128, seed=42, device="cuda"):
    """Select `size` indices from memory (N,C) covering the feature support.

    Follows Roth et al. Alg.1: fixed Gaussian projection psi (d -> proj_dim),
    then iterative argmax-min-distance selection. Adds `batch_add` points per
    round instead of one (same result in practice, ~100x faster).
    """
    gen = torch.Generator().manual_seed(seed)
    n, c = memory.shape
    dev = torch.device(device)
    proj = (torch.randn(c, proj_dim, generator=gen) / (proj_dim ** 0.5)).to(dev)
    projected = (memory.to(dev) @ proj)
    selected = [int(torch.randint(n, (1,), generator=gen))]
    chunk = 100000
    with torch.no_grad():
        min_dist = torch.full((n,), 1e18)
        for s in range(0, n, chunk):
            min_dist[s:s + chunk] = torch.cdist(
                projected[s:s + chunk], projected[selected], p=2).squeeze(1).cpu()
        min_dist[selected[0]] = -1.0
        while len(selected) < size:
            k = min(batch_add, size - len(selected))
            top = torch.topk(min_dist, k).indices.tolist()
            selected.extend(top)
            new_points = projected[top]
            for s in range(0, n, chunk):
                dist = torch.cdist(projected[s:s + chunk], new_points, p=2).min(dim=1).values
                min_dist[s:s + chunk] = torch.minimum(min_dist[s:s + chunk], dist.cpu())
            for t in top:
                min_dist[t] = -1.0
    return torch.tensor(selected)
