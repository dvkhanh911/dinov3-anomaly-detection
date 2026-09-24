"""PatchCore scoring: bank self-kNN + Eq.7 reweighting + Gaussian smoothing."""

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import gaussian_blur


def bank_self_knn(bank, neighbors=9):
    """(M,C) GPU bank -> (M, neighbors+1) neighbour indices, self first."""
    out = []
    with torch.no_grad():
        for s in range(0, bank.shape[0], 2000):
            dist = torch.cdist(bank[s:s + 2000], bank, p=2)
            _, idx = dist.topk(neighbors + 1, dim=1, largest=False)
            out.append(idx[:, :neighbors + 1])
    return torch.cat(out, dim=0)


def score_queries(queries, bank, neighbor_idx):
    """One NN pass -> (per-patch raw distances on CPU, Eq.7 image score).

    Eq.7: s = (1 - softmax(d_self, d_1..d_b)[0]) * s*, stable softmax form
    following the anomalib reference implementation (includes s* itself).
    """
    queries = queries.to(bank.device)
    raw, back = [], []
    for j in range(0, queries.shape[0], 2048):
        dist, ix = torch.cdist(queries[j:j + 2048], bank, p=2).min(dim=1)
        raw.append(dist.cpu())
        back.append(ix.cpu())
    raw = torch.cat(raw)
    back = torch.cat(back)
    top = int(torch.argmax(raw))
    dist_star = float(raw[top])
    neighbours = neighbor_idx[int(back[top])]  # (b+1,), self first
    dist_10 = torch.cdist(queries[top:top + 1], bank[neighbours], p=2).squeeze(0)
    weight = 1.0 - F.softmax(dist_10, dim=0)[0]
    return raw, float(weight) * dist_star


def smooth_map(anomaly_map, kernel=17, sigma=4.0):
    """Gaussian smoothing of the upsampled anomaly map (PatchCore §3.3)."""
    return gaussian_blur(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        kernel_size=[kernel, kernel], sigma=[sigma, sigma]).squeeze()
