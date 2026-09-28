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


def fuse_anomaly_map(anomaly_map, seg_mask, alpha=1.0):
    """Fuse PatchCore anomaly map with predicted objectness/region mask.

    Residual Baseline Formula:
        refined_map = anomaly_map * (alpha + (1.0 - alpha) * seg_mask)

    Properties:
        - alpha in [0, 1].
        - When alpha = 1.0: refined_map == anomaly_map (100% PatchCore baseline).
        - When alpha < 1.0: background anomalies are softly attenuated by (1 - alpha),
          preventing overly aggressive suppression of true defects while removing
          spurious background fixture false positives.

    Args:
        anomaly_map: (H, W) or (B, H, W) PatchCore anomaly map (float tensor or ndarray).
        seg_mask: (H, W) or (B, H, W) objectness confidence / probability in [0, 1].
        alpha: Residual baseline weight in [0.0, 1.0]. Default 1.0 (baseline).

    Returns:
        Refined anomaly map in the same type (tensor or ndarray) as input.
    """
    is_numpy = not isinstance(anomaly_map, torch.Tensor)
    if is_numpy:
        import numpy as np
        map_t = torch.from_numpy(np.asarray(anomaly_map, dtype=np.float32))
        seg_t = torch.from_numpy(np.asarray(seg_mask, dtype=np.float32))
    else:
        map_t = anomaly_map
        seg_t = seg_mask.to(device=map_t.device, dtype=map_t.dtype)

    if seg_t.shape != map_t.shape:
        # Interpolate seg_t if spatial dimensions don't match
        seg_t = F.interpolate(
            seg_t.view(1, 1, *seg_t.shape[-2:]),
            size=map_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view_as(map_t)

    seg_clamped = torch.clamp(seg_t, min=0.0, max=1.0)
    alpha = float(np.clip(alpha, 0.0, 1.0) if is_numpy else torch.clamp(torch.tensor(alpha), 0.0, 1.0))
    refined = map_t * (alpha + (1.0 - alpha) * seg_clamped)

    if is_numpy:
        return refined.cpu().numpy()
    return refined


