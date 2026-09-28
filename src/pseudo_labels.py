"""Unsupervised pseudo region mask generation from DINOv3 patch features and PatchCore cues.

Architecture & Dual-Signal Design:
---------------------------------
Pseudo-mask generation relies on TWO distinct, complementary unsupervised signals:

1. Signal 1 (Objectness / Region Saliency - S_feat):
   - Extracted from self-supervised DINOv3 representations (PCA of patch tokens,
     perimeter background contrast, or 2-cluster feature assignment).
   - Meaning: S_feat identifies the coarse footprint of the inspected component
     (commutator body) versus flat industrial background / mounting fixtures.
   - S_feat produces the primary objectness prior.

2. Signal 2 (PatchCore Spatial Structural Guidance - S_pc):
   - Extracted from patch nearest-neighbor distance against the normal memory bank:
     d(p) = min_{m in Memory} ||f(p) - m||_2.
   - Meaning: On normal images, S_pc reflects local patch structural contrast,
     functional facets, and fine surface texture variations relative to uniform background.
   - Crucial constraint: S_pc is NEVER treated directly as a defect mask or semantic class,
     but as an auxiliary spatial signal to sharpen object boundaries and suppress uniform
     non-informative background regions.

Combination & Refinement Pipeline:
---------------------------------
1. Normalization:
   - S_feat and S_pc are individually min-max normalized per image to [0, 1]:
     S_norm = (S - S_min) / (S_max - S_min + eps).
2. Combination:
   - S_combined = (1 - w_pc) * S_feat_norm + w_pc * S_pc_norm, where w_pc in [0.1, 0.4].
3. Upsampling:
   - Bilinear interpolation to target image resolution (H_image, W_image).
4. Spatial Smoothing:
   - 2D Gaussian blur (sigma=2.0) to eliminate isolated single-patch noise artifacts
     and ensure contiguous region masks.
5. Final Normalization & Thresholding:
   - Re-normalized to [0, 1].
   - Soft pseudo-masks (continuous probabilities in [0, 1]) are retained by default
     to preserve uncertainty at object boundaries during BCE training.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import gaussian_blur


def compute_border_mask(h_patch: int, w_patch: int, border_ratio: float = 0.08) -> torch.Tensor:
    """Boolean mask indicating perimeter patches (industrial fixture/background)."""
    mask = torch.zeros((h_patch, w_patch), dtype=torch.bool)
    pad_h = max(1, int(h_patch * border_ratio))
    pad_w = max(1, int(w_patch * border_ratio))
    mask[:pad_h, :] = True
    mask[-pad_h:, :] = True
    mask[:, :pad_w] = True
    mask[:, -pad_w:] = True
    return mask


class FeatureClusteringPseudoLabeler:
    """Extract objectness prior (Signal 1) from self-supervised DINOv3 features.

    Supports:
    - 'pca': 1st principal component of patch tokens (well-known DINO foreground clustering).
    - 'border_cosine': Cosine dissimilarity from perimeter background patch features.
    - 'kmeans': 2-cluster KMeans clustering on patch features.
    """

    def __init__(
        self,
        method: str = "pca",
        border_ratio: float = 0.08,
    ):
        self.method = method.lower()
        self.border_ratio = border_ratio

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Generate coarse objectness saliency map from DINOv3 features.

        Args:
            features: [B, C, H_patch, W_patch] spatial feature map.

        Returns:
            torch.Tensor: [B, 1, H_patch, W_patch] normalized objectness in [0, 1].
        """
        b, c, hp, wp = features.shape
        device = features.device
        border_mask = compute_border_mask(hp, wp, self.border_ratio).to(device)

        maps = []
        for i in range(b):
            f_img = features[i]  # [C, hp, wp]
            tokens = f_img.permute(1, 2, 0).reshape(-1, c)  # [N, C], N = hp*wp

            if self.method == "pca":
                saliency = self._saliency_pca(tokens, hp, wp, border_mask)
            elif self.method == "border_cosine":
                saliency = self._saliency_border_cosine(tokens, hp, wp, border_mask)
            elif self.method == "kmeans":
                saliency = self._saliency_kmeans(tokens, hp, wp, border_mask)
            else:
                saliency = self._saliency_pca(tokens, hp, wp, border_mask)

            maps.append(saliency)

        return torch.stack(maps, dim=0)

    def _saliency_pca(
        self,
        tokens: torch.Tensor,
        hp: int,
        wp: int,
        border_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute objectness using 1st principal component of patch features."""
        tokens_centered = tokens - tokens.mean(dim=0, keepdim=True)
        try:
            _, _, v = torch.pca_lowrank(tokens_centered, q=1, center=False)
            proj = (tokens_centered @ v[:, :1]).squeeze(1)  # [N]
        except Exception:
            v = torch.randn(tokens.shape[1], 1, device=tokens.device)
            for _ in range(5):
                v = tokens_centered.T @ (tokens_centered @ v)
                v = v / (torch.norm(v) + 1e-8)
            proj = (tokens_centered @ v).squeeze(1)

        proj_2d = proj.reshape(hp, wp)

        # Polarity check: background perimeter should have lower value than object center
        border_val = proj_2d[border_mask].mean()
        center_mask = ~border_mask
        center_val = proj_2d[center_mask].mean()

        if border_val > center_val:
            proj_2d = -proj_2d

        # Min-max normalize to [0, 1]
        p_min, p_max = proj_2d.min(), proj_2d.max()
        norm_map = (proj_2d - p_min) / (p_max - p_min + 1e-8)
        return norm_map.unsqueeze(0)  # [1, hp, wp]

    def _saliency_border_cosine(
        self,
        tokens: torch.Tensor,
        hp: int,
        wp: int,
        border_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dissimilarity against perimeter background patch features."""
        tokens_norm = F.normalize(tokens, p=2, dim=1)  # [N, C]
        border_indices = border_mask.reshape(-1).nonzero(as_tuple=True)[0]
        bg_proto = tokens_norm[border_indices].mean(dim=0, keepdim=True)
        bg_proto = F.normalize(bg_proto, p=2, dim=1)

        cos_sim = (tokens_norm @ bg_proto.T).squeeze(1)
        dissim = 1.0 - cos_sim
        dissim_2d = dissim.reshape(hp, wp)

        d_min, d_max = dissim_2d.min(), dissim_2d.max()
        norm_map = (dissim_2d - d_min) / (d_max - d_min + 1e-8)
        return norm_map.unsqueeze(0)

    def _saliency_kmeans(
        self,
        tokens: torch.Tensor,
        hp: int,
        wp: int,
        border_mask: torch.Tensor,
    ) -> torch.Tensor:
        """2-cluster unsupervised region assignment."""
        tokens_np = tokens.detach().cpu().numpy()
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=256, n_init=3)
        labels = km.fit_predict(tokens_np)
        labels_2d = torch.tensor(labels, device=tokens.device).reshape(hp, wp).float()

        c0_border = (labels_2d == 0)[border_mask].sum().item()
        c1_border = (labels_2d == 1)[border_mask].sum().item()

        if c0_border > c1_border:
            fg_map = (labels_2d == 1).float()
        else:
            fg_map = (labels_2d == 0).float()

        return fg_map.unsqueeze(0)


class PseudoMaskRefiner:
    """Refine and smooth coarse objectness and PatchCore signals into pseudo region masks."""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        spatial_smooth: bool = True,
        smooth_sigma: float = 2.0,
        soft_labels: bool = True,
        threshold: Optional[float] = 0.5,
    ):
        self.target_size = target_size
        self.spatial_smooth = spatial_smooth
        self.smooth_sigma = smooth_sigma
        self.soft_labels = soft_labels
        self.threshold = threshold

    def refine(
        self,
        saliency_map: torch.Tensor,
        patchcore_score: Optional[torch.Tensor] = None,
        patchcore_weight: float = 0.25,
    ) -> torch.Tensor:
        """Combine feature objectness (Signal 1) and PatchCore spatial cues (Signal 2).

        Formula:
            S_combined = (1 - w_pc) * S_feat + w_pc * S_pc

        Args:
            saliency_map: [B, 1, hp, wp] coarse objectness from DINOv3 features.
            patchcore_score: Optional [B, 1, hp, wp] or [B, 1, H, W] PatchCore spatial distances.
            patchcore_weight: Weight given to PatchCore spatial guidance [0.0, 1.0].

        Returns:
            torch.Tensor: [B, 1, H_target, W_target] pseudo region mask in [0, 1].
        """
        combined = saliency_map

        if patchcore_score is not None and patchcore_weight > 0.0:
            if patchcore_score.shape[-2:] != combined.shape[-2:]:
                patchcore_score = F.interpolate(
                    patchcore_score, size=combined.shape[-2:], mode="bilinear", align_corners=False
                )
            # Normalize patchcore score per sample to [0, 1]
            b = combined.shape[0]
            norm_pc = []
            for i in range(b):
                pc = patchcore_score[i]
                p_min, p_max = pc.min(), pc.max()
                norm_pc.append((pc - p_min) / (p_max - p_min + 1e-8))
            norm_pc = torch.stack(norm_pc, dim=0)

            # Convex combination of Signal 1 (objectness) and Signal 2 (spatial guidance)
            combined = (1.0 - patchcore_weight) * combined + patchcore_weight * norm_pc

        # Upsample to target image resolution
        if self.target_size is not None:
            combined = F.interpolate(
                combined, size=self.target_size, mode="bilinear", align_corners=False
            )

        # Spatial Gaussian smoothing to suppress single-patch noise
        if self.spatial_smooth and self.smooth_sigma > 0:
            kernel_size = int(2 * round(2 * self.smooth_sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            combined = gaussian_blur(
                combined, kernel_size=[kernel_size, kernel_size], sigma=[self.smooth_sigma, self.smooth_sigma]
            )

        # Per-sample min-max normalization to [0, 1]
        refined = []
        for i in range(combined.shape[0]):
            m = combined[i]
            m_min, m_max = m.min(), m.max()
            m_norm = (m - m_min) / (m_max - m_min + 1e-8)
            refined.append(m_norm)
        refined = torch.stack(refined, dim=0)

        # Soft pseudo-masks vs hard thresholding
        if not self.soft_labels and self.threshold is not None:
            refined = (refined >= self.threshold).float()

        return refined


class UnsupervisedPseudoLabelGenerator:
    """Unified generator for unsupervised pseudo region masks from DINOv3 + PatchCore."""

    def __init__(
        self,
        method: str = "pca",
        border_ratio: float = 0.08,
        use_patchcore: bool = True,
        patchcore_weight: float = 0.25,
        spatial_smooth: bool = True,
        smooth_sigma: float = 2.0,
        soft_labels: bool = True,
        threshold: Optional[float] = 0.5,
    ):
        self.clusterer = FeatureClusteringPseudoLabeler(method=method, border_ratio=border_ratio)
        self.use_patchcore = use_patchcore
        self.patchcore_weight = patchcore_weight if use_patchcore else 0.0
        self.smooth_sigma = smooth_sigma
        self.spatial_smooth = spatial_smooth
        self.soft_labels = soft_labels
        self.threshold = threshold

    def generate(
        self,
        features: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        patchcore_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate unsupervised pseudo region masks.

        Requires two signals when use_patchcore=True:
        1. DINOv3 spatial feature map (Signal 1: objectness).
        2. PatchCore memory distances (Signal 2: spatial structural guidance).

        Args:
            features: [B, C, H_patch, W_patch] spatial feature map.
            target_size: Target image size (H_image, W_image) to upsample.
            patchcore_scores: Optional [B, 1, hp, wp] PatchCore anomaly distances.

        Returns:
            torch.Tensor: [B, 1, H_target, W_target] pseudo region mask in [0, 1].
        """
        raw_saliency = self.clusterer(features)

        refiner = PseudoMaskRefiner(
            target_size=target_size,
            spatial_smooth=self.spatial_smooth,
            smooth_sigma=self.smooth_sigma,
            soft_labels=self.soft_labels,
            threshold=self.threshold,
        )

        pseudo_mask = refiner.refine(
            raw_saliency,
            patchcore_score=patchcore_scores if self.use_patchcore else None,
            patchcore_weight=self.patchcore_weight,
        )
        return pseudo_mask
