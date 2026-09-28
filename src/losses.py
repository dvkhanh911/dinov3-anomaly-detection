"""Loss functions for training Auxiliary Unsupervised Segmentation Head."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class AuxiliarySegmentationLoss(nn.Module):
    """Total loss = L_pseudo + lambda_consistency * L_consistency.

    L_pseudo:
        Binary Cross Entropy with Logits against unsupervised pseudo-mask.
        Supports both soft pseudo-labels [0, 1] and binary masks.

    L_consistency:
        Mean Squared Error between predicted probabilities of original image
        and augmented image: MSE(sigmoid(P1), sigmoid(P2)).
        If horizontal flip is used, P2 is flipped back before computing MSE.

    Note on Geometric Transformations:
        Currently, consistency loss supports photometric augmentations (e.g.
        color jitter / noise) and horizontal flip with spatial re-alignment.
        Arbitrary affine or non-rigid deformations would require backward flow
        warping, which is omitted to keep the auxiliary pipeline lightweight and robust.
    """

    def __init__(
        self,
        pseudo_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.pseudo_weight = pseudo_weight
        self.consistency_weight = consistency_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        pseudo_mask: torch.Tensor,
        aug_logits: Optional[torch.Tensor] = None,
        is_hflip: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred_logits: [B, 1, H, W] raw logits for original images.
            pseudo_mask: [B, 1, H, W] target pseudo-mask in [0, 1].
            aug_logits: Optional [B, 1, H, W] raw logits for augmented images.
            is_hflip: Whether aug_logits corresponds to horizontal flip of original.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: total_loss and loss breakdown metrics.
        """
        # Ensure pseudo_mask matches spatial resolution of pred_logits
        if pseudo_mask.shape[-2:] != pred_logits.shape[-2:]:
            pseudo_mask = F.interpolate(
                pseudo_mask, size=pred_logits.shape[-2:], mode="bilinear", align_corners=False
            )

        l_pseudo = self.bce_loss(pred_logits, pseudo_mask)
        total_loss = self.pseudo_weight * l_pseudo
        loss_dict = {"pseudo_loss": l_pseudo.item()}

        l_consistency = torch.tensor(0.0, device=pred_logits.device)
        if aug_logits is not None and self.consistency_weight > 0.0:
            p1 = torch.sigmoid(pred_logits)
            p2 = torch.sigmoid(aug_logits)

            if is_hflip:
                # Flip back horizontally along width dimension
                p2 = torch.flip(p2, dims=[-1])

            l_consistency = F.mse_loss(p1, p2)
            total_loss = total_loss + self.consistency_weight * l_consistency
            loss_dict["consistency_loss"] = l_consistency.item()
        else:
            loss_dict["consistency_loss"] = 0.0

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
