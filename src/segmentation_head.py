"""Auxiliary Unsupervised Segmentation Head for DINOv3 feature maps."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def tokens_to_spatial(
    features: torch.Tensor,
    h_patch: Optional[int] = None,
    w_patch: Optional[int] = None,
) -> torch.Tensor:
    """Ensure features are in spatial layout [B, C, H_patch, W_patch].

    Supports:
    - [B, C, H_patch, W_patch] -> returned as-is
    - [B, N, C] -> reshaped to [B, C, H_patch, W_patch]
    - [C, H_patch, W_patch] -> unsqueezed to [1, C, H_patch, W_patch]
    """
    if features.dim() == 4:
        # Already [B, C, H, W]
        return features

    if features.dim() == 3:
        if features.shape[0] > 0 and features.shape[1] > 0 and features.shape[2] > 0:
            # Check if shape is [C, H, W]
            if h_patch is not None and w_patch is not None:
                if features.shape[1] == h_patch and features.shape[2] == w_patch:
                    return features.unsqueeze(0)
                # Check if shape is [B, N, C]
                if features.shape[1] == h_patch * w_patch:
                    b, n, c = features.shape
                    return features.permute(0, 2, 1).reshape(b, c, h_patch, w_patch)

            # If no h_patch, w_patch given, check if square or guess
            b, n, c = features.shape
            side = int(n ** 0.5)
            if side * side == n:
                return features.permute(0, 2, 1).reshape(b, c, side, side)

    raise ValueError(
        f"Cannot convert feature tensor of shape {features.shape} to spatial layout. "
        f"Expected [B, C, H, W] or [B, N, C] with valid (h_patch, w_patch)."
    )


class AuxiliarySegmentationHead(nn.Module):
    """Lightweight 2D convolutional segmentation head for frozen DINOv3 features.

    Architecture:
        DINOv3 feature map [B, C, H_patch, W_patch]
                ↓
        Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
                ↓
        ReLU
                ↓
        Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                ↓
        ReLU
                ↓
        Conv2d(hidden_dim, num_classes, kernel_size=1)
                ↓
        Mask logits [B, 1, H_patch, W_patch]
                ↓
        Bilinear Interpolate -> [B, 1, H_image, W_image]
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 1,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        out_size: Optional[Tuple[int, int]] = None,
        h_patch: Optional[int] = None,
        w_patch: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: [B, C, H_patch, W_patch] or [B, N, C]
            out_size: Optional (H_image, W_image) to upsample logits.
            h_patch: Optional patch height if features are [B, N, C].
            w_patch: Optional patch width if features are [B, N, C].

        Returns:
            logits: [B, num_classes, H_out, W_out]
        """
        x = tokens_to_spatial(features, h_patch=h_patch, w_patch=w_patch)

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected feature channel dimension {self.in_channels}, but got {x.shape[1]}."
            )

        logits = self.net(x)

        if out_size is not None:
            if self.upsample_mode in ("bilinear", "bicubic"):
                logits = F.interpolate(
                    logits,
                    size=out_size,
                    mode=self.upsample_mode,
                    align_corners=self.align_corners,
                )
            else:
                logits = F.interpolate(
                    logits,
                    size=out_size,
                    mode=self.upsample_mode,
                )

        return logits

    @torch.no_grad()
    def predict_mask(
        self,
        features: torch.Tensor,
        out_size: Optional[Tuple[int, int]] = None,
        threshold: float = 0.5,
        return_prob: bool = False,
    ) -> torch.Tensor:
        """Predict binary mask or probability map.

        Args:
            features: Input features from backbone.
            out_size: (H_image, W_image) target resolution.
            threshold: Binarization threshold on sigmoid probabilities.
            return_prob: If True, returns sigmoid probabilities in [0, 1].

        Returns:
            torch.Tensor: [B, 1, H, W]
        """
        logits = self.forward(features, out_size=out_size)
        probs = torch.sigmoid(logits)
        if return_prob:
            return probs
        return (probs >= threshold).float()
