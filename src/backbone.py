"""DINOv3 ConvNeXt feature extractor (timm, stages.1+stages.2)."""

import timm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Dinov3Backbone(nn.Module):
    """ConvNeXt-Base pretrained with DINOv3, two-hierarchy patch features.

    Mirrors PatchCore patch features (§3.1): 3x3 avg-pool neighbourhood
    aggregation (stride 1) + bilinear upsampling of the deeper hierarchy
    to the shallower resolution, concatenated along channels.
    """

    def __init__(self, name="convnext_base.dinov3_lvd1689m", out_indices=(1, 2),
                 pretrained=True, pool=3):
        super().__init__()
        self.model = timm.create_model(
            name, pretrained=pretrained, features_only=True,
            exportable=True, out_indices=tuple(out_indices),
        ).eval()
        self.pool = pool

    @property
    def out_channels(self):
        return sum(self.model.feature_info.channels())

    @torch.no_grad()
    def embed(self, x):
        """x: (1,3,H,W) normalized tensor -> (C,h,w) patch embedding."""
        f1, f2 = self.model(x)
        f1 = F.avg_pool2d(f1, self.pool, 1, 1)
        f2 = F.avg_pool2d(f2, self.pool, 1, 1)
        f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f1, f2], dim=1).squeeze(0)
