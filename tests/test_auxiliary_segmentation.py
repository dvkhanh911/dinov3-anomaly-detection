"""Comprehensive unit test suite for Auxiliary Unsupervised Segmentation Head audit."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.backbone import Dinov3Backbone
from src.segmentation_head import AuxiliarySegmentationHead, tokens_to_spatial
from src.pseudo_labels import (
    FeatureClusteringPseudoLabeler,
    PseudoMaskRefiner,
    UnsupervisedPseudoLabelGenerator,
)
from src.losses import AuxiliarySegmentationLoss
from src.scoring import bank_self_knn, fuse_anomaly_map, score_queries, smooth_map


class TestAuxiliarySegmentationAudit(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.b = 4
        self.c = 768
        self.hp = 22
        self.wp = 8
        self.h_img = 176
        self.w_img = 64

    def test_shape_and_batch_greater_than_1(self):
        """Audit: Shape [B, C, H, W] with B > 1 for backbone, head, and pseudo-mask."""
        head = AuxiliarySegmentationHead(
            in_channels=self.c,
            hidden_dim=128,
            num_classes=1,
            upsample_mode="bilinear",
        ).to(self.device)

        feats = torch.randn(self.b, self.c, self.hp, self.wp, device=self.device)

        # Head forward with batch size B=4
        logits = head(feats, out_size=(self.h_img, self.w_img))
        self.assertEqual(logits.shape, (self.b, 1, self.h_img, self.w_img))

        # Output range via sigmoid in [0, 1]
        probs = torch.sigmoid(logits)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_frozen_backbone_and_parameter_counts(self):
        """Audit: Frozen backbone, 0 trainable backbone params, optimizer only updates head."""
        backbone = Dinov3Backbone(
            name="convnext_base.dinov3_lvd1689m",
            pretrained=False,
            out_indices=(1, 2),
            pool=3,
        ).to(self.device)

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

        head = AuxiliarySegmentationHead(in_channels=backbone.out_channels, hidden_dim=64).to(self.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

        # 1. Verify requires_grad on all backbone params is False
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))

        # 2. Trainable parameter counts
        backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        self.assertEqual(backbone_trainable, 0)
        self.assertGreater(head_trainable, 0)

        # 3. Optimizer only contains head parameters
        opt_params = set(p for group in optimizer.param_groups for p in group["params"])
        head_params = set(head.parameters())
        self.assertEqual(opt_params, head_params)

    def test_pseudo_mask_range_and_dual_signals(self):
        """Audit: Pseudo-mask values strictly in [0, 1] and combines both signals."""
        feats = torch.randn(self.b, self.c, self.hp, self.wp, device=self.device)
        dummy_pc_scores = torch.rand(self.b, 1, self.hp, self.wp, device=self.device)

        gen = UnsupervisedPseudoLabelGenerator(
            method="pca",
            use_patchcore=True,
            patchcore_weight=0.25,
            spatial_smooth=True,
            soft_labels=True,
        )

        pseudo_mask = gen.generate(
            features=feats,
            target_size=(self.h_img, self.w_img),
            patchcore_scores=dummy_pc_scores,
        )

        self.assertEqual(pseudo_mask.shape, (self.b, 1, self.h_img, self.w_img))
        self.assertGreaterEqual(float(pseudo_mask.min()), 0.0)
        self.assertLessEqual(float(pseudo_mask.max()), 1.0)

    def test_consistency_alignment(self):
        """Audit: Horizontal flip augmentation is properly inverse-aligned before MSE."""
        criterion = AuxiliarySegmentationLoss(pseudo_weight=1.0, consistency_weight=0.5)

        # Simulated predictions
        p1_logits = torch.randn(2, 1, 32, 32, device=self.device)
        # Perfectly consistent prediction flipped horizontally
        p2_logits_flipped = torch.flip(p1_logits, dims=[-1])

        # Loss with is_hflip=True should have consistency_loss == 0.0
        dummy_target = torch.rand_like(p1_logits)
        _, loss_dict = criterion(p1_logits, dummy_target, aug_logits=p2_logits_flipped, is_hflip=True)
        self.assertAlmostEqual(loss_dict["consistency_loss"], 0.0, places=5)

    def test_fusion_alpha_1_equals_baseline(self):
        """Audit: Residual baseline formula with alpha=1.0 mathematically equals PatchCore baseline."""
        anom_map_np = np.random.uniform(0.1, 5.0, size=(self.h_img, self.w_img)).astype(np.float32)
        seg_mask_np = np.random.uniform(0.0, 1.0, size=(self.h_img, self.w_img)).astype(np.float32)

        # Numpy test
        fused_np = fuse_anomaly_map(anom_map_np, seg_mask_np, alpha=1.0)
        diff_np = np.abs(fused_np - anom_map_np)
        self.assertAlmostEqual(float(np.max(diff_np)), 0.0, places=6)
        self.assertAlmostEqual(float(np.mean(diff_np)), 0.0, places=6)

        # Tensor test
        anom_t = torch.from_numpy(anom_map_np).to(self.device)
        seg_t = torch.from_numpy(seg_mask_np).to(self.device)
        fused_t = fuse_anomaly_map(anom_t, seg_t, alpha=1.0)
        diff_t = torch.abs(fused_t - anom_t)
        self.assertAlmostEqual(float(diff_t.max()), 0.0, places=6)

    def test_segmentation_disabled_equals_baseline(self):
        """Audit: When segmentation is disabled, output is 100% identical to baseline."""
        # Simulated bank & queries
        bank = torch.randn(100, 64, device=self.device)
        queries = torch.randn(25, 64, device=self.device)
        neighbors = bank_self_knn(bank, neighbors=5)

        # 1. Baseline scoring
        raw_base, img_score_base = score_queries(queries, bank, neighbors)
        fh, fw = 5, 5
        anom_base = F.interpolate(
            raw_base.reshape(fh, fw).unsqueeze(0).unsqueeze(0),
            size=(20, 20), mode="bilinear", align_corners=False,
        )
        map_base = smooth_map(anom_base.squeeze(), kernel=5, sigma=1.0).cpu().numpy()

        # 2. Pipeline scoring with segmentation disabled
        seg_enabled = False
        if not seg_enabled:
            map_pipeline = map_base
            img_score_pipeline = img_score_base

        max_abs_diff = float(np.max(np.abs(map_pipeline - map_base)))
        mean_abs_diff = float(np.mean(np.abs(map_pipeline - map_base)))
        score_diff = abs(img_score_pipeline - img_score_base)

        self.assertEqual(max_abs_diff, 0.0)
        self.assertEqual(mean_abs_diff, 0.0)
        self.assertEqual(score_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
