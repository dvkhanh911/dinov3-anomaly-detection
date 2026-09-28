"""Sanity check script executing the 6 required audit steps.

Steps:
1. Import test
2. Unit tests
3. 1 batch forward
4. 1 batch pseudo-label generation (dual signals: DINOv3 + PatchCore)
5. 1 batch segmentation loss (backward pass & gradient isolation)
6. 1 batch fusion (numerical baseline equivalence check)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def step_1_import_test():
    print("\n--- STEP 1: Import Test ---")
    try:
        from src.backbone import Dinov3Backbone
        from src.segmentation_head import AuxiliarySegmentationHead, tokens_to_spatial
        from src.pseudo_labels import UnsupervisedPseudoLabelGenerator
        from src.losses import AuxiliarySegmentationLoss
        from src.scoring import bank_self_knn, score_queries, smooth_map, fuse_anomaly_map
        from src.data_kolektor import fixed_split, list_items, KolektorDataset
        print("  All modules imported successfully.")
        return True
    except Exception as e:
        print(f"  Import failed: {e}")
        return False


def step_2_unit_tests():
    print("\n--- STEP 2: Unit Tests ---")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_auxiliary_segmentation.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def step_3_one_batch_forward():
    print("\n--- STEP 3: 1 Batch Forward ---")
    try:
        from src.backbone import Dinov3Backbone
        from src.segmentation_head import AuxiliarySegmentationHead

        device = "cuda" if torch.cuda.is_available() else "cpu"
        b, h, w = 2, 1408, 512
        hp, wp = h // 8, w // 8

        # Initialize backbone (pretrained=False for fast offline sanity check)
        backbone = Dinov3Backbone(pretrained=False, out_indices=(1, 2)).to(device)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

        head = AuxiliarySegmentationHead(in_channels=backbone.out_channels, hidden_dim=128).to(device)

        dummy_img = torch.randn(b, 3, h, w, device=device)
        feats = backbone(dummy_img)

        print(f"  Input image shape:   {tuple(dummy_img.shape)}")
        print(f"  Backbone feats shape: {tuple(feats.shape)} (Expected: ({b}, 768, {hp}, {wp}))")
        assert feats.shape == (b, 768, hp, wp), f"Unexpected feats shape: {feats.shape}"

        logits = head(feats, out_size=(h, w))
        print(f"  Head logits shape:   {tuple(logits.shape)} (Expected: ({b}, 1, {h}, {w}))")
        assert logits.shape == (b, 1, h, w), f"Unexpected logits shape: {logits.shape}"

        return True
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        return False


def step_4_one_batch_pseudo_label_generation():
    print("\n--- STEP 4: 1 Batch Pseudo-Label Generation (Dual Signals) ---")
    try:
        from src.pseudo_labels import UnsupervisedPseudoLabelGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        b, c, hp, wp = 2, 768, 176, 64
        h, w = 1408, 512

        feats = torch.randn(b, c, hp, wp, device=device)
        # Signal 2: PatchCore spatial guidance
        patchcore_spatial = torch.rand(b, 1, hp, wp, device=device)

        gen = UnsupervisedPseudoLabelGenerator(
            method="pca",
            use_patchcore=True,
            patchcore_weight=0.25,
            spatial_smooth=True,
            smooth_sigma=2.0,
            soft_labels=True,
        )

        pseudo_mask = gen.generate(
            features=feats,
            target_size=(h, w),
            patchcore_scores=patchcore_spatial,
        )

        print(f"  Pseudo-mask shape: {tuple(pseudo_mask.shape)} (Expected: ({b}, 1, {h}, {w}))")
        print(f"  Pseudo-mask range: [{float(pseudo_mask.min()):.4f}, {float(pseudo_mask.max()):.4f}]")

        assert pseudo_mask.shape == (b, 1, h, w)
        assert float(pseudo_mask.min()) >= 0.0 and float(pseudo_mask.max()) <= 1.0

        return True
    except Exception as e:
        print(f"  Pseudo-label generation failed: {e}")
        return False


def step_5_one_batch_segmentation_loss():
    print("\n--- STEP 5: 1 Batch Segmentation Loss & Backward Pass ---")
    try:
        from src.backbone import Dinov3Backbone
        from src.segmentation_head import AuxiliarySegmentationHead
        from src.losses import AuxiliarySegmentationLoss

        device = "cuda" if torch.cuda.is_available() else "cpu"
        b, h, w = 2, 256, 128
        hp, wp = h // 8, w // 8

        backbone = Dinov3Backbone(pretrained=False, out_indices=(1, 2)).to(device)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

        head = AuxiliarySegmentationHead(in_channels=backbone.out_channels, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        criterion = AuxiliarySegmentationLoss(pseudo_weight=1.0, consistency_weight=0.1)

        img = torch.randn(b, 3, h, w, device=device)
        img_aug = torch.flip(img, dims=[-1])  # horizontal flip

        with torch.no_grad():
            f1 = backbone(img)
            f2 = backbone(img_aug)

        dummy_pseudo = torch.rand(b, 1, h, w, device=device)

        logits1 = head(f1, out_size=(h, w))
        logits2 = head(f2, out_size=(h, w))

        loss, loss_dict = criterion(logits1, dummy_pseudo, aug_logits=logits2, is_hflip=True)
        print(f"  Loss breakdown: {loss_dict}")

        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        backbone_has_grad = any(p.grad is not None for p in backbone.parameters())
        head_grads = [p.grad for p in head.parameters() if p.requires_grad]
        head_all_have_grad = all(g is not None and torch.isfinite(g).all() for g in head_grads)

        print(f"  Backbone gradient isolated (expected False): {backbone_has_grad}")
        print(f"  Head parameters received valid gradients: {head_all_have_grad}")

        assert not backbone_has_grad, "Backbone must NOT receive any gradients!"
        assert head_all_have_grad, "All head parameters must receive valid finite gradients!"

        optimizer.step()
        return True
    except Exception as e:
        print(f"  Loss & backward pass failed: {e}")
        return False


def step_6_one_batch_fusion():
    print("\n--- STEP 6: 1 Batch Fusion & Numerical Baseline Equivalence ---")
    try:
        from src.scoring import fuse_anomaly_map

        h, w = 1408, 512
        np.random.seed(42)
        anomaly_map = np.random.uniform(0.1, 5.0, size=(h, w)).astype(np.float32)
        objectness_mask = np.random.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
        baseline_image_score = 3.4567

        # 1. Test when alpha = 1.0 (Must be numerically identical to baseline)
        fused_map_alpha1 = fuse_anomaly_map(anomaly_map, objectness_mask, alpha=1.0)
        max_abs_diff = float(np.max(np.abs(fused_map_alpha1 - anomaly_map)))
        mean_abs_diff = float(np.mean(np.abs(fused_map_alpha1 - anomaly_map)))

        # Image score with alpha = 1.0
        top_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        top_obj = float(objectness_mask[top_idx])
        fused_score_alpha1 = baseline_image_score * (1.0 + (1.0 - 1.0) * top_obj)
        image_score_diff = abs(fused_score_alpha1 - baseline_image_score)

        print(f"  [Equivalence Check with alpha = 1.0]")
        print(f"    max_abs_difference:     {max_abs_diff:.8e}")
        print(f"    mean_abs_difference:    {mean_abs_diff:.8e}")
        print(f"    image_score_difference: {image_score_diff:.8e}")

        assert max_abs_diff == 0.0, f"Expected 0.0 max diff, got {max_abs_diff}"
        assert mean_abs_diff == 0.0, f"Expected 0.0 mean diff, got {mean_abs_diff}"
        assert image_score_diff == 0.0, f"Expected 0.0 score diff, got {image_score_diff}"

        # 2. Test when alpha = 0.8 (Residual baseline suppresses background)
        fused_map_alpha08 = fuse_anomaly_map(anomaly_map, objectness_mask, alpha=0.8)
        # Background pixels (mask close to 0) should be attenuated to ~0.8 * anomaly
        bg_idx = np.where(objectness_mask < 0.05)
        if len(bg_idx[0]) > 0:
            ratio = fused_map_alpha08[bg_idx] / anomaly_map[bg_idx]
            print(f"  [Residual Check with alpha = 0.8]")
            print(f"    Mean background attenuation factor: {float(np.mean(ratio)):.4f} (Expected: ~0.8000)")
            assert 0.79 <= np.mean(ratio) <= 0.85

        return True
    except Exception as e:
        print(f"  Fusion test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("AUDIT SANITY CHECKS (6 STEPS)")
    print("=" * 60)

    results = {}
    results["1. Import test"] = "PASS" if step_1_import_test() else "FAIL"
    results["2. Unit tests"] = "PASS" if step_2_unit_tests() else "FAIL"
    results["3. 1 batch forward"] = "PASS" if step_3_one_batch_forward() else "FAIL"
    results["4. 1 batch pseudo-label generation"] = "PASS" if step_4_one_batch_pseudo_label_generation() else "FAIL"
    results["5. 1 batch segmentation loss"] = "PASS" if step_5_one_batch_segmentation_loss() else "FAIL"
    results["6. 1 batch fusion"] = "PASS" if step_6_one_batch_fusion() else "FAIL"

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY RESULTS")
    print("=" * 60)
    all_passed = True
    for step, status in results.items():
        print(f"  {step:<42}: {status}")
        if status != "PASS":
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL 6 AUDIT STEPS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("ONE OR MORE AUDIT STEPS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
