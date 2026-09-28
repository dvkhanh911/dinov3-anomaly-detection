"""Train Auxiliary Unsupervised Segmentation Head on normal images without ground-truth masks.

Zero defect mask / label leakage:
- Only normal images from fixed train split are used.
- DINOv3 Backbone is frozen and kept in eval mode.
- Only the segmentation head parameters are trained.
- Checkpoint saves only the head weights, optimizer state, epoch, and config.

Usage:
    python scripts/train_segmentation.py --config configs/train_segmentation.yaml
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone import Dinov3Backbone
from src.coreset import greedy_coreset
from src.data_kolektor import fixed_split, list_items, KolektorDataset
from src.losses import AuxiliarySegmentationLoss
from src.pseudo_labels import UnsupervisedPseudoLabelGenerator
from src.scoring import bank_self_knn, score_queries
from src.segmentation_head import AuxiliarySegmentationHead


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Auxiliary Unsupervised Segmentation Head.")
    parser.add_argument("--config", default="configs/train_segmentation.yaml", help="Path to config yaml.")
    parser.add_argument("--out", default=None, help="Output directory for checkpoints.")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    print(f"Using device: {device}", flush=True)

    out_dir = args.out or os.path.dirname(cfg["segmentation"].get("checkpoint", "checkpoints/segmentation_head_best.pth"))
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, "segmentation_head_best.pth")

    w, h = cfg["data"]["width"], cfg["data"]["height"]
    base_preprocess = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["features"]["mean"], std=cfg["features"]["std"]),
    ])

    # Consistency augmentation (horizontal flip)
    aug_preprocess = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["features"]["mean"], std=cfg["features"]["std"]),
    ])

    # 1. Dataset & Split (STRICTLY NORMAL IMAGES ONLY - ZERO LEAKAGE)
    root = cfg["data"]["root"]
    all_items = list_items(root)
    train_files, _ = fixed_split(all_items, n_train_good=cfg["data"]["train_good"], seed=cfg["seed"])

    # Strict leakage assertions
    assert len(train_files) == cfg["data"]["train_good"], (
        f"Expected {cfg['data']['train_good']} normal images, got {len(train_files)}"
    )
    for f in train_files:
        assert not f.endswith("_label.bmp") and not f.endswith(".bmp"), (
            f"LEAKAGE ERROR: Ground-truth label/mask file found in training set: {f}"
        )
    print(f"Verified: {len(train_files)} normal training images. Zero defect masks or labels loaded.", flush=True)

    train_dataset = KolektorDataset(
        root=root,
        file_list=train_files,
        transform=base_preprocess,
        aug_transform=aug_preprocess,
    )
    batch_size = cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 2. Frozen DINOv3 Backbone
    print(f"Initializing DINOv3 Backbone ({cfg['backbone']['name']})...", flush=True)
    backbone = Dinov3Backbone(
        name=cfg["backbone"]["name"],
        out_indices=cfg["backbone"]["out_indices"],
        pretrained=cfg["backbone"]["pretrained"],
        pool=cfg["features"]["pool"],
    ).to(device)

    # Explicitly freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    in_channels = backbone.out_channels
    print(f"Backbone frozen successfully. Output channels = {in_channels}", flush=True)

    # 3. PatchCore Memory Bank (for Dual-Signal Pseudo-Labeling: Signal 2)
    use_pc_guidance = cfg["pseudo_label"].get("use_patchcore", True)
    bank, neighbors = None, None
    if use_pc_guidance:
        print("Extracting normal train features for PatchCore spatial structural guidance...", flush=True)
        train_features_list = []
        from PIL import Image
        with torch.no_grad():
            for rel_path in train_files:
                img_real = base_preprocess(Image.open(os.path.join(root, rel_path)).convert("RGB"))
                feat = backbone.embed(img_real.unsqueeze(0).to(device))
                train_features_list.append(feat.permute(1, 2, 0).reshape(-1, in_channels))

        bank_full = torch.cat(train_features_list, dim=0)
        coreset_idx = greedy_coreset(
            bank_full, cfg["coreset"]["size"],
            proj_dim=cfg["coreset"]["proj_dim"],
            batch_add=cfg["coreset"]["batch_add"],
            seed=cfg["seed"], device=device,
        )
        bank = bank_full[coreset_idx].to(device).contiguous()
        neighbors = bank_self_knn(bank, cfg["scoring"]["neighbors"])
        del bank_full
        print(f"PatchCore bank built with shape {tuple(bank.shape)}", flush=True)

    # 4. Auxiliary Segmentation Head & Optimizer
    head = AuxiliarySegmentationHead(
        in_channels=in_channels,
        hidden_dim=cfg["segmentation"].get("hidden_dim", 256),
        num_classes=cfg["segmentation"].get("num_classes", 1),
        upsample_mode="bilinear",
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    # Audit trainable parameters: Backbone must be 0, Head must be > 0
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Audit Parameter Check -> Backbone trainable params: {backbone_trainable}")
    print(f"Audit Parameter Check -> Auxiliary Segmentation Head trainable params: {head_trainable}")
    assert backbone_trainable == 0, f"Backbone must have 0 trainable parameters, but got {backbone_trainable}!"
    assert head_trainable > 0, "Auxiliary segmentation head must have trainable parameters!"

    # Verify optimizer parameter group ONLY contains head parameters
    opt_params = set(p for group in optimizer.param_groups for p in group["params"])
    head_params = set(head.parameters())
    assert opt_params == head_params, "Optimizer must ONLY update auxiliary segmentation head parameters!"



    # 5. Pseudo-Label Generator & Loss
    pseudo_gen = UnsupervisedPseudoLabelGenerator(
        method=cfg["pseudo_label"].get("clustering_type", "pca"),
        use_patchcore=use_pc_guidance and bank is not None,
        patchcore_weight=float(cfg["pseudo_label"].get("patchcore_weight", 0.2)),
        spatial_smooth=cfg["pseudo_label"].get("spatial_smooth", True),
        smooth_sigma=float(cfg["pseudo_label"].get("smooth_sigma", 2.0)),
        soft_labels=cfg["pseudo_label"].get("soft_labels", True),
        threshold=cfg["pseudo_label"].get("threshold", 0.5),
    )

    criterion = AuxiliarySegmentationLoss(
        pseudo_weight=float(cfg["loss"].get("pseudo_weight", 1.0)),
        consistency_weight=float(cfg["loss"].get("consistency_weight", 0.1)),
    )

    # Helper function to compute PatchCore spatial anomaly cues for normal images
    def get_patchcore_scores(features_tensor: torch.Tensor) -> torch.Tensor:
        if bank is None or neighbors is None:
            return None
        b, c, hp, wp = features_tensor.shape
        scores = []
        for i in range(b):
            queries = features_tensor[i].permute(1, 2, 0).reshape(-1, c)
            raw, _ = score_queries(queries, bank, neighbors)
            scores.append(raw.reshape(hp, wp).to(device))
        return torch.stack(scores, dim=0).unsqueeze(1)  # [B, 1, hp, wp]

    # 6. Training Loop
    epochs = int(cfg["training"].get("epochs", 20))
    print(f"Starting training Auxiliary Segmentation Head for {epochs} epochs...", flush=True)

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        head.train()
        epoch_total_loss = 0.0
        epoch_pseudo_loss = 0.0
        epoch_consistency_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            aug_images = batch["aug_image"].to(device)

            optimizer.zero_grad()

            # Forward frozen backbone without gradients
            with torch.no_grad():
                feats = backbone(images)
                aug_feats = backbone(aug_images)

                pc_scores = get_patchcore_scores(feats) if use_pc_guidance else None
                pseudo_masks = pseudo_gen.generate(
                    features=feats,
                    target_size=(h, w),
                    patchcore_scores=pc_scores,
                )

            # Forward segmentation head
            pred_logits = head(feats, out_size=(h, w))
            aug_logits = head(aug_feats, out_size=(h, w))

            # Compute combined loss (horizontal flip consistency)
            loss, loss_dict = criterion(
                pred_logits=pred_logits,
                pseudo_mask=pseudo_masks,
                aug_logits=aug_logits,
                is_hflip=True,
            )

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss_dict["total_loss"]
            epoch_pseudo_loss += loss_dict["pseudo_loss"]
            epoch_consistency_loss += loss_dict["consistency_loss"]
            num_batches += 1

        avg_total = epoch_total_loss / max(1, num_batches)
        avg_pseudo = epoch_pseudo_loss / max(1, num_batches)
        avg_consistency = epoch_consistency_loss / max(1, num_batches)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_total:.5f} | "
            f"Pseudo Loss: {avg_pseudo:.5f} | "
            f"Consistency Loss: {avg_consistency:.5f}",
            flush=True,
        )

        # Save best checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "segmentation_head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": cfg,
            }, best_ckpt_path)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best checkpoint saved to: {best_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
