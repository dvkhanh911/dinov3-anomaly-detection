"""Evaluate PatchCore + Auxiliary Unsupervised Segmentation Head on KolektorSDD.

Capabilities:
1. Baseline PatchCore mode (when segmentation.enabled = false):
   Behaves 100% identically to eval_baseline.py.
2. Auxiliary Segmentation Head mode (when segmentation.enabled = true):
   Evaluates PatchCore anomaly map, predicted object mask, and fused anomaly map.
3. Visualization output:
   Saves comparative multi-panel images:
   [Original Image | DINOv3 Feature PCA | Pseudo Mask | Predicted Mask | PatchCore Anomaly | Fused Anomaly | GT Mask]
4. Metrics reporting:
   Image-level AUROC, AUPR, best F1 + Pixel-level AUROC, AUPR.

Usage:
    python scripts/eval_segmentation.py --config configs/train_segmentation.yaml --out results/eval_seg
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone import Dinov3Backbone
from src.coreset import greedy_coreset
from src.data_kolektor import fixed_split, list_items, load_mask
from src.metrics import summarize
from src.pseudo_labels import UnsupervisedPseudoLabelGenerator
from src.scoring import bank_self_knn, fuse_anomaly_map, score_queries, smooth_map
from src.segmentation_head import AuxiliarySegmentationHead


def visualize_feature_pca(feats_chw: torch.Tensor, out_hw: tuple) -> np.ndarray:
    """Project high-dim DINOv3 feature channels to RGB via 3-component PCA."""
    c, hp, wp = feats_chw.shape
    tokens = feats_chw.permute(1, 2, 0).reshape(-1, c)
    tokens_centered = tokens - tokens.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(tokens_centered, q=3, center=False)
        rgb = (tokens_centered @ v[:, :3]).reshape(hp, wp, 3)
    except Exception:
        # Fallback to first 3 channels
        rgb = feats_chw[:3].permute(1, 2, 0)

    rgb = rgb.detach().cpu().numpy()
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
    resized = Image.fromarray(rgb_uint8).resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.array(resized)


def normalize_to_colormap(arr_2d: np.ndarray, colormap: str = "jet") -> Image.Image:
    """Convert float 2D map in [0, 1] to a colorful PIL Image."""
    arr_2d = np.clip(arr_2d, 0.0, 1.0)
    # Simple, dependency-free jet colormap lookup
    # 4 color points: 0.0 -> Blue, 0.33 -> Cyan, 0.66 -> Yellow, 1.0 -> Red
    r = np.clip(1.5 - np.abs(4.0 * arr_2d - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * arr_2d - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * arr_2d - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore + Auxiliary Segmentation.")
    parser.add_argument("--config", default="configs/train_segmentation.yaml", help="Path to config yaml.")
    parser.add_argument("--checkpoint", default=None, help="Path to head checkpoint (overrides config).")
    parser.add_argument("--out", default="results/segmentation_eval", help="Output directory.")
    parser.add_argument("--visualize", action="store_true", default=True, help="Save comparison visualizations.")
    parser.add_argument("--num_vis", type=int, default=20, help="Max test images to visualize.")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    os.makedirs(args.out, exist_ok=True)
    vis_dir = os.path.join(args.out, "visualizations")
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)

    w, h = cfg["data"]["width"], cfg["data"]["height"]
    preprocess = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["features"]["mean"], std=cfg["features"]["std"]),
    ])

    backbone = Dinov3Backbone(
        name=cfg["backbone"]["name"],
        out_indices=cfg["backbone"]["out_indices"],
        pretrained=cfg["backbone"]["pretrained"],
        pool=cfg["features"]["pool"],
    ).to(device)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    root = cfg["data"]["root"]
    train_files, test_files = fixed_split(
        list_items(root), n_train_good=cfg["data"]["train_good"], seed=cfg["seed"]
    )
    print(f"Dataset split: train={len(train_files)} (good only), test={len(test_files)}", flush=True)

    def embed(rel_path):
        img = preprocess(Image.open(os.path.join(root, rel_path)).convert("RGB"))
        return backbone.embed(img.unsqueeze(0).to(device))

    # Build PatchCore memory bank from train_files
    print("Extracting normal train features for PatchCore memory bank...", flush=True)
    bank_full = torch.cat([
        embed(f).permute(1, 2, 0).reshape(-1, backbone.out_channels) for f in train_files
    ], dim=0)
    print(f"Memory {tuple(bank_full.shape)} -> greedy coreset...", flush=True)
    bank = bank_full[greedy_coreset(
        bank_full, cfg["coreset"]["size"],
        proj_dim=cfg["coreset"]["proj_dim"],
        batch_add=cfg["coreset"]["batch_add"],
        seed=cfg["seed"], device=device,
    )].to(device).contiguous()
    del bank_full
    neighbors = bank_self_knn(bank, cfg["scoring"]["neighbors"])

    # Load Auxiliary Segmentation Head if enabled
    seg_enabled = cfg.get("segmentation", {}).get("enabled", True)
    head = None
    if seg_enabled:
        ckpt_path = args.checkpoint or cfg["segmentation"].get("checkpoint", "checkpoints/segmentation_head_best.pth")
        head = AuxiliarySegmentationHead(
            in_channels=backbone.out_channels,
            hidden_dim=cfg["segmentation"].get("hidden_dim", 256),
            num_classes=cfg["segmentation"].get("num_classes", 1),
            upsample_mode="bilinear",
        ).to(device)

        if os.path.exists(ckpt_path):
            print(f"Loading segmentation head weights from {ckpt_path}...", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device)
            head.load_state_dict(ckpt["segmentation_head"])
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Running with initialized head.", flush=True)
        head.eval()
    else:
        print("Segmentation head is DISABLED. Running exact baseline PatchCore evaluation.", flush=True)

    pseudo_gen = UnsupervisedPseudoLabelGenerator(
        method=cfg["pseudo_label"].get("clustering_type", "pca"),
        use_patchcore=False,  # For visualization on test images, use pure feature clustering to avoid memory overhead
        spatial_smooth=cfg["pseudo_label"].get("spatial_smooth", True),
        smooth_sigma=float(cfg["pseudo_label"].get("smooth_sigma", 2.0)),
        soft_labels=True,
    )

    mw, mh = cfg["metrics"]["pixel_width"], cfg["metrics"]["pixel_height"]
    img_true, img_score_pc, img_score_fused = [], [], []
    pix_true, pix_score_pc, pix_score_fused = [], [], []

    fusion_enabled = cfg.get("fusion", {}).get("enabled", True) and seg_enabled
    fusion_alpha = float(cfg.get("fusion", {}).get("alpha", 1.0))
    fusion_mode = cfg.get("fusion", {}).get("mode", "multiplicative")

    csv_path = os.path.join(args.out, "scores.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "is_defect", "score_pc", "score_fused"])

        vis_count = 0
        for i, rel_path in enumerate(test_files):
            feats = embed(rel_path)  # [C, hp, wp]
            _, _, fh, fw = (0, 0, *feats.shape[-2:])
            queries = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])
            raw, image_score = score_queries(queries, bank, neighbors)

            # PatchCore anomaly map
            anomaly = F.interpolate(
                raw.reshape(fh, fw).unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="bilinear", align_corners=False,
            )
            anomaly_map = smooth_map(
                anomaly.squeeze(), kernel=cfg["scoring"]["gauss_kernel"],
                sigma=cfg["scoring"]["gauss_sigma"],
            ).cpu().numpy()

            # Segmentation Head inference & Fusion
            pred_mask_np = None
            pseudo_mask_np = None
            if seg_enabled and head is not None:
                with torch.no_grad():
                    pred_logits = head(feats.unsqueeze(0), out_size=(h, w))
                    pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
                    pred_mask_np = pred_prob

                    if args.visualize and vis_count < args.num_vis:
                        pseudo_t = pseudo_gen.generate(feats.unsqueeze(0), target_size=(h, w))
                        pseudo_mask_np = pseudo_t.squeeze().cpu().numpy()

                if fusion_enabled:
                    fused_map = fuse_anomaly_map(
                        anomaly_map, pred_mask_np, alpha=fusion_alpha
                    )
                    # Image-level fused score: scaled by objectness at peak anomaly coordinate
                    top_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                    top_objectness = float(pred_mask_np[top_idx])
                    fused_image_score = float(image_score * (fusion_alpha + (1.0 - fusion_alpha) * top_objectness))
                else:
                    fused_map = anomaly_map
                    fused_image_score = image_score
            else:
                fused_map = anomaly_map
                fused_image_score = image_score

            # Ground truth mask (used ONLY for test evaluation metric)
            mask_path = os.path.join(root, rel_path.replace(".jpg", "_label.bmp"))
            gt_mask = load_mask(mask_path, w, h)
            defective = bool(gt_mask.any())

            img_true.append(int(defective))
            img_score_pc.append(image_score)
            img_score_fused.append(fused_image_score)
            writer.writerow([rel_path, int(defective), f"{image_score:.4f}", f"{fused_image_score:.4f}"])

            # Metric-resolution pixel maps
            small_mask = load_mask(mask_path, mw, mh)
            pix_true.append(small_mask.reshape(-1))

            small_map_pc = np.array(Image.fromarray(anomaly_map.astype(np.float32)).resize((mw, mh), Image.BILINEAR))
            pix_score_pc.append(small_map_pc.reshape(-1))

            small_map_fused = np.array(Image.fromarray(fused_map.astype(np.float32)).resize((mw, mh), Image.BILINEAR))
            pix_score_fused.append(small_map_fused.reshape(-1))

            # Visualizations (prioritize defective images and select normal images)
            if args.visualize and vis_count < args.num_vis and (defective or vis_count < 5):
                vis_count += 1
                orig_img = Image.open(os.path.join(root, rel_path)).convert("RGB").resize((w, h))
                pca_img = Image.fromarray(visualize_feature_pca(feats, (h, w)))

                # Normalize anomaly maps for display
                def to_vis_img(m):
                    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
                    return normalize_to_colormap(m_norm)

                pc_vis = to_vis_img(anomaly_map)
                fused_vis = to_vis_img(fused_map)
                pred_vis = to_vis_img(pred_mask_np if pred_mask_np is not None else np.zeros((h, w)))
                pseudo_vis = to_vis_img(pseudo_mask_np if pseudo_mask_np is not None else np.zeros((h, w)))
                gt_vis = Image.fromarray((gt_mask.astype(np.uint8) * 255)).convert("RGB")

                # Compose multi-panel grid: [Orig | DINO PCA | Pseudo | Pred Mask | PatchCore | Fused | GT]
                panels = [orig_img, pca_img, pseudo_vis, pred_vis, pc_vis, fused_vis, gt_vis]
                total_w = sum(p.width for p in panels)
                grid = Image.new("RGB", (total_w, h))
                curr_x = 0
                for p in panels:
                    grid.paste(p, (curr_x, 0))
                    curr_x += p.width

                safe_name = rel_path.replace("/", "_").replace(".jpg", ".png")
                grid.save(os.path.join(vis_dir, f"vis_{safe_name}"))

            if (i + 1) % 50 == 0:
                print(f"  evaluated {i + 1}/{len(test_files)}", flush=True)

    # Compute and display metrics
    y_true_pix = np.concatenate(pix_true)
    summary_pc = summarize(img_true, img_score_pc, y_true_pix, np.concatenate(pix_score_pc))
    print("\n--- BASELINE PATCHCORE RESULTS ---", flush=True)
    print(" ".join(f"{k}={v:.4f}" for k, v in summary_pc.items()), flush=True)

    summary_fused = None
    if fusion_enabled:
        summary_fused = summarize(img_true, img_score_fused, y_true_pix, np.concatenate(pix_score_fused))
        print("\n--- FUSED PATCHCORE + AUXILIARY SEGMENTATION RESULTS ---", flush=True)
        print(" ".join(f"{k}={v:.4f}" for k, v in summary_fused.items()), flush=True)

    result_txt_path = os.path.join(args.out, "result.txt")
    with open(result_txt_path, "w") as fh:
        fh.write("=== Baseline PatchCore ===\n")
        for k, v in summary_pc.items():
            fh.write(f"{k}={v:.4f}\n")
        if summary_fused is not None:
            fh.write("\n=== Fused (PatchCore + Segmentation) ===\n")
            for k, v in summary_fused.items():
                fh.write(f"{k}={v:.4f}\n")

    print(f"\nEvaluation complete. Results saved to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
