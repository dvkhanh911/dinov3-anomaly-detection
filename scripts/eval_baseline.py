"""Faithful-PatchCore + DINOv3 baseline eval on KolektorSDD (single split).

Reproduces: image_AUROC=0.9507 image_AUPR=0.8274 F1=0.7292
            pixel_AUROC=0.9887 pixel_AUPR=0.2232
Usage: python scripts/eval_baseline.py --config configs/baseline_dinov3.yaml
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone import Dinov3Backbone
from src.coreset import greedy_coreset
from src.data_kolektor import fixed_split, list_items, load_mask
from src.metrics import summarize
from src.scoring import bank_self_knn, score_queries, smooth_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_dinov3.yaml")
    parser.add_argument("--out", default="results/baseline")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

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

    root = cfg["data"]["root"]
    train_files, test_files = fixed_split(
        list_items(root), n_train_good=cfg["data"]["train_good"], seed=cfg["seed"])
    print(f"train={len(train_files)} test={len(test_files)}", flush=True)

    def embed(rel_path):
        img = preprocess(Image.open(os.path.join(root, rel_path)).convert("RGB"))
        return backbone.embed(img.unsqueeze(0).to(device))

    print("extracting train features...", flush=True)
    bank_full = torch.cat([
        embed(f).permute(1, 2, 0).reshape(-1, backbone.out_channels) for f in train_files
    ], dim=0)
    print(f"memory {tuple(bank_full.shape)} -> greedy coreset...", flush=True)
    bank = bank_full[greedy_coreset(
        bank_full, cfg["coreset"]["size"],
        proj_dim=cfg["coreset"]["proj_dim"],
        batch_add=cfg["coreset"]["batch_add"],
        seed=cfg["seed"], device=device)].to(device).contiguous()
    del bank_full
    neighbors = bank_self_knn(bank, cfg["scoring"]["neighbors"])

    mw, mh = cfg["metrics"]["pixel_width"], cfg["metrics"]["pixel_height"]
    img_true, img_score, pix_true, pix_score = [], [], [], []
    with open(os.path.join(args.out, "scores.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "is_defect", "score"])
        for i, rel_path in enumerate(test_files):
            feats = embed(rel_path)
            _, _, fh, fw = (0, 0, *feats.shape[-2:])
            queries = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])
            raw, image_score = score_queries(queries, bank, neighbors)
            anomaly = F.interpolate(
                raw.reshape(fh, fw).unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="bilinear", align_corners=False)
            anomaly = smooth_map(
                anomaly.squeeze(), kernel=cfg["scoring"]["gauss_kernel"],
                sigma=cfg["scoring"]["gauss_sigma"]).cpu().numpy()
            mask = load_mask(os.path.join(root, rel_path.replace(".jpg", "_label.bmp")), w, h)
            defective = bool(mask.any())
            img_true.append(int(defective))
            img_score.append(image_score)
            writer.writerow([rel_path, int(defective), f"{image_score:.4f}"])
            small_map = np.array(Image.fromarray(anomaly.astype(np.float32)).resize((mw, mh), Image.BILINEAR))
            small_mask = load_mask(
                os.path.join(root, rel_path.replace(".jpg", "_label.bmp")), mw, mh)
            pix_true.append(small_mask.reshape(-1))
            pix_score.append(small_map.reshape(-1))
            if (i + 1) % 50 == 0:
                print(f"  tested {i + 1}/{len(test_files)}", flush=True)

    summary = summarize(img_true, img_score, np.concatenate(pix_true), np.concatenate(pix_score))
    print("RESULT " + " ".join(f"{k}={v:.4f}" for k, v in summary.items()), flush=True)
    with open(os.path.join(args.out, "result.txt"), "w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}={v:.4f}\n")


if __name__ == "__main__":
    main()
