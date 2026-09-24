# dinov3-anomaly-detection

Industrial anomaly detection on **KolektorSDD** with a **DINOv3 ConvNeXt-Base** backbone
and faithful **PatchCore** (Roth et al., CVPR 2022) scoring.

## Baseline result (KolektorSDD)

| Setup | imgAUROC | imgAUPR | F1 | pixAUROC | pixAUPR |
|---|---|---|---|---|---|
| Faithful PatchCore (greedy + Eq.7) + DINOv3 | **0.9507** | **0.8274** | **0.7292** | 0.9887 | 0.2537 |
| Approx (random bank + max) + DINOv3 | 0.9375 | 0.7639 | 0.7083 | 0.9903 | 0.2232 |
| Faithful PatchCore + WideResNet50 (paper recipe) | 0.8452 | 0.5156 | 0.5294 | 0.9363 | 0.1364 |

Protocol: 512x1408 input, train = 50 good images, test = 52 defective + 297 good
(fixed seed 42). Tiled 27-tile 3-fold CV mean: imgAUROC 0.9247 (faithful) vs 0.9109 (approx).

## Method

- Backbone: `timm` DINOv3-pretrained ConvNeXt-Base (`convnext_base.dinov3_lvd1689m`),
  hierarchies `stages.1` + `stages.2`, 3x3 avg-pool neighbourhood (PatchCore p=3, s=1).
- Coreset: greedy minimax facility location (Alg.1) with JL projection to 128-d
  (official default), 10k patches (~PatchCore-1.8%).
- Scoring: nearest-neighbour + Eq.7 reweighting (b=9, stable-softmax form),
  anomaly map bilinear-upsampled + Gaussian smoothing (sigma=4).
- KolektorSDD labels are BMP masks with values {0, 1} (binarize with `> 0`);
  defects cover only ~0.1-1% of pixels. Train/test split keeps seed 42 fixed.

## Run

```bash
pip install -r requirements.txt
# point configs/baseline_dinov3.yaml:data:root at your KolektorSDD folder (kos01..kos50)
python scripts/eval_baseline.py --config configs/baseline_dinov3.yaml --out results/baseline
```

Needs the DINOv3 license accepted on Hugging Face (timm downloads weights from `timm/convnext_base.dinov3_lvd1689m`).

## References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
  Code: https://github.com/amazon-research/patchcore-inspection
- Simeoni et al., *DINOv3*, 2025. https://github.com/facebookresearch/dinov3
- Tabernik et al., *Segmentation-based deep-learning approach for surface-defect detection*
  (KolektorSDD), JIM 2020. https://www.vicos.si/resources/kolektorsdd
- Rolih et al., *Divide and Conquer* (tiled ensemble), CVPRW 2024.
- Jiang et al., *SoftPatch*, NeurIPS 2022 (tried: degrades on clean Kolektor, see issues).
