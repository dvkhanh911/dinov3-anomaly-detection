# dinov3-anomaly-detection

Industrial anomaly detection on **KolektorSDD** with a **DINOv3 ConvNeXt-Base** backbone, **PatchCore** (Roth et al., CVPR 2022) memory bank scoring, and an optional **Auxiliary Unsupervised Region Segmentation Head**.

---

## 1. Overview & Architecture

This repository explores unsupervised surface anomaly detection on industrial commutator images. The architecture combines frozen pre-trained foundation features from DINOv3 with PatchCore memory retrieval and an auxiliary lightweight head trained strictly on normal images without defect or ground-truth segmentation masks.

### Architecture Pipeline

```text
                                Input Image (512x1408)
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │  Frozen DINOv3 Backbone │
                             │  (ConvNeXt-Base, timm)  │
                             └────────────┬────────────┘
                                          │
                            768-Channel Spatial Feature Map
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
                   ▼                                             ▼
       ┌───────────────────────┐                     ┌───────────────────────┐
       │      Branch 1:        │                     │      Branch 2:        │
       │       PatchCore       │                     │ DINOv3 Feature PCA +  │
       │   Memory Bank (10k)   │                     │ PatchCore Spatial     │
       └───────────┬───────────┘                     │ Structural Guidance   │
                   │                                 └───────────┬───────────┘
                   ▼                                             │
             Patch Anomaly                                       ▼
                 Scores                              Pseudo Region Mask
                   │                                             │
                   ▼                                             ▼
             PatchCore Map                           ┌───────────────────────┐
                   │                                 │ Auxiliary             │
                   │                                 │ Segmentation Head     │
                   │                                 │ (Trainable: 2.36M)    │
                   │                                 └───────────┬───────────┘
                   │                                             │
                   │                                             ▼
                   │                                    Predicted Region Mask
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   Residual Fusion   │
                               │  Map * (α + (1-α)*M)│
                               └──────────┬──────────┘
                                          │
                                          ▼
                                 Fused Anomaly Map
                                          │
                                          ▼
                             Image-Level Anomaly Score
```

- **Frozen Backbone**: The DINOv3 backbone is completely frozen (`requires_grad = False`, eval mode) during segmentation head training.
- **Auxiliary Head Training**: Optimization updates only the lightweight Auxiliary Segmentation Head parameters.

---

## 2. Unsupervised Setting & Zero Leakage

- **Training Data**: Strictly 50 normal (good) images from the fixed KolektorSDD split.
- **No Ground-Truth Masks**: Ground-truth defect annotations and BMP segmentation masks are **never used** during training.
- **Pseudo Region Mask Generation**:
  - **Signal 1 (Objectness / Region Signal)**: Derived from the 1st principal component of DINOv3 patch features (or perimeter background contrast), delineating the object body from flat fixtures.
  - **Signal 2 (PatchCore Spatial Structural Guidance)**: Derived from patch nearest-neighbor distances against the normal memory bank, capturing local facet and structural contours.
  - **Refinement**: Both signals are min-max normalized per sample, convexly blended, spatially Gaussian-smoothed, and output as continuous soft pseudo region masks in $[0, 1]$.
  - The PatchCore spatial signal is an auxiliary continuous spatial cue, not a discrete semantic class label.

---

## 3. Dataset (KolektorSDD)

- **Total Images**: 399 electrical commutator surface images (boards `kos01` .. `kos50`).
- **Fixed Split** (seed 42):
  - **Train**: 50 normal (good) images.
  - **Test**: 349 images (52 defective + 297 normal).
- **Resolution**: $512 \times 1408$ pixels. Evaluation labels are binary masks where defects occupy roughly $0.1\% - 1\%$ of pixels.

---

## 4. DINOv3 Feature Extraction

- **Backbone**: `convnext_base.dinov3_lvd1689m` via `timm`.
- **Feature Hierarchies**: Stages 1 and 2 (`out_indices=(1, 2)`):
  - `stages.1`: 256 channels, spatial resolution $\frac{H}{8} \times \frac{W}{8}$ ($176 \times 64$).
  - `stages.2`: 512 channels, spatial resolution $\frac{H}{16} \times \frac{W}{16}$ ($88 \times 32$).
- **Aggregation**:
  - $3 \times 3$ average pooling (stride 1, padding 1) applied to both stages (PatchCore neighbourhood aggregation).
  - Deeper hierarchy (`stages.2`) is bilinearly upsampled to match `stages.1` resolution.
  - Concatenation along channel dimension yields a 768-channel feature map $[B, 768, 176, 64]$.

---

## 5. PatchCore Baseline

- **Memory Bank**: Normal patch features extracted from 50 training images.
- **Greedy Coreset**: Subsampled via Johnson-Lindenstrauss random projection (128-d) to a fixed memory bank size of shape `(10000, 768)` (~1.8% of patches).
- **Scoring**: Nearest-neighbour search with Eq. 7 local neighbourhood reweighting ($b=9$), bilinear interpolation to image size, and Gaussian smoothing ($\sigma=4.0$).

---

## 6. Auxiliary Segmentation Head

Designed as a lightweight 2D convolutional head receiving frozen DINOv3 spatial feature maps:

```text
Input Feature Map [B, 768, H_patch, W_patch]
         ↓
Conv2d(768, 256, kernel_size=3, padding=1) + ReLU
         ↓
Conv2d(256, 256, kernel_size=3, padding=1) + ReLU
         ↓
Conv2d(256, 1, kernel_size=1)
         ↓
Bilinear Upsampling to Image Resolution [B, 1, H, W]
```

- **Input**: $[B, 768, H_{patch}, W_{patch}]$
- **Output**: $[B, 1, H_{image}, W_{image}]$
- **Trainable Parameters**: 2,360,065 (~2.36M)
- **Backbone Trainable Parameters**: 0 (frozen)
- **Loss**: Combined loss $\mathcal{L} = \mathcal{L}_{pseudo} + \lambda \mathcal{L}_{consistency}$
  - $\mathcal{L}_{pseudo}$: BCE with logits against soft pseudo region masks.
  - $\mathcal{L}_{consistency}$: Invariance loss under horizontal flip with inverse spatial alignment.

---

### 7. Residual Fusion

To prevent overly aggressive suppression of true defects located near region boundaries, fusion adopts a residual formulation with baseline weight $\alpha \in [0, 1]$:

$$M_{\text{refined}} = M_{\text{anomaly}} \odot \left( \alpha + (1 - \alpha) \cdot M_{\text{seg}} \right)$$

- **When $\alpha = 1.0$**: $M_{\text{refined}} \equiv M_{\text{anomaly}}$ (exact PatchCore baseline).
- **When $\alpha < 1.0$**: Spurious background fixture anomalies are attenuated by factor $\alpha$, while true object anomalies are preserved.

---

## 8. Experimental Results

Evaluated on the single fixed split of KolektorSDD (50 good train, 349 test):

### Baseline PatchCore

| Metric | Score |
|---|---:|
| Image AUROC | 0.9468 |
| Image AUPR | 0.8160 |
| Image Best F1 | 0.7477 |
| Pixel AUROC | 0.9888 |
| Pixel AUPR | 0.2541 |

### PatchCore + Auxiliary Unsupervised Region Segmentation ($\alpha = 0.8$)

| Metric | Score |
|---|---:|
| Image AUROC | 0.9460 |
| Image AUPR | 0.8232 |
| Image Best F1 | 0.7647 |
| Pixel AUROC | 0.9908 |
| Pixel AUPR | 0.2664 |

### Observations

- **Image AUROC**: Changes marginally from 0.9468 to 0.9460 (-0.0008).
- **Image AUPR**: Increases from 0.8160 to 0.8232 (+0.0072).
- **Image Best F1**: Increases from 0.7477 to 0.7647 (+0.0170).
- **Pixel AUROC**: Increases from 0.9888 to 0.9908 (+0.0020).
- **Pixel AUPR**: Increases from 0.2541 to 0.2664 (+0.0123).
- **Note**: These numbers reflect a single experimental run with $\alpha = 0.8$ on KolektorSDD and do not constitute a generalized conclusion. Comprehensive ablation studies across diverse $\alpha$ values, pooling configurations, and datasets are required to systematically analyze the impact of residual region fusion.

---

## 9. Project Structure

```text
dinov3-anomaly-detection/
├── configs/
│   ├── baseline_dinov3.yaml          # PatchCore baseline configuration
│   └── train_segmentation.yaml       # Auxiliary segmentation & fusion configuration
├── scripts/
│   ├── eval_baseline.py              # Pure PatchCore baseline evaluation
│   ├── eval_segmentation.py          # Dual evaluation with fusion & visualizations
│   └── train_segmentation.py         # Training pipeline for segmentation head
├── src/
│   ├── __init__.py
│   ├── backbone.py                   # DINOv3 ConvNeXt feature extractor
│   ├── coreset.py                    # Greedy minimax coreset sampling
│   ├── data_kolektor.py              # KolektorSDD loader & dataset abstraction
│   ├── losses.py                     # Pseudo-mask BCE + spatially aligned consistency loss
│   ├── metrics.py                    # Image/pixel AUROC, AUPR, best-F1
│   ├── pseudo_labels.py              # Dual-signal unsupervised pseudo region generator
│   ├── scoring.py                    # PatchCore scoring & residual baseline fusion
│   └── segmentation_head.py          # Auxiliary 2D Conv segmentation head
├── tests/
│   ├── sanity_check.py               # 6-step audit and gradient isolation verification
│   └── test_auxiliary_segmentation.py# Comprehensive unit test suite
├── README.md
└── requirements.txt
```

---

## 10. Quick Start

### Installation

```bash
pip install -r requirements.txt
```

> **Note**: Requires Hugging Face access for DINOv3 model weights (`timm/convnext_base.dinov3_lvd1689m`).

### Verification & Sanity Checks

Run the 6-step audit verification:

```bash
python tests/sanity_check.py
```

### Training the Auxiliary Segmentation Head

Train strictly on normal images with frozen DINOv3 backbone:

```bash
python scripts/train_segmentation.py --config configs/train_segmentation.yaml
```

Checkpoints are saved to `checkpoints/segmentation_head_best.pth`.

### Evaluation & Visualization

Evaluate PatchCore + Auxiliary Segmentation with multi-panel comparison visualizations:

```bash
python scripts/eval_segmentation.py --config configs/train_segmentation.yaml --out results/segmentation_eval --visualize
```

### Running Original Baseline

To reproduce the unmodified PatchCore baseline:

```bash
python scripts/eval_baseline.py --config configs/baseline_dinov3.yaml --out results/baseline
```

Or set `segmentation.enabled: false` in `configs/train_segmentation.yaml`.

---

## References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Simeoni et al., *DINOv3*, 2025.
- Tabernik et al., *Segmentation-based deep-learning approach for surface-defect detection* (KolektorSDD), JIM 2020.
