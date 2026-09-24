"""Image/pixel AUROC + AUPR + best-F1 (F1 swept over score quantiles)."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize(y_true_img, y_score_img, y_true_pix, y_score_pix, n_thresholds=60):
    y_true_img = np.asarray(y_true_img)
    y_score_img = np.asarray(y_score_img)
    img_auroc = roc_auc_score(y_true_img, y_score_img)
    img_aupr = average_precision_score(y_true_img, y_score_img)
    best_f1, best_thr = 0.0, 0.0
    for thr in np.quantile(y_score_img, np.linspace(0.05, 0.95, n_thresholds)):
        pred = (y_score_img >= thr).astype(int)
        tp = int(((pred == 1) & (y_true_img == 1)).sum())
        fp = int(((pred == 1) & (y_true_img == 0)).sum())
        fn = int(((pred == 0) & (y_true_img == 1)).sum())
        f1 = 2 * tp / max(1, (2 * tp + fp + fn))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    pix_auroc = roc_auc_score(y_true_pix, y_score_pix)
    pix_aupr = average_precision_score(y_true_pix, y_score_pix)
    return {
        "image_AUROC": img_auroc, "image_AUPR": img_aupr,
        "image_bestF1": best_f1, "image_thr": best_thr,
        "pixel_AUROC": pix_auroc, "pixel_AUPR": pix_aupr,
    }
