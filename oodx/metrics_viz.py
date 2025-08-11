import numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

def compute_metrics(id_scores, ood_scores, name, outdir):
    id_s = id_scores.numpy(); ood_s = ood_scores.numpy()
    y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])  # 0=ID, 1=OOD
    y_score = np.concatenate([id_s, ood_s])

    auroc = roc_auc_score(y_true, y_score)
    aupr_out = average_precision_score(y_true, y_score)
    aupr_in  = average_precision_score(1 - y_true, -y_score)

    fpr_i, tpr_i, thr_i = roc_curve(1 - y_true, -y_score)
    idx = np.argmin(np.abs(tpr_i - 0.95))
    fpr95 = fpr_i[idx]; tnr95 = 1.0 - fpr95

    fpr_o, tpr_o, thr_o = roc_curve(y_true, y_score)
    fnr_id = 1 - tpr_o
    det_err = np.min(0.5*(fnr_id + fpr_o))

    Path(outdir).mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr_o, tpr_o, label=f"{name} (AUROC={auroc:.3f})")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR (OOD)"); plt.ylabel("TPR (OOD)")
    plt.title(f"ROC - {name}"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/roc_{name}.png", bbox_inches="tight"); plt.close()

    from sklearn.metrics import precision_recall_curve
    prec_o, rec_o, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec_o, prec_o, label=f"AUPR-Out={aupr_out:.3f}")
    plt.xlabel("Recall (OOD)"); plt.ylabel("Precision (OOD)")
    plt.title(f"PR (OOD positive) - {name}"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/pr_out_{name}.png", bbox_inches="tight"); plt.close()

    prec_i, rec_i, _ = precision_recall_curve(1 - y_true, -y_score)
    plt.figure()
    plt.plot(rec_i, prec_i, label=f"AUPR-In={aupr_in:.3f}")
    plt.xlabel("Recall (ID)"); plt.ylabel("Precision (ID)")
    plt.title(f"PR (ID positive) - {name}"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/pr_in_{name}.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.hist(id_s, bins=50, alpha=0.6, label="ID")
    plt.hist(ood_s, bins=50, alpha=0.6, label="OOD")
    plt.title(f"Score hist - {name} (higher => more OOD)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/hist_{name}.png", bbox_inches="tight"); plt.close()

    return {
        "AUROC": auroc,
        "AUPR_OUT": aupr_out,
        "AUPR_IN": aupr_in,
        "FPR@95TPR_ID": fpr95,
        "TNR@95TPR_ID": tnr95,
        "DetectionError": det_err
    }

def visualize_features(id_feats, ood_feats, outdir, title="Feature space"):
    X = torch.cat([id_feats, ood_feats]).numpy()
    y = np.array([0]*len(id_feats) + [1]*len(ood_feats))
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, metric="euclidean", random_state=42)
        Z = reducer.fit_transform(X); tag = "umap"
    else:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X); tag = "tsne"
    plt.figure()
    plt.scatter(Z[y==0,0], Z[y==0,1], s=6, alpha=0.6, label="ID")
    plt.scatter(Z[y==1,0], Z[y==1,1], s=6, alpha=0.6, label="OOD")
    plt.legend(); plt.title(f"{title} ({tag.upper()})"); plt.grid(True, alpha=0.3)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/features_{tag}.png", bbox_inches="tight"); plt.close()
