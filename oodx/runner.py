from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

from .utils import set_seed, get_device
from .data import build_loaders_from_kaggle, save_sample_grid
from .models import BackboneWithHead
from .train import train
from .eval_utils import collect_logits_feats, head_logits_from_feats
from . import scoring
from .metrics_viz import compute_metrics, visualize_features

def run(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    id_ds, ood_ds, id_loader, ood_loader = build_loaders_from_kaggle(
        args.id_dataset, args.ood_dataset, img_size=args.img_size,
        batch_size=args.batch_size, workers=args.workers
    )
    save_sample_grid(id_loader,  Path(args.outdir)/"id_samples.png",  "ID Dataset Sample Batch")
    save_sample_grid(ood_loader, Path(args.outdir)/"ood_samples.png", "OOD Dataset Sample Batch")

    num_classes = len(id_ds.classes)

    # Model
    model = BackboneWithHead(args, num_classes).to(device)

    # Train
    model = train(model, id_loader, ood_loader, args.epochs, args.lr, args.use_oe, args.oe_lambda, device, args)

    # Collect logits/features
    print("Collecting logits/features ...")
    id_logits, id_feats, y_id   = collect_logits_feats(model, id_loader, device)
    ood_logits, ood_feats, _    = collect_logits_feats(model, ood_loader, device)

    # Optional feature shaping (ReAct/ASH) and recompute logits
    if args.enable_react:
        print("Fitting ReAct threshold ...")
        react_thr = scoring.react_fit_percentile(id_feats, percent=args.react_percent)
        id_feats_r  = scoring.react_apply(id_feats, react_thr)
        ood_feats_r = scoring.react_apply(ood_feats, react_thr)
        id_logits_r  = head_logits_from_feats(model, id_feats_r)
        ood_logits_r = head_logits_from_feats(model, ood_feats_r)
    else:
        id_feats_r, ood_feats_r = id_feats, ood_feats
        id_logits_r, ood_logits_r = id_logits, ood_logits

    if args.enable_ash:
        print("Applying ASH ...")
        id_feats_r  = scoring.ash_apply(id_feats_r, p=args.ash_p)
        ood_feats_r = scoring.ash_apply(ood_feats_r, p=args.ash_p)
        id_logits_r  = head_logits_from_feats(model, id_feats_r)
        ood_logits_r = head_logits_from_feats(model, ood_feats_r)

    # Standard scores
    scores = {}
    scores["MSP"]       = (1.0 - torch.softmax(id_logits_r, dim=1).max(1).values,
                           1.0 - torch.softmax(ood_logits_r, dim=1).max(1).values)
    scores["MaxLogit"]  = (-id_logits_r.max(1).values, -ood_logits_r.max(1).values)
    scores["Energy_T1"] = (-( -torch.logsumexp(id_logits_r/1.0,  dim=1)),
                           -( -torch.logsumexp(ood_logits_r/1.0, dim=1)))
    scores["Energy_T2"] = (-( -torch.logsumexp(id_logits_r/2.0,  dim=1)),
                           -( -torch.logsumexp(ood_logits_r/2.0, dim=1)))

    # ODIN
    if args.enable_odin:
        print("Computing ODIN (pre-norm FGSM) ...")
        mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
        scores["ODIN"] = (
            scoring.odin_pre_norm_score(model, id_loader,  device, args.odin_eps, args.odin_temp, mean, std),
            scoring.odin_pre_norm_score(model, ood_loader, device, args.odin_eps, args.odin_temp, mean, std),
        )

    # Mahalanobis / kNN
    print("Fitting Mahalanobis ...")
    means, inv_cov = scoring.fit_mahalanobis(id_feats_r, y_id, num_classes)
    scores["Mahalanobis"] = (scoring.score_mahalanobis(id_feats_r, means, inv_cov),
                             scoring.score_mahalanobis(ood_feats_r, means, inv_cov))

    print("Fitting kNN ...")
    nbrs = scoring.fit_knn(id_feats_r, n_neighbors=max(50, min(200, len(id_feats_r)//4)))
    scores["kNN_k10"] = (scoring.score_knn(nbrs, id_feats_r, 10),
                         scoring.score_knn(nbrs, ood_feats_r, 10))

    # ViM
    if args.enable_vim:
        print("Fitting ViM ...")
        mu, v = scoring.vim_fit(id_feats_r, id_logits_r)
        scores["ViM"] = (scoring.vim_score(mu, v, id_feats_r, id_logits_r, alpha=args.vim_alpha),
                         scoring.vim_score(mu, v, ood_feats_r, ood_logits_r, alpha=args.vim_alpha))

    # AdaSCALE heuristic
    if args.enable_adascale:
        print("Computing AdaSCALE heuristic ...")
        scores["AdaSCALE"] = (scoring.adascale_score(id_feats_r, id_logits_r, gamma=args.adascale_gamma),
                              scoring.adascale_score(ood_feats_r, ood_logits_r, gamma=args.adascale_gamma))

    # Metrics + plots
    all_metrics = {}
    for name, (sid, sood) in scores.items():
        print(f"\n=== {name} Metrics ===")
        m = compute_metrics(sid, sood, name=name, outdir=args.outdir)
        for k, v in m.items():
            print(f"{k:>15}: {v:.4f}")
        all_metrics[name] = m

    # Feature viz
    print("\nRendering feature visualizations ...")
    visualize_features(id_feats, ood_feats, outdir=args.outdir, title="Backbone penultimate features")

    # Summary
    print("\n==== SUMMARY (↑ AUROC/AUPR/TNR95, ↓ FPR95/DetErr) ====")
    header = f"{'Method':12} | AUROC  | AUPR_O | AUPR_I | FPR@95 | TNR@95 | DetErr"
    print(header); print("-"*len(header))
    for name, m in all_metrics.items():
        print(f"{name:12} | {m['AUROC']:.4f} | {m['AUPR_OUT']:.4f} | {m['AUPR_IN']:.4f} | "
              f"{m['FPR@95TPR_ID']:.4f} | {m['TNR@95TPR_ID']:.4f} | {m['DetectionError']:.4f}")
    print(f"\nSaved figures to: {args.outdir}/")
